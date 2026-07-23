"""V50 training sanity watcher: polls active_training.log and, each time hands-trained crosses
a new 50k boundary (and at the end), appends the RAW latest telemetry block plus a compact
computed sanity line to versions/v50/sanity_50k.md -- so the run can be evaluated in 50k steps
the next morning. Read-only on the training run; exits at 250k, on a dead/stale log, or a 13h cap.

Run (background):  .venv/Scripts/python.exe tools/training_monitor/sanity_50k_watch.py
"""
import os
import re
import time

REPO = r"c:\REPO\Antigravity\AIPoker"
LOG = os.path.join(REPO, "active_training.log")
OUT = os.path.join(REPO, "versions", "v50", "sanity_50k.md")
CKPT_DIR = os.path.join(REPO, "versions", "v50", "weights", "checkpoints")

TARGET = 250000
STEP = 50000
POLL_S = 60
STALE_EXIT_S = 40 * 60          # log unchanged this long -> assume run ended
MAX_RUNTIME_S = 13 * 3600


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def tail(path, n=70):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()[-n:]
    except OSError:
        return []


def latest_hands(lines):
    h = None
    for ln in lines:
        m = re.search(r"Hands Simulated:\s*([\d,]+)\s*/", ln)
        if m:
            h = int(m.group(1).replace(",", ""))
    return h


def grab(lines, pat, cast=str):
    for ln in reversed(lines):
        m = re.search(pat, ln)
        if m:
            try:
                return cast(m.group(1))
            except (ValueError, TypeError):
                return None
    return None


def sanity_line(lines):
    val = grab(lines, r"Val Loss:\s*([\d.]+)", float)
    trn = grab(lines, r"Train Loss:\s*([\d.]+)", float)
    hero = grab(lines, r"Seat 0 Hero.*?:\s*([+\-][\d.]+)\s*BB/100", float)
    ent = grab(lines, r"Action Entropy:\s*([\d.]+)", float)
    avg_act = grab(lines, r"Avg Active Players:\s*([\d.]+)", float)
    speed = grab(lines, r"Sim Speed:\s*([\d,]+)\s*hands/sec", lambda s: int(s.replace(",", "")))
    # action usage line: Fold x% | Call x% | r33 x% | r66 x% | rPot x% | All-In x%
    au = {}
    for k in ("Fold", "Call", "r33", "r66", "rPot", "All-In"):
        au[k] = grab(lines, rf"{re.escape(k)}\s+([\d.]+)%", float)
    raises = sum(v for k, v in au.items() if k in ("r33", "r66", "rPot") and v is not None)
    # equity->action monotonicity: air should mostly fold, nuts should mostly commit
    air_fold = grab(lines, r"<20%.*?\|\s*([\d.]+)%", float)
    nuts_line = next((ln for ln in reversed(lines) if ">80%" in ln), "")
    nm = re.findall(r"([\d.]+)%", nuts_line)
    nuts_commit = None
    if len(nm) >= 6:               # fold call r33 r66 rPot allin
        nuts_commit = float(nm[2]) + float(nm[3]) + float(nm[4]) + float(nm[5])
    flags = []
    if air_fold is not None and air_fold < 70:
        flags.append("air-fold<70%")
    if nuts_commit is not None and nuts_commit < 50:
        flags.append("nuts-commit<50%")
    if raises < 3:
        flags.append("raise-buckets<3%(middle-gear?)")
    if ent is not None and ent < 0.5:
        flags.append("entropy<0.5(collapse?)")
    verdict = "OK" if not flags else "WATCH: " + ", ".join(flags)
    return (f"SANITY: val={val} train={trn} heroBB/100={hero} entropy={ent} "
            f"avgActive={avg_act} speed={speed}/s | usage fold={au['Fold']} call={au['Call']} "
            f"r33={au['r33']} r66={au['r66']} rPot={au['rPot']} allin={au['All-In']} "
            f"(raise-buckets={raises:.1f}%) | airFold={air_fold}% nutsCommit={nuts_commit}% "
            f"-> {verdict}")


def checkpoints():
    try:
        return sorted(f for f in os.listdir(CKPT_DIR) if f.endswith(".pth"))
    except OSError:
        return []


def snapshot(threshold, lines, note=""):
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(f"\n\n## ~{threshold//1000}k hands  ({now()}){note}\n\n")
        f.write(sanity_line(lines) + "\n")
        f.write(f"checkpoints present: {checkpoints()}\n\n")
        f.write("```\n" + "".join(lines) + "\n```\n")


def main():
    if not os.path.exists(OUT):
        with open(OUT, "w", encoding="utf-8") as f:
            f.write("# V50 training sanity checks (every ~50k hands)\n\n"
                    "Raw `active_training.log` telemetry block captured at each 50k crossing, "
                    "plus a computed one-line sanity read. Newest sections appended at the bottom.\n"
                    "Battery candidates: the `main_hands*.pth` under `weights/checkpoints/`.\n")
    next_th = STEP
    started = time.time()
    last_size = -1
    last_change = time.time()
    while True:
        time.sleep(POLL_S)
        if time.time() - started > MAX_RUNTIME_S:
            break
        try:
            size = os.path.getsize(LOG)
        except OSError:
            continue
        if size != last_size:
            last_size, last_change = size, time.time()
        lines = tail(LOG)
        hands = latest_hands(lines)
        if hands is None:
            continue
        while hands >= next_th and next_th <= TARGET:
            snapshot(next_th, lines)
            next_th += STEP
        if hands >= TARGET:
            break
        if time.time() - last_change > STALE_EXIT_S:
            snapshot(hands, lines, note=" [LOG WENT STALE -- run likely ended early]")
            break


if __name__ == "__main__":
    main()

import sys
import os
import json
import re
import time
import ast

HISTORY_FILENAME = "telemetry_history.json"
MAX_HISTORY_TICKS = 50


def _acquire_lock(lock_path, timeout=5.0, poll=0.05):
    """Exclusive-create lockfile mutex. The watcher's background poll and any manual
    invocation of this script both write telemetry_history.json -- without this, two
    concurrent read-modify-write cycles can race and silently drop ticks (lost update)."""
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            if time.time() > deadline:
                return False
            time.sleep(poll)


def _release_lock(lock_path):
    try:
        os.remove(lock_path)
    except OSError:
        pass


def append_history(script_dir, telemetry):
    """Persist telemetry ticks to disk (FIFO, capped) so the dashboard's history
    survives a browser refresh or a server restart -- it is no longer rebuilt
    purely in-memory in the client JS."""
    history_path = os.path.join(script_dir, HISTORY_FILENAME)
    lock_path = history_path + ".lock"

    if not _acquire_lock(lock_path):
        return  # another writer is mid-update; skip this tick rather than risk a race

    try:
        history = []
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r', encoding='utf-8') as hf:
                    loaded = json.load(hf)
                if isinstance(loaded, list):
                    history = loaded
            except (json.JSONDecodeError, OSError):
                history = []

        # Dedup: only append if progress actually advanced since the last recorded tick
        # (the watcher polls every 5s regardless of whether the log advanced).
        if not history or history[-1].get("progress") != telemetry.get("progress"):
            history.append(telemetry)
            if len(history) > MAX_HISTORY_TICKS:
                history = history[-MAX_HISTORY_TICKS:]
            # Write-then-rename so a concurrent dashboard fetch never sees a half-written file.
            tmp_path = history_path + ".tmp"
            with open(tmp_path, 'w', encoding='utf-8') as hf:
                json.dump(history, hf, indent=2)
            os.replace(tmp_path, history_path)
    finally:
        _release_lock(lock_path)

def parse_run_header(lines):
    """The opponent pool / live-player-count banner is printed once per run (before the
    repeating dashboard blocks), identically across versions v12-v16:
        Opponent Pool:   ['fish', 'tag', 'nit', 'past']  weights=[0.4, 0.2, 0.2, 0.2]
        Live Players:    6  (Hero + 5 opponents)
    Scan the FULL file (not just the latest dashboard block) and take the LAST match, so a
    log that got appended across multiple launches reflects the currently-running config."""
    header = {}
    for line in lines:
        if "Opponent Pool:" in line:
            m = re.search(r"Opponent Pool:\s*(\[.*?\])\s*weights=(\[.*?\])", line)
            if m:
                try:
                    header["opponent_pool"] = ast.literal_eval(m.group(1))
                    header["opponent_pool_weights"] = ast.literal_eval(m.group(2))
                except (ValueError, SyntaxError):
                    pass
        elif "Live Players:" in line:
            m = re.search(r"Live Players:\s*(\d+)\s*\(Hero \+ (\d+) opponents\)", line)
            if m:
                header["configured_seats"] = int(m.group(1))
                header["configured_opponents"] = int(m.group(2))
    return header


def parse_training_log(logfile):
    if not os.path.exists(logfile):
        print(f"Error: Log file {logfile} not found.")
        return

    try:
        with open(logfile, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(logfile, 'r', encoding='utf-16') as f:
            lines = f.readlines()

    # Find the last dashboard block
    start_idx = -1
    end_idx = -1

    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if line.startswith("+===============") and "Loss Q" in lines[i-1] if i > 0 else False:
            end_idx = i
        elif line.startswith("+===============") and "SELF-PLAY" in lines[i+1] if i < len(lines)-1 else False:
            if end_idx != -1:
                start_idx = i
                break

    if start_idx != -1 and end_idx != -1:
        dashboard = "".join(lines[start_idx:end_idx+1])
        print("```text\n" + dashboard.strip() + "\n```")
        
        # Parse into JSON
        telemetry = {
            "progress": "",
            "progress_percent": "",
            "epoch": "",
            "stage": "",
            "eta": "",
            "train_loss": "",
            "val_loss": "",
            "entropy": "",
            "seats": [],
            "equity_matrix": []
        }
        
        dash_lines = dashboard.strip().split("\n")
        for line in dash_lines:
            if "Active Personality:" in line:
                m = re.search(r"Active Personality:\s+(\w+)", line)
                if m: telemetry["personality"] = m.group(1).strip()
            elif "Hands Simulated:" in line:
                m = re.search(r"Hands Simulated:\s+([0-9,]+).*?\(\s*([\d.]+)%\)", line)
                if m:
                    telemetry["progress"] = m.group(1)
                    telemetry["progress_percent"] = m.group(2)
            elif "Training Epoch:" in line:
                m = re.search(r"Training Epoch:\s+(\d+)", line)
                if m: telemetry["epoch"] = m.group(1)
            elif "Curriculum Stage:" in line:
                m = re.search(r"Curriculum Stage:\s+(.*?)\s*\|", line)
                if m: telemetry["stage"] = m.group(1).strip()
            elif "ETA:" in line:
                m = re.search(r"ETA:\s+(.*?)\s*\|", line)
                if m: telemetry["eta"] = m.group(1).strip()
            elif "Action Entropy:" in line:
                m = re.search(r"Action Entropy:\s+([\d.]+)", line)
                if m: telemetry["entropy"] = m.group(1)
            elif "Train Loss" in line and "Val Loss" in line:
                m = re.search(r"Train Loss.*?:\s+([\d.]+).*?Val Loss:\s+([\d.]+)", line)
                if m:
                    telemetry["train_loss"] = m.group(1)
                    telemetry["val_loss"] = m.group(2)
            elif "Loss Q:" in line and "Pi:" in line:
                m = re.search(r"Loss Q:\s*([\d.]+)\s*\|\s*Pi:\s*([\d.]+)\s*\|\s*Bluff:\s*([\d.]+)\s*\|\s*Str:\s*([\d.]+)\s*\|\s*Eq:\s*([\d.]+)", line)
                if m:
                    telemetry["loss_breakdown"] = {
                        "q": m.group(1), "pi": m.group(2), "bluff": m.group(3),
                        "str": m.group(4), "eq": m.group(5)
                    }
            elif "Seat " in line and "BB/100" in line:
                m = re.search(r"-\s+(Seat\s+\d+\s+[^:]+):\s+([\+\-]?[\d.]+)\s+BB/100\s+\(VPIP:\s*([\d.]+)%\s+AGG:\s*([\d.]+)%\)\s+\[R:(\d+)\s+F:(\d+)\s+AI:(\d+)", line)
                if m:
                    telemetry["seats"].append({
                        "name": m.group(1).strip(),
                        "bb100": float(m.group(2)),
                        "vpip": float(m.group(3)),
                        "agg": float(m.group(4)),
                        "r": int(m.group(5)),
                        "f": int(m.group(6)),
                        "ai": int(m.group(7))
                    })
            elif "|  EXPLOITATION SCOREBOARD (Net BB/100 Matrix):" in line:
                if "exploitation_matrix" not in telemetry:
                    telemetry["exploitation_matrix"] = []
            elif "exploitation_matrix" in telemetry and line.startswith("|") and ("Hero " in line or "Seat " in line or "Opp " in line) and "Winner" not in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 7:
                    row_name = parts[0]
                    # Handle cell values (some might be '-')
                    def parse_val(v):
                        if v == '-': return 0.0
                        try: return float(v)
                        except: return 0.0
                    
                    telemetry["exploitation_matrix"].append({
                        "name": row_name,
                        "hero": parse_val(parts[1]),
                        "s1": parse_val(parts[2]),
                        "s2": parse_val(parts[3]),
                        "s3": parse_val(parts[4]),
                        "s4": parse_val(parts[5]),
                        "s5": parse_val(parts[6])
                    })
            elif "%" in line and ("<20%" in line or "20-40%" in line or "40-60%" in line or "60-80%" in line or ">80%" in line):
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 4:
                    bracket = parts[0]
                    # The action %-columns run from parts[1] until the first non-% cell (avg end
                    # street). Detecting them dynamically handles BOTH the old 5-action layout
                    # (Fold/Call/Raise/RR/All-In) and the V14 6-action layout (Fold/Call/r33/r66/
                    # rPot/All-In) without breaking the currently-running old-format log.
                    pct = []
                    j = 1
                    while j < len(parts) and parts[j].endswith('%'):
                        pct.append(parts[j])
                        j += 1
                    rest = parts[j:]  # [n_hands], [free], avg_end_street, net_chips, [won], [lost]
                    if len(pct) == 6:
                        labels = ['Fold', 'Call', 'r33', 'r66', 'rPot', 'All-In']
                    elif len(pct) == 5:
                        labels = ['Fold', 'Call', 'Raise', 'RR', 'All-In']
                    else:
                        labels = [f'A{k}' for k in range(len(pct))]
                    # N Hands and Free are both plain integers (possibly comma-grouped, e.g.
                    # "1,234"), unlike avg_end_street/net_chips/won/lost which always contain a '.'
                    # (printed with .1f/+7.1f/+6.0f -- +6.0f still has no '.', but is signed, so the
                    # plain-unsigned-integer check below still tells them apart) -- detect each
                    # positionally so this stays backward-compatible with logs from any prior format
                    # (no N Hands column, N Hands but no Free, or both).
                    is_int_cell = lambda v: re.fullmatch(r"[\d,]+", v or "") is not None
                    has_n_hands = len(rest) > 0 and is_int_cell(rest[0])
                    n_hands = rest[0] if has_n_hands else ""
                    rest2 = rest[1:] if has_n_hands else rest
                    has_free = len(rest2) > 0 and is_int_cell(rest2[0])
                    free_checks = rest2[0] if has_free else ""
                    rest3 = rest2[1:] if has_free else rest2
                    telemetry["equity_matrix"].append({
                        "bracket": bracket,
                        "action_cols": [[lab, val] for lab, val in zip(labels, pct)],
                        "n_hands": n_hands,
                        "free_checks": free_checks,
                        "avg_end_street": rest3[0] if len(rest3) > 0 else "",
                        "net_chips": rest3[1] if len(rest3) > 1 else "",
                        "won_chips": rest3[2] if len(rest3) > 2 else "",
                        "lost_chips": rest3[3] if len(rest3) > 3 else ""
                    })
            elif "ACTION USAGE (all decisions)" in line:
                # V14 size-selection histogram over all hero decisions.
                vals = re.findall(r"([\d.]+)%", line)
                labs = ['Fold', 'Call', 'r33', 'r66', 'rPot', 'All-In']
                if len(vals) >= 6:
                    telemetry["action_usage"] = [[labs[i], vals[i] + '%'] for i in range(6)]
            elif "ALL-IN WinRate" in line:
                mw = re.search(r"ALL-IN WinRate\s+([\d.]+)%\s+\(n=(\d+)\)", line)
                if mw:
                    telemetry["allin_winrate"] = mw.group(1) + '%'
                    telemetry["allin_n"] = mw.group(2)
                jm = re.findall(r"(Blue|Green|Yellow|Red)\s+([\d.]+)%", line)
                if jm:
                    telemetry["jam_by_color"] = [[c, v + '%'] for c, v in jm]

        # Seat activity: a "Live Players: 6" run still prints a fixed 6-row seat table even when
        # the sampled Opponent Pool excludes some archetypes (e.g. no 'maniac') -- those rows stay
        # at all-zero R/F/AI for the whole run. Flag per-seat + summarize how many are actually
        # playing, rather than leaving it to be re-derived ad hoc by the dashboard.
        for idx, seat in enumerate(telemetry["seats"]):
            seat["active"] = (idx == 0) or (seat["r"] + seat["f"] + seat["ai"] > 0)
        telemetry["active_seat_count"] = sum(1 for s in telemetry["seats"] if s["active"])

        # configured_seats/configured_opponents (below) come from the run-header banner and are
        # the run's designed table size; may be absent on very old logs without that banner.
        telemetry.update(parse_run_header(lines))

        # Write JSON to same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(script_dir, "telemetry.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(telemetry, jf, indent=2)

        append_history(script_dir, telemetry)

    else:
        print("Could not find a complete dashboard block in the log.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_training_log.py <logfile>")
    else:
        parse_training_log(sys.argv[1])

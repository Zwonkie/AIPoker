"""
Verifies the single-telemetry-file contract documented in .agents/AGENTS.md:
every versions/*/self_play/train.py must tee stdout to <repo_root>/active_training.log,
which is the one file this dashboard watches. Run this after scaffolding a new version
(e.g. copying v16 -> v17) to catch a dropped or edited Tee line before it trains silently
out of the dashboard's view.

Read-only: does not touch any training script or log.
"""
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Matches: os.path.join(repo_root, 'active_training.log') / os.path.join(..., "active_training.log")
CONTRACT_RE = re.compile(r"""os\.path\.join\([^)]*['"]active_training\.log['"]\)""")
TEE_RE = re.compile(r"""sys\.stdout\s*=\s*Tee\(""")


def find_train_scripts():
    versions_dir = os.path.join(REPO_ROOT, 'versions')
    if not os.path.isdir(versions_dir):
        return
    for version in sorted(os.listdir(versions_dir)):
        script = os.path.join(versions_dir, version, 'self_play', 'train.py')
        if os.path.isfile(script):
            yield version, script


def check(version, script_path):
    with open(script_path, 'r', encoding='utf-8') as f:
        src = f.read()

    has_path = bool(CONTRACT_RE.search(src))
    has_tee = bool(TEE_RE.search(src))

    if has_path and has_tee:
        return True, "OK"
    missing = []
    if not has_path:
        missing.append("no 'active_training.log' path built from repo_root")
    if not has_tee:
        missing.append("no 'sys.stdout = Tee(...)' redirect")
    return False, "; ".join(missing)


def main():
    results = list(find_train_scripts())
    if not results:
        print("No versions/*/self_play/train.py scripts found.")
        return 1

    ok_count = 0
    for version, script_path in results:
        passed, detail = check(version, script_path)
        status = "OK  " if passed else "FAIL"
        print(f"[{status}] {version:<10} {detail}")
        if passed:
            ok_count += 1

    print(f"\n{ok_count}/{len(results)} versions comply with the single-telemetry-file contract.")
    return 0 if ok_count == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())

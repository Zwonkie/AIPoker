"""Model verification CLI -- the single command to run after any training run.

Loads a version's model generically via shared/registry.py (no version-specific code needed
here), runs the FAST scenario-check curriculum from checks.py (always), and optionally the
SLOW simulated-hand checks (--full). Prints a PASS/WARN/FAIL/SKIP table and exits non-zero on
any FAIL so this can gate a workflow later if desired.

Usage:
  .venv/Scripts/python.exe -m tools.model_verify.run --version v16
  .venv/Scripts/python.exe -m tools.model_verify.run --version v16 --full
  .venv/Scripts/python.exe -m tools.model_verify.run --version v16 --full --update-baseline
"""
import argparse
import glob
import importlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from shared.registry import get_manifest, load_model
from tools.model_verify.checks import FAST_CHECKS, SLOW_CHECKS, RunCtx, CHECK_DOCS

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_BASELINES_PATH = os.path.join(os.path.dirname(__file__), 'baselines.json')


def _load_baselines():
    if os.path.exists(_BASELINES_PATH):
        with open(_BASELINES_PATH, 'r') as f:
            return json.load(f)
    return {}


def _save_baselines(baselines):
    with open(_BASELINES_PATH, 'w') as f:
        json.dump(baselines, f, indent=2, sort_keys=True)


def _find_frozen_predecessor(version_id):
    """Pick the frozen-predecessor weights file for `beats_frozen_predecessor`.

    Fixed 2026-07-16 (found while evaluating v17_gauntlet, which deliberately keeps THREE
    frozen_*.pth files -- frozen_v15/v16/v17 -- as opponent-pool inputs, not just one).
    Alphabetical-first (the old behavior) picks the OLDEST file (frozen_v15 < frozen_v16 <
    frozen_v17), which is almost always the wrong benchmark -- a version's predecessor
    comparison should be its most recent frozen ancestor. Prefer the HIGHEST parsed version
    number; fall back to alphabetical-last, then alphabetical-first, if none parse.
    """
    weights_dir = os.path.join(_REPO_ROOT, 'versions', version_id, 'weights')
    hits = sorted(glob.glob(os.path.join(weights_dir, 'frozen_*.pth')))
    if not hits:
        return None
    def _version_num(path):
        m = re.search(r'v(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else -1
    best = max(hits, key=_version_num)
    return os.path.basename(best)


def _load_range_aware_flag(version_id):
    import yaml
    config_path = os.path.join(_REPO_ROOT, 'versions', version_id, 'self_play', 'config.yaml')
    if not os.path.exists(config_path):
        return False
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    return bool((cfg.get('training') or {}).get('range_aware_equity', False))


def main():
    parser = argparse.ArgumentParser(description="Run the model_verify curriculum against a trained version.")
    parser.add_argument('--version', required=True, help="Version id, e.g. v16")
    parser.add_argument('--weights', default='expert_main.pth', help="Weights filename within the version's weights/ dir")
    parser.add_argument('--full', action='store_true', help="Also run the SLOW simulated-hand checks")
    parser.add_argument('--n-hands-style', type=int, default=3000, help="Hands per field for the VPIP-vs-style check")
    parser.add_argument('--n-hands-field', type=int, default=4000, help="Hands per field for BB/100 + frozen-predecessor checks")
    parser.add_argument('--update-baseline', action='store_true', help="Write measured BB/100 numbers as the new baseline for this version")
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dump-json', default=None,
                        help="Write full results (incl. raw per-scenario data) to this path. "
                             "Defaults to tools/model_verify/results/<version>__<weights>.json")
    args = parser.parse_args()

    manifest = get_manifest(args.version)
    model = load_model(args.version, args.weights, device=args.device)
    action_keys = tuple(manifest.action_space)

    rc = RunCtx(version_id=args.version, model=model, manifest=manifest, action_keys=action_keys,
                device=args.device, baselines=_load_baselines(),
                n_hands_style=args.n_hands_style, n_hands_field=args.n_hands_field)

    print(f"model_verify | version={args.version} weights={args.weights} action_space={action_keys}")
    print(f"{'':2}{'check':<28} {'status':<6} detail")
    print("-" * 100)

    results = []

    for check_id, description, issue_ref, fn in FAST_CHECKS:
        result = fn(rc)
        results.append((check_id, result))
        print(f"  {check_id:<28} {result.status:<6} {result.detail}")

    if args.full:
        rc.sim_module = importlib.import_module(f"versions.{args.version}.self_play.simulator")
        rc.range_aware = _load_range_aware_flag(args.version)
        rc.frozen_predecessor_filename = _find_frozen_predecessor(args.version)
        print("-" * 100)
        for check_id, description, issue_ref, fn in SLOW_CHECKS:
            result = fn(rc)
            results.append((check_id, result))
            print(f"  {check_id:<28} {result.status:<6} {result.detail}")
    else:
        print(f"  (skipped {len(SLOW_CHECKS)} slow checks -- pass --full to include them)")

    print("-" * 100)
    counts = {'PASS': 0, 'WARN': 0, 'FAIL': 0, 'SKIP': 0}
    for _, r in results:
        counts[r.status] += 1
    print(f"  {counts['PASS']} PASS, {counts['WARN']} WARN, {counts['FAIL']} FAIL, {counts['SKIP']} SKIP")

    if args.update_baseline and 'bb100' in rc.collected:
        baselines = rc.baselines
        baselines.setdefault(args.version, {})['bb100'] = rc.collected['bb100']
        _save_baselines(baselines)
        print(f"  baseline updated for {args.version}: {rc.collected['bb100']}")

    dump_path = args.dump_json or os.path.join(
        os.path.dirname(__file__), 'results', f"{args.version}__{args.weights}.json")
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    payload = {
        "version": args.version,
        "weights": args.weights,
        "action_space": list(action_keys),
        "full": args.full,
        "checks": [
            {"id": check_id, "status": r.status, "detail": r.detail, "data": r.data,
             "doc": CHECK_DOCS.get(check_id)}
            for check_id, r in results
        ],
    }
    with open(dump_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  raw results written to {dump_path}")

    return 1 if counts['FAIL'] > 0 else 0


if __name__ == '__main__':
    sys.exit(main())

"""[V43 / B2] Diagnostic probe: WHERE and HOW BIG is the nash_pushfold_vs_chart regression?

`nash_pushfold_vs_chart` went 83% (V29) -> 78% (V40) -> 78% (V41), and the error direction FLIPPED:
V29 folded where Nash shoves, V40/V41 shove where Nash folds. Nothing in the Fable review addresses
it, and no fix should be written before the failing cells are actually enumerated -- the V20 lesson
(a plausible hypothesis, `policy_tightness_bb`, died the moment its own numbers were computed).

This runs the SAME check the suite runs, against several checkpoints, and prints the disagreement
set per model plus what changed between them. Pure forward passes -- no simulator, no training.

Run:  .venv/Scripts/python.exe -m versions.v50.self_play.probe_nash_regression
"""
import os
import sys
import json
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))   # self_play -> v43 -> versions -> repo root
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.model_verify.checks import RunCtx, _find, _aggressive_indices          # noqa: E402
from tools.model_verify.run import load_model, get_manifest                        # noqa: E402
from tools.model_verify.scenarios import build_ctx, run_policy                     # noqa: E402

_SOLVED = os.path.join(_REPO_ROOT, 'tools', 'model_verify', 'nash', 'nash_solved.json')
_CLEAR_HI, _CLEAR_LO = 0.95, 0.05
_NEUTRAL_POSITION = 2
POT_BB, CALL_BB = 1.5, 0.5

# (version_id, weights) -- V29 is the last checkpoint that scored 83%.
MODELS = [
    ('v29', 'expert_main.pth'),
    ('v40', 'expert_main.pth'),
    ('v41', 'expert_main.pth'),
]


def score(version_id, weights):
    manifest = get_manifest(version_id)
    model = load_model(version_id, weights, device='cpu')
    action_keys = tuple(manifest.action_space)
    rc = RunCtx(version_id=version_id, model=model, manifest=manifest,
                action_keys=action_keys, device='cpu', baselines={})
    ai, fi = _find(action_keys, 'allin'), _find(action_keys, 'fold')
    agg_idx = _aggressive_indices(action_keys)

    with open(_SOLVED) as f:
        solved = json.load(f)

    cells = {}
    for S in solved['stacks']:
        st = solved['per_stack'][str(S)]
        for h in solved['hands']:
            fq = st['sb_jam_freq'][h]
            if fq < _CLEAR_HI and fq > _CLEAR_LO:
                continue
            nash = 'shove' if fq >= _CLEAR_HI else 'fold'
            ctx = build_ctx(equity=solved['eq_vs_random'][h], stack_bb=S, pot_bb=POT_BB,
                            call_bb=CALL_BB, num_active_opp=1, position=_NEUTRAL_POSITION,
                            street=0, contract_version=manifest.contract_version)
            policy, q = run_policy(model, ctx, action_keys, device='cpu')
            p_fold = policy[action_keys[fi]]
            agg = sum(policy[action_keys[i]] for i in agg_idx)
            lean = 'shove' if agg > p_fold else 'fold'
            cells[(h, S)] = dict(nash=nash, lean=lean, ok=(lean == nash), agg=agg, fold=p_fold,
                                 eq=solved['eq_vs_random'][h],
                                 allin=policy[action_keys[ai]],
                                 q_allin=q[action_keys[ai]] if q else None,
                                 q_fold=q[action_keys[fi]] if q else None)
    return cells


def main():
    all_cells = {}
    for v, w in MODELS:
        try:
            all_cells[v] = score(v, w)
            n = len(all_cells[v])
            ok = sum(c['ok'] for c in all_cells[v].values())
            print(f"{v:6} {ok}/{n} = {ok/n:.1%}")
        except Exception as e:
            print(f"{v:6} FAILED: {e!r}")

    print()
    print("=" * 100)
    print("DISAGREEMENTS BY DIRECTION (per model)")
    print("=" * 100)
    for v, cells in all_cells.items():
        over = [k for k, c in cells.items() if not c['ok'] and c['nash'] == 'fold']   # shoves too wide
        under = [k for k, c in cells.items() if not c['ok'] and c['nash'] == 'shove']  # too tight
        print(f"{v:6} over-shove (Nash folds, model commits): {len(over):3}   "
              f"under-shove (Nash shoves, model folds): {len(under):3}")

    print()
    print("=" * 100)
    print("CELLS V29 GOT RIGHT AND V41 GETS WRONG  (the regression itself)")
    print("=" * 100)
    if 'v29' in all_cells and 'v41' in all_cells:
        by_stack = defaultdict(list)
        for k, c41 in all_cells['v41'].items():
            c29 = all_cells['v29'].get(k)
            if c29 and c29['ok'] and not c41['ok']:
                by_stack[k[1]].append((k[0], c41))
        for S in sorted(by_stack):
            rows = by_stack[S]
            print(f"\n  {S}bb -- {len(rows)} cells")
            for h, c in sorted(rows, key=lambda r: -r[1]['agg'])[:12]:
                c29 = all_cells['v29'][(h, S)]
                print(f"    {h:5} eq={c['eq']:.3f} nash={c['nash']:5} | "
                      f"V41 agg={c['agg']:.2f} fold={c['fold']:.2f} allin={c['allin']:.2f}"
                      f"  <-  V29 agg={c29['agg']:.2f} fold={c29['fold']:.2f}")

    print()
    print("=" * 100)
    print("CELLS V41 GETS RIGHT AND V29 GOT WRONG  (what the change bought)")
    print("=" * 100)
    if 'v29' in all_cells and 'v41' in all_cells:
        gained = [(k, c) for k, c in all_cells['v41'].items()
                  if c['ok'] and not all_cells['v29'].get(k, {}).get('ok', True)]
        print(f"  {len(gained)} cells")
        for (h, S), c in sorted(gained, key=lambda r: r[0][1])[:15]:
            print(f"    {h:5}@{S}bb eq={c['eq']:.3f} nash={c['nash']:5} V41 agg={c['agg']:.2f} fold={c['fold']:.2f}")


if __name__ == '__main__':
    main()

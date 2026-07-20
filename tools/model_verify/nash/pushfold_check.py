"""VAL-1 runtime check: does the model's learned push/fold policy agree with heads-up Nash
on unambiguous reference spots?

Pure lookup + forward pass. Loads the static nash_chart.json (equities already baked in by
precompute_equities.py) and, for each cell, builds a HU shove-or-fold context and reads the
model's policy via scenarios.run_policy. No simulator, no equity computation, no version code.

This is the first EXTERNAL ground-truth axis in the suite -- every other check tests hero
against this project's own simulator/bots/ancestors. It is intentionally WARN-gated, never
FAIL: a 6-max cash-trained model is not REQUIRED to match a heads-up push/fold subgame, so
this is a calibrated directional sanity signal, not a deploy gate.

Caveats baked into the reporting:
  - Heads-up + SB position is mildly out-of-distribution for a 6-max model; position is held
    at a neutral value (the push/fold decision is dominated by equity + stack geometry).
  - Cells with stack_bb < 5 are below the model's 5bb training floor -- reported separately,
    excluded from the pass metric.
"""
import json
import os

from tools.model_verify.scenarios import build_ctx, run_policy

_CHART_PATH = os.path.join(os.path.dirname(__file__), 'nash_chart.json')

# HU push/fold geometry (0.5 SB / 1.0 BB): at the SB's first decision the dead pot is
# SB + BB = 1.5bb and the cheap price to not-fold is completing the 0.5bb.
_POT_BB = 1.5
_CALL_BB = 0.5
_NEUTRAL_POSITION = 2   # held fixed -- HU position is OOD for the 6-max contract (see docstring)

# PASS if in-range agreement is at least this; else WARN.
_PASS_THRESHOLD = 0.75


def _load_chart():
    with open(_CHART_PATH, 'r') as f:
        return json.load(f)


def check_nash_pushfold_vs_chart(rc):
    """[VAL-1] External axis: compare the model's shove-vs-fold lean to curated Nash cells."""
    # Lazy import to avoid a circular import (checks.py imports this module for FAST_CHECKS).
    from tools.model_verify.checks import CheckResult, _find, _aggressive_indices

    ai = _find(rc.action_keys, 'allin')
    fi = _find(rc.action_keys, 'fold')
    if ai is None or fi is None:
        return CheckResult('SKIP', "action space lacks ALLIN and/or FOLD -- push/fold axis N/A")
    if not os.path.exists(_CHART_PATH):
        return CheckResult('SKIP', "nash_chart.json missing -- run tools.model_verify.nash.precompute_equities")

    # Nash push/fold assumes a shove-OR-fold-only game. The model has a discretized sizing
    # action space (raise_33/66/pot/allin), so "Nash shove" maps to "commit aggressively"
    # (ANY raise-family or all-in mass beating fold), NOT to ALLIN specifically -- otherwise a
    # model that correctly commits premiums via a pot-sized raise instead of a literal jam
    # (exactly what V29's anti-jam [BET-1] fix produces at short stacks) would read as a false
    # disagreement. The ALLIN-vs-raise sizing choice is reported separately as a nuance.
    agg_idx = _aggressive_indices(rc.action_keys)

    chart = _load_chart()
    cv = rc.manifest.contract_version
    in_agree = in_total = 0
    ood_agree = ood_total = 0
    commit_via_raise = commit_cells = 0   # among agreeing 'shove' cells: raises instead of jamming
    gross = []   # in-range disagreements (chart unambiguous, model went the other way on direction)
    data = []

    for cell in chart['cells']:
        ctx = build_ctx(
            equity=cell['equity_vs_random'],
            stack_bb=cell['stack_bb'], pot_bb=_POT_BB, call_bb=_CALL_BB,
            num_active_opp=1, position=_NEUTRAL_POSITION, street=0,
            contract_version=cv,
        )
        policy, _q = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
        p_allin, p_fold = policy[rc.action_keys[ai]], policy[rc.action_keys[fi]]
        agg_mass = sum(policy[rc.action_keys[i]] for i in agg_idx)
        model_lean = 'shove' if agg_mass > p_fold else 'fold'   # 'shove' == commit aggressively
        agrees = (model_lean == cell['nash_action'])
        raw_argmax = max(policy, key=policy.get)
        # sizing nuance: on an agreeing commit, did it jam or raise?
        prefers_allin = None
        if cell['nash_action'] == 'shove' and agrees:
            top_agg = max(agg_idx, key=lambda i: policy[rc.action_keys[i]])
            prefers_allin = (top_agg == ai)

        rec = {
            "hand": cell['hand'], "stack_bb": cell['stack_bb'],
            "nash": cell['nash_action'], "model_lean": model_lean, "agrees": agrees,
            "agg_mass": round(agg_mass, 3), "p_allin": round(p_allin, 3), "p_fold": round(p_fold, 3),
            "raw_argmax": raw_argmax, "prefers_allin": prefers_allin,
            "in_training_range": cell['in_training_range'],
        }
        data.append(rec)

        if cell['in_training_range']:
            in_total += 1
            in_agree += int(agrees)
            if not agrees:
                gross.append(f"{cell['hand']}@{cell['stack_bb']}bb "
                             f"(Nash {cell['nash_action']}, model {model_lean}: "
                             f"agg={agg_mass:.2f}/F={p_fold:.2f}, argmax={raw_argmax})")
            elif prefers_allin is not None:
                commit_cells += 1
                commit_via_raise += int(not prefers_allin)
        else:
            ood_total += 1
            ood_agree += int(agrees)

    frac = in_agree / in_total if in_total else 0.0
    detail = f"Nash commit-vs-fold agreement (in-range >=5bb): {in_agree}/{in_total} ({frac:.0%})"
    if commit_cells:
        detail += (f" | of {commit_cells} agreeing shove cells, {commit_via_raise} commit via a "
                   f"sized RAISE not a literal jam (expected post-V29 anti-jam)")
    if gross:
        shown = "; ".join(gross[:5]) + (f"; +{len(gross) - 5} more" if len(gross) > 5 else "")
        detail += f" | DIRECTION disagreements: {shown}"
    if ood_total:
        detail += f" | below-floor(<5bb): {ood_agree}/{ood_total} agree (OOD, informational)"

    status = 'PASS' if frac >= _PASS_THRESHOLD else 'WARN'
    return CheckResult(status, detail, data)

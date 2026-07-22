"""VAL-1 runtime checks: does the model's learned play agree with the in-repo heads-up Nash
push/fold solution (nash_solved.json, produced offline by solve_nash_pushfold.py)?

Two checks, both pure lookup + a single forward pass via scenarios.run_policy -- no simulator,
no equity computation, no version code:

  check_nash_pushfold_vs_chart  -- SB decision over the FULL 169 hands x every solved stack.
                                   [V47 P0.3] PRIMARY score is literal ALLIN-vs-FOLD (the binary
                                   question the Nash solution actually answers); the old
                                   commit-vs-fold composite (raise-family + all-in mass beating
                                   fold) is kept as a SECONDARY column. The composite let a model
                                   "agree" with a Nash jam via a small sized raise -- a different
                                   (dominated) strategy -- so composite movement across versions
                                   (e.g. the V40-era 83%->78%) conflated sizing-preference shifts
                                   with genuine push/fold disagreement.
  check_nash_bbcall_vs_jam      -- BB facing a jam: call/commit vs fold. This is the cleaner
                                   binary spot (facing an all-in there is no cheap-limp option to
                                   confound the read), with the model's equity input
                                   RANGE-CONDITIONED on SB's Nash jamming range (as it would be in
                                   real play once the opponent has committed).

Both are WARN-only, never FAIL: a 6-max cash-trained model is not REQUIRED to match a heads-up
push/fold subgame (HU + position is mildly OOD). They are a calibrated EXTERNAL directional axis
-- the first checks in the suite that test hero against a game-theory answer rather than this
project's own simulator/bots/ancestors.

(Superseded: nash_chart.json / precompute_equities.py were the Tier-A curated-subset version of
the SB check; the solver's eq_vs_random now covers all 169 hands, so the SB check reads
nash_solved.json instead. The curated files are kept for reference but no longer wired in.)
"""
import json
import os

from tools.model_verify.scenarios import build_ctx, run_policy

_SOLVED_PATH = os.path.join(os.path.dirname(__file__), 'nash_solved.json')

_NEUTRAL_POSITION = 2   # held fixed -- HU position is OOD for the 6-max contract
_CLEAR_HI = 0.95        # Nash freq >= this  -> unambiguous "do it"
_CLEAR_LO = 0.05        # Nash freq <= this  -> unambiguous "don't"
_PASS_THRESHOLD = 0.80  # in-range agreement below this -> WARN


def _load_solved():
    with open(_SOLVED_PATH) as f:
        return json.load(f)


def check_nash_pushfold_vs_chart(rc):
    """[VAL-1] SB open-jam: model commit-vs-fold vs the solved Nash jam range, all 169 x stacks."""
    from tools.model_verify.checks import CheckResult, _find, _aggressive_indices
    ai = _find(rc.action_keys, 'allin')
    fi = _find(rc.action_keys, 'fold')
    if ai is None or fi is None:
        return CheckResult('SKIP', "action space lacks ALLIN and/or FOLD -- push/fold axis N/A")
    if not os.path.exists(_SOLVED_PATH):
        return CheckResult('SKIP', "nash_solved.json missing -- run tools.model_verify.nash.solve_nash_pushfold")

    solved = _load_solved()
    cv = rc.manifest.contract_version
    agg_idx = _aggressive_indices(rc.action_keys)
    # HU SB-first geometry (0.5 SB / 1.0 BB): dead pot 1.5, cheap price to continue 0.5.
    POT_BB, CALL_BB = 1.5, 0.5

    agree = total = 0                 # [V47 P0.3] PRIMARY: literal ALLIN-vs-FOLD
    agree_composite = 0               # secondary: the pre-P0.3 commit-vs-fold composite
    commit_via_raise = commit_cells = 0
    gross = []
    data = []

    for S in solved['stacks']:
        st = solved['per_stack'][str(S)]
        for h in solved['hands']:
            f = st['sb_jam_freq'][h]
            if f < _CLEAR_HI and f > _CLEAR_LO:
                continue  # mixed / near-indifference cell -- skip, only score unambiguous spots
            nash = 'shove' if f >= _CLEAR_HI else 'fold'
            ctx = build_ctx(equity=solved['eq_vs_random'][h], stack_bb=S, pot_bb=POT_BB,
                            call_bb=CALL_BB, num_active_opp=1, position=_NEUTRAL_POSITION,
                            street=0, contract_version=cv)
            policy, _q = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
            p_fold = policy[rc.action_keys[fi]]
            p_allin = policy[rc.action_keys[ai]]
            agg_mass = sum(policy[rc.action_keys[i]] for i in agg_idx)
            # [V47 P0.3] Primary lean: does the model prefer the literal jam over folding --
            # the binary sub-decision the Nash chart is actually a solution to.
            model_lean = 'shove' if p_allin > p_fold else 'fold'
            composite_lean = 'shove' if agg_mass > p_fold else 'fold'
            ok = (model_lean == nash)
            total += 1
            agree += int(ok)
            agree_composite += int(composite_lean == nash)
            if not ok:
                gross.append(f"{h}@{S}bb(Nash {nash},model {model_lean}:allin={p_allin:.2f}/F={p_fold:.2f})")
            if nash == 'shove' and composite_lean == 'shove':
                commit_cells += 1
                top_agg = max(agg_idx, key=lambda i: policy[rc.action_keys[i]])
                commit_via_raise += int(top_agg != ai)
            data.append({"seat": "SB", "hand": h, "stack_bb": S, "nash": nash,
                         "model_lean": model_lean, "agrees": ok,
                         "composite_lean": composite_lean,
                         "agrees_composite": composite_lean == nash,
                         "p_allin": round(p_allin, 3),
                         "agg_mass": round(agg_mass, 3), "p_fold": round(p_fold, 3)})

    frac = agree / total if total else 0.0
    frac_composite = agree_composite / total if total else 0.0
    detail = (f"SB literal-jam-vs-fold agreement over {total} unambiguous Nash cells: {agree} "
              f"({frac:.0%}) | secondary commit-vs-fold composite (pre-P0.3 metric): "
              f"{frac_composite:.0%}")
    if commit_cells:
        detail += (f" | {commit_via_raise}/{commit_cells} composite-commits use a sized RAISE "
                   f"not a literal jam")
    if gross:
        detail += f" | {len(gross)} disagreements e.g. " + "; ".join(gross[:4])
    status = 'PASS' if frac >= _PASS_THRESHOLD else 'WARN'
    return CheckResult(status, detail, data)


def check_nash_bbcall_vs_jam(rc):
    """[VAL-1] BB facing a jam: model call/commit-vs-fold vs the solved Nash calling range.
    Cleaner binary spot than the SB check -- facing an all-in there is no cheap-limp confound --
    and the model's equity input is range-conditioned on SB's Nash jam range."""
    from tools.model_verify.checks import CheckResult, _find
    fi = _find(rc.action_keys, 'fold')
    if fi is None:
        return CheckResult('SKIP', "action space lacks FOLD -- call/fold axis N/A")
    if not os.path.exists(_SOLVED_PATH):
        return CheckResult('SKIP', "nash_solved.json missing -- run tools.model_verify.nash.solve_nash_pushfold")

    solved = _load_solved()
    cv = rc.manifest.contract_version
    agree = total = 0
    gross = []
    data = []

    for S in solved['stacks']:
        st = solved['per_stack'][str(S)]
        for h in solved['hands']:
            f = st['bb_call_freq'][h]
            if f < _CLEAR_HI and f > _CLEAR_LO:
                continue
            nash = 'call' if f >= _CLEAR_HI else 'fold'
            # BB faces SB's all-in of S: pot = SB jam (S) + BB posted blind (1.0); to call, BB
            # adds S-1.0 to match. pot_odds = (S-1)/(2S) = the Nash call-equity threshold.
            ctx = build_ctx(equity=st['bb_eq_vs_jam'][h], stack_bb=S, pot_bb=S + 1.0,
                            call_bb=max(S - 1.0, 0.0), num_active_opp=1,
                            position=_NEUTRAL_POSITION, street=0, contract_version=cv)
            policy, _q = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
            p_fold = policy[rc.action_keys[fi]]
            model_lean = 'call' if (1.0 - p_fold) > p_fold else 'fold'   # commit == anything but fold
            ok = (model_lean == nash)
            total += 1
            agree += int(ok)
            if not ok:
                gross.append(f"{h}@{S}bb(Nash {nash},model {model_lean}:F={p_fold:.2f})")
            data.append({"seat": "BB", "hand": h, "stack_bb": S, "nash": nash,
                         "model_lean": model_lean, "agrees": ok, "p_fold": round(p_fold, 3),
                         "eq_vs_jam": st['bb_eq_vs_jam'][h]})

    frac = agree / total if total else 0.0
    detail = f"BB call-vs-fold-facing-a-jam agreement over {total} unambiguous Nash cells: {agree} ({frac:.0%})"
    if gross:
        detail += f" | {len(gross)} disagreements e.g. " + "; ".join(gross[:4])
    status = 'PASS' if frac >= _PASS_THRESHOLD else 'WARN'
    return CheckResult(status, detail, data)

"""The model-verification CURRICULUM: one growing list of checks, each guarding a specific
diagnosed issue (or a general health property) so it can never silently regress in a future
version. Append new checks here as new issues get diagnosed -- that's the intended workflow;
nothing else in this tool needs to change.

Each check is a plain function `fn(rc: RunCtx) -> CheckResult`. FAST checks are single
synthetic-scenario forward passes (milliseconds each); SLOW checks run real simulated hands
via the version's own `self_play.simulator` (minutes) and are gated behind `--full`.
"""
import os
from dataclasses import dataclass, field

from tools.model_verify.scenarios import build_ctx, run_policy
from tools.model_verify.stress_bots import TieredLookupBot

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


@dataclass
class CheckResult:
    status: str   # 'PASS' | 'WARN' | 'FAIL' | 'SKIP'
    detail: str
    data: list = None   # raw per-scenario records, for external analysis/plotting (optional)


@dataclass
class RunCtx:
    version_id: str
    model: object
    manifest: object
    action_keys: tuple
    device: str = "cpu"
    # Slow-check extras (only populated when --full is requested)
    sim_module: object = None
    range_aware: bool = False
    baselines: dict = field(default_factory=dict)
    n_hands_style: int = 3000
    n_hands_field: int = 4000
    frozen_predecessor_filename: str = None
    collected: dict = field(default_factory=dict)   # raw numbers a check can stash for --update-baseline


def _find(action_keys, canonical):
    """Exact (case-insensitive) match on the WHOLE action name. Substring matching is a trap
    here -- 'all' in 'call' is True in Python, so a naive `needle in key` search silently
    matches CALL when looking for ALLIN. Callers must pass the full canonical name."""
    canonical = canonical.lower()
    for i, k in enumerate(action_keys):
        if k.lower() == canonical:
            return i
    return None


def _aggressive_indices(action_keys):
    """RAISE-family or ALLIN indices, using prefix/exact checks (not substring) for the same
    reason _find is exact -- 'all' in 'call' would otherwise wrongly count CALL as aggressive."""
    return [i for i, k in enumerate(action_keys) if k.lower().startswith('raise') or k.lower() == 'allin']


# =====================================================================================
# FAST checks -- synthetic single-decision scenarios, run every invocation.
# =====================================================================================

def check_equity_ablation_monotonic(rc):
    fold_i = _find(rc.action_keys, 'fold')
    aggr_idx = _aggressive_indices(rc.action_keys)
    if fold_i is None or not aggr_idx:
        return CheckResult('SKIP', 'action space has no fold/raise-like actions')
    equities = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]
    fold_probs, aggr_probs, data = [], [], []
    for eq in equities:
        ctx = build_ctx(equity=eq, stack_bb=40, pot_bb=10, call_bb=5, num_active_opp=2,
                        contract_version=rc.manifest.contract_version)
        policy, _ = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
        fold_probs.append(policy[rc.action_keys[fold_i]])
        aggr_probs.append(sum(policy[rc.action_keys[i]] for i in aggr_idx))
        data.append({"equity": eq, "policy": policy})
    fold_drop = fold_probs[0] - fold_probs[-1]
    aggr_rise = aggr_probs[-1] - aggr_probs[0]
    detail = f"P(fold) {fold_probs[0]:.2f}->{fold_probs[-1]:.2f}, P(aggressive) {aggr_probs[0]:.2f}->{aggr_probs[-1]:.2f}"
    status = 'PASS' if (fold_drop > 0.15 and aggr_rise > 0.15) else 'FAIL'
    if status == 'FAIL':
        detail += " -- expected fold to drop and aggression to rise as equity rises"
    return CheckResult(status, detail, data)


def check_free_check_low_fold(rc):
    """FOLD is strictly dominated when checking is free, and core/decision.py masks it out
    before sampling live (`free_check` logic) -- that mask is the actual, proven mitigation.
    This check tracks how much RAW residual mass the mask is covering, as a WARN-only health
    signal, not a deploy gate: V15 (validated, deployed, known-good) carries up to ~0.35 here,
    so a hard FAIL threshold would flag a already-shipped-safely model as broken."""
    fold_i = _find(rc.action_keys, 'fold')
    if fold_i is None:
        return CheckResult('SKIP', 'no fold action')
    worst = 0.0
    data = []
    for eq in (0.1, 0.3, 0.5, 0.7, 0.9):
        for stack in (8, 40, 100):
            ctx = build_ctx(equity=eq, stack_bb=stack, pot_bb=3, call_bb=0, num_active_opp=2,
                            contract_version=rc.manifest.contract_version)
            policy, _ = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
            worst = max(worst, policy[rc.action_keys[fold_i]])
            data.append({"equity": eq, "stack_bb": stack, "policy": policy})
    detail = f"max raw P(fold) when call_bb=0 across sweep: {worst:.3f}"
    status = 'PASS' if worst < 0.05 else 'WARN'
    if status == 'WARN':
        detail += " (covered by decision.py's free-check mask, but worth tracking)"
    return CheckResult(status, detail, data)


def check_air_folds_mostly(rc):
    fold_i = _find(rc.action_keys, 'fold')
    if fold_i is None:
        return CheckResult('SKIP', 'no fold action')
    probs, data = [], []
    for stack in (8, 20, 40, 80):
        ctx = build_ctx(equity=0.12, stack_bb=stack, pot_bb=12, call_bb=8, num_active_opp=2,
                        contract_version=rc.manifest.contract_version)
        policy, _ = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
        probs.append(policy[rc.action_keys[fold_i]])
        data.append({"stack_bb": stack, "policy": policy})
    avg = sum(probs) / len(probs)
    detail = f"avg P(fold) at ~12% equity facing a real bet: {avg:.2f}  (per-stack {[round(p,2) for p in probs]})"
    status = 'PASS' if avg > 0.40 else 'WARN'
    if status == 'WARN':
        detail += " -- air not folding enough"
    return CheckResult(status, detail, data)


def check_nuts_aggressive_mostly(rc):
    aggr_idx = _aggressive_indices(rc.action_keys)
    if not aggr_idx:
        return CheckResult('SKIP', 'no raise/allin actions')
    probs, data = [], []
    for stack in (8, 20, 40, 80):
        ctx = build_ctx(equity=0.92, stack_bb=stack, pot_bb=12, call_bb=8, num_active_opp=2,
                        contract_version=rc.manifest.contract_version)
        policy, _ = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
        probs.append(sum(policy[rc.action_keys[i]] for i in aggr_idx))
        data.append({"stack_bb": stack, "policy": policy})
    avg = sum(probs) / len(probs)
    detail = f"avg aggressive mass at ~92% equity: {avg:.2f}"
    status = 'PASS' if avg > 0.40 else 'WARN'
    if status == 'WARN':
        detail += " -- nuts not betting/raising enough"
    return CheckResult(status, detail, data)


def check_deep_stack_ood_guard(rc):
    """Regression test for the V14 live incident (versions/v15/SPECS.md [P0]): hero jammed K9o
    (43% equity) for a full 20bb all-in into a single limper -- ALL-IN was the model's argmax.
    Sweeps the marginal-equity x 15-40bb x single-modest-bet neighborhood around that spot."""
    allin_i = _find(rc.action_keys, 'allin')
    if allin_i is None:
        return CheckResult('SKIP', 'no all-in action')
    worst = 0.0
    detail_worst = ""
    data = []
    for eq in (0.35, 0.40, 0.43, 0.48, 0.55):
        for stack in (15, 20, 25, 30, 40):
            ctx = build_ctx(equity=eq, stack_bb=stack, pot_bb=2.5, call_bb=1.0, num_active_opp=1,
                            contract_version=rc.manifest.contract_version)
            policy, _ = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
            p_allin = policy[rc.action_keys[allin_i]]
            argmax = max(policy, key=policy.get)
            is_allin_argmax = argmax == rc.action_keys[allin_i]
            data.append({"equity": eq, "stack_bb": stack, "policy": policy,
                         "argmax": argmax, "argmax_is_allin": is_allin_argmax})
            if is_allin_argmax and p_allin > worst:
                worst = p_allin
                detail_worst = f"eq={eq} stack={stack}bb -> ALL-IN argmax @ {p_allin:.2f}"
    if worst == 0.0:
        return CheckResult('PASS', "no marginal-equity/deep-stack/single-modest-bet cell jams all-in", data)
    return CheckResult('FAIL', f"[V14 P0 regression] deep-stack OOD trash-jam: {detail_worst}", data)


def check_short_stack_polarization(rc):
    """Tracks [P3]: at a clear shove-or-fold spot (short stack, profitable equity, facing a
    raise sized near the stack), CALL shouldn't be sitting between fold and shove. WARN-only --
    this is being watched post-hoc, not gated, per the V16 [P4] plan discussion."""
    call_i = _find(rc.action_keys, 'call')
    if call_i is None:
        return CheckResult('SKIP', 'no call action')
    probs, data = [], []
    for eq in (0.50, 0.55, 0.60, 0.65):
        for stack in (6, 8, 10):
            ctx = build_ctx(equity=eq, stack_bb=stack, pot_bb=stack * 0.9, call_bb=stack * 0.85, num_active_opp=1,
                            contract_version=rc.manifest.contract_version)
            policy, _ = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
            probs.append(policy[rc.action_keys[call_i]])
            data.append({"equity": eq, "stack_bb": stack, "policy": policy})
    avg = sum(probs) / len(probs)
    detail = f"avg P(call) in clear shove-or-fold spots: {avg:.2f}"
    status = 'PASS' if avg < 0.20 else 'WARN'
    if status == 'WARN':
        detail += " -- [P3] residual flatting where theory says push/fold"
    return CheckResult(status, detail, data)


def check_action_diversity(rc):
    """Guards against the historical raise-everything/call-everything collapse that the
    regret-matching actor target was built to prevent (see train.py's own comments): ONE
    action dominating argmax regardless of equity/stack, not just "few distinct winners".
    NOTE: a healthy policy can legitimately have fold/call/allin win every argmax while the
    raise buckets still carry real, smoothly-varying probability mass without ever peaking --
    that's the known, accepted "no middle gear" bimodal-sizing characteristic
    (versions/v15/SPECS.md [P1]), not a collapse. This check only flags true monopolization."""
    from collections import Counter
    counter = Counter()
    total = 0
    data = []
    for eq in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95):
        for stack in (8, 25, 60):
            ctx = build_ctx(equity=eq, stack_bb=stack, pot_bb=10, call_bb=4, num_active_opp=2,
                            contract_version=rc.manifest.contract_version)
            policy, _ = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
            argmax = max(policy, key=policy.get)
            counter[argmax] += 1
            total += 1
            data.append({"equity": eq, "stack_bb": stack, "policy": policy, "argmax": argmax})
    dominant, dominant_count = counter.most_common(1)[0]
    share = dominant_count / total
    detail = f"argmax distribution across the equity x stack grid: {dict(counter)}"
    status = 'PASS' if share < 0.85 else 'FAIL'
    if status == 'FAIL':
        detail += f" -- '{dominant}' monopolizes {share:.0%} of the grid, possible action collapse"
    return CheckResult(status, detail, data)


def check_no_nan_or_crash(rc):
    """Regression guard for the NaN-inference crashes fixed by removing key_padding_mask
    (commit 55a1bc9) -- edge-case seat counts / stacks / streets should never produce NaN."""
    edge_cases = [
        dict(equity=0.5, stack_bb=0.5, pot_bb=1, call_bb=0.4, num_active_opp=0),
        dict(equity=0.0, stack_bb=200, pot_bb=0.01, call_bb=0.0, num_active_opp=5),
        dict(equity=1.0, stack_bb=400, pot_bb=999, call_bb=399, num_active_opp=5, opp_vpip=0.45, opp_agg=0.85),
        dict(equity=0.5, stack_bb=5, pot_bb=5, call_bb=5, num_active_opp=1, street=3),
    ]
    data = []
    for i, kw in enumerate(edge_cases):
        try:
            ctx = build_ctx(**kw, contract_version=rc.manifest.contract_version)
            policy, q = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
        except Exception as e:
            data.append({"case": i, **kw, "outcome": f"{type(e).__name__}: {e}", "ok": False})
            return CheckResult('FAIL', f"edge case {i} raised {type(e).__name__}: {e}", data)
        vals = list(policy.values()) + list(q.values())
        is_nan = any(v != v for v in vals)  # NaN != NaN
        data.append({"case": i, **kw, "policy": policy, "outcome": "NaN" if is_nan else "clean", "ok": not is_nan})
        if is_nan:
            return CheckResult('FAIL', f"edge case {i} produced NaN output", data)
    return CheckResult('PASS', f"{len(edge_cases)} edge cases: no NaN/crash", data)


def check_hand_strength_sensitivity(rc):
    """[V20_preflopEq] Is the model actually USING the `hand_strength` context feature (ctx[36]),
    or did it land in the tensor with zero effective weight? Holds equity/stack/pot/field FIXED
    and swings only hand_strength (0.15 "weak card shape at this equity" vs 0.85 "strong card
    shape") -- a synthetic ablation: in real play hand_strength is never this decoupled from
    equity, but perturbing it alone here isolates whether the network reads that slot at all.
    WARN-only / diagnostic (no established expected magnitude yet -- this is the first version
    to carry this feature), not a pass/fail gate."""
    if getattr(rc.manifest, 'context_dim', 35) < 37:
        return CheckResult('SKIP', 'model context has no hand_strength feature (context_dim<37)')
    diffs, data = [], []
    for eq in (0.35, 0.50, 0.65, 0.80):
        for stack in (15, 40):
            ctx_lo = build_ctx(equity=eq, stack_bb=stack, pot_bb=10, call_bb=4, num_active_opp=2,
                                contract_version=rc.manifest.contract_version, hand_strength=0.15)
            ctx_hi = build_ctx(equity=eq, stack_bb=stack, pot_bb=10, call_bb=4, num_active_opp=2,
                                contract_version=rc.manifest.contract_version, hand_strength=0.85)
            policy_lo, _ = run_policy(rc.model, ctx_lo, rc.action_keys, device=rc.device)
            policy_hi, _ = run_policy(rc.model, ctx_hi, rc.action_keys, device=rc.device)
            tv_dist = 0.5 * sum(abs(policy_hi[k] - policy_lo[k]) for k in rc.action_keys)
            diffs.append(tv_dist)
            data.append({"equity": eq, "stack_bb": stack, "policy_lo_hs": policy_lo,
                         "policy_hi_hs": policy_hi, "total_variation": round(tv_dist, 4)})
    avg = sum(diffs) / len(diffs)
    detail = f"avg policy shift (total variation) between hand_strength=0.15 vs 0.85 at fixed equity: {avg:.3f}"
    status = 'PASS' if avg > 0.03 else 'WARN'
    if status == 'WARN':
        detail += " -- negligible response; hand_strength may not be contributing (or was absorbed elsewhere)"
    return CheckResult(status, detail, data)


def check_equity_edge_sensitivity(rc):
    """[V20_preflopEq] Same idea as check_hand_strength_sensitivity but for `equity_edge`
    (ctx[35], real-play value = equity*(num_active+1)). Holds equity/stack/pot/field FIXED and
    swings only equity_edge to a value well below vs well above what the real formula would give
    for that (equity, num_active) pair -- probing whether the network reads this ctx slot as
    informative beyond what it could already derive from equity + the existing active-opponent
    count feature. WARN-only / diagnostic, not a pass/fail gate (first version to carry this
    feature -- no established expected magnitude yet)."""
    if getattr(rc.manifest, 'context_dim', 35) < 37:
        return CheckResult('SKIP', 'model context has no equity_edge feature (context_dim<37)')
    diffs, data = [], []
    for eq in (0.35, 0.50, 0.65, 0.80):
        for num_active in (1, 3):
            real_value = eq * (num_active + 1)
            ctx_lo = build_ctx(equity=eq, stack_bb=30, pot_bb=10, call_bb=4, num_active_opp=num_active,
                                contract_version=rc.manifest.contract_version,
                                equity_edge=max(0.0, real_value - 0.8))
            ctx_hi = build_ctx(equity=eq, stack_bb=30, pot_bb=10, call_bb=4, num_active_opp=num_active,
                                contract_version=rc.manifest.contract_version,
                                equity_edge=real_value + 0.8)
            policy_lo, _ = run_policy(rc.model, ctx_lo, rc.action_keys, device=rc.device)
            policy_hi, _ = run_policy(rc.model, ctx_hi, rc.action_keys, device=rc.device)
            tv_dist = 0.5 * sum(abs(policy_hi[k] - policy_lo[k]) for k in rc.action_keys)
            diffs.append(tv_dist)
            data.append({"equity": eq, "num_active_opp": num_active, "real_value": round(real_value, 2),
                         "policy_lo_edge": policy_lo, "policy_hi_edge": policy_hi,
                         "total_variation": round(tv_dist, 4)})
    avg = sum(diffs) / len(diffs)
    detail = f"avg policy shift (total variation) between low vs high equity_edge at fixed equity/field: {avg:.3f}"
    status = 'PASS' if avg > 0.03 else 'WARN'
    if status == 'WARN':
        detail += " -- negligible response; equity_edge may be redundant with equity+active_opp_count"
    return CheckResult(status, detail, data)


def check_committed_sensitivity(rc):
    """[V22] Same idea as check_hand_strength_sensitivity/check_equity_edge_sensitivity but for
    the new `opp_committed_this_hand_bb`/`hero_committed_this_hand_bb` features (ctx[37:43]).
    Holds equity/stack/pot/field FIXED and swings only the lead opponent's committed-this-hand
    amount between 0bb (hasn't put in anything, e.g. a flat call so far) and near-stack-deep (a
    big earlier bet/raise this hand) -- probing whether the network reads these ctx slots as
    informative beyond what it could already derive from `opp_stack` (remaining) +
    `call_amount`/`pot_size` alone. WARN-only / diagnostic, not a pass/fail gate (first version to
    carry this feature -- no established expected magnitude yet). SKIP if context_dim<43."""
    if getattr(rc.manifest, 'context_dim', 35) < 43:
        return CheckResult('SKIP', 'model context has no committed-this-hand feature (context_dim<43)')
    diffs, data = [], []
    for eq in (0.35, 0.50, 0.65, 0.80):
        for stack in (20, 60):
            ctx_lo = build_ctx(equity=eq, stack_bb=stack, pot_bb=10, call_bb=4, num_active_opp=2,
                                contract_version=rc.manifest.contract_version,
                                per_opp_committed_bb=[0.0, 0.0])
            ctx_hi = build_ctx(equity=eq, stack_bb=stack, pot_bb=10, call_bb=4, num_active_opp=2,
                                contract_version=rc.manifest.contract_version,
                                per_opp_committed_bb=[stack * 0.6, 0.0])
            policy_lo, _ = run_policy(rc.model, ctx_lo, rc.action_keys, device=rc.device)
            policy_hi, _ = run_policy(rc.model, ctx_hi, rc.action_keys, device=rc.device)
            tv_dist = 0.5 * sum(abs(policy_hi[k] - policy_lo[k]) for k in rc.action_keys)
            diffs.append(tv_dist)
            data.append({"equity": eq, "stack_bb": stack,
                         "policy_lo_committed": policy_lo, "policy_hi_committed": policy_hi,
                         "total_variation": round(tv_dist, 4)})
    avg = sum(diffs) / len(diffs)
    detail = f"avg policy shift (total variation) between opp committed=0bb vs =60%stack at fixed equity/stack: {avg:.3f}"
    status = 'PASS' if avg > 0.03 else 'WARN'
    if status == 'WARN':
        detail += " -- negligible response; opp_committed_this_hand_bb may be redundant with opp_stack/call_amount"
    return CheckResult(status, detail, data)


def check_pot_type_sensitivity(rc):
    """[V23] Same idea as check_committed_sensitivity but for `pot_type` (ctx[43], bucketed
    0=limped/1=single-raised/2=3-bet+). Holds equity/stack/pot/call/field FIXED and swings only
    pot_type between 0 (limped) and 2 (3-bet+) -- probing whether the network reads this ctx slot
    as informative beyond what it could already derive from call_amount/pot_size/committed alone
    (a big call_amount can arise from one big bet OR a raise war -- pot_type distinguishes them).
    WARN-only / diagnostic, not a pass/fail gate (first version to carry this feature -- no
    established expected magnitude yet). SKIP if context_dim<44."""
    if getattr(rc.manifest, 'context_dim', 35) < 44:
        return CheckResult('SKIP', 'model context has no pot_type feature (context_dim<44)')
    diffs, data = [], []
    for eq in (0.35, 0.50, 0.65, 0.80):
        for stack in (20, 60):
            ctx_lo = build_ctx(equity=eq, stack_bb=stack, pot_bb=10, call_bb=4, num_active_opp=2,
                                contract_version=rc.manifest.contract_version, pot_type=0)
            ctx_hi = build_ctx(equity=eq, stack_bb=stack, pot_bb=10, call_bb=4, num_active_opp=2,
                                contract_version=rc.manifest.contract_version, pot_type=2)
            policy_lo, _ = run_policy(rc.model, ctx_lo, rc.action_keys, device=rc.device)
            policy_hi, _ = run_policy(rc.model, ctx_hi, rc.action_keys, device=rc.device)
            tv_dist = 0.5 * sum(abs(policy_hi[k] - policy_lo[k]) for k in rc.action_keys)
            diffs.append(tv_dist)
            data.append({"equity": eq, "stack_bb": stack,
                         "policy_lo_pottype": policy_lo, "policy_hi_pottype": policy_hi,
                         "total_variation": round(tv_dist, 4)})
    avg = sum(diffs) / len(diffs)
    detail = f"avg policy shift (total variation) between pot_type=limped vs =3bet+ at fixed equity/stack: {avg:.3f}"
    status = 'PASS' if avg > 0.03 else 'WARN'
    if status == 'WARN':
        detail += " -- negligible response; pot_type may be redundant with call_amount/pot_size/committed"
    return CheckResult(status, detail, data)


# =====================================================================================
# SENSITIVITY SWEEPS -- one clean parameter axis per check, so a collapsed/flatlined/
# wrong-direction response is visible at a glance (as a line chart or small heatmap) rather than
# buried in an aggregate pass/fail number. Complements the spot-checks above: those probe SPECIFIC
# known-bad neighborhoods (the V14 trash-jam, the short-stack polarization spot); these sweep a
# whole practical range of ONE (or two) input dimensions and watch for "nothing moved at all",
# which is the signature of a feature that silently isn't load-bearing (wrong tensor index, a
# dead weight, a normalization bug) rather than a specific behavioral defect. WARN-only by design
# -- there usually isn't a single "correct" curve shape to gate on, only "did it respond at all".
# =====================================================================================

def _policy_sweep_range(records, action_keys):
    """Robust flatline detector across a full parameter sweep: max, over actions, of that
    action's probability range (max-min) across every point in the sweep. Endpoint-only TV
    (first vs last) can miss a real non-monotonic bump in the middle -- e.g. argmax swinging
    to a different action mid-sweep and back to the SAME action at both ends -- this doesn't."""
    return max(max(r["policy"][k] for r in records) - min(r["policy"][k] for r in records)
               for k in action_keys)


def check_stack_full_sweep(rc):
    """Full-range 1D stack sweep (5-180bb) at a fixed marginal spot (~50% equity facing a
    proportionally-sized bet), independent of the narrow OOD-guard neighborhoods above (those
    only cover 6-40bb). Bet/pot sizes scale WITH stack so every point stays a realistic
    ~20%-pot bet rather than a nonsensical tiny-bet-at-huge-stack or huge-bet-at-tiny-stack.
    WARN-only flatline detector: stack alone has no single "correct" direction (short-stack
    push/fold vs deep-stack pot-control are both legitimate depending on the rest of the read),
    so this only flags total non-responsiveness across the whole practical range -- NOTE for
    V20+ (contract_version>=4): stack-derived features clamp at 50bb by design (the live-safety
    clamp, see versions/v20/SPECS.md), so a flat tail from 60bb-180bb on those models is the
    KNOWN clamp behavior, not a new defect -- the signal that matters is whether 5bb-40bb (fully
    unclamped) shows movement."""
    stacks = [5, 10, 15, 25, 40, 60, 90, 130, 180]
    data = []
    for s in stacks:
        ctx = build_ctx(equity=0.50, stack_bb=s, pot_bb=s * 0.2, call_bb=s * 0.1, num_active_opp=2,
                        contract_version=rc.manifest.contract_version)
        policy, _ = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
        data.append({"stack_bb": s, "policy": policy, "argmax": max(policy, key=policy.get)})
    rng = _policy_sweep_range(data, rc.action_keys)
    detail = (f"max action-probability range across the full stack sweep: {rng:.3f}  "
              f"(argmax path: {[d['argmax'] for d in data]})")
    status = 'PASS' if rng > 0.05 else 'WARN'
    if status == 'WARN':
        detail += " -- flat across the whole practical stack range; stack feature may not be load-bearing"
    return CheckResult(status, detail, data)


def check_position_sweep(rc):
    """Full 1D sweep over hero_position (0-5) at a fixed marginal preflop spot. Regression
    watch for the historical hero_position bug (universal, training-only -- fixed in V19, see
    auto-memory v19-p0-position-fix / versions/v19/SPECS.md). WARN-only flatline detector --
    position's "correct" direction isn't a single scalar prior (it depends on the rest of the
    table), so this only flags total non-responsiveness."""
    fold_i = _find(rc.action_keys, 'fold')
    if fold_i is None:
        return CheckResult('SKIP', 'no fold action')
    data = []
    for p in range(6):
        ctx = build_ctx(equity=0.45, stack_bb=40, pot_bb=3, call_bb=1.5, position=p,
                        num_active_opp=2, street=0, contract_version=rc.manifest.contract_version)
        policy, _ = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
        data.append({"position": p, "policy": policy, "argmax": max(policy, key=policy.get)})
    fold_vals = [d["policy"][rc.action_keys[fold_i]] for d in data]
    spread = max(fold_vals) - min(fold_vals)
    detail = f"P(fold) across position 0-5: {[round(v, 2) for v in fold_vals]} (spread {spread:.3f})"
    status = 'PASS' if spread > 0.03 else 'WARN'
    if status == 'WARN':
        detail += " -- policy barely varies by position; position feature may not be load-bearing"
    return CheckResult(status, detail, data)


# Opponent-style archetypes mirroring the HUD-color -> VPIP/AGG convention used across every
# version's contract.py (e.g. versions/v20_preflopEq/core/contract.py VPIP_MAP/AGG_MAP: Blue=nit,
# Green=tag, Yellow=loose/station, Red=loose-aggressive). Defined locally (not imported) so this
# tool stays version-agnostic -- these are illustrative reference points, not a hard dependency.
STYLE_ARCHETYPES = [
    (0, 'Blue (nit)', 0.10, 0.18),
    (1, 'Green (tag)', 0.22, 0.46),
    (2, 'Yellow (station)', 0.30, 0.63),
    (3, 'Red (maniac)', 0.45, 0.85),
]


def check_opponent_style_sweep(rc):
    """2D sweep: equity x opponent-style-archetype (Blue/Green/Yellow/Red), P(fold) facing an
    identical bet. A dimension model_verify never probed before. Poker prior: the SAME bet from a
    tighter/more-aggressive villain should play scarier (more folds) than the identical bet from a
    looser villain, at fixed hero equity. WARN-only / diagnostic -- first version to carry this
    check, but it directly probes the [P6] backlog concern (versions/v16/SPECS.md: "opponent-
    action attribution") from a different angle: not the action-sequence gap P6 tracks, but
    whether the per-opponent VPIP/AGG color inputs the model has ALWAYS received are actually
    load-bearing at all."""
    fold_i = _find(rc.action_keys, 'fold')
    if fold_i is None:
        return CheckResult('SKIP', 'no fold action')
    data = []
    # [2026-07-17] 0.25 and 0.45 were both already saturated (fold~1.0 / fold~0.0) for every
    # style, so the original 3-point sweep couldn't tell "opponent style doesn't matter" apart
    # from "opponent style shifts WHERE the fold/continue threshold sits, but not by much once
    # you're already past it" -- the two additional points sample the actual transition zone
    # between those saturated ends, where a real per-style continuation-threshold shift (if the
    # model has one) would actually show up.
    for eq in (0.25, 0.30, 0.35, 0.45, 0.65):
        for style_idx, style_name, vpip, agg in STYLE_ARCHETYPES:
            ctx = build_ctx(equity=eq, stack_bb=40, pot_bb=10, call_bb=6, num_active_opp=1,
                            opp_vpip=vpip, opp_agg=agg, contract_version=rc.manifest.contract_version)
            policy, _ = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
            data.append({"equity": eq, "style_idx": style_idx, "style": style_name,
                         "policy": policy, "argmax": max(policy, key=policy.get)})
    by_eq = {}
    for d in data:
        by_eq.setdefault(d["equity"], []).append(d)
    spreads = []
    for eq, rows in by_eq.items():
        rows.sort(key=lambda r: r["style_idx"])
        fold_vals = [r["policy"][rc.action_keys[fold_i]] for r in rows]
        spreads.append(max(fold_vals) - min(fold_vals))
    avg_spread = sum(spreads) / len(spreads)
    detail = f"avg P(fold) spread across Blue->Red opponent archetypes (same bet, fixed equity): {avg_spread:.3f}"
    status = 'PASS' if avg_spread > 0.03 else 'WARN'
    if status == 'WARN':
        detail += " -- fold rate barely moves with opponent style; per-opponent VPIP/AGG read may not be load-bearing"
    return CheckResult(status, detail, data)


# Exact base_vpip/base_agg_freq of the 4 named archetypes populating the training pool
# (opponent_bots.py's TAG/LAG/NIT/CALLING_STATION) -- deliberately NOT the same as STYLE_ARCHETYPES
# above, which uses idealized/representative spectrum points (its own "Red (maniac)" point, for
# instance, doesn't match the real CALLING_STATION archetype at all: loose+PASSIVE in reality
# (agg=0.15) vs loose+aggressive as modeled there (agg=0.85)). See [OPP-8].
# Ordered tightest/most-fold-prone (NIT) -> loosest/least-fold-prone (CALLING_STATION), matching
# each archetype's real base_fold_to_pressure ordering (0.85/0.60/0.45/0.15) for a meaningful
# gradient when rendered as a heatmap axis.
REAL_ARCHETYPES = [
    (0, 'NIT', 0.11, 0.25),
    (1, 'TAG', 0.22, 0.45),
    (2, 'LAG', 0.32, 0.55),
    (3, 'CALLING_STATION', 0.45, 0.15),
]


def check_allin_exploits_opponent_foldiness(rc):
    """[OPP-8] Does hero's own all-in FREQUENCY actually differentiate by opponent archetype,
    proportional to how differently those archetypes really respond to a shove? Complements
    `opponent_style_sweep` (an abstract Blue->Red VPIP/AGG spectrum) by feeding the EXACT
    base_vpip/base_agg_freq of the four named archetypes that actually populate the training pool.
    Motivation (2026-07-18): a direct probe against `_ev_target_fold_decision` (the same function
    used to build hero's own training targets) shows NIT folds to an all-in ~98% of the time at a
    realistic price, vs CALLING_STATION ~0%, at the IDENTICAL price -- driven by each archetype's
    independent `base_fold_to_pressure` (NIT=0.85, CALLING_STATION=0.15), a trait NEVER itself fed
    to the model as an input feature (only inferred indirectly via the coarse 4-bucket VPIP/AGG
    color read). A policy that's actually exploiting this should show a LARGE P(all-in) spread
    across these 4 real archetypes at a fixed, decent equity -- not just whatever
    `opponent_style_sweep`'s synthetic spectrum shows. WARN-only / diagnostic (informational, not
    gated) -- see OFK known-shortcomings-backlog [OPP-8] for the full investigation.

    CAVEAT (2026-07-18, user-caught methodology point -- do not over-read a low spread as pure
    exploitable waste): this sweep feeds a FLAT equity value per cell, i.e. it asks "given the
    opponent genuinely has this equity right now, does the policy react to who they are" -- it does
    NOT weight by how OFTEN each archetype actually presents that equity level in real play. NIT's
    tight preflop selection (VPIP 0.11) means its real postflop range is selection-skewed strong --
    a genuine 45%-equity NIT spot is rarer than this synthetic sweep implies, since most weak NIT
    hands never survive to see a flop at all. A low spread here is still evidence the model can't
    directly observe the underlying fold_to_pressure trait, but it does NOT by itself prove hero is
    leaving significant real-game EV on the table -- that requires weighting by each archetype's
    true conditional-on-continuing range, which this check deliberately does not attempt."""
    allin_i = _find(rc.action_keys, 'allin')
    if allin_i is None:
        return CheckResult('SKIP', 'no all-in action')
    data = []
    for eq in (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
        for idx, name, vpip, agg in REAL_ARCHETYPES:
            ctx = build_ctx(equity=eq, stack_bb=25, pot_bb=7.5, call_bb=3.75, num_active_opp=1,
                            opp_vpip=vpip, opp_agg=agg, contract_version=rc.manifest.contract_version)
            policy, _ = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
            argmax = max(policy, key=policy.get)
            data.append({"equity": eq, "archetype": name, "archetype_idx": idx, "policy": policy,
                         "argmax": argmax, "argmax_is_allin": argmax == rc.action_keys[allin_i]})
    by_eq = {}
    for d in data:
        by_eq.setdefault(d["equity"], []).append(d)
    spreads = []
    for eq, rows in by_eq.items():
        allin_vals = [r["policy"][rc.action_keys[allin_i]] for r in rows]
        spreads.append(max(allin_vals) - min(allin_vals))
    avg_spread = sum(spreads) / len(spreads)
    detail = f"avg P(all-in) spread across NIT/TAG/LAG/CALLING_STATION at fixed equity: {avg_spread:.3f}"
    status = 'PASS' if avg_spread > 0.08 else 'WARN'
    if status == 'WARN':
        detail += " -- [OPP-8] all-in frequency barely differs by real opponent archetype despite wildly different actual fold-to-pressure"
    return CheckResult(status, detail, data)


def check_allin_vs_nextbest_qgap(rc):
    """[BET-1 diagnostic, V28] The FIRST permanent, reusable measurement of the all-in-vs-next-best
    Q-value gap. Every number quoted for this gap in versions/v21_auxhead..v26/SPECS.md (reported
    as a 1.35x-1.78x ratio across versions) was run via a one-off, non-committed script -- five
    versions of "did the gap get better or worse" were each measured a different way. This check
    reuses the SAME eq x stack grid `check_deep_stack_ood_guard` already sweeps (directly
    comparable to that check's own numbers) and the SAME 4 real archetypes [OPP-8] already defines,
    and reports a full BREAKDOWN (by stack depth, by opponent archetype) using the raw Q-values
    `run_policy` already returns (and every other check discards) -- not a single aggregate figure.
    That's the actual localization ("where does the gap concentrate") the prior five versions
    never had, which is the point: know WHERE before picking a fix.

    Reports the gap as `q_allin - max(other actions' q)`, normalized by the scenario's own pot_bb
    (so the two breakdowns, which use different pot sizes, are comparable to each other) --
    NOT the historical ratio framing, since that ratio was never well-defined near a zero/negative
    next-best Q (FOLD's own critic target is a hardcoded 0.0 by construction, see
    `_mc_target_evs_sized`, so a ratio against it can blow up or flip sign). WARN-only / diagnostic
    -- first version to carry this check, so the 0.15-of-pot threshold below is a provisional
    starting point, not an established baseline; the main value is the breakdown table itself.

    [Caught before shipping] The by-stack/by-archetype breakdown reports the WORST (max) cell in
    each slice, NOT the average. An early version of this check averaged across the equity sweep
    and reported PASS even at stack=40bb, where Q_allin swings from -6.12 (eq=0.35) to +2.97
    (eq=0.55) -- a steep, almost threshold-like jump, much sharper than call/raise's much more
    gradual rise over the same range -- because the other 4 (correctly negative) cells in that
    stack bucket diluted the one cell that actually matches `check_deep_stack_ood_guard`'s own
    FAIL at that exact spot. Averaging across a swept dimension hides exactly the localization
    this check exists to surface -- fixed to report the max (worst) cell per slice instead.
    """
    allin_i = _find(rc.action_keys, 'allin')
    if allin_i is None:
        return CheckResult('SKIP', 'no all-in action')
    allin_key = rc.action_keys[allin_i]

    def _gap_frac(q, pot_bb):
        other_best = max(v for k, v in q.items() if k != allin_key)
        return (q[allin_key] - other_best) / max(pot_bb, 1e-6)

    # -- Breakdown 1: by stack depth (same eq x stack grid as check_deep_stack_ood_guard). --
    by_stack, stack_data = {}, []
    for stack in (15, 20, 25, 30, 40):
        gaps = []
        for eq in (0.35, 0.40, 0.43, 0.48, 0.55):
            ctx = build_ctx(equity=eq, stack_bb=stack, pot_bb=2.5, call_bb=1.0, num_active_opp=1,
                            contract_version=rc.manifest.contract_version)
            _, q = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
            gap = _gap_frac(q, 2.5)
            gaps.append(gap)
            stack_data.append({"stack_bb": stack, "equity": eq, "qvals": q, "gap_frac_of_pot": round(gap, 4)})
        by_stack[stack] = max(gaps)   # worst cell, not the average -- see docstring caveat

    # -- Breakdown 2: by opponent archetype, same MARGINAL-equity band as breakdown 1 (0.35-0.55,
    # NOT check_allin_exploits_opponent_foldiness's own wider 0.15-0.85 sweep). Deliberately
    # narrower than OPP-8's own range: an early version of this check reused that wider band and
    # its "worst cell" always landed at eq=0.85 -- shoving for value at 85% equity is often
    # correctly profitable, not the BET-1 defect (marginal-equity overshoving) this check exists to
    # localize. Keeping both breakdowns on the SAME equity band makes them directly comparable and
    # keeps this check pointed at the actual phenomenon, not legitimate high-equity value shoves.
    by_archetype, archetype_data = {}, []
    for idx, name, vpip, agg in REAL_ARCHETYPES:
        gaps = []
        for eq in (0.35, 0.40, 0.43, 0.48, 0.55):
            ctx = build_ctx(equity=eq, stack_bb=25, pot_bb=7.5, call_bb=3.75, num_active_opp=1,
                            opp_vpip=vpip, opp_agg=agg, contract_version=rc.manifest.contract_version)
            _, q = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
            gap = _gap_frac(q, 7.5)
            gaps.append(gap)
            archetype_data.append({"archetype": name, "archetype_idx": idx, "equity": eq, "qvals": q,
                                   "gap_frac_of_pot": round(gap, 4)})
        by_archetype[name] = max(gaps)   # worst cell, not the average -- see docstring caveat

    # Flat combined list (not a nested dict) so render_report.py's bespoke card can `.filter()`
    # it into the two sub-views by which shape each record has (stack_bb+equity vs
    # archetype+equity) -- matching the "paired two-col card" convention already used for
    # air_folds_mostly/nuts_aggressive_mostly. The two per-breakdown averages are still in
    # `detail` above for anyone just reading the text summary.
    data = stack_data + archetype_data

    worst_stack = max(by_stack, key=by_stack.get)
    worst_archetype = max(by_archetype, key=by_archetype.get)
    worst_overall = max(by_stack[worst_stack], by_archetype[worst_archetype])
    detail = (f"WORST-cell Q-gap (allin - next-best, as fraction of pot) by stack: "
              + ", ".join(f"{s}bb={g:+.2f}" for s, g in by_stack.items())
              + " | by archetype: "
              + ", ".join(f"{n}={g:+.2f}" for n, g in by_archetype.items())
              + f" | worst overall: stack={worst_stack}bb ({by_stack[worst_stack]:+.2f}), "
                f"archetype={worst_archetype} ({by_archetype[worst_archetype]:+.2f})")
    status = 'WARN' if worst_overall > 0.15 else 'PASS'
    if status == 'WARN':
        detail += " -- [BET-1] a meaningful gap survives somewhere in the breakdown; see raw_stack/raw_archetype in the JSON dump to localize it further"
    return CheckResult(status, detail, data)


def check_opponent_color_isolated_ablation(rc):
    """[OPP-5 diagnostic] `check_opponent_style_sweep` already sweeps opp_vpip/opp_agg across the
    full REALISTIC archetype range (Blue 0.10/0.18 -> Red 0.45/0.85) and found essentially flat
    P(fold) (spread <=0.004). That leaves two very different explanations open: (a) the training
    population never rewarded differentiating by style enough for the network to bother -- a
    training-population artifact, same mechanism as [BET-1] -- or (b) the VPIP/AGG inputs are
    genuinely dead (a wiring/normalization bug), in which case NOTHING would move them, no matter
    how extreme. This check distinguishes the two by (1) pushing WELL past the realistic range
    (0.0 vs 1.0, not 0.10 vs 0.85) and (2) testing the ctx's TWO separate VPIP/AGG representations
    IN ISOLATION: the table-level scalar (ctx[7]/ctx[8], set once per decision) vs the per-seat
    block (5 opponent slots x [active,pos,stack,vpip,agg], see build_ctx) -- mirroring
    check_hand_strength_sensitivity/check_equity_edge_sensitivity's isolate-one-slot-at-extremes
    approach rather than opponent_style_sweep's realistic-archetype sweep. Uses full-policy total
    variation (all actions), not just P(fold), so it also catches a response that shows up in
    raise-sizing rather than the fold/continue line. WARN-only / diagnostic -- first version to
    carry this check, still investigating [OPP-5], not yet a pass/fail gate."""
    fixed = dict(equity=0.45, stack_bb=40, pot_bb=10, call_bb=6, num_active_opp=2,
                 contract_version=rc.manifest.contract_version)
    data = {}
    # (1) Table-level scalar isolated: per-seat block held at neutral defaults throughout.
    ctx_lo = build_ctx(**fixed, opp_vpip=0.0, opp_agg=0.0, per_opp_vpip=[0.30, 0.30], per_opp_agg=[0.40, 0.40])
    ctx_hi = build_ctx(**fixed, opp_vpip=1.0, opp_agg=1.0, per_opp_vpip=[0.30, 0.30], per_opp_agg=[0.40, 0.40])
    policy_lo, _ = run_policy(rc.model, ctx_lo, rc.action_keys, device=rc.device)
    policy_hi, _ = run_policy(rc.model, ctx_hi, rc.action_keys, device=rc.device)
    tv_scalar = 0.5 * sum(abs(policy_hi[k] - policy_lo[k]) for k in rc.action_keys)
    data['table_scalar'] = {"policy_lo": policy_lo, "policy_hi": policy_hi, "total_variation": round(tv_scalar, 4)}
    # (2) Per-seat block isolated: table-level scalar held at the default (0.30/0.40) throughout.
    ctx_lo2 = build_ctx(**fixed, opp_vpip=0.30, opp_agg=0.40, per_opp_vpip=[0.0, 0.0], per_opp_agg=[0.0, 0.0])
    ctx_hi2 = build_ctx(**fixed, opp_vpip=0.30, opp_agg=0.40, per_opp_vpip=[1.0, 1.0], per_opp_agg=[1.0, 1.0])
    policy_lo2, _ = run_policy(rc.model, ctx_lo2, rc.action_keys, device=rc.device)
    policy_hi2, _ = run_policy(rc.model, ctx_hi2, rc.action_keys, device=rc.device)
    tv_seat = 0.5 * sum(abs(policy_hi2[k] - policy_lo2[k]) for k in rc.action_keys)
    data['per_seat_block'] = {"policy_lo": policy_lo2, "policy_hi": policy_hi2, "total_variation": round(tv_seat, 4)}

    detail = f"table-scalar TV (0.0 vs 1.0): {tv_scalar:.3f} | per-seat-block TV (0.0 vs 1.0): {tv_seat:.3f}"
    status = 'PASS' if (tv_scalar > 0.03 or tv_seat > 0.03) else 'WARN'
    if status == 'WARN':
        detail += (" -- flat even at synthetic extremes (0.0 vs 1.0, well past any realistic archetype); "
                   "looks like a genuinely dead/unwired input, not just training-population insensitivity")
    else:
        detail += (" -- responds at extremes but opponent_style_sweep found realistic-archetype values flat; "
                   "reads as a training-population artifact (network CAN use this input, population never taught it to care within realistic bounds), not dead wiring")
    return CheckResult(status, detail, data)


def check_hand_strength_sweep(rc):
    """Full 5-point curve companion to check_hand_strength_sensitivity's 2-point TV check -- same
    synthetic ablation (hand_strength swung independently of equity; never decoupled like this in
    real play, isolates whether the network reads ctx[36] at all), but shaped for a line chart so
    a flatline OR a non-monotonic kink is visible, not just one aggregate number. SKIP if the
    model's context has no hand_strength feature (context_dim<37)."""
    if getattr(rc.manifest, 'context_dim', 35) < 37:
        return CheckResult('SKIP', 'model context has no hand_strength feature (context_dim<37)')
    values = [0.1, 0.3, 0.5, 0.7, 0.9]
    data = []
    for hs in values:
        ctx = build_ctx(equity=0.45, stack_bb=40, pot_bb=10, call_bb=4, num_active_opp=2,
                        contract_version=rc.manifest.contract_version, hand_strength=hs)
        policy, _ = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
        data.append({"hand_strength": hs, "policy": policy, "argmax": max(policy, key=policy.get)})
    rng = _policy_sweep_range(data, rc.action_keys)
    detail = f"max action-probability range across the hand_strength={values[0]}-{values[-1]} sweep: {rng:.3f}"
    status = 'PASS' if rng > 0.03 else 'WARN'
    if status == 'WARN':
        detail += " -- negligible response across the full sweep"
    return CheckResult(status, detail, data)


def check_equity_edge_sweep(rc):
    """Full multi-point curve companion to check_equity_edge_sensitivity's 2-point TV check,
    sweeping the raw equity_edge override directly (not via num_active) at fixed
    equity/stack/field, shaped for a line chart. SKIP if the model's context has no equity_edge
    feature (context_dim<37)."""
    if getattr(rc.manifest, 'context_dim', 35) < 37:
        return CheckResult('SKIP', 'model context has no equity_edge feature (context_dim<37)')
    values = [0.4, 0.8, 1.2, 1.6, 2.4, 3.2]
    data = []
    for edge in values:
        ctx = build_ctx(equity=0.45, stack_bb=40, pot_bb=10, call_bb=4, num_active_opp=2,
                        contract_version=rc.manifest.contract_version, equity_edge=edge)
        policy, _ = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
        data.append({"equity_edge": edge, "policy": policy, "argmax": max(policy, key=policy.get)})
    rng = _policy_sweep_range(data, rc.action_keys)
    detail = f"max action-probability range across the equity_edge={values[0]}-{values[-1]} sweep: {rng:.3f}"
    status = 'PASS' if rng > 0.03 else 'WARN'
    if status == 'WARN':
        detail += " -- negligible response across the full sweep"
    return CheckResult(status, detail, data)


def check_multiway_shortstack_aggression(rc):
    """[BET-3] Aggression must not collapse as opponent count rises at short stacks.

    Live-confirmed (V29, Double-or-Nothing): the model plays a correct aggressive short-stack
    push/fold range HEADS-UP but refuses to raise at all with 3+ opponents (won't jam even at
    90% equity), folding clear short jams -- the exact live "too tight / not aggressive"
    complaint. This isolates that collapse by sweeping ONLY num_active_opp at a fixed
    short-stack / jam-worthy spot, the one condition no other check exercises (nuts_aggressive,
    action_diversity, and the VAL-1 Nash checks all run at 1-2 opponents)."""
    agg_idx = _aggressive_indices(rc.action_keys)
    if not agg_idx:
        return CheckResult('SKIP', 'no aggressive actions in action space')
    cv = rc.manifest.contract_version
    data, collapses = [], []
    for stack in (5, 6, 8):
        for eq in (0.55, 0.65):
            aggs = {}
            for opp in (1, 2, 3, 4):
                ctx = build_ctx(equity=eq, stack_bb=stack, pot_bb=2.0, call_bb=1.0,
                                num_active_opp=opp, position=2, street=0,
                                contract_version=cv, hand_strength=0.62, equity_edge=eq * (opp + 1))
                pol, _q = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
                aggs[opp] = sum(pol[rc.action_keys[i]] for i in agg_idx)
            hu, mw = aggs[1], aggs[3]
            data.append({"stack_bb": stack, "equity": eq, "hu_agg": round(hu, 3),
                         "three_way_agg": round(mw, 3),
                         "agg_by_opp": {o: round(a, 3) for o, a in aggs.items()}})
            if hu >= 0.30 and mw < 0.10:   # HU wants to commit, 3-way craters to ~nothing
                collapses.append(f"{stack}bb/eq{eq}: HU {hu:.2f}->3way {mw:.2f}")
    profile = "; ".join(f"{d['stack_bb']}bb/eq{d['equity']}: {d['hu_agg']:.2f}->{d['three_way_agg']:.2f}"
                        for d in data)
    if collapses:
        return CheckResult('WARN', f"[BET-3] multiway aggression COLLAPSE in {len(collapses)}/{len(data)} "
                           f"short-stack cells (HU aggression vanishes by 3-way): {profile}", data)
    return CheckResult('PASS', f"multiway aggression holds up (HU->3way agg): {profile}", data)


# VAL-1 external ground-truth axis (self-contained plug-in -- see tools/model_verify/nash/).
from tools.model_verify.nash.pushfold_check import (
    check_nash_pushfold_vs_chart, check_nash_bbcall_vs_jam)

FAST_CHECKS = [
    ("equity_ablation_monotonic", "P(fold) falls / P(aggressive) rises as equity rises",
     "general health", check_equity_ablation_monotonic),
    ("nash_pushfold_vs_chart", "SB commit/fold lean agrees with in-repo heads-up Nash over all 169 hands x stacks (EXTERNAL ground truth, WARN-only)",
     "VAL-1 -- external GTO reference axis (SB jam)", check_nash_pushfold_vs_chart),
    ("nash_bbcall_vs_jam", "BB call/fold-facing-a-jam agrees with in-repo heads-up Nash, range-conditioned equity (EXTERNAL ground truth, WARN-only)",
     "VAL-1 -- external GTO reference axis (BB call)", check_nash_bbcall_vs_jam),
    ("free_check_low_fold", "never wants to fold a free option (call_amount == 0)",
     "train/serve fold-mask consistency", check_free_check_low_fold),
    ("air_folds_mostly", "~12% equity facing a real bet folds more than it doesn't",
     "V14 spot-test baseline", check_air_folds_mostly),
    ("nuts_aggressive_mostly", "~92% equity bets/raises more than it doesn't",
     "V14 spot-test baseline", check_nuts_aggressive_mostly),
    ("deep_stack_ood_guard", "no marginal-equity/15-40bb/single-modest-bet cell jams all-in",
     "V14 P0 -- live K9o 20bb trash-jam", check_deep_stack_ood_guard),
    ("short_stack_polarization", "CALL not dominant in a clear short-stack shove-or-fold spot",
     "P3 -- preflop flattening (WARN, tracked not gated)", check_short_stack_polarization),
    ("multiway_shortstack_aggression", "aggression doesn't collapse from heads-up to 3+ opponents at short stacks",
     "BET-3 -- live-confirmed V29 multiway passivity (WARN, tracked)", check_multiway_shortstack_aggression),
    ("action_diversity", "at least most actions appear as argmax somewhere in the grid",
     "V11 raise-/call-everything collapse", check_action_diversity),
    ("no_nan_or_crash", "edge-case seat/stack/street combos never NaN or throw",
     "commit 55a1bc9 -- NaN inference crashes", check_no_nan_or_crash),
    ("hand_strength_sensitivity", "policy responds to hand_strength at fixed equity (SKIP if context_dim<37)",
     "V20_preflopEq -- is the new feature actually load-bearing", check_hand_strength_sensitivity),
    ("equity_edge_sensitivity", "policy responds to equity_edge at fixed equity/field (SKIP if context_dim<37)",
     "V20_preflopEq -- is the new feature actually load-bearing", check_equity_edge_sensitivity),
    ("committed_sensitivity", "policy responds to opp_committed_this_hand_bb at fixed equity/stack (SKIP if context_dim<43)",
     "V22 -- is the new entry-sizing feature actually load-bearing", check_committed_sensitivity),
    ("pot_type_sensitivity", "policy responds to pot_type (limped vs 3bet+) at fixed equity/stack (SKIP if context_dim<44)",
     "V23 -- is the new pot_type feature actually load-bearing", check_pot_type_sensitivity),
    ("stack_full_sweep", "full 5-180bb stack sweep at a fixed marginal spot doesn't flatline",
     "sensitivity sweep -- stack", check_stack_full_sweep),
    ("position_sweep", "full 0-5 position sweep at a fixed marginal spot doesn't flatline",
     "sensitivity sweep -- position / V19 hero_position regression watch", check_position_sweep),
    ("opponent_style_sweep", "P(fold) responds to opponent VPIP/AGG archetype (Blue->Red) at fixed equity",
     "sensitivity sweep -- opponent style / P6-adjacent", check_opponent_style_sweep),
    ("allin_exploits_opponent_foldiness", "P(all-in) spreads meaningfully across the 4 REAL named archetypes (NIT/TAG/LAG/CALLING_STATION) at fixed equity",
     "OPP-8 diagnostic -- does hero exploit each archetype's real (unobserved) fold-to-pressure trait", check_allin_exploits_opponent_foldiness),
    ("allin_vs_nextbest_qgap", "all-in's Q-value advantage over the next-best action, broken down by stack depth and by opponent archetype",
     "BET-1 diagnostic, V28 -- first permanent/reproducible measurement of the shove-preference gap", check_allin_vs_nextbest_qgap),
    ("opponent_color_isolated_ablation", "policy responds to opp VPIP/AGG at synthetic extremes (0.0 vs 1.0), table-scalar and per-seat-block isolated separately",
     "OPP-5 diagnostic -- dead input vs training-population artifact", check_opponent_color_isolated_ablation),
    ("hand_strength_sweep", "full 5-point hand_strength curve doesn't flatline (SKIP if context_dim<37)",
     "sensitivity sweep -- hand_strength", check_hand_strength_sweep),
    ("equity_edge_sweep", "full multi-point equity_edge curve doesn't flatline (SKIP if context_dim<37)",
     "sensitivity sweep -- equity_edge", check_equity_edge_sweep),
]


# =====================================================================================
# SLOW checks -- real simulated hands via the version's own simulator. --full only.
# =====================================================================================

def _run_field(rc, pool, weights, stack_bb_range, n_hands, equity_sims=120):
    sim = rc.sim_module.SixMaxSimulator(bb_size=10.0, equity_sims=equity_sims,
                                        hero_personality='main', bootstrap_alpha=0.0)
    sim.hero_model = rc.model
    sim.opponent_pool_styles = pool
    sim.opponent_pool_weights = weights
    sim.live_players = 6
    sim.fixed_stack_bb = stack_bb_range
    sim.disable_exploration = True
    sim.range_aware_equity = rc.range_aware
    # BUG FIX (2026-07-15): simulator.py reads this via getattr(self, 'policy_temperature', 1.0)
    # -- it is NEVER pre-declared in __init__, so hasattr() on a fresh instance is always False
    # and this assignment was silently skipped, leaving every eval running at temp=1.0 (the
    # TRAINING-exploration policy) instead of 0.5 (live serve). That's exactly the documented
    # trap: raw temp=1.0 rollout policy LOSES (-6 to -10 BB/100, VPIP 60%+) because it's not what
    # deploys. Always assign unconditionally -- Python allows setting a new attribute regardless
    # of whether it existed before, so the hasattr guard was never doing anything useful.
    sim.policy_temperature = 0.5   # match live serve (core/decision.py LIVE_POLICY_TEMPERATURE)
    for i in range(n_hands):
        sim.simulate_hand(current_hand=500000 + i)
    h = sim.seat_histories[0]
    bb100 = (h['profit'] / 10.0) / max(1, n_hands) * 100.0
    vpip = (h['vpip_acts'] / h['vpip_ops'] * 100.0) if h.get('vpip_ops') else 0.0
    return bb100, vpip


def check_vpip_adapts_to_style(rc):
    """[P4] regression gate: hero VPIP must move with opponent tightness. This is the exact
    test that originally caught the flat-VPIP bug (versions/v16/SPECS.md [P4])."""
    if rc.sim_module is None:
        return CheckResult('SKIP', 'requires --full (simulator not loaded)')
    parts, data, ok = [], [], True
    for depth_label, stack_range in (('short', [5, 14]), ('deep', [30, 50])):
        bb100_tight, vpip_tight = _run_field(rc, ['nit'], [1.0], stack_range, rc.n_hands_style)
        bb100_loose, vpip_loose = _run_field(rc, ['fish'], [1.0], stack_range, rc.n_hands_style)
        delta = vpip_loose - vpip_tight
        ok = ok and delta >= 5.0
        parts.append(f"{depth_label}: tight={vpip_tight:.1f}% loose={vpip_loose:.1f}% (delta {delta:+.1f}pts)")
        data.append({"depth": depth_label, "stack_range": stack_range,
                     "vpip_tight": round(vpip_tight, 1), "vpip_loose": round(vpip_loose, 1),
                     "delta": round(delta, 1), "bb100_tight": round(bb100_tight, 1),
                     "bb100_loose": round(bb100_loose, 1), "pass": delta >= 5.0})
    detail = "; ".join(parts)
    if ok:
        return CheckResult('PASS', detail, data)
    return CheckResult('FAIL', detail + " -- [P4] VPIP not conforming to opponent tightness (need >=5pt delta)", data)


def check_bb100_vs_standard_fields(rc):
    """Pure-policy winrate vs standard field presets, diffed against the stored baseline for
    this version (tools/model_verify/baselines.json) so a quiet regression shows up as WARN
    even when nothing is FAILing outright."""
    if rc.sim_module is None:
        return CheckResult('SKIP', 'requires --full (simulator not loaded)')
    fields = [
        ('loose_short', ['fish', 'tag', 'nit'], [0.45, 0.30, 0.25], [5, 14]),
        ('loose_deep',  ['fish', 'tag', 'nit'], [0.45, 0.30, 0.25], [30, 50]),
        ('tight_short', ['nit', 'tag'], [0.5, 0.5], [5, 14]),
        ('tight_deep',  ['nit', 'tag'], [0.5, 0.5], [30, 50]),
    ]
    baseline = (rc.baselines.get(rc.version_id) or {}).get('bb100', {})
    measured = {}
    lines, data, warn = [], [], False
    for key, pool, weights, stack_range in fields:
        bb100, vpip = _run_field(rc, pool, weights, stack_range, rc.n_hands_field)
        measured[key] = round(bb100, 1)
        prior = baseline.get(key)
        regressed = prior is not None and (bb100 - prior) < -15.0
        if regressed:
            warn = True
            lines.append(f"{key}: {bb100:+.1f} BB/100 (baseline {prior:+.1f}, DOWN {prior - bb100:.1f}) VPIP {vpip:.0f}%")
        else:
            note = f", baseline {prior:+.1f}" if prior is not None else ", no baseline recorded yet"
            lines.append(f"{key}: {bb100:+.1f} BB/100{note} VPIP {vpip:.0f}%")
        data.append({"field": key, "pool": pool, "stack_range": stack_range,
                     "bb100": round(bb100, 1), "vpip": round(vpip, 1), "baseline": prior,
                     "regressed": regressed})
    rc.collected['bb100'] = measured   # available to run.py for --update-baseline
    detail = " | ".join(lines)
    return CheckResult('WARN' if warn else 'PASS', detail, data)


def check_beats_frozen_predecessor(rc):
    """Every version must beat a frozen snapshot of its immediate predecessor (the
    `frozen_v{N-1}.pth` benchmark pattern used since V15)."""
    if rc.sim_module is None:
        return CheckResult('SKIP', 'requires --full (simulator not loaded)')
    if not rc.frozen_predecessor_filename:
        return CheckResult('SKIP', 'no frozen_v*.pth found in this version\'s weights dir')
    from shared.registry import load_model
    try:
        frozen_model = load_model(rc.version_id, rc.frozen_predecessor_filename, device=rc.device)
    except Exception as e:
        return CheckResult('SKIP', f'could not load {rc.frozen_predecessor_filename}: {e}')
    sim = rc.sim_module.SixMaxSimulator(bb_size=10.0, equity_sims=120,
                                        hero_personality='main', bootstrap_alpha=0.0)
    sim.hero_model = rc.model
    sim.opponent_pool_styles = ['fish', 'tag', 'nit', 'past']
    sim.opponent_pool_weights = [0.40, 0.20, 0.20, 0.20]
    sim.live_players = 6
    sim.fixed_stack_bb = [5, 50]
    sim.disable_exploration = True
    sim.range_aware_equity = rc.range_aware
    sim.disable_past_self = False
    sim.past_model = frozen_model
    # BUG FIX (2026-07-15): simulator.py reads this via getattr(self, 'policy_temperature', 1.0)
    # -- it is NEVER pre-declared in __init__, so hasattr() on a fresh instance is always False
    # and this assignment was silently skipped, leaving every eval running at temp=1.0 (the
    # TRAINING-exploration policy) instead of 0.5 (live serve). That's exactly the documented
    # trap: raw temp=1.0 rollout policy LOSES (-6 to -10 BB/100, VPIP 60%+) because it's not what
    # deploys. Always assign unconditionally -- Python allows setting a new attribute regardless
    # of whether it existed before, so the hasattr guard was never doing anything useful.
    sim.policy_temperature = 0.5
    n_hands = rc.n_hands_field
    for i in range(n_hands):
        sim.simulate_hand(current_hand=700000 + i)
    h = sim.seat_histories[0]
    bb100 = (h['profit'] / 10.0) / max(1, n_hands) * 100.0
    detail = f"vs field incl. frozen predecessor ({rc.frozen_predecessor_filename}): {bb100:+.1f} BB/100 over {n_hands} hands"
    data = [{"opponent": rc.frozen_predecessor_filename, "bb100": round(bb100, 1),
             "n_hands": n_hands, "field": "fish/tag/nit/past(frozen)"}]
    if bb100 > 0:
        return CheckResult('PASS', detail, data)
    return CheckResult('FAIL', detail + " -- must beat the frozen predecessor benchmark", data)


def check_beats_offformula_stress(rc):
    """Stress test vs a DELIBERATELY DIFFERENT opponent functional form (`TieredLookupBot`,
    stress_bots.py) -- a discrete, PRICE-INSENSITIVE equity-tier lookup table that tightens by
    STREET rather than by bet size. Every training-pool bot (fish/tag/nit/maniac) is built from
    the SAME continuous pot-odds-linear formula family (`opponent_bots.py`), just
    reparameterized/jittered -- so a win-rate that holds up against re-weighted versions of that
    formula doesn't prove the hero generalizes past it. This swaps in a genuinely different shape
    to probe that. Raised in discussion 2026-07-15 (concern: hero may be learning structure
    specific to the training formula rather than transferable poker principles)."""
    if rc.sim_module is None:
        return CheckResult('SKIP', 'requires --full (simulator not loaded)')
    n_hands = rc.n_hands_field
    results = {}
    for depth_label, stack_range in (('short', [5, 14]), ('deep', [30, 50])):
        sim = rc.sim_module.SixMaxSimulator(bb_size=10.0, equity_sims=120,
                                            hero_personality='main', bootstrap_alpha=0.0)
        sim.hero_model = rc.model
        sim.tag_heuristic = TieredLookupBot()   # per-instance override; see stress_bots.py
        sim.opponent_pool_styles = ['tag']
        sim.opponent_pool_weights = [1.0]
        sim.live_players = 6
        sim.fixed_stack_bb = stack_range
        sim.disable_exploration = True
        sim.range_aware_equity = rc.range_aware
        if hasattr(sim, 'policy_temperature'):
            sim.policy_temperature = 0.5
        for i in range(n_hands):
            sim.simulate_hand(current_hand=900000 + i)
        h = sim.seat_histories[0]
        bb100 = (h['profit'] / 10.0) / max(1, n_hands) * 100.0
        vpip = (h['vpip_acts'] / h['vpip_ops'] * 100.0) if h.get('vpip_ops') else 0.0
        results[depth_label] = (bb100, vpip)
    detail = "; ".join(f"{d}: {bb100:+.1f} BB/100 (VPIP {vpip:.0f}%)" for d, (bb100, vpip) in results.items())
    data = [{"depth": d, "bb100": round(bb100, 1), "vpip": round(vpip, 1)} for d, (bb100, vpip) in results.items()]
    worst = min(bb100 for bb100, _ in results.values())
    if worst > -10.0:
        return CheckResult('PASS', detail, data)
    if worst > -30.0:
        return CheckResult('WARN', detail + " -- notable dropoff vs an off-formula opponent, possible overfit to the training pool's specific shape", data)
    return CheckResult('FAIL', detail + " -- large dropoff vs an off-formula opponent, strong overfit signal", data)


SLOW_CHECKS = [
    ("vpip_adapts_to_style", "hero VPIP moves with opponent tightness at short + deep stacks",
     "P4 -- VPIP-vs-style flatness", check_vpip_adapts_to_style),
    ("beats_offformula_stress", "winrate holds up vs a structurally different (not just reweighted) opponent",
     "generalization-vs-overfitting probe, 2026-07-15 discussion", check_beats_offformula_stress),
    ("bb100_vs_standard_fields", "winrate vs loose/tight fields at short+deep, diffed vs baseline",
     "general health / regression tracking", check_bb100_vs_standard_fields),
    ("beats_frozen_predecessor", "positive BB/100 vs a field including the frozen predecessor",
     "deploy gate (V15+)", check_beats_frozen_predecessor),
]

ALL_CHECKS = FAST_CHECKS + SLOW_CHECKS


# =====================================================================================
# PLAIN-ENGLISH DOCS -- one entry per check_id, surfaced in the rendered HTML report so a human
# reading it doesn't need to open this source file to know what's being tested. Keep in sync
# with ALL_CHECKS (a check with no entry here just renders without the explainer, it's not an
# error) -- add one whenever a new check is added above.
# =====================================================================================
CHECK_DOCS = {
    "nash_pushfold_vs_chart": dict(
        what="SB open-jam decision: compares the model's commit-vs-fold lean against an IN-REPO-SOLVED heads-up Nash push/fold equilibrium (SB jam vs BB call, chip-EV), over all 169 starting hands x every solved stack depth (5-20bb). The first check that tests hero against an EXTERNAL game-theory answer rather than this project's own simulator/bots. Nash 'shove' is scored as 'commit aggressively' (any raise-family or all-in mass beating fold), since the model has a discretized sizing action space.",
        expect="On UNAMBIGUOUS Nash cells (jam-freq near 0 or 1; mixed hands are skipped), the model should lean the same way -- commit its Nash jam range, fold the rest. The jam-vs-sized-raise split is reported separately (a model that commits via raise_pot instead of a literal jam still AGREES on direction).",
        if_not="WARN-only, never a deploy gate -- a 6-max cash model isn't required to match a heads-up subgame, and HU/position is mildly OOD. But low agreement, or a gross error like folding a premium or committing pure trash deep, is a real external red flag independent of how clean the self-referential checks look.",
    ),
    "nash_bbcall_vs_jam": dict(
        what="BB-facing-a-jam decision: compares the model's call/commit-vs-fold lean against the in-repo-solved Nash BB calling range, over all 169 hands x stacks. The cleaner binary spot -- facing an all-in there is no cheap-limp option to muddy the read -- with the model's equity input RANGE-CONDITIONED on SB's Nash jamming range (as it would be in real play once the opponent has committed).",
        expect="On unambiguous Nash cells, the model should call (commit) with hands whose equity vs the jam range clears the pot-odds threshold and fold the rest. Because facing a jam collapses the action space to call-or-fold, agreement here is a purer test of the model's short-stack calling discipline than the SB check.",
        if_not="WARN-only. A gross error (folding AA to a jam, or calling off with trash that can't be getting the right price) is a real external red flag. Calibrate expectations for HU/position OOD, but the range-conditioned equity makes this the more trustworthy of the two axes.",
    ),
    "multiway_shortstack_aggression": dict(
        what="At a fixed short-stack, jam-worthy spot, sweeps ONLY the number of opponents (1 to 4) and measures how much aggressive (raise/all-in) mass the model keeps. Isolates a live-confirmed failure where the model plays aggressively heads-up but goes passive multiway.",
        expect="Aggression should stay meaningful as opponents are added -- multiway warrants SOME tightening, but a strong short-stack hand should still want to commit, not drop to zero raises the moment a third player is in.",
        if_not="If heads-up aggression vanishes by 3 opponents (the V29 pattern -- won't raise even at 90% equity, folds clear short jams), the model is a multiway call/fold machine. This is exactly the live 'too tight / not aggressive' complaint in Double-or-Nothing (all multiway short stacks), and the reason a clean heads-up scorecard didn't predict live play.",
    ),
    "equity_ablation_monotonic": dict(
        what="Sweeps the model's win probability (equity) from very weak (5%) to very strong (95%), holding everything else fixed, and watches how often it folds vs bets/raises.",
        expect="Folding should become rare and betting/raising should become common as equity rises -- a smooth swing from mostly-fold at low equity to mostly-aggressive at high equity.",
        if_not="If fold doesn't drop or aggression doesn't rise, the model isn't using its own equity input correctly -- a fundamental, likely training-breaking bug (wrong feature index, dead equity pathway, or a catastrophic collapse).",
    ),
    "free_check_low_fold": dict(
        what="Checks how often the model wants to fold when there's nothing to call (a free option) -- folding here is always a mistake since checking costs nothing.",
        expect="Fold probability should be near zero whenever the price to continue is zero.",
        if_not="A high raw rate here isn't necessarily a live bug (core/decision.py masks FOLD out of live sampling when checking is free), but a rising trend over versions means the model's own preference is drifting worse and the live mask is doing more of the work to paper over it.",
    ),
    "air_folds_mostly": dict(
        what="Tests a clearly bad hand (~12% equity) facing a real bet across a range of stack depths.",
        expect="The model should fold more often than not -- weak hands facing money should mostly give up.",
        if_not="A model that keeps calling/raising with air is over-continuing with trash hands, bleeding chips broadly -- this was the historical V16 regression this check was built to catch.",
    ),
    "nuts_aggressive_mostly": dict(
        what="Tests a near-certain winning hand (~92% equity) facing a real bet across stack depths.",
        expect="The model should bet/raise more often than not -- strong hands should build the pot, not slow-play into passivity.",
        if_not="A model that mostly calls/folds here is leaving value on the table with its best hands -- a passive-value-hand bug.",
    ),
    "deep_stack_ood_guard": dict(
        what="Re-creates the exact spot that caused a real-money mistake: a marginal hand (35-55% equity) at a moderate-to-deep stack (15-40bb) facing one modest bet from a single opponent.",
        expect="ALL-IN should never be the model's top choice here -- shoving a marginal hand for stacks this deep into a small bet is a severe, high-variance mistake.",
        if_not="If ALL-IN wins, the model is repeating the live incident that motivated this check (a K9o 20bb trash-jam) -- a hard, deploy-blocking regression, not a stylistic quirk.",
    ),
    "short_stack_polarization": dict(
        what="Tests short-stack (6-10bb) spots with decent equity (50-65%) facing a raise sized near the whole stack -- a classic 'push or fold' situation.",
        expect="CALL should be rare -- with a stack this short and a raise this big, the correct play is almost always shove or fold, not a flat call sitting in between.",
        if_not="A high call rate means the model hasn't learned short-stack push/fold theory and is leaking value through a 'mushy middle' action -- tracked as an open backlog item, not yet gating deploys (WARN-only).",
    ),
    "action_diversity": dict(
        what="Sweeps equity and stack together and checks which action wins (argmax) across the whole grid.",
        expect="Different regions of the grid should prefer different actions. A HEALTHY model can still have only 3-4 actions ever win (e.g. raise buckets legitimately never peak -- a known 'no middle gear' pattern) as long as no ONE action dominates almost everything.",
        if_not="If one single action wins over ~85%+ of the whole grid regardless of equity/stack, that's a policy collapse -- the model has stopped discriminating between situations, historically caused by a 'raise-everything' or 'call-everything' training failure.",
    ),
    "no_nan_or_crash": dict(
        what="Feeds the model deliberately extreme/edge-case inputs (near-zero stacks, zero opponents, huge pots, max street) that a real table could occasionally produce.",
        expect="The model should always return a valid, finite policy -- no crashes, no NaN outputs.",
        if_not="A NaN or crash means the model would go completely unresponsive (or the live client would error out) the moment a rare real-table edge case occurs -- previously caused by a padding-mask bug, now a permanent regression guard.",
    ),
    "hand_strength_sensitivity": dict(
        what="Artificially swings the model's 'hand_strength' input (how strong the raw two cards are, independent of position/opponents) between weak and strong while holding equity fixed -- an ablation that could never happen in real play, used to check whether the model actually reads that input slot.",
        expect="The policy should shift meaningfully between the weak and strong settings -- proof the feature carries real weight in the network, not just padding along for the ride.",
        if_not="A flat/negligible response means this input may be dead weight -- either never learned as useful, or a wiring bug feeding it to the wrong place. WARN-only: no established 'correct' amount of movement yet, this is a first-pass load-bearing check.",
    ),
    "equity_edge_sensitivity": dict(
        what="Same idea as hand_strength, but for 'equity_edge' -- a precomputed 'how much better than average for this field size is my equity' signal. Swings it independently of equity/opponent-count to isolate whether the network reads it.",
        expect="The policy should respond meaningfully to this input on its own.",
        if_not="A flat response suggests the feature may be redundant with (fully derivable from) equity + opponent-count, and the network never bothered to use it separately -- not dangerous, just a wasted input, worth knowing before investing more design effort in it.",
    ),
    "committed_sensitivity": dict(
        what="Holds equity/stack/pot/field fixed and swings only how much an opponent has ALREADY put into this hand's pot -- from nothing (0bb) to a substantial chunk (60% of their stack) -- checking whether the policy responds.",
        expect="The policy should respond meaningfully -- an opponent who already has 60% of their stack in the pot this hand reads very differently from one who hasn't put in anything yet, even at identical remaining stack and hand equity.",
        if_not="A flat response suggests this new feature isn't load-bearing yet (may need more training exposure, or the network is deriving everything it needs from opp_stack/call_amount already) -- not dangerous on its own, but means the entry-sizing addition isn't paying off yet.",
    ),
    "pot_type_sensitivity": dict(
        what="Holds equity/stack/pot/call fixed and swings only whether this hand has been limped/unraised vs 3-bet (or more) -- checking whether the policy responds to the STRUCTURE of the action, not just the raw money amounts.",
        expect="The policy should respond meaningfully -- a 3-bet pot implies a much stronger range behind the bet than a limped pot, even at an identical call price.",
        if_not="A flat response suggests pot_type isn't load-bearing yet -- the network may already be deriving everything it needs from call_amount/committed, or hasn't had enough training exposure to use this new signal.",
    ),
    "stack_full_sweep": dict(
        what="Sweeps stack depth from very short (5bb) to very deep (180bb) at a fixed, middling equity spot with a bet sized proportionally to the stack, watching whether/how the chosen action changes.",
        expect="SOME meaningful change in behavior across that huge range -- a stack-blind model would be a real gap, since push/fold, pot-control, and deep-stack play all call for different approaches.",
        if_not="A totally flat policy across 5bb-180bb means stack depth isn't influencing decisions at all. For V20-family models a flat TAIL past ~50bb is an accepted, deliberate side effect of a live-safety clamp -- but flatness across the WHOLE range (including 5-40bb, never clamped) would be a real defect.",
    ),
    "position_sweep": dict(
        what="Sweeps table position (0 through 5) at a fixed marginal spot, checking whether fold rate changes at all by seat.",
        expect="Some spread -- position is one of the most basic levers in poker strategy (tighter early, looser on the button), so a healthy model should show at least a little sensitivity.",
        if_not="A flatline means the model isn't differentiating by seat at all in this spot -- a hardcoded/default-position bug existed once before (since fixed in V19) and this check exists specifically to catch a repeat.",
    ),
    "opponent_style_sweep": dict(
        what="Holds hero's own equity fixed and swaps only the opponent's read style (from a tight/aggressive 'nit' to a loose/aggressive 'maniac') while facing an identical bet, checking whether fold rate shifts.",
        expect="Facing the SAME bet, hero should fold more against a villain whose range is scarier (tighter-but-betting = stronger range) and less against a looser villain, at the same equity.",
        if_not="No shift at all means the model may be ignoring the opponent-style inputs entirely and just reacting to its own hand/equity in a vacuum -- a real exploitability gap (a fixed strategy is easier to beat than an adjusting one), and ties into a known open backlog item about per-opponent modeling.",
    ),
    "opponent_color_isolated_ablation": dict(
        what="Follow-up to opponent_style_sweep: pushes the opponent VPIP/AGG inputs to synthetic extremes (0.0 vs 1.0, well past any real archetype) and tests the table-level scalar and the per-seat block separately, rather than sweeping both together across realistic values.",
        expect="If opponent_style_sweep found no response within the realistic range, this check tells us why: a response HERE (at extremes) but not there means the network can read the input but the training population never taught it to care within realistic bounds. No response even here means the input is likely dead/unwired.",
        if_not="Flat even at 0.0 vs 1.0 is stronger evidence of a genuine wiring/normalization bug than opponent_style_sweep alone could show -- worth a direct code audit of how VPIP_MAP/AGG_MAP values reach the model before assuming it's purely a training-population artifact.",
    ),
    "allin_exploits_opponent_foldiness": dict(
        what="Sweeps hero's own all-in probability across equity, feeding the EXACT VPIP/AGG of the four real named archetypes that populate the training pool (NIT/TAG/LAG/CALLING_STATION) -- not an abstract Blue-Red spectrum, the literal opponents hero actually trains against.",
        expect="Since these four archetypes have wildly different real fold-to-a-shove rates AT A GIVEN EQUITY (e.g. NIT folds an all-in ~98% of the time at a realistic price when it genuinely holds marginal equity, CALLING_STATION ~0%, driven by a `fold_to_pressure` trait the model is never directly shown), a well-exploiting policy should shove NIT somewhat more often than CALLING_STATION at the same hero equity.",
        if_not="A small spread is evidence the model can't directly observe the underlying fold_to_pressure trait (it only infers tight-vs-loose via a coarse VPIP/AGG color). CAVEAT: don't over-read this as pure exploitable waste -- this sweep feeds a flat equity per cell and doesn't weight by how OFTEN each archetype actually presents that equity in real play; NIT's tight preflop selection means a genuine marginal-equity NIT spot is rarer in practice than the sweep implies, since NIT's real postflop range is already selection-skewed strong. See OFK known-shortcomings-backlog [OPP-8] for the full investigation, including this correction.",
    ),
    "allin_vs_nextbest_qgap": dict(
        what="Measures the raw Q-value gap between ALL-IN and whichever OTHER action the model rates second-best, reporting the WORST (max) cell in each slice -- broken down by stack depth (15-40bb) and by the 4 real named opponent archetypes (NIT/TAG/LAG/CALLING_STATION) -- not a single aggregate number, a full breakdown table.",
        expect="This is the FIRST committed, reproducible version of a metric that's been quoted informally (as a 1.35x-1.78x ratio) since V21_auxhead -- every prior number was measured with a different one-off script. No established 'correct' magnitude yet; the value is in localizing WHERE any gap concentrates (which stack depths, which opponent reads) rather than re-measuring the same aggregate a sixth way. Reports the WORST cell deliberately, not an average across the equity sweep -- an early version of this check averaged and hid a real spike (all-in swinging from strongly disfavored to strongly favored over a narrow equity band at one specific stack depth) under four other correctly-behaved cells.",
        if_not="A large gap surviving in a specific slice of the breakdown (e.g. concentrated at one stack depth or one archetype) is a much sharper lead for a targeted fix than the old aggregate ratio ever was -- see [BET-1] in the OFK backlog for the investigation this feeds.",
    ),
    "hand_strength_sweep": dict(
        what="Full 5-point curve version of hand_strength_sensitivity (same ablation, more points) so a flatline OR a non-monotonic kink is visible on a chart, not just one aggregate number.",
        expect="Same as hand_strength_sensitivity: a meaningful, ideally smooth response across the curve.",
        if_not="Same as hand_strength_sensitivity -- WARN-only, diagnostic.",
    ),
    "equity_edge_sweep": dict(
        what="Full multi-point curve version of equity_edge_sensitivity (same ablation, more points) so a flatline OR a non-monotonic kink is visible on a chart, not just one aggregate number.",
        expect="Same as equity_edge_sensitivity: a meaningful, ideally smooth response across the curve.",
        if_not="Same as equity_edge_sensitivity -- WARN-only, diagnostic.",
    ),
    "vpip_adapts_to_style": dict(
        what="Runs real simulated hands against an all-tight ('nit') table vs an all-loose ('fish') table, and measures how often hero voluntarily enters a pot (VPIP) against each.",
        expect="Hero's VPIP should be noticeably HIGHER against the loose/weak table than the tight table (by at least ~5 percentage points) -- playing more hands where opponents are more likely to be weak, fewer where they're not.",
        if_not="A flat VPIP regardless of opponent tightness means hero isn't adapting its starting-hand selection to the table at all -- the exact bug this check was built to catch (a real historical regression).",
    ),
    "bb100_vs_standard_fields": dict(
        what="Runs real simulated hands against four standard opponent-field presets (loose/tight x short/deep stacks) and measures hero's win rate (BB/100), compared against the last recorded baseline for this version.",
        expect="Win rate should hold roughly steady or improve versus the stored baseline in all four fields.",
        if_not="A drop of more than 15 BB/100 in any field vs baseline is flagged as a quiet regression -- the model got WORSE against a standard opponent mix, even though nothing else may be failing.",
    ),
    "beats_frozen_predecessor": dict(
        what="Plays real simulated hands against a mixed field that includes a frozen snapshot of this version's immediate predecessor, so the new model literally plays against the old one.",
        expect="The new version should have a positive win rate (profit) in that field -- a straightforward 'did we actually get better' test.",
        if_not="A loss (negative BB/100) here means the new version may not actually be an improvement over what it's replacing -- a serious deploy-readiness concern.",
    ),
    "beats_offformula_stress": dict(
        what="Swaps in a structurally different opponent (a discrete equity-tier lookup bot, not just a reweighted version of the same continuous pot-odds formula every training opponent uses) to see if hero's win rate holds up against a genuinely different playing style.",
        expect="Win rate should stay positive or only mildly negative -- evidence the model learned general poker principles, not just how to beat its own training-opponent formula.",
        if_not="A big win-rate collapse (below -30 BB/100) is a strong overfitting signal -- the model may have learned to exploit quirks specific to the training bots' math rather than transferable strategy.",
    ),
}

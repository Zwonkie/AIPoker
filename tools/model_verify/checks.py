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


FAST_CHECKS = [
    ("equity_ablation_monotonic", "P(fold) falls / P(aggressive) rises as equity rises",
     "general health", check_equity_ablation_monotonic),
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
    ("action_diversity", "at least most actions appear as argmax somewhere in the grid",
     "V11 raise-/call-everything collapse", check_action_diversity),
    ("no_nan_or_crash", "edge-case seat/stack/street combos never NaN or throw",
     "commit 55a1bc9 -- NaN inference crashes", check_no_nan_or_crash),
    ("hand_strength_sensitivity", "policy responds to hand_strength at fixed equity (SKIP if context_dim<37)",
     "V20_preflopEq -- is the new feature actually load-bearing", check_hand_strength_sensitivity),
    ("equity_edge_sensitivity", "policy responds to equity_edge at fixed equity/field (SKIP if context_dim<37)",
     "V20_preflopEq -- is the new feature actually load-bearing", check_equity_edge_sensitivity),
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

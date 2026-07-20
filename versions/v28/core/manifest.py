"""V28 manifest -- SAME architecture/contract/opponent pool as V27 (context_dim=44,
contract_version=7 -- see versions/v25/SPECS.md for the multi-street `_rollout_continuation_ev`
mechanism, versions/v26/SPECS.md for the TreeOpponent real-data pool, versions/v27/SPECS.md for the
[VAL-3]/[OPP-7] fixes).

V28 changes ONLY `_mc_target_evs_sized`'s per-size EV target: adds a closed-form risk/variance
penalty, applied UNIFORMLY to every sized action (raise_33/raise_66/raise_pot/allin alike), not an
`is_allin`-special-cased patch. Motivated by [BET-1]: a permanent `allin_vs_nextbest_qgap`
`model_verify` check (added in V26/V27's own diagnostic work) showed the all-in-vs-next-best Q-gap
gets WORSE the deeper the stack -- backwards from correct theory. Root cause: the target is a raw
point-estimate EV with no risk-awareness, and because a bigger bet's `raise_size` scales with stack
while its outcome variance scales too, the same marginal equity edge produces a linearly bigger raw
EV number at deeper stacks with nothing counteracting the fact that all-in is a much higher-variance
action than a smaller raise.

Fix: `Var[X] = E[X^2] - E[X]^2` computed IN CLOSED FORM from the SAME three quantities
`_mc_target_evs_sized` already computes per size (`p_all_fold`, `true_equity`,
`base_pot_if_called`/`raise_size`) -- no new sampling. `risk_adjusted_ev = raw_ev -
RISK_AVERSION_COEFFICIENT * sqrt(Var[X])`, subtracted before `evs.append(...)` for every sized
action. All-in's `raise_size` is far larger than a smaller raise's, so its variance (which scales
with `raise_size^2`) is naturally far larger too -- the SAME coefficient penalizes all-in the most
because its outcome genuinely IS riskier, not because of a hardcoded carve-out.

Calibrated via a standalone script (mirrors versions/v23/self_play/calibrate_bet1.py /
versions/v25/self_play/calibrate_multistreet_ev.py's established pattern) BEFORE this training run
-- see versions/v28/SPECS.md for the calibration result and chosen coefficient.

Base: copied from `versions/v27` (VAL-3/OPP-7 fixes, all TreeOpponent/real-data-pool
infrastructure, deep-stack curriculum, entry-sizing, `pot_type`, multi-street EV fix all inherited
unchanged). Fresh weights, no `--resume_path`, per [VAL-5]. Kept V27 (not V26) as the base per
explicit user decision (2026-07-19): V27's two fixes were both verified correct in isolation before
training, despite V27's own run showing a real regression cluster (VPIP doubled, action diversity
narrowed, position_sweep newly WARN) alongside its head-to-head win over V26 -- see
versions/v27/SPECS.md "Results" and OFK backlog [OPP-7] for the full accounting; NOT re-litigated
here.

See: versions/v27/SPECS.md ([VAL-3]/[OPP-7] + the regression cluster) | versions/v26/SPECS.md
(opponent pool) | versions/v25/SPECS.md (EV-fix mechanism) |
.agents/skills/OFK/references/known-shortcomings-backlog.md ([BET-1])
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v28",
    context_dim=44,                  # UNCHANGED from V25/V26/V27 -- no context/feature change
    contract_version=7,              # UNCHANGED -- training-loop-only change
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v28.core.model:PokerEVModelV4",
    contract_class="versions.v28.core.contract:ContractV12",
    weights_dir="versions/v28/weights",
    status="training",
    milestone=False,
)

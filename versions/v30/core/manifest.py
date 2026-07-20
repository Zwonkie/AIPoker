"""V29 manifest -- two changes, both flagged as V29's scope by explicit user direction
(2026-07-20): [OPP-2] per-opponent-seat raise attribution (a real context/architecture change) and
a critic-consistency filter on the actor's training target (a training-loop-only change, no
contract impact). Base: copied from `versions/v28` (risk-adjusted target, all TreeOpponent/
real-data-pool infrastructure, deep-stack curriculum, entry-sizing, `pot_type`, multi-street EV fix
all inherited unchanged -- see versions/v28/SPECS.md).

## Change 1: [OPP-2] per-opponent-seat raise attribution (context_dim 44->54, contract_version 7->8)

Before V29, the model only ever saw "someone raised this hand" via `pot_type` (a hand-level
aggregate bucket: limped/single-raised/3-bet+) -- never WHICH specific opponent seat did it. A
seat's only per-seat signal was its STATIC, cross-hand VPIP/AGG HUD color -- no way to tell "this
specific villain has been the aggressor all hand" from "this villain limped in and everyone else
did the raising." Flagged since the V16 ROADMAP as [P6]/[OPP-2], unaddressed until now.

Fix: two new per-opponent-seat boolean arrays, threaded the same way `committed`/
`opponents_committed` already are (simulator.py's per-hand betting loop -> `add_decision` ->
HandRecordV4.decision_points -> vectorize_hand_samples, AND simulator.py's `table_state` dict ->
`_query_model_decide` -> `SeatState`, for every OTHER NN-backed opponent's own queries too, not
just hero's):
  - `raised_this_hand[seat]`: has this specific seat raised at least once so far this hand.
  - `raised_this_street[seat]`: did this specific seat raise on the CURRENT betting street.
Both reset at the natural points (`raised_this_hand` once per hand alongside `committed`;
`raised_this_street` once per street alongside `street_committed`), set True in BOTH the hero's own
raise branch and the opponent-bot raise branch of the betting loop.

Appended (not inserted -- every existing index 0-43 is stable) as 10 new context features,
ctx[44:49]=`opp_raised_this_hand` and ctx[49:54]=`opp_raised_this_street` (same 5-seat order as the
existing per-opponent block). `core/board_state.py`'s shared `SeatState` dataclass gained the two
matching fields (`raised_this_hand`/`raised_this_street`, default False) -- optional/additive, inert
for every earlier version's contract exactly like `committed` was before V22.

`tools/model_verify/scenarios.py`'s `build_ctx` (the FAST-check synthetic context builder) extended
for `contract_version>=8` with matching optional params, both defaulting to all-zero so every
existing FAST check's call sites (which don't pass them) are unaffected.

**Known limitation, deliberately deferred**: this fix is REAL and FUNCTIONAL for training (verified
via versions/v29/self_play/simulator.py's own hand loop, exercised by every training hand and by
`model_verify --full`, which runs entirely against the version's own simulator -- see
tools/model_verify/run.py). It is NOT yet functional in LIVE serving: `core/table_state.py`
(PHPHelp.py's live game-state tracker) has no per-seat raise/aggression tracking of any kind today
(confirmed by direct code search before starting this) -- extending it safely would mean touching
the SAME live game-state code the currently-active V28 model depends on, unsupervised, with no user
present to catch a live-play regression. Deferred as flagged follow-up rather than risked this pass;
V29's live bridge (if/when wired) will feed these 10 features as a constant 0 (inert), same
degraded-but-safe posture `committed`/`hero_committed`/`pot_type` were ALREADY silently in live
(discovered as a byproduct of this investigation -- see versions/v29/SPECS.md "Also found" section).

## Change 2: critic-consistency filter (training-loop only, no contract impact)

`regret_match_policy_torch`'s fold-relative regret matching gives ALLIN real actor-target
probability mass any time its Q merely beats the ~0 fold baseline -- it never compares ALLIN
against the actual BEST alternative action. New `critic_consistency_margin` (config knob,
0.0=off): if any OTHER action's Q beats ALLIN's own Q by more than this margin, ALLIN's regret is
zeroed outright. Calibrated against the frozen V28 checkpoint's real Q-values across
`deep_stack_ood_guard`'s own eq x stack grid (versions/v29/self_play/calibrate_critic_consistency.py) --
**found to be a PARTIAL fix, documented honestly**: at eq=0.43 (all 5 stack depths), V28's own
critic already ranks ALLIN below RAISE_POT by a wide margin while still clearing the fold baseline
-- exactly the spurious-weight case this fixes. But at eq=0.48/0.55 (the check's OTHER 10 failing
cells), ALLIN is the critic's own genuine Q-argmax -- there this filter is a correct no-op, since
the problem lives in what the CRITIC learned, not in how the actor's target is built from it. See
`risk_aversion_coefficient`'s bump (0.10->0.15) alongside this for that other half. Applied ONLY to
ALLIN specifically (not a general all-pairs dominance rule) -- an all-pairs version, tested against
the same calibration data, collapsed legitimate raise-size mixing (call/raise_33/raise_66/raise_pot)
to a single surviving action in the large majority of grid cells, a much bigger unintended hit to
`action_diversity`/[BET-2] than intended. See `regret_match_policy_torch`'s own docstring in
versions/v29/self_play/train.py for the full derivation.

See: versions/v28/SPECS.md (risk-adjusted target, BET-1 mechanism) | versions/v27/SPECS.md
([VAL-3]/[OPP-7]) | .agents/skills/OFK/references/known-shortcomings-backlog.md ([BET-1], [OPP-2])
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v30",
    context_dim=54,                  # unchanged from V29 -- [V30, BET-3] is a training-TARGET fix only
    contract_version=8,              # unchanged: no tensor-schema change, so v29/v30 are contract-compatible
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v30.core.model:PokerEVModelV4",
    contract_class="versions.v30.core.contract:ContractV12",
    weights_dir="versions/v30/weights",
    status="training",
    milestone=False,
)

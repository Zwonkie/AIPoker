"""V24_extreme manifest -- a deliberately EXTREME, throwaway DIAGNOSTIC, not a production
candidate. Same code/mechanism as V24 (decoupled EV-target fold model + `bot_bluff_perc`
"show of strength" bonus for non-all-in raises) -- only the PARAMETERS and training recipe change.

**Why**: V24's calibrated settings (`RAISE_RESPECT_BOOST=0.10`, realistic per-archetype
`bot_bluff_perc`) demonstrably created a real, non-degenerate fold-equity gradient favoring raises
over all-in when tested in ISOLATION (direct EV-arithmetic checks on `_ev_target_fold_decision`
itself) -- but a full 150k-hand retrain showed NO improvement on `action_diversity`/
`deep_stack_ood_guard`, and the allin-vs-next-best Q-value gap actually WIDENED at every tested
stack depth (see versions/v24/SPECS.md). Ambiguous result: is the mechanism fundamentally unable to
overcome all-in's other advantages at this population/architecture, or was the calibrated
magnitude simply too subtle to show up over a full, complex training run alongside everything
else? This diagnostic isolates that question.

**What's different from V24**:
- `bot_bluff_perc` pushed to near-0 for ALL FOUR archetypes (maximizing "respect" probability to
  ~98%, not aiming for a realistic calibration) -- see `opponent_bots.py`.
- `RAISE_RESPECT_BOOST` pushed far above the calibrated 0.10 -- see `simulator.py`.
- `target_hands: 50000` (fast turnaround) instead of 150000.
- Simplified curriculum: `stack_depth_mix` removed in favor of a single `fixed_stack_bb` (~35bb,
  matching `deep_stack_ood_guard`'s own failure zone) and `disable_bootstrap: true` (skip the
  heuristic-anchoring warmup entirely), so training converges quickly on just the one variable
  under test instead of interacting with a full multi-phase curriculum.

**Reading the result**: if THIS run also shows no movement on `action_diversity`/the Q-value gap,
that's strong evidence the opponent-response-shaping approach is a structural dead end for this
problem (redirect to an explicit overbet EV discount in the target computation instead, per the
OFK backlog's revised suggestion). If it DOES move the needle at this extreme setting, that
confirms the mechanism is viable and the search should shift to finding a calibrated value between
V24's (too weak) and this run's (deliberately too strong, not production-safe) settings.

Base: copied from `versions/v24` (`versions/v24/SPECS.md`, `versions/v23/SPECS.md`, and the OFK
backlog's `[BET-1]` entry have the full history/root-cause chain this diagnostic follows from).

See: versions/v24/SPECS.md | .agents/skills/OFK/references/known-shortcomings-backlog.md [BET-1] |
versions/v24_extreme/SPECS.md (this diagnostic's own results)
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v24_extreme",
    context_dim=44,                  # UNCHANGED from V24 -- no context/feature change
    contract_version=7,              # UNCHANGED from V24 -- target-EV computation only, no contract change
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v24_extreme.core.model:PokerEVModelV4",
    contract_class="versions.v24_extreme.core.contract:ContractV12",
    weights_dir="versions/v24_extreme/weights",
    status="diagnostic",
    milestone=False,
)

"""V21 manifest — clone of V20_preflopEq_AI (versions/v20_preflopEq_AI). IDENTICAL architecture/
tensor schema (context_dim=37, contract_version=5, same PokerEVModelV4 arch) -- this is a
STRUCTURAL CLEANUP build, not a new-behavior experiment. Same opponent pool composition too
(deliberately unchanged, so any behavior delta is attributable to the cleanup, not a confound).

Motivation: a structural-soundness review of the core train/sim loop (independent of any specific
model-behavior bug) found several places where the actor-target pipeline, exploration mix, and
curriculum timing had accumulated dead branches or duplicated-but-drifting mechanisms rather than
one clean mechanism per job. See `versions/v21/SPECS.md` for the full itemized list and rationale;
summary of what changed vs v20_preflopEq_AI:

  1. Deleted the critic-side preflop tightness prior (`TIGHTNESS_PENALTY_BB`/`ENTRY_EQUITY_MARGIN`
     + the `disable_target_shaping` flag end-to-end) -- superseded by range-aware equity, and
     validated dormant since V17.
  2. `model_share: 0.80 -> 0.95` (eps stays 0.05) -- removes the permanent 15% steady-state
     heuristic-anchor floor past bootstrap; exception-path fallback on a model-query error is
     unaffected.
  3. Phase-4 dynamic-active-players fold-weights reweighted toward 3-5-handed starting tables
     (`[0.40,0.25,0.20,0.10,0.05] -> [0.10,0.25,0.30,0.25,0.10]` for `[0,1,2,3,4]` folded).
  4. Bootstrap/cutover timing shifted 10k hands earlier: bootstrap decay 10k-30k -> 5k-20k;
     `ACTOR_CRITIC_CUTOVER_HANDS` 30k -> 20k; Phase-4 threshold 50k -> 40k; Phase-5 (disabled)
     75k -> 65k; dashboard phase labels fixed to reflect what's actually running.
  5. Actor-target pipeline: the realization-discount math (`POLICY_TIGHTNESS_BB`/`_PIVOT`) was
     implemented twice independently (dataset-time in `vectorize_hand_samples`, live in
     `regret_match_policy_torch`) -- factored into one shared helper. `Y_pol`/`pol_t` (the
     pre-cutover dataset policy target) is no longer computed at all once
     `hands_done >= ACTOR_CRITIC_CUTOVER_HANDS`, since the post-cutover loss never reads it
     (was previously computed and discarded for ~87% of a 150k run).
  6. Live-serve sampling temperature (`_stack_scaled_temperature` in `core/decision.py`) --
     DEFERRED to this version's deploy step, not part of this training-code change (see
     SPECS.md item 6b): that file is shared/not version-namespaced, and this version isn't wired
     into live serving yet.

`beats_frozen_predecessor` benchmark: `frozen_v20_preflopEq_AI.pth` (this version's immediate
parent's final 150k-hand model, SAME architecture) -- should run cleanly, no scale mismatch.

See: versions/v20_preflopEq_AI/SPECS.md (parent) | versions/v21/SPECS.md (this version)
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v21",
    context_dim=37,                 # UNCHANGED from v20_preflopEq_AI -- same tensor schema
    contract_version=5,              # UNCHANGED -- structural cleanup only, no contract change
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v21.core.model:PokerEVModelV4",
    contract_class="versions.v21.core.contract:ContractV12",
    weights_dir="versions/v21/weights",
    status="active",
    milestone=False,
)

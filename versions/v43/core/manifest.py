"""V43 manifest -- corrective-prior cleanup: back to learning from correct inputs.

Base: cloned from `versions/v41` (fresh weights, not resumed -- per [VAL-5]).

Contract is BYTE-IDENTICAL to V29/V40/V41 (context_dim=54, contract_version=8): every change is in
the TRAINING LOOP, never in the tensor schema. Checkpoints stay contract-compatible and the live
bridge needs no new wiring.

V40/V41 fixed root causes in the training DATA (betting round no longer ends on a check; CALL no
longer exempt from the variance penalty/continuation credit; dead blinds, degraded NN opponents,
symmetric stacks, the min-raise rule, [OPP-7] at the tensor boundary). Several training-loop knobs
predate those fixes and existed to suppress behaviour the broken data was producing. V43 removes the
ones that impose an answer the model should LEARN, keeps the one measurement says is load-bearing,
and re-scales the two that are coupled:

  REMOVED  realization discount (`policy_tightness_bb`, V12) -- imposed entry tightness. Measured
           first: no effect at depth, but entry rate at eq<=0.35 goes 0.59 -> 0.79 without it, so
           EXPECT entry range to widen and check it.
  REMOVED  ALLIN critic-consistency veto (`critic_consistency_margin`, V29) -- measured near-inert
           (ALLIN target share 0.37-0.43 with, 0.41-0.43 without).
  KEPT     variance penalty (`risk_aversion_coefficient`) -- its [BET-1] pathology is NOT gone at
           the source: removing it triples the ALLIN-vs-next-best gap's growth with stack depth.
  CHANGED  TARGET_CLIP_BB 40 -> 100, matching STACK_CEIL_BB and the curriculum ceiling (review
           T-M5). At 40 the clip truncated 23.4% of realized go-forward returns.
  CHANGED  risk_aversion_coefficient 0.15 -> 0.20, REQUIRED BY the clip change -- the 40bb clip was
           an undeclared deep-stack all-in dampener, so 0.15 only covered what it left over.

Both removed knobs FAIL LOUD if a config still sets them -- a removed knob must never be a silent
no-op. See versions/v43/SPECS.md for every measurement behind these.
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v43",
    context_dim=54,                  # unchanged from V29 -- no contract change in this version
    contract_version=8,              # unchanged from V29 -- V29/V40 checkpoints are contract-compatible
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v43.core.model:PokerEVModelV4",
    contract_class="versions.v43.core.contract:ContractV12",
    weights_dir="versions/v43/weights",
    status="active",                 # NOT TRAINED YET -- V41 is still the live model
    milestone=False,
)

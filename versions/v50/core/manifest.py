"""V50 manifest -- wider seat x depth curriculum, 250k hands.

Base: cloned from `versions/v48` (fresh weights, not resumed -- per [VAL-5]).

CONTRACT UNCHANGED (context_dim=54, contract_version=9 -- identical to V48/V47/V44): V50 is a
CURRICULUM-ONLY change. The only edit vs V48 is `table_stack_joint_mix` in config.yaml (owner
targets, 2026-07-23): seats restricted to the real DoN range 3-6 (heads-up cut; floor = hero +
2 opponents), depths 2-100bb with a short/tight 3-handed endgame row, marginals fitted from the
full 4,492-hand hero corpus. No simulator, contract, target, or veto change -- V48's slice is
byte-identical except the mix and target_hands. Feature semantics/index/scale are V48's, so the
live bridge, live_features(), and v48_engine carry over unchanged; a v50_engine is a
declaration-only clone at deploy time. See versions/v50/SPECS.md for the full rationale and the
deep_stack_ood_guard OOD note.
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v50",
    context_dim=54,                  # unchanged
    contract_version=9,              # unchanged -- no tensor-schema change (curriculum-only)
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v50.core.model:PokerEVModelV4",
    contract_class="versions.v50.core.contract:ContractV12",
    weights_dir="versions/v50/weights",
    status="active",                 # NOT TRAINED YET -- V48 is the live model / rollback
    milestone=False,
)

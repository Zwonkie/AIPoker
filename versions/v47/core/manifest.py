"""V47 manifest -- opponent-behavior realism + target alignment.

Base: cloned from `versions/v44` (fresh weights, not resumed -- per [VAL-5]).

CONTRACT UNCHANGED (context_dim=54, contract_version=9 -- identical to V44): every V47 change is
simulator/training-side (opponent raise-size repertoire [#6], chip-identical bucket collapse [M9],
occupant-true fold models [M4], sub-5bb curriculum band [VAL-1(A)], training-loop hygiene
[M6/M7]). Feature semantics, index layout, and scaling are byte-for-byte V44's, so V44's live
game-state work, bridge, and live_features() carry over and the V47 engine is a declaration-only
clone of v44_engine.py. See versions/v47/SPECS.md for the full scope, verification plan, and
acceptance gates.
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v47",
    context_dim=54,                  # unchanged -- ctx[35] changes MEANING, not width
    contract_version=9,              # BUMPED from 8: ctx[35] semantics changed, V43 weights are
                                     # not behaviourally compatible even though they would load
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v47.core.model:PokerEVModelV4",
    contract_class="versions.v47.core.contract:ContractV12",
    weights_dir="versions/v47/weights",
    status="active",                 # NOT TRAINED YET -- V44 is the registered candidate, V43 live
    milestone=False,
)

"""V47 manifest -- opponent-behavior realism + target alignment.

Base: cloned from `versions/v44` (fresh weights, not resumed -- per [VAL-5]).

CONTRACT UNCHANGED (context_dim=54, contract_version=9 -- identical to V44): every V47 change is
simulator/training-side (opponent raise-size repertoire [#6], chip-identical bucket collapse [M9],
occupant-true fold models [M4], sub-5bb curriculum band [VAL-1(A)], training-loop hygiene
[M6/M7]). Feature semantics, index layout, and scaling are byte-for-byte V44's, so V44's live
game-state work, bridge, and live_features() carry over and the V47 engine is a declaration-only
clone of v44_engine.py. See versions/v48/SPECS.md for the full scope, verification plan, and
acceptance gates.
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v48",
    context_dim=54,                  # unchanged -- ctx[35] changes MEANING, not width
    contract_version=9,              # BUMPED from 8: ctx[35] semantics changed, V43 weights are
                                     # not behaviourally compatible even though they would load
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v48.core.model:PokerEVModelV4",
    contract_class="versions.v48.core.contract:ContractV12",
    weights_dir="versions/v48/weights",
    status="active",                 # trained 100k + deployed live 2026-07-23; now the rollback
                                     # behind V50 (curriculum retrain) and V50's frozen predecessor
    milestone=True,                  # MILESTONE (2026-07-24): 3rd ever (after V13, V41). The
                                     # table-geometry foundation -- true N-handed dealing, first
                                     # 3-max Nash axis, restored opponent-style; base of the whole
                                     # 3-6 seat line incl. V50. See versions/v48/MILESTONE.md.
                                     # Do NOT delete versions/v48/weights/expert_main.pth.
)

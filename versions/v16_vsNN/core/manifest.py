"""V16_vsNN manifest — EXPLORATORY side-branch of V16, not a production candidate.

Warm-started from versions/v16/weights/expert_main.pth (110,057 hands), continued 75k more hands
on a 3-seat table against TWO neural-net opponents instead of the usual heuristic pool: a true
self-play lagged mirror ('past') and a static frozen V15 checkpoint (repurposing the dormant
`nit_model` slot). See versions/v16_vsNN/SPECS.md for the full rationale (probing whether the hero
generalizes past the training-pool heuristics' specific formula shape, or has overfit to it).

Tensor + action schema is IDENTICAL to v16/v15 (context_dim=35, contract_version=3, same 6-action
space) -- no architecture changes here, only the opponent population + table size.

See: .agents/skills/OFK/references/versioned-architecture-guardrails.md
     versions/v16_vsNN/SPECS.md   |   versions/v16/SPECS.md (parent line)
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v16_vsNN",
    context_dim=35,                 # input schema UNCHANGED (35-feature context)
    contract_version=3,             # UNCHANGED from v15: same discretized 6-action space
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v16_vsNN.core.model:PokerEVModelV4",
    contract_class="versions.v16_vsNN.core.contract:ContractV12",
    weights_dir="versions/v16_vsNN/weights",
    status="active",
    milestone=False,                # v13 remains the kept milestone/fallback
)

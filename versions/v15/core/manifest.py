"""V15 manifest — clones V14 (versions/v14).

V15 keeps V14's discretized 6-action bet-size space and equity-primary architecture UNCHANGED;
it only widens the training stack distribution (DoN-shaped depth mixture) and adds a frozen-V14
expert opponent, over 200k hands. See versions/v15/SPECS.md.

The tensor + action schema is IDENTICAL to v14 (context_dim=35, contract_version=3, same 6-action
space), so v14 weights are load-compatible — used as the FROZEN opponent (frozen_v14.pth) and as an
optional warm-start (default run is FRESH; see SPECS "Training recipe decision").

See: .agents/skills/OFK/references/versioned-architecture-guardrails.md
     versions/v15/SPECS.md   |   versions/v13/VALIDATED_FINDINGS.md (inherited, still locked)
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v15",
    context_dim=35,                 # input schema UNCHANGED (35-feature context)
    contract_version=3,             # UNCHANGED from v14: same discretized 6-action space
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v15.core.model:PokerEVModelV4",
    contract_class="versions.v15.core.contract:ContractV12",
    weights_dir="versions/v15/weights",
    status="active",
    milestone=False,                # v13 remains the kept milestone/fallback
)

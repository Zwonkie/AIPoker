"""V12 version manifest — the single source of truth for what "v12" is.

The runtime loads v12 ONLY through this. To start v13: copy versions/v12 -> versions/v13,
clear versions/v13/weights, and edit the fields below (bump contract_version only if the
tensor schema changes).

See: .agents/skills/OFK/references/versioned-architecture-guardrails.md
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v12",
    context_dim=35,                 # matches ContractV8V9 (v12) / model state_proj input
    contract_version=2,             # inherited from v11's 35-feature contract (unchanged schema)
    action_space=("fold", "call", "raise"),
    model_class="versions.v12d.core.model:PokerEVModelV4",
    contract_class="versions.v12d.core.contract:ContractV12",
    weights_dir="versions/v12/weights",
    status="active",
)

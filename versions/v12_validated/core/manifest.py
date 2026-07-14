"""V12_VALIDATED manifest — the single source of truth for the VALIDATED V12 foundation.

This is the frozen, test-verified foundation that future versions (v13+) should copy.
Its fixes are documented and evidence-backed in VALIDATED_FINDINGS.md (same folder) and
must not be changed except through the testing workflow described there.

The runtime loads this ONLY through this manifest. To start v13: copy versions/v12_validated
-> versions/v13, clear versions/v13/weights, and edit the fields below (bump contract_version
only if the tensor schema changes — it did NOT change here, only the model internals did).

See: .agents/skills/OFK/references/versioned-architecture-guardrails.md
     versions/v12_validated/VALIDATED_FINDINGS.md
     versions/v13/SPECS.md
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v12_validated",
    context_dim=35,                 # UNCHANGED 35-feature context (contract schema is stable)
    contract_version=2,             # tensor schema unchanged from v11/v12; only model internals changed
    action_space=("fold", "call", "raise"),
    model_class="versions.v12_validated.core.model:PokerEVModelV4",
    contract_class="versions.v12_validated.core.contract:ContractV12",
    weights_dir="versions/v12_validated/weights",
    status="active",
)

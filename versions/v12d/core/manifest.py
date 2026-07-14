"""V12D — DEPRECATED scratch/diagnostic version.

This was the working copy used to diagnose and fix the V12 training foundation. Its
validated result has been promoted to `versions/v12_validated/` (the canonical foundation).
Kept only for its diagnostic history; do NOT build on it. Given a distinct version_id so it
no longer collides with the real "v12" in shared.registry.

See: versions/v12_validated/VALIDATED_FINDINGS.md
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v12d",
    context_dim=35,
    contract_version=2,
    action_space=("fold", "call", "raise"),
    model_class="versions.v12d.core.model:PokerEVModelV4",
    contract_class="versions.v12d.core.contract:ContractV12",
    weights_dir="versions/v12d/weights",
    status="deprecated",
)

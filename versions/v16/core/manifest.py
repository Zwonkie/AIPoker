"""V16 manifest — clones V15 (versions/v15).

V16 keeps V15's discretized 6-action bet-size space, equity-primary architecture, and stack
curriculum UNCHANGED. The only change is the preflop CALL/FOLD target: `_mc_target_evs_sized`
now uses range-aware equity (vs each opponent's VPIP-implied range) instead of oracle equity
vs literal dealt cards, specifically for street_idx==0, so the preflop entry decision finally
carries the opponent-tightness signal RAISE already had via fold-out probability. See
versions/v16/SPECS.md [P4].

The tensor + action schema is IDENTICAL to v15 (context_dim=35, contract_version=3, same 6-action
space), so v15 weights are load-compatible — used as the FROZEN opponent (frozen_v15.pth) and as an
optional warm-start.

See: .agents/skills/OFK/references/versioned-architecture-guardrails.md
     versions/v16/SPECS.md   |   versions/v13/VALIDATED_FINDINGS.md (inherited, still locked)
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v16",
    context_dim=35,                 # input schema UNCHANGED (35-feature context)
    contract_version=3,             # UNCHANGED from v15: same discretized 6-action space
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v16.core.model:PokerEVModelV4",
    contract_class="versions.v16.core.contract:ContractV12",
    weights_dir="versions/v16/weights",
    status="active",
    milestone=False,                # v13 remains the kept milestone/fallback
)

"""V14 manifest — built from the V13 milestone (versions/v13).

V14's spine is a discretized BET-SIZE action space (see versions/v14/SPECS.md). This first
step (P1b) makes the opponent bots SIZE-AWARE (fold more to bigger bets) so that sizing becomes
learnable; the action-space widening + per-size targets (P1/P1a) and the contract bump come next.

Until the action space widens, the tensor schema is UNCHANGED from v13 (context_dim=35,
contract_version=2), so v13 weights load as a warm start. Bump contract_version when the
{fold,call,raise_33,raise_66,raise_pot,allin} action space lands.

See: .agents/skills/OFK/references/versioned-architecture-guardrails.md
     versions/v14/SPECS.md   |   versions/v13/VALIDATED_FINDINGS.md (inherited, still locked)
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v14",
    context_dim=35,                 # input schema UNCHANGED (35-feature context); only the OUTPUT grows
    contract_version=3,             # bumped: discretized bet-size action space (6 actions) landed
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v14.core.model:PokerEVModelV4",
    contract_class="versions.v14.core.contract:ContractV12",
    weights_dir="versions/v14/weights",
    status="active",
    milestone=False,                # v13 remains the kept milestone/fallback; v14 is in-progress
)

"""Live inference engine for V20_preflopEq (versions/v20_preflopEq), 75k-hand checkpoint.

Same 6-action bet-size head order and PokerEVModelV4 arch as V20, but a WIDER context
(context_dim 35->37, contract_version 4->5): two new appended features, `equity_edge`
(equity*(num_active+1)) and `hand_strength` (preflop_equities.csv lookup / cheap postflop MC) --
see versions/v20_preflopEq/core/contract.py and SPECS.md. Also carries the Finding 2 fix (hero's
range-aware equity now splits opponents into front/already-acted [guaranteed in] vs
after/still-to-act [normal VPIP roll], instead of one flat roll for everyone) and a Finding 1 fix
(unknown HUD color -> Yellow instead of dropped, in PHPHelp.py's opp_colors construction --
applies regardless of active model). This is a DIFFERENT input SCALE+WIDTH than every prior sized
model -- CANNOT reuse `core/decision.py`'s `bridge_v13` OR `bridge_v20`; needs its own bridge built
from `versions.v20_preflopEq.core.contract`, gated by `is_v20_preflopEq` in decision.py.

Loads `expert_main.pth` (the 75k final checkpoint from the first production run).

**model_verify --full @ 75k (2026-07-17): 11 PASS / 1 WARN / 1 FAIL / 1 SKIP.**
`vpip_adapts_to_style` PASS (short +6.6pt, deep +7.1pt -- the metric most directly downstream of
the Finding 2 equity fix). `bb100_vs_standard_fields` PASS, positive across all 4 fields
(loose_short +16.8, loose_deep +36.2, tight_short +22.8, tight_deep +61.1 BB/100 -- recorded as
this version's baseline, no prior existed). `beats_offformula_stress` PASS (+31.3/+66.0 BB/100).
Both new-feature sensitivity checks PASS (hand_strength 0.107, equity_edge 0.641 avg policy shift)
-- confirmed load-bearing, not inert padding. `deep_stack_ood_guard` FAIL and `free_check_low_fold`
WARN are the SAME long-standing soft spots V19/V20 also carry at maturity -- not introduced or
worsened here. `beats_frozen_predecessor` SKIPs: copied V20's weights in as `frozen_v20.pth`
but it can't load (context_dim 35 vs 37) -- no per-model contract-selection mechanism exists yet,
so there is NO direct head-to-head number against V20 specifically, only the field/style/
generalization checks above.

Deployed as the active live model per explicit user decision (2026-07-17), accepting the missing
frozen-V20 comparison in exchange for the strong, broad field/style/generalization evidence above.
V20 (`V20ModelEngine`) stays in the registry, fully intact, as the rollback.
"""
import os
import torch

from versions.v20_preflopEq.core.model import PokerEVModelV4 as V20PreflopEqModel
from versions.v20_preflopEq.core.manifest import MANIFEST as V20_PREFLOPEQ_MANIFEST
from shared.manifest import load_state_dict as load_ckpt_state

# Same head order as V14/V15/V17/V17_gauntlet/V19/V20 (shared 6-action contract).
V20_PREFLOPEQ_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V20PreflopEqModelEngine:
    is_v20_preflopEq = True
    is_v20 = False
    is_v19 = False
    is_v17_gauntlet = False
    is_v17 = False
    is_v15 = False
    is_v14 = False
    is_v13 = False
    is_v11 = False

    def __init__(self, weight_name: str = "expert_main.pth", device: str = "cpu"):
        self.device = torch.device(device)
        self.model = V20PreflopEqModel().to(self.device)
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        weight_path = os.path.join(repo_root, "versions", "v20_preflopEq", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(weight_path, V20_PREFLOPEQ_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            self.loaded = False
            print(f"WARNING: could not load V20_preflopEq weights at {weight_path}: {e}. Model outputs garbage.")

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V20_PREFLOPEQ_ACTION_KEYS."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V20_PREFLOPEQ_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V20_PREFLOPEQ_ACTION_KEYS)}
        return {k: float(probs[i]) for i, k in enumerate(V20_PREFLOPEQ_ACTION_KEYS)}

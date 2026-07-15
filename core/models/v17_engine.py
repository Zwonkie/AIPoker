"""Live inference engine for the V17 model (versions/v17).

V17 has the IDENTICAL 6-action bet-size contract as V14/V15/V16 (same PokerEVModelV4 arch, same
35-dim ContractV12 input, same {FOLD,CALL,RAISE_33,RAISE_66,RAISE_POT,ALLIN} head order). It
differs only in TRAINING: the actor's regret-matching target is routed through the critic's own
(detached) Q-values once past 30k hands (fold-relative baseline), instead of a fresh noisy
per-hand simulator sample all the way through -- see versions/v17/SPECS.md for the full trace.
So live serving reuses the entire V14/V15 path -- same bridge, same `_v14_size_to_slider` sizing,
same executor mapping -- with just the weights swapped.

Validated (2026-07-15): `tools/model_verify --full` 10 PASS/1 WARN/1 FAIL (the one FAIL is the
same pre-existing deep-stack OOD defect every version in this line carries, tracked separately as
V18 [P0], not a V17 regression). Fixes the air/draws overcontinuation problem cleanly
(air_folds_mostly 0.62->1.00) WITHOUT v16_foldregret's style-flip regression -- vpip_adapts_to_style
PASSES (short +9.7pt / deep +5.8pt), and loose_deep BB/100 actually IMPROVES over V16 (+62.1->+90.3)
instead of collapsing like foldregret's -11.6. Beats frozen-V16 by +87.5 BB/100 over 4000 hands.
"""
import os
import torch

from versions.v17.core.model import PokerEVModelV4 as V17Model
from versions.v17.core.manifest import MANIFEST as V17_MANIFEST
from shared.manifest import load_state_dict as load_ckpt_state

# Same head order as V14/V15 (shared 6-action contract).
V17_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V17ModelEngine:
    is_v17 = True
    is_v15 = False  # distinct flag, but decision.py routes v14+v15+v17 through the same sized path
    is_v14 = False
    is_v13 = False
    is_v11 = False

    def __init__(self, weight_name: str = "expert_main.pth", device: str = "cpu"):
        self.device = torch.device(device)
        self.model = V17Model().to(self.device)
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        weight_path = os.path.join(repo_root, "versions", "v17", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(weight_path, V17_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            self.loaded = False
            print(f"WARNING: could not load V17 weights at {weight_path}: {e}. Model outputs garbage.")

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V17_ACTION_KEYS."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V17_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V17_ACTION_KEYS)}
        return {k: float(probs[i]) for i, k in enumerate(V17_ACTION_KEYS)}

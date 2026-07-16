"""Live inference engine for the V17_gauntlet model (versions/v17_gauntlet).

V17_gauntlet has the IDENTICAL 6-action bet-size contract as V14/V15/V17 (same PokerEVModelV4 arch,
same 35-dim ContractV12 input, same {FOLD,CALL,RAISE_33,RAISE_66,RAISE_POT,ALLIN} head order). It
differs from V17 only in TRAINING: the opponent pool was WIDENED (intended: scripted heuristics +
frozen V15 in the `nit` seat + frozen V16 in the `tag` seat + a true lagged self-play mirror in the
`past` seat). CORRECTION (found 2026-07-16): a wiring bug silently nullified the `tag` seat's model
load, so this checkpoint actually trained against the TAG heuristic bot there, not frozen V16 --
`nit` (frozen V15) and `past` (lagged mirror) worked as intended. See versions/v17_gauntlet/SPECS.md
"CORRECTION" for the full trace; fixed going forward in versions/v18's opponent-architecture
refactor. So live serving reuses the entire V14/V15/V17 path -- same bridge, same
`_v14_size_to_slider` sizing, same executor mapping -- with just the weights swapped.

Validated (2026-07-16): `tools/model_verify --full` 10 PASS/1 WARN/1 FAIL (the one FAIL is the same
pre-existing deep-stack OOD defect every version in this line carries, tracked as V18 [P0], not a
regression). Beats V17 on 3 of 4 `bb100_vs_standard_fields` fields (loose_short +28.9->+32.5,
tight_short +18.4->+26.8, tight_deep +32.6->+35.4; loose_deep came down +90.3->+69.8 but stayed
strongly positive -- reads as a more balanced policy, not a collapse). `vpip_adapts_to_style`
deep-stack delta more than doubled V17's (+5.8pt->+12.3pt). Beats its own immediate parent
frozen-V17 by +84.3 BB/100 over 4000 hands.
"""
import os
import torch

from versions.v17_gauntlet.core.model import PokerEVModelV4 as V17GauntletModel
from versions.v17_gauntlet.core.manifest import MANIFEST as V17_GAUNTLET_MANIFEST
from shared.manifest import load_state_dict as load_ckpt_state

# Same head order as V14/V15/V17 (shared 6-action contract).
V17_GAUNTLET_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V17GauntletModelEngine:
    is_v17_gauntlet = True
    is_v17 = False  # distinct flag, but decision.py routes v14+v15+v17+v17_gauntlet through the same sized path
    is_v15 = False
    is_v14 = False
    is_v13 = False
    is_v11 = False

    def __init__(self, weight_name: str = "expert_main.pth", device: str = "cpu"):
        self.device = torch.device(device)
        self.model = V17GauntletModel().to(self.device)
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        weight_path = os.path.join(repo_root, "versions", "v17_gauntlet", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(weight_path, V17_GAUNTLET_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            self.loaded = False
            print(f"WARNING: could not load V17_gauntlet weights at {weight_path}: {e}. Model outputs garbage.")

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V17_GAUNTLET_ACTION_KEYS."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V17_GAUNTLET_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V17_GAUNTLET_ACTION_KEYS)}
        return {k: float(probs[i]) for i, k in enumerate(V17_GAUNTLET_ACTION_KEYS)}

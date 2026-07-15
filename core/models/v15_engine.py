"""Live inference engine for the V15 model (versions/v15).

V15 has the IDENTICAL 6-action bet-size contract as V14 (same PokerEVModelV4 arch, same 35-dim
ContractV12 input, same {FOLD,CALL,RAISE_33,RAISE_66,RAISE_POT,ALLIN} head order). It differs only
in TRAINING: a DoN-shaped stack mixture (5-50bb, fixes v14's deep-stack OOD) + a frozen-V14 expert
opponent, over 200k hands. So live serving reuses the entire V14 path — same bridge, same
`_v14_size_to_slider` sizing, same executor mapping — with just the weights swapped.

Validated (2026-07-15): fixes v14's stack-cratering deep-stack jams (wins deep vs loose, beats
frozen-V14 at all depths), holds the short game. Loose-aggressive style — crushes loose/station
fields (the live population), softer vs tight deep tables (see versions/v16/SPECS.md [P4]).
"""
import os
import torch

from versions.v15.core.model import PokerEVModelV4 as V15Model
from versions.v15.core.manifest import MANIFEST as V15_MANIFEST
from shared.manifest import load_state_dict as load_ckpt_state

# Same head order as V14 (shared 6-action contract).
V15_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V15ModelEngine:
    is_v15 = True
    is_v14 = False   # distinct flag, but decision.py routes v14+v15 through the same sized path
    is_v13 = False
    is_v11 = False

    def __init__(self, weight_name: str = "expert_main.pth", device: str = "cpu"):
        self.device = torch.device(device)
        self.model = V15Model().to(self.device)
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        weight_path = os.path.join(repo_root, "versions", "v15", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(weight_path, V15_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            self.loaded = False
            print(f"WARNING: could not load V15 weights at {weight_path}: {e}. Model outputs garbage.")

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V15_ACTION_KEYS."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V15_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V15_ACTION_KEYS)}
        return {k: float(probs[i]) for i, k in enumerate(V15_ACTION_KEYS)}

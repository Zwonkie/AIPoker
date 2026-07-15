"""Live inference engine for the V14 model (versions/v14).

V14 extends V13 with a DISCRETIZED BET-SIZE action space. The actor now outputs a
6-way policy instead of {FOLD,CALL,RAISE}:

    index:  0     1      2         3         4          5
    action: FOLD  CALL   RAISE_33  RAISE_66  RAISE_POT  ALLIN
            (fold)(call)  (0.33pot) (0.66pot) (1.0pot)   (all-in)

matching versions/v14/self_play/simulator.py's
    actions = ['fold','call'] + ['raise_0','raise_1','raise_2','raise_3']
with raise_fracs = [0.33, 0.66, 1.0, None]  (None -> all-in).

The INPUT contract is identical to V13 (same ContractV12 tensors, 35-dim ctx) — only the
head width differs (6 vs 3). So the decision engine reuses the V13 bridge and just consumes
the 6-way distribution. Sizing (raise_33/66/pot -> chip amount -> slider fraction, all-in ->
slider 1.0) is applied in core/decision.py so it exactly mirrors the training-side
simulator._raise_size_for_fraction (train/serve consistency for P1c).

Range-aware equity (as in V13) is applied upstream where board_state.equity is computed.
"""
import os
import torch

from versions.v14.core.model import PokerEVModelV4 as V14Model
from versions.v14.core.manifest import MANIFEST as V14_MANIFEST
from shared.manifest import load_state_dict as load_ckpt_state

# Policy keys in head order. RAISE_33/66/POT are pot-fraction raises; ALLIN is a full-stack shove.
V14_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V14ModelEngine:
    is_v14 = True
    is_v13 = False   # so v13-specific branches don't fire
    is_v11 = False

    def __init__(self, weight_name: str = "expert_main.pth", device: str = "cpu"):
        self.device = torch.device(device)
        self.model = V14Model().to(self.device)
        # Diagnostics: last decision's critic Q-values and raw actor policy (set in predict_ev).
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        weight_path = os.path.join(repo_root, "versions", "v14", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(weight_path, V14_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            self.loaded = False
            print(f"WARNING: could not load V14 weights at {weight_path}: {e}. Model outputs garbage.")

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V14_ACTION_KEYS.
        Named predict_ev for drop-in compatibility with the decision engine."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        # Stash 6-way critic Q (~BB vs fold) and raw actor policy for F12 diagnostics.
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V14_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V14_ACTION_KEYS)}
        return {k: float(probs[i]) for i, k in enumerate(V14_ACTION_KEYS)}

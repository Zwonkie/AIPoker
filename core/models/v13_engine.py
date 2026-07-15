"""Live inference engine for the V13 model (versions/v13).

V13 differs from the legacy engines in two ways that MUST be honored live:
  1. Architecture: the equity-primary PokerEVModelV4 in versions/v13/core/model.py (NOT the
     legacy core/models one) — different state_dict shapes; loaded via the v13 manifest.
  2. Action selection comes from the ACTOR (policy_logits softmax), not argmax(q_vals). This
     wrapper returns policy probabilities under the same {FOLD,CALL,RAISE} keys the decision
     engine argmaxes, so downstream code is unchanged.

Range-aware equity (the other half of V13) is applied upstream where board_state.equity is
computed — see PHPHelp.py / compute_range_aware_equity. This engine consumes whatever equity
is already in the context tensor.
"""
import os
import torch

from versions.v13.core.model import PokerEVModelV4 as V13Model
from versions.v13.core.manifest import MANIFEST as V13_MANIFEST
from shared.manifest import load_state_dict as load_ckpt_state


class V13ModelEngine:
    is_v13 = True
    is_v11 = False   # so legacy `is_v11` branches don't fire

    def __init__(self, weight_name: str = "expert_main.pth", device: str = "cpu"):
        self.device = torch.device(device)
        self.model = V13Model().to(self.device)
        # Diagnostics: last decision's critic Q-values and raw actor policy (set in predict_ev).
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        weight_path = os.path.join(repo_root, "versions", "v13", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(weight_path, V13_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            self.loaded = False
            print(f"WARNING: could not load V13 weights at {weight_path}: {e}. Model outputs garbage.")

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step as {FOLD,CALL,RAISE}.
        Named predict_ev for drop-in compatibility with the decision engine (which argmaxes)."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # Diagnostics: stash the critic's per-action EV (Q, ~BB vs fold) and the raw actor
        # policy for the final step. Kept as attributes (NOT added to the returned dict) so the
        # decision engine's argmax over {FOLD,CALL,RAISE} is untouched. F12 turn-diagnostics reads these.
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {"FOLD": float(q[0]), "CALL": float(q[1]), "RAISE": float(q[2])}
        self.last_policy = {"FOLD": float(probs[0]), "CALL": float(probs[1]), "RAISE": float(probs[2])}
        return {"FOLD": float(probs[0]), "CALL": float(probs[1]), "RAISE": float(probs[2])}

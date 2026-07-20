"""Live inference engine for V29 (versions/v29) -- [OPP-2] per-opponent-seat raise attribution
+ a critic-consistency filter on the actor's training target.

NEW contract (context_dim=54, contract_version=8 -- NOT the same as V25/V26/V27/V28's shared
context_dim=44/contract_version=7). This is the FIRST version since V25 to need its OWN live
bridge rather than reusing `bridge_v25` -- see core/decision.py's `bridge_v29` wiring.

Two changes this version (both by explicit user direction, 2026-07-20):
1. [OPP-2] Ten new appended context features (ctx[44:54]): per-opponent-seat
   `raised_this_hand`/`raised_this_street` -- lets the model attribute in-hand aggression to a
   SPECIFIC seat, not just a hand-level aggregate (`pot_type`) or a seat's static VPIP/AGG color.
   **Functional in training/model_verify (which runs entirely against the simulator, not this live
   bridge). NOT yet functional in LIVE serving** -- core/table_state.py has no per-seat raise
   tracking of any kind, so this engine's live bridge feeds all 10 new features as a constant 0
   (inert), the same degraded-but-safe posture V22/V23's `committed`/`hero_committed`/`pot_type`
   were ALREADY silently in live serving (discovered as a byproduct of this work -- see
   versions/v29/SPECS.md). Extending core/table_state.py is deferred as flagged follow-up, not
   attempted here (would touch the same live game-state code the currently-active model depends
   on).
2. Critic-consistency filter + a risk_aversion_coefficient bump (0.10->0.15) -- training-loop-only,
   no live-serving impact at all (regret_match_policy_torch only runs during training).

Loads `expert_main.pth`, a from-scratch 100k-hand run (fresh weights, no --resume_path, per
[VAL-5]). See versions/v29/SPECS.md for the full derivation, calibration, and
`model_verify --full` results.
"""
import os
import torch

from versions.v29.core.model import PokerEVModelV4 as V29Model
from versions.v29.core.manifest import MANIFEST as V29_MANIFEST
from shared.manifest import load_state_dict as load_ckpt_state

# Same head order as every other sized model (shared 6-action contract).
V29_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V29ModelEngine:
    is_v29 = True
    is_v28 = False
    is_v26 = False
    is_v25 = False
    is_v21_auxhead = False
    is_v20_preflopEq_AI = False
    is_v20_preflopEq = False
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
        self.model = V29Model().to(self.device)
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        weight_path = os.path.join(repo_root, "versions", "v29", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(weight_path, V29_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            self.loaded = False
            print(f"WARNING: could not load V29 weights at {weight_path}: {e}. Model outputs garbage.")

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V29_ACTION_KEYS."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V29_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V29_ACTION_KEYS)}
        # Same aux-head reads as V21_auxhead/V25/V26/V28 (identical head architecture) -- see that
        # engine's own docstring for the raw-MSE-scalar caveat (not confident categorical bands).
        self.last_aux = {
            'self_equity': float(out['equity'][0, -1]),
            'opp_strength': float(out['strength'][0, -1]),
            'opp_bluff': float(out['bluff'][0, -1]),
        }
        return {k: float(probs[i]) for i, k in enumerate(V29_ACTION_KEYS)}

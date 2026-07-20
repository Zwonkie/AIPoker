"""Live inference engine for V25 (versions/v25) -- the multi-street EV rollout fix.

Same aux-head architecture as V21_auxhead (equity/strength/bluff heads on the shared transformer
trunk, inherited unchanged through V22->V23->V24->V25), but a WIDER context (context_dim=44,
contract_version=7, vs V21_auxhead's 37/5) -- V22 appended entry-sizing features
(opp_committed_this_hand_bb x5 + hero_committed_this_hand_bb), V23 appended `pot_type`. This is a
DIFFERENT scale+width than every other model in the live registry, so it needs its OWN bridge
(core/decision.py's `bridge_v25`, versions.v25.core.contract) -- cannot share bridge_v20_preflopEq.

V25 itself changes ONLY `simulator.py`'s `_mc_target_evs_sized` (a new `_rollout_continuation_ev`
correction representing the value of streets beyond the current one -- see
versions/v25/SPECS.md for the full derivation, calibration, and results).

Loads `expert_main.pth`, the 100k-hand from-scratch confirmatory run (fresh weights, no
--resume_path, per [VAL-5]). **model_verify --full @ 100k (2026-07-18): 17 PASS / 5 WARN / 1 FAIL
/ 0 SKIP.** Mixed vs. the 50k diagnostic that prompted this run (18/2/1/1): `vpip_adapts_to_style`
held and strengthened (short +12.0pts, deep +9.4pts -- the core hypothesis this version was built
to test), `beats_frozen_predecessor` ran for the first time and PASSED (+74.0 BB/100 vs a field
including the frozen 50k snapshot). But the direct allin-vs-next-best Q-gap widened back to
1.73-1.78x (from 50k's 1.35-1.36x) and two previously-clean checks (`committed_sensitivity`,
`position_sweep`) drifted to WARN. `deep_stack_ood_guard` FAIL is the same persistent issue every
version since V19 has carried ([STACK-1] in the OFK backlog).

Deployed live (2026-07-18) for user evaluation per explicit request, despite the mixed 100k
verification -- not a claim that this is a clean, unambiguous improvement over V21_auxhead, but a
deliberate choice to get real playtest signal on the multi-street EV mechanism while the Q-gap
trend (50k vs 100k) is still being understood. V21_auxhead/V20_preflopEq_AI/V20_preflopEq/V20 all
stay fully intact in the registry as rollback options.
"""
import os
import torch

from versions.v25.core.model import PokerEVModelV4 as V25Model
from versions.v25.core.manifest import MANIFEST as V25_MANIFEST
from shared.manifest import load_state_dict as load_ckpt_state

# Same head order as every other sized model (shared 6-action contract).
V25_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V25ModelEngine:
    is_v25 = True
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
        self.model = V25Model().to(self.device)
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        weight_path = os.path.join(repo_root, "versions", "v25", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(weight_path, V25_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            self.loaded = False
            print(f"WARNING: could not load V25 weights at {weight_path}: {e}. Model outputs garbage.")

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V25_ACTION_KEYS."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V25_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V25_ACTION_KEYS)}
        # Same aux-head reads as V21_auxhead (identical head architecture) -- see that engine's own
        # docstring for the raw-MSE-scalar caveat (not confident categorical bands).
        self.last_aux = {
            'self_equity': float(out['equity'][0, -1]),
            'opp_strength': float(out['strength'][0, -1]),
            'opp_bluff': float(out['bluff'][0, -1]),
        }
        return {k: float(probs[i]) for i, k in enumerate(V25_ACTION_KEYS)}

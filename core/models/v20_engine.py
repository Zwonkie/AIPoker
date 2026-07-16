"""Live inference engine for the V20 model (versions/v20), 200k-hand checkpoint.

V20 has the SAME 6-action bet-size head order as V14/V15/V17/V17_gauntlet/V19 (same PokerEVModelV4
arch), but a DIFFERENT input SCALE (contract_version bumped 3->4): the stack/pot/call-amount
context features (ctx[1]/ctx[2]/ctx[9] + the 5 opp_stack slots) were rescaled from /400(/1000) to
/100(/250) to fit the actual 5-50bb training range this model line has used since V15 -- see
versions/v20/SPECS.md. This is why V20 CANNOT reuse `core/decision.py`'s shared `bridge_v13`
(the v13-scale contract every prior sized model was trained on) -- it needs its OWN bridge built
from `versions.v20.core.contract`, gated by `is_v20_model` in decision.py.

Loads `expert_main_200k.pth` -- a preserved snapshot cloned at the end of the 120k->200k
continuation (same run, resumed via --resume_path, not a fresh restart) so live-deployed weights
stay fixed regardless of any further training. Do NOT repoint this at `expert_main.pth` while a
future resume is in flight -- that file gets overwritten live as training continues.

**model_verify --full comparison, 120k vs 200k (2026-07-17):** 120k scored 9 PASS/1 WARN/1 FAIL/
1 SKIP; 200k scored 8 PASS/2 WARN/1 FAIL/1 SKIP -- a real tradeoff, not a strict improvement.
`deep_stack_ood_guard` (still FAIL both) narrowed sharply: 120k jammed ALL-IN flat across 15-40bb
at eq=0.48 AND eq=0.55 (~0.35-0.38 mass uniformly); 200k only fails one cell (eq=0.55, 40bb,
allin@0.23 mass). But `short_stack_polarization` flipped PASS->WARN: avg P(call) in clear
shove-or-fold spots roughly doubled (0.12->0.25) -- [P3]'s residual short-stack call-flatting got
WORSE with more training, not better, still open. `vpip_adapts_to_style` and
`beats_offformula_stress` show the same short-worse/deep-better trade. `beats_frozen_predecessor`
SKIPs by design (no cross-scale-compatible frozen checkpoint this version).

**Live-safety clamp** (found + fixed during pre-deployment smoke-testing, before the 120k ship):
the rescale's 4x resolution gain inside the 5-50bb TRAINED band trades away headroom past it -- a
flopped set of aces facing a bet showed fold probability climbing with stack depth well beyond
50bb (a real table depth), a genuine OOD extrapolation artifact. `contract.py` clamps the
stack/pot/call_amount-DERIVED context features (not the real stack used for bet sizing) to the
training ceiling, so every live query stays in-distribution regardless of real table depth --
verified fold rate holds flat (~13%) from 20bb to 150bb+.

Deployed at 200k (`expert_main_200k.pth`, a preserved snapshot) per explicit user decision --
accepting the short-stack polarization regression in exchange for the deep-stack OOD improvement.
[P3] short-stack call-flatting remains open and worth a dedicated look.
"""
import os
import torch

from versions.v20.core.model import PokerEVModelV4 as V20Model
from versions.v20.core.manifest import MANIFEST as V20_MANIFEST
from shared.manifest import load_state_dict as load_ckpt_state

# Same head order as V14/V15/V17/V17_gauntlet/V19 (shared 6-action contract).
V20_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V20ModelEngine:
    is_v20 = True
    is_v19 = False
    is_v17_gauntlet = False
    is_v17 = False
    is_v15 = False
    is_v14 = False
    is_v13 = False
    is_v11 = False

    def __init__(self, weight_name: str = "expert_main_120k.pth", device: str = "cpu"):
        self.device = torch.device(device)
        self.model = V20Model().to(self.device)
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        weight_path = os.path.join(repo_root, "versions", "v20", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(weight_path, V20_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            self.loaded = False
            print(f"WARNING: could not load V20 weights at {weight_path}: {e}. Model outputs garbage.")

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V20_ACTION_KEYS."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V20_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V20_ACTION_KEYS)}
        return {k: float(probs[i]) for i, k in enumerate(V20_ACTION_KEYS)}

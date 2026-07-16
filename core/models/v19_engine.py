"""Live inference engine for the V19 model (versions/v19).

V19 has the IDENTICAL 6-action bet-size contract as V14/V15/V17/V17_gauntlet (same PokerEVModelV4
arch, same 35-dim ContractV12 input, same {FOLD,CALL,RAISE_33,RAISE_66,RAISE_POT,ALLIN} head
order). It differs from v17_gauntlet (its training parent, via the v18 plumbing refactor) in three
targeted content fixes, not architecture: [P0] a size-aware preflop opponent fold-bar (fixes the
target-EV inflation behind the deep-stack trash-jam bug), [hero_position] every training-time model
query (Hero's own AND every opponent's) now gets its real button-relative position instead of a
silent default-to-Button, and a Past-Self VPIP mystery investigation (documented, not fixed). See
versions/v19/SPECS.md for full detail. Live serving reuses the entire V14/V15/V17/V17_gauntlet
path -- same bridge, same `_v14_size_to_slider` sizing, same executor mapping -- with just the
weights swapped.

Validated (2026-07-16): `tools/model_verify --full` 10 PASS/1 WARN/1 FAIL. The FAIL
(`deep_stack_ood_guard`) is NOT resolved by [P0] -- the failure grid shows 13/25 cells (eq>=0.43,
every stack 15-40bb) argmax to ALL-IN with a probability that's roughly FLAT across stack depth
(0.33-0.36), which doesn't match [P0]'s stack-scaling hypothesis and points instead at a threshold
effect around `policy_tightness_bb`'s "realization discount below eq 0.45" config knob -- a
different, not-yet-investigated root cause. Deployed anyway per explicit user decision: every other
gate passes strongly (`vpip_adapts_to_style` short +9.7pt/deep +6.8pt, `bb100_vs_standard_fields`
positns across all 4 fields, `beats_frozen_predecessor` +56.8 BB/100 vs the v17_gauntlet field,
`beats_offformula_stress` PASS). deep_stack_ood_guard carried forward as backlog, not a blocker.
"""
import os
import torch

from versions.v19.core.model import PokerEVModelV4 as V19Model
from versions.v19.core.manifest import MANIFEST as V19_MANIFEST
from shared.manifest import load_state_dict as load_ckpt_state

# Same head order as V14/V15/V17/V17_gauntlet (shared 6-action contract).
V19_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V19ModelEngine:
    is_v19 = True
    is_v17_gauntlet = False  # distinct flag, but decision.py routes v14+v15+v17+v17_gauntlet+v19 through the same sized path
    is_v17 = False
    is_v15 = False
    is_v14 = False
    is_v13 = False
    is_v11 = False

    def __init__(self, weight_name: str = "expert_main.pth", device: str = "cpu"):
        self.device = torch.device(device)
        self.model = V19Model().to(self.device)
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        weight_path = os.path.join(repo_root, "versions", "v19", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(weight_path, V19_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            self.loaded = False
            print(f"WARNING: could not load V19 weights at {weight_path}: {e}. Model outputs garbage.")

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V19_ACTION_KEYS."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V19_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V19_ACTION_KEYS)}
        return {k: float(probs[i]) for i, k in enumerate(V19_ACTION_KEYS)}

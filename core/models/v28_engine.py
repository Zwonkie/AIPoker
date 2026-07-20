"""Live inference engine for V28 (versions/v28) -- risk-adjusted (variance-penalized) sized-action
training target.

IDENTICAL architecture/tensor schema to V25/V26/V27 (context_dim=44, contract_version=7 -- the
aux-head trunk inherited unchanged since V21_auxhead, plus V22's entry-sizing features and V23's
pot_type). Shares V25's bridge (core/decision.py's `bridge_v25`, versions.v25.core.contract) -- no
new bridge needed, gated by `is_v28_model` alongside `is_v25_model`/`is_v26_model`.

V28 changes ONLY `_mc_target_evs_sized`'s per-size EV target: adds a closed-form risk/variance
penalty (`risk_adjusted_ev = raw_ev - 0.10 * sqrt(Var[X])`), applied UNIFORMLY to every sized
action (not an `is_allin` special case -- all-in's larger `raise_size` gives it naturally larger
variance, so the same coefficient penalizes it most on its own). See versions/v28/SPECS.md for the
full derivation, calibration, and results.

Loads `expert_main.pth`, the 100k-hand from-scratch run (fresh weights, no --resume_path, per
[VAL-5]). **model_verify --full (2026-07-19): 19 PASS / 4 WARN / 1 FAIL / 0 SKIP.** The targeted
metric -- `allin_vs_nextbest_qgap` [BET-1] -- shrank ~40-50% at every cell AND the pathological
"worse with stack depth" pattern from V27 is GONE (roughly flat 15bb-40bb instead of escalating to
+0.61 at 40bb). V27's own regression cluster substantially resolved alongside it: `action_diversity`
recovered from a 2-action collapse to 4 actions, `stack_full_sweep`'s argmax path is now a coherent
call->raise_33 progression (was all-allin), `position_sweep` recovered from WARN back to PASS
(0.653, even better than V26's own 0.378). `beats_frozen_predecessor` PASS, +29.7 BB/100 vs a
frozen V27 snapshot. `deep_stack_ood_guard` FAIL is the same persistent pre-existing issue every
version since V19 has carried ([STACK-1]), though at meaningfully lower confidence (0.33 -> 0.24).

Deployed live (2026-07-19) for user evaluation per explicit request, on the strength of the
targeted Q-gap fix plus a clean head-to-head win over V27 with no new regressions. V27/V26/V25/
V21_auxhead/V20_preflopEq_AI/V20_preflopEq/V20 all stay fully intact in the registry as rollback
options.
"""
import os
import torch

from versions.v28.core.model import PokerEVModelV4 as V28Model
from versions.v28.core.manifest import MANIFEST as V28_MANIFEST
from shared.manifest import load_state_dict as load_ckpt_state

# Same head order as every other sized model (shared 6-action contract).
V28_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V28ModelEngine:
    is_v28 = True
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
        self.model = V28Model().to(self.device)
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        weight_path = os.path.join(repo_root, "versions", "v28", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(weight_path, V28_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            self.loaded = False
            print(f"WARNING: could not load V28 weights at {weight_path}: {e}. Model outputs garbage.")

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V28_ACTION_KEYS."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V28_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V28_ACTION_KEYS)}
        # Same aux-head reads as V21_auxhead/V25/V26 (identical head architecture) -- see that
        # engine's own docstring for the raw-MSE-scalar caveat (not confident categorical bands).
        self.last_aux = {
            'self_equity': float(out['equity'][0, -1]),
            'opp_strength': float(out['strength'][0, -1]),
            'opp_bluff': float(out['bluff'][0, -1]),
        }
        return {k: float(probs[i]) for i, k in enumerate(V28_ACTION_KEYS)}

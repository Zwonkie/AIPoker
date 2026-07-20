"""Live inference engine for V26 (versions/v26) -- real-data (TreeOpponent) training pool.

IDENTICAL architecture/tensor schema to V25 (context_dim=44, contract_version=7 -- the aux-head
trunk inherited unchanged since V21_auxhead, plus V22's entry-sizing features and V23's pot_type).
Shares V25's bridge (core/decision.py's `bridge_v25`, versions.v25.core.contract) -- no new bridge
needed, gated by `is_v26_model` alongside `is_v25_model`.

V26 changes ONLY `config.yaml`'s opponent pool: 2 of 5 seats (`maniac`, `nit`) swapped from
hand-designed heuristic archetypes to `TreeOpponent` instances (versions/v25/self_play/
tree_opponent.py) -- XGBoost models fit directly on real Pluribus/WSOP full-info hand-history
decisions, not a formula and not anything trained inside this simulator's own self-play loop. See
versions/v26/SPECS.md for the full pipeline and results.

Loads `expert_main.pth`, the 100k-hand from-scratch run (fresh weights, no --resume_path, per
[VAL-5]). **model_verify --full (2026-07-18): 19 PASS / 3 WARN / 1 FAIL / 0 SKIP.** Beats a frozen
V25 snapshot head-to-head: `beats_frozen_predecessor` PASS, +42.8 BB/100 over 4000 hands (V25's
own expert_main.pth copied in as versions/v26/weights/frozen_v25.pth to enable this check). Two
checks that were WARN in V25's own 100k run recovered to PASS here (`committed_sensitivity`,
`position_sweep`) -- not conclusively attributed to the real-data opponents specifically (single
run, no seed-controlled ablation). `allin_exploits_opponent_foldiness` [OPP-8] did NOT improve
(spread 0.011, same as V25) -- the real-data opponents didn't teach hero to read opponent
foldiness any better. `deep_stack_ood_guard` FAIL is the same persistent pre-existing issue every
version since V19 has carried ([STACK-1]).

Deployed live (2026-07-19) for user evaluation per explicit request, on the strength of a clean
head-to-head win over V25 with no new regressions. V25/V21_auxhead/V20_preflopEq_AI/
V20_preflopEq/V20 all stay fully intact in the registry as rollback options.
"""
import os
import torch

from versions.v26.core.model import PokerEVModelV4 as V26Model
from versions.v26.core.manifest import MANIFEST as V26_MANIFEST
from shared.manifest import load_state_dict as load_ckpt_state

# Same head order as every other sized model (shared 6-action contract).
V26_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V26ModelEngine:
    is_v26 = True
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
        self.model = V26Model().to(self.device)
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        weight_path = os.path.join(repo_root, "versions", "v26", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(weight_path, V26_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            self.loaded = False
            print(f"WARNING: could not load V26 weights at {weight_path}: {e}. Model outputs garbage.")

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V26_ACTION_KEYS."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V26_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V26_ACTION_KEYS)}
        # Same aux-head reads as V21_auxhead/V25 (identical head architecture) -- see that engine's
        # own docstring for the raw-MSE-scalar caveat (not confident categorical bands).
        self.last_aux = {
            'self_equity': float(out['equity'][0, -1]),
            'opp_strength': float(out['strength'][0, -1]),
            'opp_bluff': float(out['bluff'][0, -1]),
        }
        return {k: float(probs[i]) for i, k in enumerate(V26_ACTION_KEYS)}

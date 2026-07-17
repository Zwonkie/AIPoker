"""Live inference engine for V20_preflopEq_AI (versions/v20_preflopEq_AI), 150k-hand checkpoint.

IDENTICAL architecture + tensor schema to V20_preflopEq (context_dim=37, contract_version=5,
same PokerEVModelV4) -- this version changed ONLY the training opponent pool (shifted toward real
NN opponents instead of mostly-heuristic bots, testing whether that reduces the shove-preference
traced to the heuristic bots' price-insensitive value-branch). Because the architecture is
unchanged, this shares `core/decision.py`'s `bridge_v20_preflopEq` bridge -- no new bridge needed,
just its own weights + registry entry, gated by `is_v20_preflopEq_AI`.

Loads `expert_main.pth` (the 150k final checkpoint).

**model_verify --full @ 150k (2026-07-17): 12 PASS / 1 WARN / 1 FAIL / 0 SKIP.**
The sizing-diversity hypothesis this version tested did NOT pan out -- `action_diversity` stayed
allin-dominant (`{fold:9, allin:11, raise_pot:1}`) and `deep_stack_ood_guard` still FAILs, same as
V20_preflopEq. But the model came out a clear overall improvement over its parent: for the first
time in this lineage `beats_frozen_predecessor` actually RAN (same architecture as V20_preflopEq,
no scale-mismatch skip) and PASSED at +53.5 BB/100 vs a field including frozen V20_preflopEq --
the first real validated win over a direct predecessor this lineage has managed. Also beats the
parent's own bb100_vs_standard_fields baseline in all 4 fields (e.g. tight_deep +65.1 vs +61.1,
loose_short +31.1 vs +16.8) and shows meaningfully stronger vpip_adapts_to_style deltas (short
+11.5pt vs +6.6pt, deep +9.6pt vs +7.1pt). Both new-feature sensitivity checks remain healthy.
See versions/v20_preflopEq_AI/SPECS.md for full detail.

Deployed live for user testing per explicit request (2026-07-17), alongside V20_preflopEq and V20
which both stay fully intact in the registry as rollback options.
"""
import os
import torch

from versions.v20_preflopEq_AI.core.model import PokerEVModelV4 as V20PreflopEqAIModel
from versions.v20_preflopEq_AI.core.manifest import MANIFEST as V20_PREFLOPEQ_AI_MANIFEST
from shared.manifest import load_state_dict as load_ckpt_state

# Same head order as V14/V15/V17/V17_gauntlet/V19/V20/V20_preflopEq (shared 6-action contract).
V20_PREFLOPEQ_AI_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V20PreflopEqAIModelEngine:
    is_v20_preflopEq_AI = True
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
        self.model = V20PreflopEqAIModel().to(self.device)
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        weight_path = os.path.join(repo_root, "versions", "v20_preflopEq_AI", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(weight_path, V20_PREFLOPEQ_AI_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            self.loaded = False
            print(f"WARNING: could not load V20_preflopEq_AI weights at {weight_path}: {e}. Model outputs garbage.")

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V20_PREFLOPEQ_AI_ACTION_KEYS."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V20_PREFLOPEQ_AI_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V20_PREFLOPEQ_AI_ACTION_KEYS)}
        return {k: float(probs[i]) for i, k in enumerate(V20_PREFLOPEQ_AI_ACTION_KEYS)}

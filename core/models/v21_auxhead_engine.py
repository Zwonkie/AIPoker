"""Live inference engine for V21_auxhead (versions/v21_auxhead), Phase 8 final candidate.

IDENTICAL architecture + tensor schema to V20_preflopEq/V20_preflopEq_AI (context_dim=37,
contract_version=5, same PokerEVModelV4 -- versions/v21_auxhead/core/contract.py and core/model.py
are byte-identical to V20_preflopEq_AI's). This version changed ONLY the training-time aux-head
loss (bluff/strength/equity heads on the shared transformer trunk, previously trained at
aux_loss_weight=0.0 since V14 -- genuinely inert). Because the architecture is unchanged, this
shares `core/decision.py`'s `bridge_v20_preflopEq` bridge -- no new bridge needed, just its own
weights + registry entry, gated by `is_v21_auxhead`.

Loads `expert_main.pth` (Phase 8: fresh 100k-hand run with the fully chosen aux configuration --
corrected `opp_bluff_prob` label gated on `last_raiser`, sqrt-dampened bluff-loss reweighting,
per-head weights bluff=0.05/strength=0.10/equity=0.05).

**model_verify --full @ Phase 8 (2026-07-17): 15 PASS / 3 WARN / 1 FAIL / 0 SKIP.** Same shape as
V21/Phase 2 baselines, no new failures. `inspect_aux_heads.py`: equity r=0.922, strength r=0.171
(best of all 8 phases in this experiment), bluff r=0.091 -- all three heads show real, non-collapsed
correlation with their training labels for the first time in this lineage. `action_diversity`
recovered to 3 actions (`fold`/`allin`/`raise_33`, a real raise_33 plateau across 5/9 stack points
in `stack_full_sweep`) vs the continuation-damaged Phase 7a arm's 2-action collapse -- confirming
that training the final aux config FRESH (not as a warm-started continuation) avoids a real,
separate diversity regression traced to the continuation mechanism itself (see
versions/v21_auxhead/SPECS.md Phase 7/8). `deep_stack_ood_guard` FAIL and `opponent_style_sweep`
WARN are the SAME pre-existing, untouched issues every version in this lineage carries
([STACK-1]/[OPP-5] in the OFK backlog).

Deployed live (2026-07-17) as the final candidate of the aux-head experiment, superseding
V20_preflopEq_AI. V20_preflopEq_AI/V20_preflopEq/V20 all stay fully intact in the registry as
rollback options.
"""
import os
import torch

from versions.v21_auxhead.core.model import PokerEVModelV4 as V21AuxheadModel
from versions.v21_auxhead.core.manifest import MANIFEST as V21_AUXHEAD_MANIFEST
from shared.manifest import load_state_dict as load_ckpt_state

# Same head order as V14/V15/V17/V17_gauntlet/V19/V20/V20_preflopEq/V20_preflopEq_AI (shared
# 6-action contract).
V21_AUXHEAD_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V21AuxheadModelEngine:
    is_v21_auxhead = True
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
        self.model = V21AuxheadModel().to(self.device)
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        weight_path = os.path.join(repo_root, "versions", "v21_auxhead", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(weight_path, V21_AUXHEAD_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            self.loaded = False
            print(f"WARNING: could not load V21_auxhead weights at {weight_path}: {e}. Model outputs garbage.")

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V21_AUXHEAD_ACTION_KEYS."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V21_AUXHEAD_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V21_AUXHEAD_ACTION_KEYS)}
        # Aux-head reads (self_equity/opp_strength/opp_bluff_prob), the FIRST time in this lineage
        # these are trained against real gradient (aux_loss_weight>0 -- see manifest docstring).
        # Raw MSE-trained scalars in ~[0,1] (see train.py: pred_bluff/pred_str/pred_eq compared
        # directly against 0-1 labels, no sigmoid). Correlations are real but modest
        # (inspect_aux_heads.py @ Phase 8: equity r=0.922, strength r=0.171, bluff r=0.091) --
        # `strength`'s predicted std (0.034) is far narrower than its label's own (0.142), i.e. the
        # head barely swings from its mean. Exposed as raw numbers, not confident categorical bands,
        # for exactly that reason -- see core/decision.py's _narrate_opponent_read.
        self.last_aux = {
            'self_equity': float(out['equity'][0, -1]),
            'opp_strength': float(out['strength'][0, -1]),
            'opp_bluff': float(out['bluff'][0, -1]),
        }
        return {k: float(probs[i]) for i, k in enumerate(V21_AUXHEAD_ACTION_KEYS)}

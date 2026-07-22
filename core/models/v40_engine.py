"""Live inference engine for V40 (versions/v40) -- the [BET-3] package from the 2026-07-20 Fable
full-stack review.

SAME contract as V29 (context_dim=54, contract_version=8). V40 changed nothing about the tensor
schema -- every fix is in the SIMULATOR or the training target -- so this engine is wired exactly
like `v29_engine.py`, and all of V29's live game-state work ([OPP-2] per-seat raise tracking in
core/table_state.py, plus the `committed`/`hero_committed`/`pot_type` and all-in stack-tracking
byproduct fixes) carries over unchanged and stays fully functional.

Three changes vs V29, all from the review's "explains live behavior already logged in the backlog"
tier (see versions/v40/SPECS.md for the measured before/after on each):
  1. [BET-3] The betting round no longer ends on a single check. Postflop `highest_bet` starts at 0,
     so the old terminator was satisfied from the street's first instant and the round closed as
     soon as the opening seat acted -- 0 of 849 postflop checks in an instrumented 1000-hand run
     were followed by anyone acting, and the BB never got its limped-pot option. The model had ZERO
     training data for check-behind, check-raise, "checked to me", delayed c-bet or BB-option nodes.
     Measured after the fix: postflop actions/hand 3.13 -> 5.16, BB-option decisions 0 -> 259/750.
  2. [BET-3] CALL is no longer exempt from the [V28/V29] variance penalty or the [V25] multi-street
     continuation credit, both of which every sized raise already received. The penalty scales with
     pot size, i.e. it bit hardest in exactly the multiway/high-equity spots where V29 refuses to
     raise live. A deliberate carve-out keeps a FREE check (`to_call == 0`) unpenalised, so it can
     never be pushed below the flat-0.0 fold baseline.
  3. [STACK-3] The ALLIN critic-consistency veto is rescoped to non-fold alternatives -- documented
     honestly as a provable no-op under the fold baseline both call sites use.

Live effect measured by model_verify: the multiway-aggression collapse that motivated the whole
package went from 6/6 short-stack cells to 3/6, with 3-way aggression at eq 0.65 rising 0.01 -> ~0.5.

**Deployment status**: deployed as an INTERIM live model on 2026-07-21 by explicit user request, to
be play-tested while V41 (which adds the simulation-realism package on top) finishes training. Its
model_verify --full was only partially completed -- all FAST checks plus `vpip_adapts_to_style` and
`beats_offformula_stress` passed; `bb100_vs_standard_fields` and `beats_frozen_predecessor` were cut
short to free CPU for V41. See versions/v40/SPECS.md and
.agents/skills/OFK/references/fable-review-resolution-log.md.
"""
import os
import torch

from versions.v40.core.model import PokerEVModelV4 as V40Model
from versions.v40.core.manifest import MANIFEST as V40_MANIFEST
from versions.v40.core.contract import ContractV12 as V40Contract
from shared.manifest import load_state_dict as load_ckpt_state

# Same head order as every other sized model (shared 6-action contract).
V40_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V40ModelEngine:
    # [v46_legacySweep] Declarations the shared layer asks for (the is_vN flags below are legacy
    # relics of the deleted decision.py ladder, kept harmless).
    is_sized = True
    display_tag = "V40"
    has_aux = True
    is_v40 = True
    is_v41 = False
    is_v29 = False
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
        self.model = V40Model().to(self.device)
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.weight_path = os.path.join(repo_root, "versions", "v40", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(self.weight_path, V40_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            # [Fable review #15] See v41_engine.py's note: the failure is recorded on `.loaded`
            # rather than raised, because the registry builds EVERY engine at init; core/decision.py
            # refuses to serve a decision from an engine whose `.loaded` is False.
            self.loaded = False
            self.load_error = repr(e)
            print(f"WARNING: could not load V40 weights at {self.weight_path}: {e}. "
                  f"This engine will REFUSE to act (see decision.py's .loaded guard).")

    def make_bridge(self):
        """[Fable review #16/H4] This engine's OWN tensor bridge -- short-circuits decision.py's
        `is_vN` substring ladder entirely. See v41_engine.make_bridge for the full rationale."""
        return V40Contract(max_seq_len=20)

    def live_features(self):
        """[V42_liveFixes] This version's OWN live equity / hand_strength implementations --
        short-circuits PHPHelp.py's two remaining per-version substring ladders, which stopped at
        'v29' and therefore served V40 and V41 vs-random equity live. See
        `v41_engine.live_features` for the full rationale."""
        return {
            'version_package': 'versions.v40',
            'range_aware_equity': True,
            'front_colors': True,
            'hand_strength': True,
        }

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V40_ACTION_KEYS."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V40_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V40_ACTION_KEYS)}
        # Same aux-head reads as V21_auxhead/V25/V26/V28/V29 (identical head architecture).
        # HUD/telemetry only -- never used for action selection.
        self.last_aux = {
            'self_equity': float(out['equity'][0, -1]),
            'opp_strength': float(out['strength'][0, -1]),
            'opp_bluff': float(out['bluff'][0, -1]),
        }
        return {k: float(probs[i]) for i, k in enumerate(V40_ACTION_KEYS)}

"""Live inference engine for V43 (versions/v43) -- the corrective-prior cleanup.

SAME contract as V29/V40/V41 (context_dim=54, contract_version=8). Nothing about the tensor schema
changed: every V43 change is in the TRAINING LOOP, so this engine is wired exactly like
`v41_engine.py` and all of the live game-state work (V29's [OPP-2] per-seat raise tracking, and
V42_liveFixes' call-amount sentinel / decimal-stake units / unknown-HUD default / occupied-ring
positions) carries over unchanged.

What V43 changed, all training-side (see versions/v43/SPECS.md for every measurement):
  REMOVED  the realization discount (`policy_tightness_bb`, V12) -- a flat bb penalty on every
           voluntary action below eq 0.45, i.e. an imposed answer to "how wide should hero enter".
  REMOVED  the ALLIN critic-consistency veto (`critic_consistency_margin`, V29) -- measured
           near-inert before removal.
  KEPT     the variance penalty -- measured still load-bearing (removing it triples the
           ALLIN-vs-next-best gap's growth with stack depth).
  CHANGED  TARGET_CLIP_BB 40 -> 100 to match STACK_CEIL_BB and the curriculum (review T-M5): at 40
           the clip truncated 23.4% of realized go-forward returns.
  CHANGED  risk_aversion_coefficient 0.15 -> 0.20, REQUIRED BY the clip change -- the 40bb clip had
           been acting as an undeclared deep-stack all-in dampener.

The premise: V40/V41 fixed the root causes in the training DATA those priors were compensating for,
so entry width and jam frequency should be LEARNED from correct inputs rather than imposed.

DEPLOYED LIVE 2026-07-21 by explicit user decision, on a MIXED scorecard and BEFORE the
head-to-head finished. Known at deploy time:
  * `vpip_adapts_to_style` FAIL -- deep-stack delta +4.2pts vs the >=5pt gate (V41 passed at
    short +5.9 / deep +7.2). The predicted cost of removing the discount: entry range widened and
    opponent-adaptation weakened.
  * `nash_bbcall_vs_jam` 47% (V41 passed) -- calls jams wider than Nash at 5bb, often never
    folding (F=0.00). This is the CLEAN binary Nash check and should not be dismissed.
  * `nash_pushfold_vs_chart` 65% (V41 78%) -- discounted, see versions/v43/SPECS.md: that check
    sums four aggressive heads against one fold head while three of them are the same 1.5bb
    min-raise.
  * Genuinely better than V41: `allin_vs_nextbest_qgap` negative at every cell,
    `opponent_style_sweep` recovered from WARN to PASS, `action_diversity` genuinely mixed,
    `short_stack_polarization` 0.15 (V41 0.19), [BET-3] still resolved.
V41 remains registered and is the rollback: set `active_model_name` back to 'Herocules (v41)'.
"""
import os
import torch

from versions.v43.core.model import PokerEVModelV4 as V43Model
from versions.v43.core.manifest import MANIFEST as V43_MANIFEST
from versions.v43.core.contract import ContractV12 as V43Contract
from shared.manifest import load_state_dict as load_ckpt_state

# Same head order as every other sized model (shared 6-action contract).
V43_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V43ModelEngine:
    is_v43 = True
    # [V43] Declares the 6-action sized contract directly instead of relying on decision.py's
    # `is_sized_model` OR-chain. That ladder is the one that silently bit the V40 deploy -- the
    # bridge was fine, but `is_sized_model` didn't recognise v40, so live emitted a bare
    # `RAISE_POT` with size=0.0 instead of an executable `RAISE_SLIDER_x`. An engine should say
    # what it is, not be recognised by substring. See the Fable review's #16/H4 "still open" list.
    is_sized = True
    # [V43] The other two ladder consumers decision.py used to resolve by substring: the live
    # reason-line label (a 13-deep nested ternary whose final `else` is "V14", so an unrecognised
    # version renders as V14 -- the HUD naming a different model than the one acting) and the
    # aux-head gate (same architecture as V21_auxhead/V25-V41, so the opponent-read line applies).
    display_tag = "V43"
    has_aux = True
    is_v41 = False
    is_v40 = False
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
        self.model = V43Model().to(self.device)
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.weight_path = os.path.join(repo_root, "versions", "v43", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(self.weight_path, V43_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            # [Fable review #15] Swallowed here on purpose (the registry builds EVERY engine at
            # init; one missing rollback checkpoint must not take the app down). `.loaded` is the
            # contract -- core/decision.py refuses to serve from an engine that failed to load.
            self.loaded = False
            self.load_error = repr(e)
            print(f"WARNING: could not load V43 weights at {self.weight_path}: {e}. "
                  f"This engine will REFUSE to act (see decision.py's .loaded guard).")

    def make_bridge(self):
        """[Fable review #16/H4] This engine's OWN tensor bridge -- short-circuits decision.py's
        `is_vN` substring ladder entirely. Same 54/8 schema as V29/V40/V41, but served from V43's
        own contract module so a future divergence cannot silently misalign."""
        return V43Contract(max_seq_len=20)

    def live_features(self):
        """[V42_liveFixes] This version's OWN live input-feature implementations, so PHPHelp's
        per-version ladders cannot forget it -- the failure that served V40 and V41 vs-random
        equity and a constant hand_strength while they were the live model."""
        return {
            'version_package': 'versions.v43',
            'range_aware_equity': True,
            'front_colors': True,
            'hand_strength': True,
        }

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V43_ACTION_KEYS."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V43_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V43_ACTION_KEYS)}
        self.last_aux = {
            'self_equity': float(out['equity'][0, -1]),
            'opp_strength': float(out['strength'][0, -1]),
            'opp_bluff': float(out['bluff'][0, -1]),
        }
        return {k: float(probs[i]) for i, k in enumerate(V43_ACTION_KEYS)}

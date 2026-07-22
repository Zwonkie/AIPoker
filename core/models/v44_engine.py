"""Live inference engine for V44 (versions/v44) -- effective-contested-field `equity_edge`.

CONTRACT CHANGE vs V29/V40/V41/V43: contract_version 8 -> 9. `context_dim` is UNCHANGED at 54 --
ctx[35] (`equity_edge`) keeps its index and width but changes MEANING: it now normalizes by the
EFFECTIVE contested field (E[k|k>=1], each still-to-act opponent rolled at its VPIP) instead of the
nominal opponent count. Because the width is identical, a V43 checkpoint would LOAD into this engine
and behave wrongly -- so this engine gets its OWN bridge (make_bridge -> versions.v44 ContractV12)
and the manifest's contract_version bump is what makes a cross-load fail loud rather than silent.

Everything else is V43 unchanged (realization discount + ALLIN veto removed, TARGET_CLIP_BB 100,
risk_aversion_coefficient 0.20), and all live game-state work carries over identically (V29's
[OPP-2] per-seat raise tracking, V42_liveFixes' call-amount sentinel / decimal-stake units /
occupied-ring positions / front-colors-by-committed-chips / is_active-monotonic-within-a-hand).

What V44 changed and why (see versions/v44/SPECS.md for every measurement):
  `equity_edge` exists to say "strong FOR THIS FIELD SIZE" and no model in the lineage used it,
  because its denominator (nominal count) disagreed with the field `equity` is actually measured
  against (the rolled/conditional contested field). V44 makes the denominators agree.

Result (model_verify --full, 2026-07-22): 21 PASS / 6 WARN / 0 FAIL.
  * `vpip_adapts_to_style` PASS for the FIRST time in the project ([P4], short +6.1 / deep +5.9 vs
    the >=5pt gate V43 failed at +5.4/+4.2) -- the metric this version targeted.
  * `beats_frozen_predecessor` +91.7 BB/100 vs frozen V43 (V43 was +74.6 vs frozen V41).
  * Cost: `committed_sensitivity` and `pot_type_sensitivity` dropped to WARN (0.05 -> ~0.023) --
    V44 attends less to those two features, plausibly now redundant with the sharper edge.

  The LIVE side of `equity_edge` needs the caller to supply `BoardState.effective_field`. PHPHelp's
  decision path builds `colors_in_pot` / `colors_still_to_act` already; the field is set there the
  same way `equity` and `hand_strength` are. If a construction site leaves it at 0.0, the V44
  contract falls back to the nominal count (i.e. exactly V43's feature), so an un-updated caller
  degrades gracefully rather than feeding a silently mis-scaled ctx[35].

DEPLOYED LIVE 2026-07-22 by explicit user decision, on the strength of the 0-FAIL scorecard and the
first-ever [P4] pass. V43 remains registered and is the one-line rollback: set `active_model_name`
back to 'Herocules (v43)'.
"""
import os
import torch

from versions.v44.core.model import PokerEVModelV4 as V44Model
from versions.v44.core.manifest import MANIFEST as V44_MANIFEST
from versions.v44.core.contract import ContractV12 as V44Contract
from shared.manifest import load_state_dict as load_ckpt_state

# Same head order as every other sized model (shared 6-action contract).
V44_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V44ModelEngine:
    is_v44 = True
    # Declares the sized 6-action contract, the display label, and the aux-head presence directly,
    # so none of decision.py's `is_vN` OR-chains / nested-ternary ladders can misroute it -- the
    # three failure modes that silently bit the V40 deploy (unsized RAISE_POT size=0.0, HUD label
    # falling through to "V14", lost aux line). An engine says what it is; it is not recognised by
    # substring.
    is_sized = True
    display_tag = "V44"
    has_aux = True
    # Every legacy is_vN flag the ladders probe -- all False, so no fallback path can claim this
    # engine as some older version by substring.
    is_v43 = False
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
        self.model = V44Model().to(self.device)
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.weight_path = os.path.join(repo_root, "versions", "v44", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(self.weight_path, V44_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            # [Fable review #15] Swallowed here on purpose (the registry builds EVERY engine at
            # init; one missing rollback checkpoint must not take the app down). `.loaded` is the
            # contract -- core/decision.py refuses to serve from an engine that failed to load.
            self.loaded = False
            self.load_error = repr(e)
            print(f"WARNING: could not load V44 weights at {self.weight_path}: {e}. "
                  f"This engine will REFUSE to act (see decision.py's .loaded guard).")

    def make_bridge(self):
        """[Fable review #16/H4] This engine's OWN tensor bridge. Critically here it is NOT
        interchangeable with V43's: same 54 width, but ctx[35] means the effective-field edge, so
        serving V44 weights through a V43 bridge (or vice versa) would feed the wrong denominator.
        The own-bridge dispatch makes that impossible."""
        return V44Contract(max_seq_len=20)

    def live_features(self):
        """[V42_liveFixes] This version's OWN live input-feature implementations, so PHPHelp's
        per-version ladders cannot forget it -- the failure that served V40/V41 vs-random equity and
        a constant hand_strength while they were the live model."""
        return {
            'version_package': 'versions.v44',
            'range_aware_equity': True,
            'front_colors': True,
            'hand_strength': True,
        }

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V44_ACTION_KEYS."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V44_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V44_ACTION_KEYS)}
        self.last_aux = {
            'self_equity': float(out['equity'][0, -1]),
            'opp_strength': float(out['strength'][0, -1]),
            'opp_bluff': float(out['bluff'][0, -1]),
        }
        return {k: float(probs[i]) for i, k in enumerate(V44_ACTION_KEYS)}

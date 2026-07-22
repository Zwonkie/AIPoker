"""Live inference engine for V47 (versions/v47) -- opponent-behavior realism + target alignment.

CONTRACT UNCHANGED vs V44 (context_dim=54, contract_version=9): every V47 change is
simulator/training-side (opponent raise-size repertoire [#6], chip-identical bucket collapse [M9],
occupant-true fold models [M4], sub-5bb curriculum [VAL-1(A)], training hygiene [M6/M7]), so this
engine is a declaration-clone of v44_engine.py pointed at the versions/v47 slice -- plus ONE new
serve-side declaration:

  `collapse_aliased_allin = True` -- [V47, Change 2 / M9] train/serve invariant pair: training
  collapses the actor-target mass of any raise bucket whose chips equal the shove onto the
  canonical ALLIN bucket (train.py's aliased masks); decision.py's sampler reads this flag and
  masks the chip-identical duplicate buckets the same way before sampling. Engines that trained
  WITHOUT the collapse must not declare it (their bucket mass is real, not duplicated).

NOT YET REGISTERED / NOT LIVE -- deploy is one registry line in core/decision.py after the
versions/v47/SPECS.md acceptance gates pass. See that SPECS for scope, gates, and rollback.
"""

import os
import torch

from versions.v47.core.model import PokerEVModelV4 as V47Model
from versions.v47.core.manifest import MANIFEST as V47_MANIFEST
from versions.v47.core.contract import ContractV12 as V47Contract
from shared.manifest import load_state_dict as load_ckpt_state

# Same head order as every other sized model (shared 6-action contract).
V47_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V47ModelEngine:
    is_v47 = True
    # Declares the sized 6-action contract, the display label, and the aux-head presence directly,
    # so none of decision.py's `is_vN` OR-chains / nested-ternary ladders can misroute it -- the
    # three failure modes that silently bit the V40 deploy (unsized RAISE_POT size=0.0, HUD label
    # falling through to "V14", lost aux line). An engine says what it is; it is not recognised by
    # substring.
    is_sized = True
    display_tag = "V47"
    has_aux = True
    # [V47, Change 2 / M9] see module docstring -- the serve half of the aliased-bucket collapse.
    collapse_aliased_allin = True
    # Every legacy is_vN flag the ladders probe -- all False, so no fallback path can claim this
    # engine as some older version by substring.
    is_v44 = False
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
        self.model = V47Model().to(self.device)
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.weight_path = os.path.join(repo_root, "versions", "v47", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(self.weight_path, V47_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            # [Fable review #15] Swallowed here on purpose (the registry builds EVERY engine at
            # init; one missing rollback checkpoint must not take the app down). `.loaded` is the
            # contract -- core/decision.py refuses to serve from an engine that failed to load.
            self.loaded = False
            self.load_error = repr(e)
            print(f"WARNING: could not load V47 weights at {self.weight_path}: {e}. "
                  f"This engine will REFUSE to act (see decision.py's .loaded guard).")

    def make_bridge(self):
        """[Fable review #16/H4] This engine's OWN tensor bridge. Critically here it is NOT
        interchangeable with V43's: same 54 width, but ctx[35] means the effective-field edge, so
        serving V44 weights through a V43 bridge (or vice versa) would feed the wrong denominator.
        The own-bridge dispatch makes that impossible."""
        return V47Contract(max_seq_len=20)

    def live_features(self):
        """[V42_liveFixes] This version's OWN live input-feature implementations, so PHPHelp's
        per-version ladders cannot forget it -- the failure that served V40/V41 vs-random equity and
        a constant hand_strength while they were the live model."""
        return {
            'version_package': 'versions.v47',
            'range_aware_equity': True,
            'front_colors': True,
            'hand_strength': True,
        }

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V47_ACTION_KEYS."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V47_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V47_ACTION_KEYS)}
        self.last_aux = {
            'self_equity': float(out['equity'][0, -1]),
            'opp_strength': float(out['strength'][0, -1]),
            'opp_bluff': float(out['bluff'][0, -1]),
        }
        return {k: float(probs[i]) for i, k in enumerate(V47_ACTION_KEYS)}

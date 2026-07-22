"""Live inference engine for V48 (versions/v48) -- true short-handed geometry + generalized chip-identity collapse.

Declaration-clone of v47_engine.py pointed at the versions/v48 slice; ONE serve-side declaration
changes: `collapse_aliased_buckets = True` replaces V47's narrower `collapse_aliased_allin`
([V48, Change 0] -- train/serve invariant pair with versions/v48 simulator/train aliased masks).

NOT YET REGISTERED / NOT LIVE -- deploy is one registry line in core/decision.py after the
versions/v48/SPECS.md acceptance gates pass.
"""

import os
import torch

from versions.v48.core.model import PokerEVModelV4 as V48Model
from versions.v48.core.manifest import MANIFEST as V48_MANIFEST
from versions.v48.core.contract import ContractV12 as V48Contract
from shared.manifest import load_state_dict as load_ckpt_state

# Same head order as every other sized model (shared 6-action contract).
V48_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V48ModelEngine:
    is_v48 = True
    # Declares the sized 6-action contract, the display label, and the aux-head presence directly,
    # so none of decision.py's `is_vN` OR-chains / nested-ternary ladders can misroute it -- the
    # three failure modes that silently bit the V40 deploy (unsized RAISE_POT size=0.0, HUD label
    # falling through to "V14", lost aux line). An engine says what it is; it is not recognised by
    # substring.
    is_sized = True
    display_tag = "V48"
    has_aux = True
    # [V48, Change 0] see module docstring -- the serve half of the GENERALIZED chip-identity
    # collapse (decision.py groups raise buckets by resolved slider chips, keeps ALLIN for the
    # shove group / lowest-index otherwise, masks duplicates). Supersedes V47's allin-only flag
    # for this engine; do NOT also declare collapse_aliased_allin (the generalized mask covers it).
    collapse_aliased_buckets = True
    collapse_aliased_allin = False
    # Every legacy is_vN flag the ladders probe -- all False, so no fallback path can claim this
    # engine as some older version by substring.
    is_v47 = False
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
        self.model = V48Model().to(self.device)
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.weight_path = os.path.join(repo_root, "versions", "v48", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(self.weight_path, V48_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            # [Fable review #15] Swallowed here on purpose (the registry builds EVERY engine at
            # init; one missing rollback checkpoint must not take the app down). `.loaded` is the
            # contract -- core/decision.py refuses to serve from an engine that failed to load.
            self.loaded = False
            self.load_error = repr(e)
            print(f"WARNING: could not load V48 weights at {self.weight_path}: {e}. "
                  f"This engine will REFUSE to act (see decision.py's .loaded guard).")

    def make_bridge(self):
        """[Fable review #16/H4] This engine's OWN tensor bridge. Critically here it is NOT
        interchangeable with V43's: same 54 width, but ctx[35] means the effective-field edge, so
        serving V44 weights through a V43 bridge (or vice versa) would feed the wrong denominator.
        The own-bridge dispatch makes that impossible."""
        return V48Contract(max_seq_len=20)

    def live_features(self):
        """[V42_liveFixes] This version's OWN live input-feature implementations, so PHPHelp's
        per-version ladders cannot forget it -- the failure that served V40/V41 vs-random equity and
        a constant hand_strength while they were the live model."""
        return {
            'version_package': 'versions.v48',
            'range_aware_equity': True,
            'front_colors': True,
            'hand_strength': True,
        }

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V48_ACTION_KEYS."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V48_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V48_ACTION_KEYS)}
        self.last_aux = {
            'self_equity': float(out['equity'][0, -1]),
            'opp_strength': float(out['strength'][0, -1]),
            'opp_bluff': float(out['bluff'][0, -1]),
        }
        return {k: float(probs[i]) for i, k in enumerate(V48_ACTION_KEYS)}

"""Live inference engine for V50 (versions/v50) -- V48 geometry, WIDER seat x depth curriculum.

Declaration-clone of v48_engine.py pointed at the versions/v50 slice. V50 is a CURRICULUM-ONLY
retrain of V48: the contract is UNCHANGED (context_dim=54, contract_version=9) and every serve-side
declaration is identical to V48 -- including `collapse_aliased_buckets = True`, the serve half of
the generalized chip-identity collapse ([V48, Change 0]). The only difference vs V48 is training
data (raw-empirical seats 3-6 / depth 2-100bb mix, 250k fresh hands), which lives entirely in
versions/v50/self_play, so nothing about the live tensor path changes here.

Because the contract matches V48, this engine's bridge is interchangeable-in-width with V48's, but
it deliberately builds its OWN versions.v50 contract/bridge anyway (guardrails: one engine, one
declared bridge; no cross-version bridge sharing) so a future V50 contract change cannot silently
serve through V48's.
"""

import os
import torch

from versions.v50.core.model import PokerEVModelV4 as V50Model
from versions.v50.core.manifest import MANIFEST as V50_MANIFEST
from versions.v50.core.contract import ContractV12 as V50Contract
from shared.manifest import load_state_dict as load_ckpt_state

# Same head order as every other sized model (shared 6-action contract).
V50_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V50ModelEngine:
    is_v50 = True
    # Declares the sized 6-action contract, the display label, and the aux-head presence directly,
    # so none of decision.py's dispatch can misroute it -- an engine says what it is; it is not
    # recognised by substring (the three failure modes that silently bit the V40 deploy).
    is_sized = True
    display_tag = "V50"
    has_aux = True
    # [V48, Change 0 -- inherited unchanged] serve half of the GENERALIZED chip-identity collapse
    # (decision.py groups raise buckets by resolved slider chips, keeps ALLIN for the shove group /
    # lowest-index otherwise, masks duplicates). Train/serve invariant pair with the versions/v50
    # simulator/train aliased masks (which are the V48 masks, unchanged). Do NOT also declare
    # collapse_aliased_allin (the generalized mask covers it).
    collapse_aliased_buckets = True
    collapse_aliased_allin = False
    # Every legacy is_vN flag any defensive check probes -- all False, so no fallback path can claim
    # this engine as some older version by substring.
    is_v48 = False
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
        self.model = V50Model().to(self.device)
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.weight_path = os.path.join(repo_root, "versions", "v50", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(self.weight_path, V50_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            # [Fable review #15] Swallowed here on purpose (the registry builds EVERY engine at
            # init; one missing rollback checkpoint must not take the app down). `.loaded` is the
            # contract -- core/decision.py refuses to serve from an engine that failed to load.
            self.loaded = False
            self.load_error = repr(e)
            print(f"WARNING: could not load V50 weights at {self.weight_path}: {e}. "
                  f"This engine will REFUSE to act (see decision.py's .loaded guard).")

    def make_bridge(self):
        """[Fable review #16/H4] This engine's OWN tensor bridge. Same 54 width and cv9 meaning as
        V48 (V50 is a curriculum-only retrain), but built from the versions.v50 contract so a future
        V50 contract change can never silently serve through V48's bridge."""
        return V50Contract(max_seq_len=20)

    def live_features(self):
        """[V42_liveFixes] This version's OWN live input-feature implementations, so PHPHelp's
        per-version ladders cannot forget it -- the failure that served V40/V41 vs-random equity and
        a constant hand_strength while they were the live model. Identical set to V48."""
        return {
            'version_package': 'versions.v50',
            'range_aware_equity': True,
            'front_colors': True,
            'hand_strength': True,
        }

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V50_ACTION_KEYS."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V50_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V50_ACTION_KEYS)}
        self.last_aux = {
            'self_equity': float(out['equity'][0, -1]),
            'opp_strength': float(out['strength'][0, -1]),
            'opp_bluff': float(out['bluff'][0, -1]),
        }
        return {k: float(probs[i]) for i, k in enumerate(V50_ACTION_KEYS)}

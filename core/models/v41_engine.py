"""Live inference engine for V41 (versions/v41) -- the simulation-realism package from the
2026-07-20 Fable full-stack review.

SAME contract as V29/V40 (context_dim=54, contract_version=8). Nothing about the tensor schema
changed in V40 or V41 -- every fix in both versions is in the SIMULATOR or in how a training-time
opponent query is encoded, never in what the live bridge feeds. So this engine is wired exactly
like `v29_engine.py` and the live game-state work done for V29 ([OPP-2] per-seat raise tracking in
core/table_state.py, plus the `committed`/`hero_committed`/`pot_type` and all-in stack-tracking
byproduct fixes) carries over unchanged and stays fully functional.

Lineage: V29 -> V40 (BET-3 package) -> V41 (simulation realism). V40 fixed the betting round
ending on any check, CALL's exemption from the variance penalty and the multi-street continuation
credit, and rescoped the ALLIN critic-consistency veto. V41 then fixed, each with a measured
before/after:
  - dead blinds (a pre-folded seat could post a blind and never act -- 47.6% of flop-reaching
    hands, now 0%), so hero no longer learns that stealing prints money against seats that
    physically cannot defend;
  - NN opponents playing a degraded self (range-aware equity was gated to hero only although the
    lagged-self mirror trained on it; `call_amount` was passed as `pot_odds * pot_size`, which is
    NOT to_call -- a pot-sized bet reached the network at half its real size);
  - all six stacks always identical (opponents are now 0.35x-2.0x log-uniform around hero's
    curriculum depth -- hero had literally never seen a covered or covering opponent), plus the two
    NLH rule bugs that symmetry was hiding: the min-raise floor (`to_call + last increment`, not
    always +1bb) and short all-ins incorrectly re-opening betting;
  - [OPP-7] being defeated at the tensor boundary: V27's remap keyed opponent slots by ABSOLUTE
    seat number while `ContractV12.to_tensors` reads only `seat_1..seat_5`, so the real hero went
    to a `seat_0` key the encoder never reads. Measured: V40 dropped the hero on 128 of 128
    NN-opponent queries, V41 on zero.

Loads `expert_main.pth`, a from-scratch 100k-hand run (fresh weights, no --resume_path, per
[VAL-5]). See versions/v41/SPECS.md for the full derivations and
.agents/skills/OFK/references/fable-review-resolution-log.md for per-finding status.
"""
import os
import torch

from versions.v41.core.model import PokerEVModelV4 as V41Model
from versions.v41.core.manifest import MANIFEST as V41_MANIFEST
from versions.v41.core.contract import ContractV12 as V41Contract
from shared.manifest import load_state_dict as load_ckpt_state

# Same head order as every other sized model (shared 6-action contract).
V41_ACTION_KEYS = ("FOLD", "CALL", "RAISE_33", "RAISE_66", "RAISE_POT", "ALLIN")


class V41ModelEngine:
    # [v46_legacySweep] Declarations the shared layer asks for (the is_vN flags below are legacy
    # relics of the deleted decision.py ladder, kept harmless).
    is_sized = True
    display_tag = "V41"
    has_aux = True
    is_v41 = True
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
        self.model = V41Model().to(self.device)
        self.last_q_vals = None
        self.last_policy = None
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.weight_path = os.path.join(repo_root, "versions", "v41", "weights", weight_name)
        try:
            self.model.load_state_dict(load_ckpt_state(self.weight_path, V41_MANIFEST))
            self.model.eval()
            self.loaded = True
        except Exception as e:
            # [Fable review #15] The failure is still swallowed here rather than raised, because
            # the registry constructs EVERY engine at PokerDecisionEngine.__init__ and one missing
            # rollback checkpoint must not take the whole app down. `self.loaded` is the contract:
            # core/decision.py refuses to serve a decision from an engine whose `.loaded` is False
            # instead of playing random weights at a real table. Do not "fix" this by removing the
            # try/except without moving the guard somewhere that still runs.
            self.loaded = False
            self.load_error = repr(e)
            print(f"WARNING: could not load V41 weights at {self.weight_path}: {e}. "
                  f"This engine will REFUSE to act (see decision.py's .loaded guard).")

    def make_bridge(self):
        """[Fable review #16/H4] This engine's OWN tensor bridge.

        `core/decision.py` calls this once at startup and uses the result instead of walking its
        `is_vN` substring ladder. That ladder's failure mode is silent and severe -- a registry entry
        the ladder doesn't recognise used to fall through to `bridge_v9`, throw, get caught, and fold
        every hand while play continued -- and its substring matching would collide a future 'v41b'
        with 'v41'. Declaring the contract here means the version that owns it also owns how its
        tensors are built, which is the direction the OFK guardrails point (manifest-driven dispatch,
        not more `is_vN` branching).
        """
        return V41Contract(max_seq_len=20)

    def live_features(self):
        """[V42_liveFixes] This version's OWN live input-feature implementations.

        Same argument as `make_bridge()` above, for the two OTHER hand-maintained ladders the review
        flagged (#16/H4's "still open" list): `PHPHelp.py` selected `compute_range_aware_equity` and
        `preflop_hand_strength` by walking a per-version substring chain that stopped at 'v29'. V40
        and V41 were both deployed without being added to it, so V41 played live on **vs-random
        equity** -- the feature it is most sensitive to, and one it was never trained on -- and a
        constant `hand_strength=0.5`, with nothing thrown and nothing logged. Declaring here means a
        version cannot be forgotten by a file that doesn't know it exists.

        V41 inherits both implementations unchanged from V29 through V40 (neither version touched
        `compute_range_aware_equity` or the contract's `preflop_hand_strength`), but they are
        imported from `versions.v41` regardless: the version serves its OWN copy, exactly as
        `make_bridge()` returns its own contract instance rather than sharing V29's.
        """
        return {
            'version_package': 'versions.v41',
            'range_aware_equity': True,
            'front_colors': True,      # front (already in) / after (still to act) split -- V20_preflopEq
            'hand_strength': True,     # contract index 36
        }

    def predict_ev(self, hole, board, ctx, act) -> dict:
        """Returns the ACTOR policy probabilities for the final step keyed by V41_ACTION_KEYS."""
        with torch.no_grad():
            out = self.model(hole.to(self.device), board.to(self.device),
                             ctx.to(self.device), act.to(self.device))
        logits = out["policy_logits"][0, -1, :]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        q = out["q_vals"][0, -1, :].cpu().numpy()
        self.last_q_vals = {k: float(q[i]) for i, k in enumerate(V41_ACTION_KEYS)}
        self.last_policy = {k: float(probs[i]) for i, k in enumerate(V41_ACTION_KEYS)}
        # Same aux-head reads as V21_auxhead/V25/V26/V28/V29 (identical head architecture) -- see
        # that engine's own docstring for the raw-MSE-scalar caveat (not confident categorical
        # bands). Still unused for action selection; HUD/telemetry only.
        self.last_aux = {
            'self_equity': float(out['equity'][0, -1]),
            'opp_strength': float(out['strength'][0, -1]),
            'opp_bluff': float(out['bluff'][0, -1]),
        }
        return {k: float(probs[i]) for i, k in enumerate(V41_ACTION_KEYS)}

"""[V25] TreeOpponent -- an opponent whose decisions come from a real XGBoost model fit on
observed human play (Pluribus/WSOP full-hole-card hand histories), not a hand-written formula.

Motivation (2026-07-18 discussion): every opponent in this simulator, heuristic or NN, has so far
either been hand-designed (FuzzyPlayerArchetype's fold/continue formulas) or trained inside this
same simulator (lagged-self, sharing whatever blind spots the training population already has).
Neither can express behavior we didn't already assume the shape of. Pluribus's released hand
histories give real cards for every decision (including folds), so a per-cluster tree model can be
fit directly on (real equity, price, street, stack, ...) -> action, capturing whatever nonlinear
shape real players' decisions actually have -- not ours.

Clustering note: the 4 clusters here are NOT designed to match TAG/LAG/NIT/CALLING_STATION --
per explicit instruction, they're whatever a Gaussian Mixture Model found in the real data
(identity-agnostic: clustered on OBSERVED BEHAVIOR, not player-name labels, since Pluribus's
anonymized code-names are a rotating pool of ~13 pros, not one name = one consistent person). All
4 turned out fairly similar to each other (Pluribus's human opponents were elite, similarly-skilled
professionals, not a diverse recreational population) -- this is an honest property of the source
data, not a clustering bug. See versions/v25/SPECS.md for the full pipeline and this caveat.

Interface note: `decide_preflop`/`decide_postflop` must return one of {'fold','call','raise'} to
match every other Opponent in this simulator (HeuristicOpponent/NNOpponent) -- the simulator sizes
ANY opponent's 'raise' with one fixed 0.75x-pot rule (simulator.py, ~line 671), the same treatment
every heuristic archetype already gets; this class doesn't (yet) carry a specific bet-size through
that pipeline. The XGBoost model itself predicts 6 classes (fold/call/raise_33/raise_66/raise_pot/
allin, matching hero's own action space, since that's what the real bet sizes were bucketed into
for training) -- the 4 raise buckets are collapsed to 'raise' at the interface boundary. Extending
the Opponent interface to carry a real predicted size through is a real, separate follow-up (see
SPECS.md), not attempted here.

Safety note: trees extrapolate badly outside the range they were trained on, and this simulator's
own curriculum deliberately sweeps stack depths/prices real hands rarely reach. Every input is
clipped to the SAME range used at training time (see CLIP_RANGES) before querying the model --
prevents wild extrapolation on out-of-distribution corners, not a personality choice.
"""
import os

import numpy as np
import xgboost as xgb

_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'weights', 'tree_opponents')

# Matches train_xgboost.py's own normalization exactly -- MUST stay in lockstep with whatever
# built the .json models in weights/tree_opponents/, or predictions silently go stale/wrong.
_BB = 100.0
_STACK_CEIL_BB = 200.0
_STREET_CEIL = 3.0
_NUM_OPP_CEIL = 5.0

ACTIONS = ("fold", "call", "raise_33", "raise_66", "raise_pot", "allin")
_RAISE_ACTIONS = {"raise_33", "raise_66", "raise_pot", "allin"}

# Clip ranges applied to the RAW (equity, pot_odds, street, stack_bb, num_active_opp) inputs
# before normalizing -- keeps queries inside (or close to) what the real Pluribus/WSOP data
# actually covered. street/num_active_opp are already hard-bounded by the game itself; equity/
# pot_odds are mathematically in [0,1] already; stack_bb is the one dimension real training
# hands (100bb-deep cash game) may not cover as widely as this simulator's own curriculum does.
CLIP_RANGES = {
    "equity": (0.0, 1.0),
    "pot_odds": (0.0, 1.0),
    "stack_bb": (0.0, _STACK_CEIL_BB),
    "num_active_opp": (0.0, _NUM_OPP_CEIL),
}


class TreeOpponent:
    """Mirrors the Opponent base interface (opponents.py) directly -- kept standalone rather than
    subclassing Opponent to avoid a circular import (opponents.py doesn't need to know about this
    module); `simulator.py`'s call sites only ever call `.decide_preflop`/`.decide_postflop`/
    `.apply_forcing_*`/`.recording_bot`/`.label`, which this class provides the same way
    HeuristicOpponent/NNOpponent do."""

    _boosters = {}  # class-level cache: cluster_id -> xgb.Booster, loaded once per process

    def __init__(self, cluster_id, recording_bot, style=None, display_name=None, forced=False):
        self.cluster_id = cluster_id
        self.recording_bot = recording_bot  # HUD telemetry + fallback on any query failure
        self.style = style or f"tree_{cluster_id}"
        self.forced = forced
        self.display_name = display_name or f"RealPlay-{cluster_id}"
        self.kind = 'Tree'
        self.booster = self._load_booster(cluster_id)

    @classmethod
    def _load_booster(cls, cluster_id):
        if cluster_id not in cls._boosters:
            path = os.path.join(_WEIGHTS_DIR, f"xgb_cluster_{cluster_id}.json")
            booster = xgb.Booster()
            booster.load_model(path)
            # [V26 hang fix, 2026-07-18] XGBoost spins up its OWN internal OpenMP thread pool by
            # default (one per Booster/process). Training runs this inside 8 separate spawned
            # worker PROCESSES (Windows multiprocessing.Pool, spawn not fork) -- 8 processes each
            # independently grabbing their own multi-threaded pool oversubscribes the machine's
            # real core count and reproducibly DEADLOCKED (confirmed: 1-worker test succeeded in
            # 14.8s, 8-worker test hung past a 90s timeout, every time). Pinning each Booster to a
            # single thread makes every worker's XGBoost calls single-threaded, matching how the
            # rest of this simulator is already purely single-threaded-per-worker -- eliminates
            # the oversubscription entirely rather than trying to tune thread counts.
            booster.set_param({'nthread': 1})
            cls._boosters[cluster_id] = booster
        return cls._boosters[cluster_id]

    @property
    def label(self):
        return f"{self.display_name} ({self.kind})"

    @staticmethod
    def _clip(value, key):
        lo, hi = CLIP_RANGES[key]
        return min(max(value, lo), hi)

    def _predict_action(self, equity, pot_odds, street_idx, stack, num_opps):
        try:
            eq = self._clip(equity, "equity")
            po = self._clip(pot_odds, "pot_odds")
            stack_bb = self._clip(stack / _BB, "stack_bb")
            n_opp = self._clip(num_opps, "num_active_opp")

            features = np.array([[
                eq, po, street_idx / _STREET_CEIL, stack_bb / _STACK_CEIL_BB, n_opp / _NUM_OPP_CEIL,
            ]], dtype=np.float32)
            proba = self.booster.predict(xgb.DMatrix(features))[0]
            # SAMPLE from the predicted distribution, not argmax -- a static, always-most-likely
            # bot is fully predictable/exploitable; every other opponent in this pool has some
            # randomness (fuzzed traits, bluff_freq rolls) for the same reason.
            action_idx = np.random.choice(len(ACTIONS), p=proba / proba.sum())
            action = ACTIONS[action_idx]
            return "raise" if action in _RAISE_ACTIONS else action
        except Exception:
            return None  # caller falls back to recording_bot

    def decide_preflop(self, equity, pot_odds, *_a, force_heuristic=False, **_kw):
        # Opponent.decide_preflop's full positional signature is (equity, pot_odds, pot_size,
        # stack, num_opps, cards, ...) -- `_a` = (pot_size, stack, num_opps, cards, ...) here.
        if force_heuristic:
            return self.recording_bot.decide_preflop(equity, pot_odds)
        stack = _a[1] if len(_a) > 1 else 10000
        num_opps = _a[2] if len(_a) > 2 else 1
        result = self._predict_action(equity, pot_odds, street_idx=0, stack=stack, num_opps=num_opps)
        return result if result is not None else self.recording_bot.decide_preflop(equity, pot_odds)

    def decide_postflop(self, equity, pot_odds, pot_size, stack, street_idx, *_a,
                         force_heuristic=False, **_kw):
        if force_heuristic:
            return self.recording_bot.decide_postflop(equity, pot_odds, pot_size, stack, street_idx)
        num_opps = _a[0] if _a else 1
        result = self._predict_action(equity, pot_odds, street_idx, stack, num_opps)
        return result if result is not None else self.recording_bot.decide_postflop(
            equity, pot_odds, pot_size, stack, street_idx)

    def apply_forcing_preflop(self, decision, opp_vpip, opp_agg):
        return decision  # never forced -- a real-data-fitted model keeps its own tendencies

    def apply_forcing_postflop(self, decision, opp_vpip, opp_agg):
        return decision

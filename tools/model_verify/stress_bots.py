"""Stress-test opponent(s) with a DELIBERATELY DIFFERENT functional form than the training
pool's `FuzzyPlayerArchetype` (versions/<v>/self_play/opponent_bots.py).

Why this exists: the training pool's bots are all built from the SAME continuous formula family
(a pot-odds-linear continue bar, `continue_bar = pot_odds + style_shift`, with Gaussian-jittered
parameters each hand). That jitter defends against memorizing one exact threshold, but not against
the hero implicitly learning structure specific to that one formula SHAPE over hundreds of
thousands of repeated hands against it. `TieredLookupBot` below is intentionally built a totally
different way -- a coarse discrete equity-tier lookup table that is PRICE-INSENSITIVE (ignores
pot_odds entirely) and tightens NON-MONOTONICALLY BY STREET (loose flop, much tighter turn/river)
rather than by bet size. If the hero's win-rate collapses specifically against this shape (while
holding up against the training-formula-family fields), that's evidence of overfitting to the
training population's specific structure rather than genuinely general play.

Implements the exact bot interface `versions/<v>/self_play/simulator.py` expects on
`self.<style>_heuristic` (`decide_preflop`, `decide_postflop`, `record_preflop`,
`record_postflop`, `start_new_hand`, `base_vpip`) so it can be dropped in via a plain attribute
override on a fresh `SixMaxSimulator` instance -- no simulator.py changes needed.
"""
import random


class TieredLookupBot:
    base_vpip = 0.30  # nominal, for the o['bot'].base_vpip telemetry read in _mc_target_evs_sized

    PREFLOP_TIERS = [  # (equity upper bound, {action: prob}) -- NO pot_odds dependence at all
        (0.30, {'fold': 1.00}),
        (0.45, {'fold': 0.70, 'call': 0.30}),
        (0.60, {'call': 0.60, 'raise': 0.40}),
        (0.75, {'call': 0.30, 'raise': 0.70}),
        (1.01, {'raise': 1.00}),
    ]
    # Tightens by STREET, not by price -- "commits light on the flop, bails deep" instead of the
    # training bots' "folds more to bigger bets" shape.
    POSTFLOP_TIERS = {
        'flop':  [(0.35, {'fold': 0.50, 'call': 0.50}),
                  (0.60, {'call': 0.60, 'raise': 0.40}),
                  (1.01, {'raise': 0.80, 'call': 0.20})],
        'turn':  [(0.55, {'fold': 0.85, 'call': 0.15}),
                  (0.75, {'call': 0.70, 'raise': 0.30}),
                  (1.01, {'raise': 0.70, 'call': 0.30})],
        'river': [(0.65, {'fold': 0.90, 'call': 0.10}),
                  (0.85, {'call': 0.75, 'raise': 0.25}),
                  (1.01, {'raise': 0.60, 'call': 0.40})],
    }

    def start_new_hand(self):
        pass  # deterministic tiers -- no per-hand Gaussian identity fuzz (a different kind of
              # variance than the training pool uses, on purpose)

    @staticmethod
    def _pick(tiers, equity):
        for cap, dist in tiers:
            if equity <= cap:
                r = random.random()
                cum = 0.0
                for action, p in dist.items():
                    cum += p
                    if r <= cum:
                        return action
                return next(iter(dist))
        return 'fold'

    def decide_preflop(self, equity, pot_odds, is_blind=False):
        action = self._pick(self.PREFLOP_TIERS, equity)
        if action == 'fold' and is_blind and pot_odds == 0:
            return 'call'
        return action

    def decide_postflop(self, equity, pot_odds, pot_size, stack, street_idx):
        street = {0: 'flop', 1: 'flop', 2: 'turn', 3: 'river'}.get(street_idx, 'river')
        return self._pick(self.POSTFLOP_TIERS[street], equity)

    def record_preflop(self, action):
        pass

    def record_postflop(self, action):
        pass

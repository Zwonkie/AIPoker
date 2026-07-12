"""
Conceptual model of the four core poker archetypes.

This is NOT a poker engine — it's a mental model expressed as code.
Numbers are illustrative approximations of tendencies, not GTO values.

Key stats modeled:
  - vpip:  % of hands they voluntarily put money in preflop (looseness)
  - pfr:   % of hands they raise preflop (aggression, preflop)
  - af:    aggression factor = (bets + raises) / calls  (postflop)
  - equity thresholds: how much raw equity they *think* they need
    to continue / bet / raise on each street
"""

from dataclasses import dataclass, field
import random


@dataclass
class Decision:
    action: str          # 'fold' | 'check' | 'call' | 'bet' | 'raise'
    reasoning: str


@dataclass
class PlayerArchetype:
    name: str

    # --- Preflop identity -------------------------------------------------
    vpip: float          # e.g. 0.22 -> plays 22% of starting hands
    pfr: float           # e.g. 0.18 -> raises 18% of starting hands
    three_bet: float     # how often they re-raise preflop

    # --- Postflop identity ------------------------------------------------
    aggression_factor: float     # >2.5 aggressive, <1.5 passive
    bluff_frequency: float       # 0..1, how often they bet with air
    fold_to_pressure: float      # 0..1, how easily they release a hand

    # --- Perceived equity needed to CONTINUE (call) per street -------------
    # Sticky players *underestimate* what they need; nits *overestimate*.
    call_threshold: dict = field(default_factory=dict)   # street -> equity
    # --- Equity needed to BET/RAISE for value ------------------------------
    value_threshold: dict = field(default_factory=dict)  # street -> equity

    # ------------------------------------------------------------------ #
    def preflop(self, hand_strength_percentile: float) -> Decision:
        """hand_strength_percentile: 0.0 (72o) .. 1.0 (AA)"""
        if hand_strength_percentile >= (1 - self.pfr):
            return Decision('raise', f'{self.name}: top {self.pfr:.0%} of hands -> raise')
        if hand_strength_percentile >= (1 - self.vpip):
            return Decision('call', f'{self.name}: playable but not raise-worthy -> limp/call')
        return Decision('fold', f'{self.name}: below my {self.vpip:.0%} VPIP range -> fold')

    # ------------------------------------------------------------------ #
    def postflop(self, street: str, equity: float, facing_bet: bool) -> Decision:
        """
        street: 'flop' | 'turn' | 'river'
        equity: our true equity vs opponent's range (0..1)
        """
        need_to_call = self.call_threshold[street]
        need_for_value = self.value_threshold[street]

        if facing_bet:
            # Aggressive players sometimes raise instead of call
            if equity >= need_for_value and self._wants_to_raise():
                return Decision('raise', f'{self.name}: {equity:.0%} equity on {street} -> raise for value')
            if equity >= need_to_call:
                return Decision('call', f'{self.name}: {equity:.0%} >= my {need_to_call:.0%} bar -> call')
            # Below threshold: do we hero-fold or float anyway?
            if random.random() > self.fold_to_pressure:
                return Decision('call', f'{self.name}: technically weak, but I do not fold -> sticky call')
            return Decision('fold', f'{self.name}: {equity:.0%} < {need_to_call:.0%} -> fold')

        # Nobody has bet -> do we attack or check?
        if equity >= need_for_value:
            return Decision('bet', f'{self.name}: value bet {street} with {equity:.0%}')
        if random.random() < self.bluff_frequency:
            return Decision('bet', f'{self.name}: bluffing {street} (bluff freq {self.bluff_frequency:.0%})')
        return Decision('check', f'{self.name}: not strong enough, not bluffing -> check')

    def _wants_to_raise(self) -> bool:
        """Aggression factor as a probability of choosing raise over call."""
        return random.random() < min(self.aggression_factor / 4, 0.95)


# ======================================================================= #
#  THE CORE FOUR
# ======================================================================= #

TAG = PlayerArchetype(
    name='TAG',
    vpip=0.22, pfr=0.18, three_bet=0.07,        # tight, but raises most hands he plays
    aggression_factor=3.0,
    bluff_frequency=0.25,                        # bluffs selectively, in good spots
    fold_to_pressure=0.60,                       # disciplined: can let go of hands
    call_threshold={'flop': 0.40, 'turn': 0.45, 'river': 0.50},   # roughly pot-odds correct
    value_threshold={'flop': 0.60, 'turn': 0.62, 'river': 0.65},
)

LAG = PlayerArchetype(
    name='LAG',
    vpip=0.32, pfr=0.26, three_bet=0.11,        # wide AND aggressive
    aggression_factor=3.8,
    bluff_frequency=0.40,                        # constant pressure, many bluffs
    fold_to_pressure=0.45,                       # fights back, floats, re-bluffs
    call_threshold={'flop': 0.33, 'turn': 0.38, 'river': 0.45},   # continues light on purpose
    value_threshold={'flop': 0.55, 'turn': 0.58, 'river': 0.60},  # value bets thinner
)

NIT = PlayerArchetype(
    name='Nit/Rock',
    vpip=0.11, pfr=0.08, three_bet=0.03,        # only premiums
    aggression_factor=1.8,                       # even good hands played cautiously
    bluff_frequency=0.03,                        # basically never bluffs
    fold_to_pressure=0.85,                       # folds to almost any real pressure
    call_threshold={'flop': 0.55, 'turn': 0.62, 'river': 0.70},   # demands way more than pot odds
    value_threshold={'flop': 0.75, 'turn': 0.80, 'river': 0.85},  # only bets near-nuts
)

CALLING_STATION = PlayerArchetype(
    name='Calling Station',
    vpip=0.45, pfr=0.06, three_bet=0.01,        # plays everything, raises nothing
    aggression_factor=0.7,                       # calls >> bets
    bluff_frequency=0.05,                        # too passive to bluff
    fold_to_pressure=0.15,                       # "I have to see it"
    call_threshold={'flop': 0.20, 'turn': 0.22, 'river': 0.25},   # any pair / any draw / ace-high
    value_threshold={'flop': 0.70, 'turn': 0.72, 'river': 0.75},  # only bets when very strong
)


# ======================================================================= #
#  DEMO: same situation, four different players
# ======================================================================= #

if __name__ == '__main__':
    random.seed(7)
    players = [TAG, LAG, NIT, CALLING_STATION]

    print('--- Preflop: dealt KJs (~top 15% hand, percentile 0.85) ---')
    for p in players:
        print(' ', p.preflop(0.85).reasoning)

    print('\n--- Flop: middle pair, ~35% equity, facing a half-pot bet ---')
    for p in players:
        print(' ', p.postflop('flop', 0.35, facing_bet=True).reasoning)

    print('\n--- River: strong hand, 68% equity, checked to us ---')
    for p in players:
        print(' ', p.postflop('river', 0.68, facing_bet=False).reasoning)

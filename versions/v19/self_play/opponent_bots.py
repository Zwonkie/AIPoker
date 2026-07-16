"""
Fuzzy Heuristics Opponent Bots for V11 Self-Play.

These bots exclusively replace the old rigid heuristics to prevent the Hero from 
overfitting to static math triggers. Base stats are "fuzzed" using Gaussian noise
at the start of every hand.

Note: We use the Aggression Frequency (AGG) approach rather than Aggression Factor (AF) 
to determine the probability of raising instead of calling when deciding to play a hand.
AGG = (Bets + Raises) / (Bets + Raises + Calls + Folds).
"""

import random

# V14 P1b: how far a bot sits off the break-even price when facing a bet. A high-fold_to_pressure
# (nit) demands equity ABOVE the price (over-folds); a low one (station) continues BELOW it. ±0.10
# at the archetype extremes (0.85 / 0.15). Tunable. See versions/v19/SPECS.md §P1b.
STYLE_SHIFT_SCALE = 0.30


class FuzzyPlayerArchetype:
    def __init__(
        self, 
        name: str, 
        base_vpip: float, 
        base_agg_freq: float,  # The Bet365 AGG approach: % of actions that are aggressive
        base_bluff_freq: float, 
        base_fold_to_pressure: float, 
        base_call_threshold: dict, 
        base_value_threshold: dict
    ):
        self.name = name
        self.base_vpip = base_vpip
        self.base_agg_freq = base_agg_freq
        self.base_bluff_freq = base_bluff_freq
        self.base_fold_to_pressure = base_fold_to_pressure
        self.base_call_threshold = base_call_threshold
        self.base_value_threshold = base_value_threshold
        
        # Current hand specific stats (fuzzed)
        self.current_vpip = base_vpip
        self.current_agg_freq = base_agg_freq
        self.current_bluff_freq = base_bluff_freq
        self.current_fold_to_pressure = base_fold_to_pressure
        self.current_call_threshold = dict(base_call_threshold)
        self.current_value_threshold = dict(base_value_threshold)
        
        # HUD tracking (still useful for telemetry, even if not fuzzed)
        self.hands_played = 0
        self.vpip_count = 0
        self.pfr_count = 0
        self.agg_bets = 0
        self.agg_calls = 0
        self.agg_folds = 0
        
    def start_new_hand(self):
        """Called by the simulator at the start of every hand to roll the fuzzy traits."""
        # Fuzz identity with small standard deviation
        self.current_vpip = max(0.01, min(0.99, random.gauss(self.base_vpip, 0.05)))
        self.current_agg_freq = max(0.01, min(0.99, random.gauss(self.base_agg_freq, 0.08)))
        self.current_bluff_freq = max(0.0, min(1.0, random.gauss(self.base_bluff_freq, 0.05)))
        self.current_fold_to_pressure = max(0.0, min(1.0, random.gauss(self.base_fold_to_pressure, 0.05)))
        
        for street, val in self.base_call_threshold.items():
            self.current_call_threshold[street] = max(0.0, min(1.0, random.gauss(val, 0.04)))
            
        for street, val in self.base_value_threshold.items():
            self.current_value_threshold[street] = max(0.0, min(1.0, random.gauss(val, 0.04)))

    @property
    def vpip(self):
        return self.vpip_count / max(1, self.hands_played)
        
    @property
    def agg_frequency(self):
        total = self.agg_bets + self.agg_calls + self.agg_folds
        return self.agg_bets / max(1, total)

    @property
    def vpip_normalized(self):
        return min(1.0, self.vpip)
    
    @property
    def agg_normalized(self):
        return min(1.0, self.agg_frequency)

    def record_preflop(self, action):
        self.hands_played += 1
        if action in ('call', 'raise'):
            self.vpip_count += 1
        if action == 'raise':
            self.pfr_count += 1
            
    def record_postflop(self, action):
        if action in ('raise', 'bet'):
            self.agg_bets += 1
        elif action == 'call':
            self.agg_calls += 1
        elif action == 'fold':
            self.agg_folds += 1

    def decide_preflop(self, equity, pot_odds, is_blind=False):
        """
        Uses raw equity to proxy hand strength percentile.
        We don't use PFR explicitly in the boardstate for Hero tracking, but the bot
        uses Aggression Frequency internally to decide to raise or call preflop.
        """
        # Top equity hands (value threshold)
        if equity >= self.current_value_threshold.get('flop', 0.60):
            return 'raise'

        # V19 P0: SIZE-AWARE preflop continue/fold bar, mirroring decide_postflop's V14 [P1b]
        # fix. Previously `pot_odds` was accepted but never used here -- a min-raise and an
        # all-in shove got the IDENTICAL simulated fold rate, which fed directly into
        # `_mc_target_evs_sized`'s per-size EV target and systematically inflated the all-in
        # target at marginal equity (the deep_stack_ood_guard trash-jam root cause, failing on
        # every version V15-v17_gauntlet). Facing a real bet, the bar now rises with bet size
        # (same `pot_odds + style_shift` formula as postflop); with no bet yet (RFI/limped pot),
        # the original flat VPIP-proxy bar is unchanged.
        facing_bet = pot_odds > 0
        if facing_bet:
            style_shift = (self.current_fold_to_pressure - 0.5) * STYLE_SHIFT_SCALE
            call_bar = min(0.95, max(0.02, pot_odds + style_shift))
        else:
            # Playable hands (VPIP threshold roughly proxied by equity)
            # E.g. VPIP 0.22 implies playing top 22% of hands -> equity > ~0.55
            # We simplify by tying it to call_thresholds
            call_bar = self.current_call_threshold.get('flop', 0.40)

        if equity >= call_bar:
            # Does the bot want to raise instead of call?
            if random.random() < self.current_agg_freq:
                return 'raise'
            return 'call'

        # Too weak
        return 'call' if is_blind and pot_odds == 0 else 'fold'

    def decide_postflop(self, equity, pot_odds, pot_size, stack, street_idx):
        street_map = {0: 'flop', 1: 'flop', 2: 'turn', 3: 'river'}
        street_str = street_map.get(street_idx, 'river')
        
        need_to_call = self.current_call_threshold.get(street_str, 0.5)
        need_for_value = self.current_value_threshold.get(street_str, 0.7)
        
        facing_bet = pot_odds > 0
        
        if facing_bet:
            # --- SIZE-AWARE continue/fold bar (V14 P1b) ----------------------------------
            # pot_odds == bet/(pot+bet) == the break-even equity to call, and it RISES with bet
            # size (½-pot->0.33, pot->0.50, 2x overbet->0.67). Anchoring the continue bar to
            # pot_odds makes the bot fold MORE to bigger bets -- the size-response signal the hero
            # needs to learn sizing. (Range view: continuing only when equity>=price yields a
            # defend frequency ~= MDF = 1 - pot_odds.) Style shifts the bar off the price:
            #   fold_to_pressure high (nit)     -> demands equity ABOVE price (over-folds)
            #   fold_to_pressure low  (station) -> continues BELOW price      (under-folds/sticky)
            # This REPLACES the old flat fold_to_pressure "sticky float" (which ignored bet size).
            style_shift = (self.current_fold_to_pressure - 0.5) * STYLE_SHIFT_SCALE
            continue_bar = min(0.95, max(0.02, pot_odds + style_shift))

            if equity >= need_for_value:
                # Value raise (strong hands raise regardless of price)
                if random.random() < self.current_agg_freq * 2.0:
                    return 'raise'
                return 'call' # slowplay

            if equity >= continue_bar:
                # Clears the size-adjusted price -> continue; sometimes raise (semi-bluff/protection)
                if random.random() < self.current_agg_freq * 1.0:
                    return 'raise'
                return 'call'

            # Below the bar -> mostly fold. (Size-scaled bluff-raise deferred to V15 — see SPECS.)
            if random.random() < self.current_bluff_freq and random.random() < self.current_agg_freq * 1.5:
                return 'raise' # bluff raise

            return 'fold'

        else:
            # Nobody has bet
            if equity >= need_for_value:
                if random.random() < self.current_agg_freq * 2.5:
                    return 'raise' # value bet
                return 'call' # slowplay check
                
            if equity >= need_to_call:
                # marginal hand
                if random.random() < self.current_agg_freq * 1.5:
                    return 'raise' # protection bet
                return 'call' # check
                
            # Weak hand
            if random.random() < self.current_bluff_freq and random.random() < self.current_agg_freq * 2.0:
                return 'raise' # bluff bet
            return 'call' # check


# ======================================================================= #
#  V11 POOL DEFINITIONS
# ======================================================================= #

TAG = FuzzyPlayerArchetype(
    name='TAG',
    base_vpip=0.22, 
    base_agg_freq=0.45,   # Bet365 AGG proxy
    base_bluff_freq=0.25,                        
    base_fold_to_pressure=0.60,                       
    base_call_threshold={'flop': 0.42, 'turn': 0.47, 'river': 0.52},   
    base_value_threshold={'flop': 0.60, 'turn': 0.62, 'river': 0.65},
)

LAG = FuzzyPlayerArchetype(
    name='LAG',
    base_vpip=0.32, 
    base_agg_freq=0.55,
    base_bluff_freq=0.40,                        
    base_fold_to_pressure=0.45,                       
    base_call_threshold={'flop': 0.37, 'turn': 0.42, 'river': 0.49},   
    base_value_threshold={'flop': 0.55, 'turn': 0.58, 'river': 0.60},  
)

NIT = FuzzyPlayerArchetype(
    name='Nit',
    base_vpip=0.11, 
    base_agg_freq=0.25,                       
    base_bluff_freq=0.03,                        
    base_fold_to_pressure=0.85,                       
    base_call_threshold={'flop': 0.50, 'turn': 0.57, 'river': 0.65},   
    base_value_threshold={'flop': 0.75, 'turn': 0.80, 'river': 0.85},  
)

CALLING_STATION = FuzzyPlayerArchetype(
    name='Calling Station',
    base_vpip=0.45, 
    base_agg_freq=0.15,                       
    base_bluff_freq=0.05,                        
    base_fold_to_pressure=0.15,                       
    base_call_threshold={'flop': 0.32, 'turn': 0.34, 'river': 0.37},   
    base_value_threshold={'flop': 0.70, 'turn': 0.72, 'river': 0.75},  
)

BOT_PROFILES = {
    'tag': TAG,
    'lag': LAG,
    'nit': NIT,
    'calling_station': CALLING_STATION,
    'fish': CALLING_STATION,
    'maniac': LAG,
    'sticky': CALLING_STATION
}

def create_opponent_pool(distribution=None):
    if distribution is None:
        distribution = {
            'tag': 0.25,
            'maniac': 0.25,
            'nit': 0.20,
            'calling_station': 0.30
        }
        
    pool = []
    for style, weight in distribution.items():
        profile_template = BOT_PROFILES.get(style, TAG)
        # Create a fresh copy for this opponent
        bot = FuzzyPlayerArchetype(
            name=f"{style.capitalize()}",
            base_vpip=profile_template.base_vpip,
            base_agg_freq=profile_template.base_agg_freq,
            base_bluff_freq=profile_template.base_bluff_freq,
            base_fold_to_pressure=profile_template.base_fold_to_pressure,
            base_call_threshold=profile_template.base_call_threshold,
            base_value_threshold=profile_template.base_value_threshold
        )
        pool.append((bot, weight))
        
    return pool

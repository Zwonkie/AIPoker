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
# at the archetype extremes (0.85 / 0.15). Tunable. See versions/v20/SPECS.md §P1b.
STYLE_SHIFT_SCALE = 0.30

# [V23, BET-1] How much the "auto-continue for value regardless of price" bar (need_for_value)
# rises per unit of pot_odds, extending the SAME price-sensitivity mechanism STYLE_SHIFT_SCALE
# already gives the marginal continue_bar up to the top of the equity range too. Previously
# need_for_value was completely flat -- a min-bet and an all-in shove got IDENTICAL continuation
# once a bot's equity cleared it, which is the root cause [BET-1] traces hero's shove-preference
# to (bigger bets extract identical value with zero extra fold-equity cost, so the critic sees no
# downside to sizing up).
#
# Calibrated 2026-07-17 via a standalone probe (versions/v23/self_play/calibrate_bet1.py, kept as
# a reusable tool -- results captured here) that directly measured P(fold) at value-tier equities across a
# pot_odds grid (0.20/0.33/0.50/0.67/0.80, i.e. ~1/4-pot up to a shove-into-a-big-pot) for all 4
# archetypes, comparing the unmodified baseline against several candidate values:
#   - Baseline confirmed the bug: P(fold) stayed ~0 regardless of bet size once equity cleared
#     need_for_value (e.g. TAG at equity=0.70: 0.003 at po=0.67 vs 0.004 at po=0.80 -- flat).
#   - A single global constant compounds UNEVENLY across archetypes: NIT (tightest, already has
#     the highest need_for_value=0.75 AND the steepest style_shift) responds far more steeply than
#     TAG/LAG at the identical constant -- at 0.15, NIT's P(fold) at equity=0.80 facing a shove
#     jumped from a 9.9% baseline to 95%+, an unrealistic "always folds a big favorite" collapse,
#     while TAG only reached 58% at that same value.
#   - 0.05 chosen as the calibrated value: creates a REAL, non-trivial fold-equity gradient across
#     ALL FOUR archetypes (TAG: 1.8%->5.4% across the tested bet sizes at equity=0.70; LAG: 24-26%
#     at equity=0.60; NIT: up to 40% at equity=0.80, the strongest response, fitting since NIT is
#     the tightest archetype -- thematically correct, not an overcorrection; CALLING_STATION:
#     18.7%->30.3% at equity=0.70) without ANY archetype collapsing into "always folds a strong
#     hand" territory (0.08+ started pushing NIT past 60%, judged too extreme for an 80%-equity
#     hand). Every archetype still NEVER folds a near-nuts hand (equity>=0.90) regardless of bet
#     size at this value -- the value_bar's min(0.98, ...) clamp keeps monster hands safe, they can
#     only be pushed from the "auto-raise-for-value" branch into the (still-continuing,
#     price-sensitive) marginal-continue branch, never all the way to fold.
VALUE_PRICE_SENSITIVITY = 0.05

# ======================================================================================= #
# [V47, Change 1 / #6 SIM-RAISE] Per-archetype raise-SIZE repertoires.
#
# Until V47 the simulator sized EVERY opponent raise as min(pot*0.75, stack) -- hero had never
# faced an open-jam, a min-raise probe, or an overbet from any opponent in this project's
# history, so its fold-vs-raise responses (and the [OPP-2] per-seat raise features) were
# calibrated against a world with exactly one opponent bet size. Each entry is a list of
# (pot_fraction, weight); fraction None = open-jam (all-in). Fractions feed the SAME
# `_raise_size_for_fraction` hero uses (min-raise floor, stack cap), so a small fraction in a
# small pot realizes as a legal MIN-RAISE -- that is how LAG's probe sizes manifest preflop --
# and every size obeys the V41 min-raise/reopen rules by construction.
#
# [V48, Change 1b] FITTED from the measured player population -- live2/historydb/population.py
# over 4,015 real hands / 99 recurring opponents (hero excluded), history/population_fit.json,
# 2026-07-22. The measured signature the hand-authored V47 tables missed: preflop raises are
# BIG (pot-plus to jam; sub-pot fractions barely exist preflop), postflop bets are SMALL
# (0.33-0.50 pot dominates every cluster). Hence STREET-SPLIT tables.
#
# Semantics note: the fit measures amount/pot-BEFORE; the sim frac is of pot-AFTER-CALL
# (`_raise_size_for_fraction`). POSTFLOP to_call is usually 0 at bet time, so fitted values
# carry over VERBATIM. PREFLOP the open geometry (pot 1.5bb, call 1bb) compresses fitted
# {<=0.75 / 1.0 / 1.5 / jam} onto sim fracs {floor->0.33 / 0.33 / 0.66 / None}; the C1
# harness's realized-class shares (min-raise/small/pot/overbet/jam) are the calibration
# check against the fit's realized classes -- adjust HERE if C1 disagrees materially.
# Keyed by FuzzyPlayerArchetype.name; `sample_raise_fraction` maps pool STYLE keys through
# BOT_PROFILES so 'maniac'->LAG, 'fish'->Calling Station etc. resolve to the right shape.
# ======================================================================================= #
RAISE_SIZE_DISTRIBUTIONS_PREFLOP = {
    'TAG':             [(0.33, 0.31), (0.66, 0.33), (None, 0.36)],
    'LAG':             [(0.33, 0.35), (0.66, 0.38), (None, 0.27)],
    'Nit':             [(0.33, 0.19), (0.66, 0.34), (None, 0.47)],
    'Calling Station': [(0.33, 0.25), (0.66, 0.42), (None, 0.33)],
}
# Postflop: fitted values verbatim (semantics align, see note above).
RAISE_SIZE_DISTRIBUTIONS = {
    'TAG':             [(0.33, 0.370), (0.50, 0.253), (0.66, 0.160), (0.75, 0.056),
                        (1.00, 0.037), (None, 0.123)],
    'LAG':             [(0.33, 0.378), (0.50, 0.313), (0.66, 0.116), (0.75, 0.039),
                        (1.00, 0.089), (1.50, 0.008), (None, 0.058)],
    'Nit':             [(0.33, 0.275), (0.50, 0.275), (0.66, 0.255), (1.00, 0.059),
                        (None, 0.137)],
    'Calling Station': [(0.33, 0.456), (0.50, 0.163), (0.66, 0.066), (0.75, 0.050),
                        (1.00, 0.150), (1.50, 0.016), (None, 0.100)],
}

# Fallback for a style/name with no entry (e.g. a stress bot): the old single 0.75-pot size,
# so an unknown opponent degrades to pre-V47 behavior rather than crashing or guessing.
_LEGACY_RAISE_DISTRIBUTION = [(0.75, 1.0)]


def raise_size_distribution_for(style_or_name, street_idx=None):
    """Resolve a pool style key ('maniac', 'fish', 'tag', ...) OR an archetype display name
    ('LAG', 'Calling Station', ...) to its raise-size distribution. [V48, Change 1b]
    street_idx 0 -> the preflop table (measured big-raise/jam-heavy shape); any other street
    (or None, the legacy street-blind call) -> the postflop table."""
    table = RAISE_SIZE_DISTRIBUTIONS_PREFLOP if street_idx == 0 else RAISE_SIZE_DISTRIBUTIONS
    if style_or_name in table:
        return table[style_or_name]
    profile = BOT_PROFILES.get(str(style_or_name).lower())
    if profile is not None and profile.name in table:
        return table[profile.name]
    return _LEGACY_RAISE_DISTRIBUTION


def sample_raise_fraction(style_or_name, street_idx=None):
    """Sample one raise event's pot fraction (None = all-in) from the archetype's repertoire."""
    dist = raise_size_distribution_for(style_or_name, street_idx=street_idx)
    fracs = [f for f, _w in dist]
    weights = [w for _f, w in dist]
    return random.choices(fracs, weights=weights, k=1)[0]


class FuzzyPlayerArchetype:
    def __init__(
        self,
        name: str,
        base_vpip: float,
        base_agg_freq: float,  # The Bet365 AGG approach: % of actions that are aggressive
        base_bluff_freq: float,
        base_fold_to_pressure: float,
        base_call_threshold: dict,
        base_value_threshold: dict,
        base_bluff_perc: float = 0.20,
    ):
        self.name = name
        self.base_vpip = base_vpip
        self.base_agg_freq = base_agg_freq
        self.base_bluff_freq = base_bluff_freq
        self.base_fold_to_pressure = base_fold_to_pressure
        self.base_call_threshold = base_call_threshold
        self.base_value_threshold = base_value_threshold
        # [V24] How often THIS bot's own raise is "for show" (representing strength it doesn't
        # actually have), as a general personality trait -- distinct from `base_bluff_freq` (which
        # only governs the narrower below-the-fold-bar bluff-RAISE frequency). Currently used one
        # way: when this bot is the one FACING a raise, `1.0 - bot_bluff_perc` is its probability of
        # "respecting" that raise as committed value (see simulator.py's `_ev_target_fold_decision`)
        # -- a bot that bluffs a lot itself assumes others do too (low respect); one that rarely
        # bluffs trusts raises more (high respect). Deliberately general-purpose for reuse in other
        # bluff-related calculations later (e.g. preflop bluff-raise frequency) -- scoped to just
        # the raise-respect mechanism for now. See versions/v28/SPECS.md.
        self.base_bluff_perc = base_bluff_perc

        # Current hand specific stats (fuzzed)
        self.current_vpip = base_vpip
        self.current_agg_freq = base_agg_freq
        self.current_bluff_freq = base_bluff_freq
        self.current_fold_to_pressure = base_fold_to_pressure
        self.current_call_threshold = dict(base_call_threshold)
        self.current_value_threshold = dict(base_value_threshold)
        self.current_bluff_perc = base_bluff_perc

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
        self.current_bluff_perc = max(0.0, min(1.0, random.gauss(self.base_bluff_perc, 0.05)))

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
        # [V47 hygiene] startswith('raise'): NN opponents record sized bucket strings
        # ('raise_0'..'raise_3') through this same funnel -- exact-match 'raise' silently
        # dropped every one of those from VPIP/PFR since V18, so an NN seat's recording-bot
        # HUD read looked far more passive than its actual play.
        self.hands_played += 1
        if action == 'call' or action.startswith('raise'):
            self.vpip_count += 1
        if action.startswith('raise'):
            self.pfr_count += 1

    def record_postflop(self, action):
        if action == 'bet' or action.startswith('raise'):   # [V47 hygiene] see record_preflop
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
        # V19 P0: SIZE-AWARE preflop continue/fold bar, mirroring decide_postflop's V14 [P1b]
        # fix. Previously `pot_odds` was accepted but never used here -- a min-raise and an
        # all-in shove got the IDENTICAL simulated fold rate, which fed directly into
        # `_mc_target_evs_sized`'s per-size EV target and systematically inflated the all-in
        # target at marginal equity (the deep_stack_ood_guard trash-jam root cause, failing on
        # every version V15-v17_gauntlet). Facing a real bet, the bar now rises with bet size
        # (same `pot_odds + style_shift` formula as postflop); with no bet yet (RFI/limped pot),
        # the original flat VPIP-proxy bar is unchanged.
        facing_bet = pot_odds > 0

        # [V23, BET-1] The old code checked the flat value threshold FIRST, unconditionally --
        # "raise regardless of price" even facing a bet, the exact mechanism [BET-1] traces the
        # hero's shove-preference to (a bigger bet extracted identical continuation as a min-bet,
        # so the critic saw no downside to sizing up). Restructured so the facing-a-bet branch
        # applies a SIZE-AWARE value bar (see VALUE_PRICE_SENSITIVITY below) instead of a flat one;
        # the no-bet-yet (RFI/open) branch keeps the original flat-threshold behavior unchanged --
        # there's no price to be sensitive to when opening the pot.
        if facing_bet:
            style_shift = (self.current_fold_to_pressure - 0.5) * STYLE_SHIFT_SCALE
            call_bar = min(0.95, max(0.02, pot_odds + style_shift))
            value_bar = min(0.98, self.current_value_threshold.get('flop', 0.60)
                             + VALUE_PRICE_SENSITIVITY * pot_odds)
            if equity >= value_bar:
                return 'raise'
        else:
            # Top equity hands (value threshold) -- unconditional RFI/open raise, unaffected by
            # this fix (no bet to be price-sensitive to yet).
            if equity >= self.current_value_threshold.get('flop', 0.60):
                return 'raise'
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
            # [V23, BET-1] need_for_value used to be flat -- "raise regardless of price" once
            # cleared, the root cause behind hero's shove-preference (see VALUE_PRICE_SENSITIVITY's
            # docstring above for the calibration). Now rises with bet size, same mechanism as
            # continue_bar above, just extended to the top of the range instead of stopping at it.
            value_bar = min(0.98, need_for_value + VALUE_PRICE_SENSITIVITY * pot_odds)

            if equity >= value_bar:
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
# [V24] `base_bluff_perc`: see FuzzyPlayerArchetype's own docstring for the mechanism. Ordering
# chosen to match each archetype's REAL identity for "how hard is this personality to represent
# strength against" (i.e. how skeptical it is of an opponent's raise, via 1 - bot_bluff_perc) --
# deliberately NOT copied from base_bluff_freq, since that trait means something different
# (CALLING_STATION's base_bluff_freq is low because it rarely bluffs when IT acts, being passive --
# that says nothing about how skeptical it is of OTHERS. A real calling station is the HARDEST
# personality to bluff/represent strength against regardless of any signal, so it gets the
# HIGHEST bluff_perc here (lowest respect for a raise), not the lowest.
TAG = FuzzyPlayerArchetype(
    name='TAG',
    base_vpip=0.22,
    base_agg_freq=0.45,   # Bet365 AGG proxy
    base_bluff_freq=0.25,
    base_fold_to_pressure=0.60,
    base_call_threshold={'flop': 0.42, 'turn': 0.47, 'river': 0.52},
    base_value_threshold={'flop': 0.60, 'turn': 0.62, 'river': 0.65},
    base_bluff_perc=0.20,
)

LAG = FuzzyPlayerArchetype(
    name='LAG',
    base_vpip=0.32,
    base_agg_freq=0.55,
    base_bluff_freq=0.40,
    base_fold_to_pressure=0.45,
    base_call_threshold={'flop': 0.37, 'turn': 0.42, 'river': 0.49},
    base_value_threshold={'flop': 0.55, 'turn': 0.58, 'river': 0.60},
    base_bluff_perc=0.35,
)

NIT = FuzzyPlayerArchetype(
    name='Nit',
    base_vpip=0.11,
    base_agg_freq=0.25,
    base_bluff_freq=0.03,
    base_fold_to_pressure=0.85,
    base_call_threshold={'flop': 0.50, 'turn': 0.57, 'river': 0.65},
    base_value_threshold={'flop': 0.75, 'turn': 0.80, 'river': 0.85},
    base_bluff_perc=0.05,
)

CALLING_STATION = FuzzyPlayerArchetype(
    name='Calling Station',
    base_vpip=0.45,
    base_agg_freq=0.15,
    base_bluff_freq=0.05,
    base_fold_to_pressure=0.15,
    base_call_threshold={'flop': 0.32, 'turn': 0.34, 'river': 0.37},
    base_value_threshold={'flop': 0.70, 'turn': 0.72, 'river': 0.75},
    # [V24] Higher than TAG/LAG despite similar reasoning -- calibration found even a small,
    # infrequent "respect the raise" bonus captured too much of CALLING_STATION's equity range
    # (its continue_bar is already very low/sticky by design, so ANY boost that applies at all
    # swings a wide swath of otherwise-continuing equities to fold -- directly contradicting a
    # calling station's real identity as the hardest personality to fold via any signal). Pushed to
    # 0.70 (30% respect-application rate) to keep the bonus rare enough not to override that.
    base_bluff_perc=0.70,
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

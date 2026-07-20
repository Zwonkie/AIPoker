"""
Headless 6-max No-Limit Hold'em Poker Simulator for Herocules V8.
Supports up to 6 players, positions SB/BB/UTG/MP/CO/BTN, rotational dealer,
multi-way betting rounds, multi-personality league NNs, bootstrap preflop charts,
and stack size curriculum learning.
"""
import random
import sys
import os
import math
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from treys import Card, Deck, Evaluator
from core.evaluator import PokerEvaluator
from core.evaluator_cuda import CudaPokerEvaluator
from versions.v24_extreme.self_play.opponent_bots import TAG, LAG, NIT, CALLING_STATION, STYLE_SHIFT_SCALE
from versions.v24_extreme.self_play.opponents import HeuristicOpponent, build_opponent_pool
from versions.v24_extreme.core.contract import preflop_hand_strength

# [V24] "Show of strength" bonus: flat boost applied to `continue_bar` (the ACTUAL fold-vs-continue
# gate at realistic price levels -- NOT need_for_value/the value bar, which direct EV-arithmetic
# calibration found never gates fold-vs-continue except when continue_bar is independently already
# high, i.e. shove-level pot odds) in SixMaxSimulator._ev_target_fold_decision when a non-all-in
# raise "reads" as committed value (a probabilistic, per-personality event via bot_bluff_perc --
# see that function's own docstring and opponent_bots.py's FuzzyPlayerArchetype). Deliberately a
# flat, categorical bump rather than a price-continuous one (like VALUE_PRICE_SENSITIVITY) -- it's
# meant to be a "this looks like a real hand" MESSAGE effect, not another price-scaling term.
#
# [V24_EXTREME] Calibrated to 0.10 in V24 (see versions/v24/SPECS.md) -- a real, non-degenerate
# effect in isolation, but a full 150k-hand retrain showed NO movement on action_diversity/
# deep_stack_ood_guard, and the allin-vs-next-best Q-value gap WIDENED rather than narrowed at
# every tested stack depth. Pushed far above the calibrated value here (deliberately NOT
# production-safe) to test whether ANY strength of this lever can move the needle at all, before
# concluding the opponent-response-shaping approach is a structural dead end. See
# versions/v24_extreme/SPECS.md.
RAISE_RESPECT_BOOST = 0.40

# Shared evaluator instances
_treys_evaluator = Evaluator()
_poker_evaluator = PokerEvaluator()
try:
    _cuda_evaluator = CudaPokerEvaluator()
except Exception as e:
    print(f"Warning: Could not initialize CudaPokerEvaluator: {e}. Falling back to CPU.")
    _cuda_evaluator = None

class HandRecordV4:
    """Stores all decision points from a single hand for Decision Transformer training."""
    
    def __init__(self, hand_id, hero_cards, opponents_profiles):
        self.hand_id = hand_id
        self.hero_cards = list(hero_cards)
        self.opponents_profiles = opponents_profiles
        self.final_hero_profit = 0.0
        self.decision_points = []
        
    def add_decision(self, step, street, board, hero_position, pot_size, big_blind,
                     call_amount, hero_stack, active_opponents_mask, opponents_stacks,
                     action_history, equity, action_taken, chips_committed_before,
                     target_evs, opp_strength, opp_bluff_prob, opp_vpip_color=None,
                     hand_strength=0.5, opponents_committed=None, raise_count_before=0):
        """Record a single decision point snapshot."""
        self.decision_points.append({
            'opp_vpip_color': opp_vpip_color,   # tightest active opp's VPIP colour (jam-by-color telemetry)
            'step': step,
            'street': street,
            'board': list(board),
            'hero_position': hero_position,
            'pot_size': pot_size,
            'big_blind': big_blind,
            'call_amount': call_amount,
            'hero_stack': hero_stack,
            'active_opponents_mask': list(active_opponents_mask),
            'opponents_stacks': list(opponents_stacks),
            'action_history': list(action_history),
            'equity': equity,
            'action': action_taken,
            'is_all_in': False,
            'committed_before': chips_committed_before,
            'target_evs': list(target_evs),
            'opp_strength': opp_strength,
            'opp_bluff_prob': opp_bluff_prob,
            # [V20_preflopEq] field-independent card-quality signal, see SixMaxSimulator._hand_strength.
            'hand_strength': hand_strength,
            # [V22] chips each of the 5 opponent slots has ALREADY put into this hand's pot (raw
            # chips, same convention as `opponents_stacks`) -- see core/contract.py's
            # opp_committed_this_hand_bb / versions/v22/SPECS.md. `committed_before` above (hero's
            # own) is the symmetric hero-side value, already recorded since V12.
            'opponents_committed': list(opponents_committed) if opponents_committed is not None else [0.0] * 5,
            # [V23] whole-hand raise count BEFORE this decision (any street, any actor) -- source
            # for the `pot_type` context feature (limped=0/single-raised=1/3-bet+=2, bucketed in
            # contract.py). See VALUE_PRICE_SENSITIVITY's docstring in opponent_bots.py for the
            # related [BET-1] fix this version also bundles.
            'raise_count_before': raise_count_before,
        })
        
    def get_training_samples(self):
        """Convert decision points to training samples with target EV."""
        samples = []
        for dp in self.decision_points:
            samples.append({
                **dp,
                'hero_cards': self.hero_cards,
                'opponents_profiles': self.opponents_profiles
            })
        return samples

# Canonical personality -> stat-bucket slot mapping.
# Cumulative VPIP/AGG/profit/exploitation are accumulated per PERSONALITY, not per
# table seat, because the style occupying a given seat is reshuffled every hand.
# The slot order matches the fixed HUD row labels in train_selfplay.print_dashboard:
#   0 Hero(Main) | 1 Maniac | 2 Nit | 3 Sticky(fish) | 4 Past Self | 5 TAG Bot
STYLE_SLOT = {
    'main': 0, 'hero': 0,
    'maniac': 1,
    'nit': 2,
    'fish': 3,
    'past': 4,
    'tag': 5,
}

# Opponent pool composition (Fix 3): weight the table toward disciplined opponents so
# the Hero isn't training against a lineup that's half deliberately-bad players. Each
# opponent seat samples independently; disciplined archetypes (tag/past/nit) dominate,
# spew-fish (maniac/sticky) are the minority. All five still appear often enough to keep
# every personality's HUD/stat bucket populated.
OPPONENT_POOL_STYLES = ['tag', 'past', 'nit', 'maniac', 'fish']
OPPONENT_POOL_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]

# ======================================================================= #
#  V13: RANGE-AWARE EQUITY  (VPIP/tightness adaptation)
# ======================================================================= #
# The hero's INPUT equity is computed vs each opponent's VPIP-color-implied RANGE instead of
# a random hand, so the equity-primary model auto-tightens vs nits / loosens vs stations.
# We only ever observe the opponent's COLOR in live play, so the exact VPIP within a bucket is
# unknown -> we model that as noise on the range percentile (uniform over the bucket, which is
# a truncated-uniform take on "gaussian on the bucket mean"; it can never leak into another
# color). This also makes the model robust to the large sampling error of an early-session HUD.
import json

_ALL_CARD_INTS = [Card.new(r + s) for r in '23456789TJQKA' for s in 'shdc']
_CARD_STR_TO_INT = {Card.int_to_str(c): c for c in _ALL_CARD_INTS}

# VPIP color -> (low, high) fraction of starting hands the opponent plays. Red is intentionally
# WIDE and uncapped-ish so a maniac's range isn't lumped with a 36%-VPIP reg (see V13 SPECS).
_COLOR_RANGE_BAND = {
    'Blue':   (0.05, 0.15),   # nit
    'Green':  (0.15, 0.26),   # TAG
    'Yellow': (0.26, 0.40),   # LAG
    'Red':    (0.40, 0.85),   # loose / station / maniac
}

# P(opponent voluntarily plays this hand) per VPIP color — used PREFLOP to fold-weight
# yet-to-act opponents so range-aware equity doesn't assume everyone is already in with their
# range (which over-counted opponents and made preflop equity absurdly low vs a tight field).
_COLOR_TO_VPIP = {'Blue': 0.10, 'Green': 0.22, 'Yellow': 0.30, 'Red': 0.45}

def _vpip_to_color(v):
    if v < 0.18: return 'Blue'
    elif v < 0.26: return 'Green'
    elif v < 0.35: return 'Yellow'
    return 'Red'

def _sample_range_pct(color, noise=True):
    lo, hi = _COLOR_RANGE_BAND.get(color, (0.26, 0.40))
    if not noise:
        return 0.5 * (lo + hi)
    return random.uniform(lo, hi)   # truncated-uniform noise within the color bucket

_PREFLOP_RANKED = None   # list of [c1_str, c2_str], strongest preflop hand -> weakest

def _get_preflop_ranked():
    """All 1326 starting-hand combos sorted by preflop equity-vs-random (best first),
    cached to disk so it is computed once ever. 'Top p%' of this list == a p-wide range."""
    global _PREFLOP_RANKED
    if _PREFLOP_RANKED is not None:
        return _PREFLOP_RANKED
    cache = os.path.join(os.path.dirname(__file__), 'preflop_ranking.json')
    if os.path.exists(cache):
        try:
            _PREFLOP_RANKED = json.load(open(cache))
            return _PREFLOP_RANKED
        except Exception:
            pass
    combos = [(_ALL_CARD_INTS[i], _ALL_CARD_INTS[j])
              for i in range(52) for j in range(i + 1, 52)]
    SIMS = 80
    scores = {}
    for (a, b) in combos:
        hero = [a, b]
        wins = 0.0
        for _ in range(SIMS):
            rem = [c for c in _ALL_CARD_INTS if c != a and c != b]
            picks = random.sample(rem, 7)
            opp, board = picks[:2], picks[2:]
            hr = _treys_evaluator.evaluate(board, hero)
            orr = _treys_evaluator.evaluate(board, opp)
            wins += 1.0 if hr < orr else (0.5 if hr == orr else 0.0)
        scores[(a, b)] = wins / SIMS
    ranked = sorted(combos, key=lambda x: -scores[x])
    _PREFLOP_RANKED = [[Card.int_to_str(a), Card.int_to_str(b)] for (a, b) in ranked]
    try:  # atomic write so concurrent workers can't corrupt the cache
        tmp = cache + f".{os.getpid()}.tmp"
        json.dump(_PREFLOP_RANKED, open(tmp, 'w'))
        os.replace(tmp, cache)
    except Exception:
        pass
    return _PREFLOP_RANKED


def compute_range_aware_equity(hero_cards_str, board_str, opp_colors, noise=True, sims=150, front_colors=None):
    """Hero equity vs each opponent's VPIP-color range (V13). SHARED by the training simulator
    and the LIVE decision path so train and serve compute equity identically (train/serve
    consistency — see versions/v20/VALIDATED_FINDINGS.md). Returns equity in [0,1], or None if
    it cannot compute (no colors / hero not 2 cards / all samples rejected) so the caller can
    fall back to its normal vs-random equity.

    [V20_preflopEq Finding 2] `front_colors`: opponents who have ALREADY acted and committed
    this betting round -- guaranteed still in the pot, so they NEVER get the preflop VPIP
    fold-roll (regardless of street). `opp_colors` keeps its original meaning ("after": still to
    act this round) and keeps the existing roll exactly as before. Previously every active
    opponent (front and after alike) got the identical flat roll despite the simulator having
    full ground-truth action-order info available at the call site -- see SPECS.md for the
    quantified impact. `front_colors=None` (the default) reproduces the OLD behavior exactly, so
    any caller that hasn't been updated to pass it is unaffected."""
    if not opp_colors and not front_colors:
        return None
    ranked = _get_preflop_ranked()
    n = len(ranked)
    hero = [_CARD_STR_TO_INT[c] for c in hero_cards_str if c in _CARD_STR_TO_INT]
    board0 = [_CARD_STR_TO_INT[c] for c in board_str if c in _CARD_STR_TO_INT]
    if len(hero) != 2:
        return None
    dead0 = set(hero) | set(board0)
    need = 5 - len(board0)
    # PREFLOP: an "after" opponent hasn't acted, so each is only IN the pot with prob = their
    # VPIP (they fold the rest). POSTFLOP (board present): unchanged, no roll applies there
    # either way (matches pre-existing behavior). "front" opponents are NEVER rolled (see above).
    is_preflop = len(board0) == 0
    wins = 0.0
    counted = 0

    def _sample_range_hand(color, used):
        p = _sample_range_pct(color, noise)
        cutoff = max(1, int(p * n))
        for _t in range(25):
            a, b = ranked[random.randrange(cutoff)]
            ai, bi = _CARD_STR_TO_INT[a], _CARD_STR_TO_INT[b]
            if ai not in used and bi not in used:
                used.add(ai); used.add(bi)
                return [ai, bi]
        return None

    for _ in range(sims):
        used = set(dead0)
        opp_hands = []
        ok = True
        # FRONT: already acted + committed this betting round -- guaranteed in, no fold-roll.
        for color in (front_colors or []):
            hand = _sample_range_hand(color, used)
            if hand is None:
                ok = False; break
            opp_hands.append(hand)
        if not ok:
            continue
        # AFTER (legacy `opp_colors`): still to act this round -- unchanged VPIP fold-roll.
        for color in opp_colors:
            if is_preflop and random.random() >= _COLOR_TO_VPIP.get(color, 0.30):
                continue   # this opponent folds preflop -> not in the pot
            hand = _sample_range_hand(color, used)
            if hand is None:
                ok = False; break
            opp_hands.append(hand)
        if not ok:
            continue
        if not opp_hands:
            # Everyone folded this sample. SKIP it (don't count) rather than scoring a win:
            # counting all-folds as wins would bundle fold-equity into every hand equally,
            # destroying hand-strength discrimination (72o and AA both ~0.9) and making the
            # model play everything. So equity here = SHOWDOWN strength CONDITIONAL on being
            # called, over a realistic (fold-weighted) opponent count. Fold-equity itself is
            # learned separately from the training targets (sim outcomes include folds).
            continue
        if need > 0:
            remaining = [c for c in _ALL_CARD_INTS if c not in used]
            full_board = board0 + random.sample(remaining, need)
        else:
            full_board = board0
        hr = _treys_evaluator.evaluate(full_board, hero)
        best_opp = min(_treys_evaluator.evaluate(full_board, h) for h in opp_hands)
        if hr < best_opp:
            wins += 1.0
        elif hr == best_opp:
            wins += 1.0 / (len(opp_hands) + 1)
        counted += 1
    if counted == 0:
        return None
    return round(wins / counted * 100) / 100.0


class SixMaxSimulator:
    """
    Headless 6-max NLH simulator for V8.
    Hero occupies Seat 0.
    Seats 1-5 are populated with league personalities (Maniac, Nit, Calling Station, Past Self, TAG),
    whose table seats are reshuffled each hand. Cumulative stats are bucketed per personality
    (see STYLE_SLOT), so telemetry attribution is stable regardless of seating.
    """
    
    def __init__(self, bb_size=10.0, equity_sims=200, hero_personality='main',
                 hero_model=None, bootstrap_alpha=0.0):
        self.bb_size = bb_size
        self.equity_sims = equity_sims
        self.hero_personality = hero_personality
        self.hero_model = hero_model

        # [V18] Opponent seats: {style: Opponent} built by opponents.build_opponent_pool(...) and
        # assigned onto this instance post-construction (same pattern hero_model/etc. already used
        # -- see simulate_worker in train.py). Replaces the five separate `self.<style>_model`
        # attributes + hardcoded elif chain every prior version had. A style with no entry here
        # (or an empty dict, e.g. before simulate_worker populates it) has no opponent assigned;
        # `simulate_hand`'s seat-assignment loop falls back to a bare TAG heuristic in that case,
        # same "always-available default" every prior version relied on.
        self.opponent_pool = {}
        self.focus_archetype = None
        
        self.bootstrap_alpha = bootstrap_alpha
        self.hand_counter = 0

        # --- Config-driven opponent lineup (overridable per run; see config.yaml) ---
        # Which styles populate opponent seats, and their sampling weights. Defaults to the
        # full module pool; the V12-D diagnostic narrows this to a tight static field.
        self.opponent_pool_styles = list(OPPONENT_POOL_STYLES)
        self.opponent_pool_weights = list(OPPONENT_POOL_WEIGHTS)
        # Players dealt in per hand INCLUDING the Hero. 6 == full ring. When < 6, exactly
        # (6 - live_players) opponent seats are pre-folded each hand, yielding short-handed
        # pots (e.g. 3 == Hero + 2 opponents) without touching the fixed 6-seat scaffolding.
        self.live_players = 6
        # Disable the Phase-3 "extreme stacks" regime (the abrupt jump at hand 30k from the
        # 80-120bb moderate band to 10-300bb, sigma=90). When True, stacks stay in the
        # moderate band for the whole run past 10k, removing the single violent stack-depth
        # discontinuity so any residual VPIP ratchet can't be blamed on deep-stack fat tails.
        self.disable_extreme_stacks = False
        # Verify-mode knobs. fixed_stack_bb (when set) overrides ALL stack curriculum to a
        # flat depth. disable_exploration removes the Hero's 5% pure-random + heuristic-anchor
        # so the generated data reflects the model's TRUE policy (read alongside disable_bootstrap).
        self.fixed_stack_bb = None
        # V15: DoN-shaped depth MIXTURE. A list of [lo, hi, weight] bands; per hand pick a band by
        # weight then sample a uniform depth within it (short-weighted with a deep tail). Takes
        # precedence over fixed_stack_bb / the curriculum so the model sees the FULL DoN depth range.
        self.stack_depth_mix = None
        self.disable_exploration = False
        # V13: hero input equity vs opponents' VPIP-color-implied ranges (opponent adaptation).
        self.range_aware_equity = False
        self.range_equity_noise = True   # noise the range percentile within the color bucket
        # V14: discretized bet-size action space. Raise buckets as pot fractions (None = all-in).
        # Full action space = [fold, call] + one raise per fraction. num_actions MUST match the model
        # heads. Overridable from config (raise_pot_fractions). Preflop the small pot-fraction raises
        # floor to the min-raise (pot is tiny), so preflop effectively = fold/call/min-raise/all-in;
        # all-in is the meaningful short-stack action. Refining preflop BB-sizing is a later step.
        self.raise_fracs = [0.33, 0.66, 1.0, None]
        self.num_actions = 2 + len(self.raise_fracs)
        # Fallback bucket for a bare 'raise' string (heuristic/personality-forcing/legacy paths):
        # the pot-sized raise (closest to the old 0.75-pot default), else the last non-all-in bucket.
        self.default_raise_bucket = next((i for i, f in enumerate(self.raise_fracs) if f == 1.0),
                                         max(0, len(self.raise_fracs) - 2))

        # Instantiate bots for heuristics fallback
        import copy
        self.nit_heuristic = copy.deepcopy(NIT)
        self.fish_heuristic = copy.deepcopy(CALLING_STATION)  # used for Sticky Caller
        self.maniac_heuristic = copy.deepcopy(LAG)
        self.tag_heuristic = copy.deepcopy(TAG)
        
        # Dynamic Opponent Profiling keyed by PERSONALITY SLOT (see STYLE_SLOT), not table seat.
        # Track VPIP (voluntary preflop entry) and AGG (postflop bet/raise ratio) per personality
        # so stats stay attributed correctly even though seats are reshuffled every hand.
        self.seat_histories = {s: {
            'vpip_ops': 0, 'vpip_acts': 0,
            'agg_ops': 0, 'agg_acts': 0,
            'profit': 0.0,
            'raises': 0, 'folds': 0, 'all_ins': 0
        } for s in range(6)}
        self.global_metrics = {'flop_players': 0, 'flop_count': 0}
        # Exploitation matrix is also indexed by personality slot on both axes.
        self.global_exploitation_net = {i: {j: 0.0 for j in range(6)} for i in range(6)}
        # Surfaced (not swallowed) model-query error counter (P1). A swallowed KeyError
        # once disabled the NN for entire V11 runs; here the first few are printed loudly
        # with a traceback so a broken inference path can never hide behind a heuristic.
        self._query_error_count = 0

    def _note_query_error(self, where, exc):
        """Surface a model-query failure instead of silently falling back to heuristics."""
        self._query_error_count += 1
        if self._query_error_count <= 5:
            import traceback
            print(f"WARNING: model query failed in {where} "
                  f"(#{self._query_error_count}): {exc!r}")
            traceback.print_exc()

    def _get_starting_stack(self, current_hand):
        """Curriculum stack sizing logic based on hand count."""
        # V15 DoN-shaped MIXTURE: [[lo,hi,weight], ...] -> weighted band pick, then uniform within.
        # Highest precedence (spans short-to-deep so ONE model covers all DoN depths).
        if self.stack_depth_mix:
            bands = self.stack_depth_mix
            lo, hi, _w = random.choices(bands, weights=[b[2] for b in bands], k=1)[0]
            return round(random.uniform(lo, hi)) * self.bb_size
        # fixed_stack_bb: a scalar = flat depth (removes stack as a variable), OR a [lo, hi] pair =
        # V14 SHORT-STACK / tournament (DoN) mode -> sample a short/medium effective depth each hand
        # (e.g. [5, 14]) so the model learns push/fold across the band. Either overrides curriculum.
        if self.fixed_stack_bb is not None:
            fs = self.fixed_stack_bb
            if isinstance(fs, (list, tuple)):
                return round(random.uniform(fs[0], fs[1])) * self.bb_size
            return fs * self.bb_size
        # Phase 3 (extreme stacks) can be disabled for diagnostics: past 30k the run then
        # stays in the Phase-2 moderate band instead of jumping to the 10-300bb regime.
        extreme = (current_hand >= 30000) and not self.disable_extreme_stacks
        if current_hand < 10000:
            std_dev = 0.0
        elif not extreme:
            std_dev = 10.0
        else:
            std_dev = 90.0

        if std_dev == 0.0:
            return 100.0 * self.bb_size

        stack_bb = random.gauss(100.0, std_dev)
        if not extreme:
            stack_bb = max(80.0, min(120.0, stack_bb))
        else:
            stack_bb = max(10.0, min(300.0, stack_bb))
            
        return round(stack_bb) * self.bb_size
        
    def _calculate_equity(self, hero_cards_str, board_str, num_opponents, specific_opponents=None):
        """Calculate Hero's equity using MC simulations."""
        if _cuda_evaluator is not None and specific_opponents is None:
            try:
                eq = _cuda_evaluator.calculate_equity_batched(
                    [hero_cards_str], 
                    [board_str], 
                    num_opponents=num_opponents, 
                    num_simulations=self.equity_sims
                )[0]
                return round(eq * 100) / 100.0
            except Exception:
                pass # Fallback to CPU

        eq, _ = _poker_evaluator.calculate_equity(
            board_str, hero_cards_str,
            num_opponents=num_opponents,
            num_simulations=self.equity_sims,
            specific_opponents=specific_opponents
        )
        return round(eq * 100) / 100.0

    def _calculate_range_aware_equity(self, hero_cards_str, board_str, opp_colors, sims=None, front_colors=None):
        """V13: hero equity vs each active opponent's VPIP-color-implied RANGE (not random).
        Each MC iteration samples every opponent a hand from its top-p% range (p noised within
        its color bucket), deals the rest of the board, and scores the showdown. Falls back to
        the standard vs-random equity if there are no opponents / it can't sample.

        [V20_preflopEq Finding 2] `front_colors`: opponents already acted+committed this round
        (guaranteed in, no fold-roll) -- see compute_range_aware_equity's docstring."""
        eq = compute_range_aware_equity(
            hero_cards_str, board_str, opp_colors,
            noise=self.range_equity_noise, sims=sims or 150, front_colors=front_colors,
        )
        if eq is None:   # empty colors / bad hero / all rejected -> vs-random fallback
            return self._calculate_equity(hero_cards_str, board_str,
                                           len(opp_colors or []) + len(front_colors or []))
        return eq

    def _hand_strength(self, cards_str, board_str):
        """[V20_preflopEq] Field-independent card-quality signal, decoupled from equity's
        opponent/field modeling -- "how good is my hand on its own merits" vs "how good given
        this field" (classic hand-strength-vs-potential framing). Preflop: O(1) lookup into
        preflop_equities.csv's 169-hand vs-1-random equity table (10k sims/hand, precomputed).
        Postflop: a cheap live vs-1-random MC call (200 sims -- no color/range modeling, much
        cheaper than the range-aware equity call already made for the main `equity` feature).
        See versions/v20_preflopEq/SPECS.md."""
        if not board_str:
            return preflop_hand_strength(cards_str[0], cards_str[1])
        eq, _ = _poker_evaluator.calculate_equity(board_str, cards_str, num_opponents=1, num_simulations=200)
        return round(eq * 100) / 100.0

    def _query_model_decide(self, model, hand_cards, equity, pot_size, call_amount, hero_stack, num_opponents, table_state_dict=None, model_state_history=None, hero_actions_history=None, sample=True):
        """Decide an action from the model's ACTOR (policy) head.

        V12: the action is drawn from the policy distribution `softmax(policy_logits)`
        (sampled during self-play for exploration; argmax when `sample=False` for
        deterministic eval). This replaces the V11 `argmax(q_vals)`, which let one
        over-estimated Q head capture every decision (raise-/call-everything collapse).
        Falls back to Q-argmax only for a legacy checkpoint with no policy head.
        """
        from core.board_state import BoardState, SeatState, HUDStats
        from versions.v24_extreme.core.contract import ContractV12

        board_cards = table_state_dict.get('board', []) if table_state_dict else []
        street_idx = table_state_dict.get('street', 0) if table_state_dict else 0
        opp_profiles = table_state_dict.get('opponents_profiles', {}) if table_state_dict else {}
        street_map = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River"}
        street_str = street_map.get(street_idx, "Preflop")

        # V19 [hero_position fix]: BoardState.hero_position previously always defaulted to 0
        # (Button) here -- it was never set for ANY query, hero's own turn or any opponent's.
        # `table_state_dict` now carries the current actor's own seat + the button seat (set
        # once per actor's turn in simulate_hand), so every query -- hero's and each opponent
        # NN's -- gets ITS OWN real button-relative position, matching the live-serve path
        # (core/table_state.py's `to_board_state`, the only other call site that ever set this).
        actor_seat = table_state_dict.get('actor_seat', 0) if table_state_dict else 0
        button_seat = table_state_dict.get('button_seat', 0) if table_state_dict else 0
        actor_position = (actor_seat - button_seat) % 6

        # [V20_preflopEq] hand_strength was already computed once (alongside `equity`) at the
        # betting-loop call site and stashed on table_state_dict -- reused here rather than
        # recomputed, same pattern `equity` itself already uses. Defaults to neutral (0.5) for
        # any call site that hasn't been updated to set it (e.g. a diagnostic script).
        hand_strength = table_state_dict.get('hand_strength', 0.5) if table_state_dict else 0.5

        # [V22] Full-table committed-this-hand array (absolute seat 0-5, hero==0), stashed on
        # table_state_dict by the betting-loop call site (see simulate_hand). `actor_seat` above
        # already identifies WHICH seat this query is for -- reuse it so hero_committed reflects
        # the querying actor's own committed amount (mirrors how `hero_stack`/`hand_cards` already
        # mean "this actor's own", not literally seat 0), same pattern as `hand_strength` above.
        committed_list = table_state_dict.get('committed') if table_state_dict else None
        hero_committed_val = (committed_list[actor_seat]
                              if committed_list and actor_seat < len(committed_list) else 0.0)

        # [V23] whole-hand raise count so far, bucketed into pot_type (0=limped/unraised,
        # 1=single-raised, 2=3-bet+) -- same for every actor's query (a hand-level property, not
        # actor-relative like committed above).
        raise_count = table_state_dict.get('raise_count', 0) if table_state_dict else 0
        pot_type_val = min(2, raise_count)

        board_state = BoardState(
            community_cards=board_cards,
            hero_cards=hand_cards,
            pot_size=pot_size,
            hero_stack=hero_stack,
            street=street_str,
            big_blind=self.bb_size,
            call_amount=call_amount,
            equity=equity,
            hero_position=actor_position,
            hand_strength=hand_strength,
            hero_committed=hero_committed_val,
            pot_type=pot_type_val,
        )
        for idx in range(5):
            seat_key = f"seat_{idx+1}"
            is_active = (idx < num_opponents)

            vpip_col = "Blue"
            agg_col = "Blue"

            if is_active and seat_key in opp_profiles:
                # Use the acting hand's per-seat personality profile (correct VPIP/AGG floats).
                prof = opp_profiles[seat_key]
                v_val = prof.get('vpip', 0.30)
                a_val = prof.get('agg', 0.40)

                if v_val >= 0.35: vpip_col = "Red"
                elif v_val >= 0.26: vpip_col = "Yellow"
                elif v_val >= 0.18: vpip_col = "Green"

                if a_val >= 0.71: agg_col = "Red"
                elif a_val >= 0.56: agg_col = "Yellow"
                elif a_val >= 0.36: agg_col = "Green"

            # [V22] committed uses the REAL per-seat value (committed_list[idx+1], absolute seat
            # index) rather than the `stack=hero_stack` placeholder above (a pre-existing
            # approximation this change doesn't touch) -- committed[] is already a full,
            # correctly-indexed table array, no approximation needed.
            opp_committed_val = (committed_list[idx + 1]
                                 if is_active and committed_list and (idx + 1) < len(committed_list) else 0.0)
            board_state.seats[seat_key] = SeatState(
                name=f"Opponent {idx+1}",
                is_active=is_active,
                stack=hero_stack if is_active else 0.0,
                hud=HUDStats(
                    vpip_color=vpip_col,
                    agg_color=agg_col
                ),
                committed=opp_committed_val,
            )
            
        if model_state_history is not None:
            model_state_history.append(board_state)
            states_to_pass = model_state_history
        else:
            states_to_pass = [board_state]
            
        bridge = ContractV12()
        h_t, b_t, c_t, a_t = bridge.to_tensors(states_to_pass, hero_actions=hero_actions_history)
        
        device = model.device if hasattr(model, 'device') else next(model.parameters()).device
        
        with torch.no_grad():
            preds = model(h_t.to(device), b_t.to(device), c_t.to(device), a_t.to(device))

        # V14: [fold, call, raise_0..raise_{K-1}] where raise_k uses self.raise_fracs[k] (None=all-in).
        actions = ['fold', 'call'] + [f'raise_{k}' for k in range(len(self.raise_fracs))]

        # V12 ACTOR path: choose from the policy distribution.
        if isinstance(preds, dict) and 'policy_logits' in preds:
            logits = preds['policy_logits'].squeeze(0)[-1]
            # policy_temperature<1 sharpens toward the mode. Training rollouts keep the
            # default 1.0 (on-policy diversity); eval/live set 0.5 to MATCH the deployed
            # serve config in core/decision.py (train/serve consistency).
            _temp = getattr(self, 'policy_temperature', 1.0)
            probs = torch.softmax(logits / _temp, dim=-1)
            # When checking is free (no bet to call) folding is never correct: zero its
            # mass and renormalize so the model never folds a free option.
            if call_amount == 0:
                probs = probs.clone()
                probs[0] = 0.0
                total = probs.sum()
                probs = probs / total if total > 0 else torch.tensor([0.0, 1.0, 0.0], device=probs.device)
            if sample:
                idx = int(torch.multinomial(probs, 1).item())
            else:
                idx = int(torch.argmax(probs).item())
            return actions[idx]

        # Legacy fallback (checkpoint without a policy head): argmax over the critic Q.
        q_vals = preds['q_vals'].squeeze(0)[-1] if isinstance(preds, dict) else preds.squeeze(0)[-1]
        available_evs = {'fold': q_vals[0].item(), 'call': q_vals[1].item(), 'raise': q_vals[2].item()}
        if call_amount == 0:
            available_evs['fold'] = -9999.0
        return max(available_evs, key=available_evs.get)

    def _calculate_mc_target_evs(self, hero_cards, pot, to_call, hero_stack, street_idx, active_opponents, board_str):
        """Monte Carlo Target EV evaluation using exact opponent profile simulations and True Equity."""
        
        # Calculate True Equity using the actual hole cards of the opponents
        opp_hands = [opp['cards'] for opp in active_opponents]
        if opp_hands:
            true_equity = self._calculate_equity(hero_cards, board_str, len(active_opponents), specific_opponents=opp_hands)
        else:
            true_equity = 1.0 # If no opponents left, equity is 100%
            
        ev_fold = 0.0
        
        # Call EV calculation
        pot_after_call = pot + to_call
        ev_call = true_equity * pot_after_call - to_call
        
        # Raise EV calculation
        raise_size = min(pot * 0.75, hero_stack)
        raise_size = max(raise_size, to_call + self.bb_size)
        raise_size = min(raise_size, hero_stack)
        
        raise_increment = raise_size - to_call
        new_pot = pot + raise_size + raise_increment
        pot_odds = raise_increment / max(1.0, new_pot)
        
        fold_probs = []
        for opp in active_opponents:
            bot = opp['bot']
            opp_stack = opp['stack']
            opp_cards = opp['cards']
            opp_equity = self._calculate_equity(opp_cards, board_str, 1)
            
            f_count = 0
            for _ in range(10):
                if street_idx == 0:
                    d = bot.decide_preflop(opp_equity, pot_odds)
                else:
                    d = bot.decide_postflop(opp_equity, pot_odds, new_pot, opp_stack, street_idx)
                if d == 'fold':
                    f_count += 1
            fold_probs.append(f_count / 10.0)
            
        p_all_fold = 1.0
        max_opp_equity = 0.0
        for p, opp_eq in zip(fold_probs, [self._calculate_equity(opp['cards'], board_str, 1) for opp in active_opponents]):
            p_all_fold *= p
            max_opp_equity = max(max_opp_equity, opp_eq)
            
        opp_bluff_prob = 1.0 if max_opp_equity < 0.33 and len(active_opponents) > 0 else 0.0
            
        # Showdown EV if called
        ev_raise_if_called = true_equity * (pot + 2.0 * raise_size - to_call) - raise_size
        ev_raise = p_all_fold * pot + (1.0 - p_all_fold) * ev_raise_if_called

        return [ev_fold, ev_call, ev_raise], max_opp_equity, opp_bluff_prob

    def _raise_size_for_fraction(self, frac, pot, to_call, hero_stack):
        """Chips the hero commits for a raise of `frac`×pot (frac None -> ALL-IN), clamped to a
        legal min-raise (to_call + bb) and the hero stack. Shared by the sized EV target and (later)
        the betting loop so train/serve sizing is identical."""
        if frac is None:
            return hero_stack                          # all-in
        rs = min(pot * frac, hero_stack)
        rs = max(rs, to_call + self.bb_size)           # at least a min-raise
        return min(rs, hero_stack)

    def _ev_target_fold_decision(self, bot, equity, pot_odds, street_idx, is_allin):
        """[V24] DECOUPLED fold model used ONLY for `_mc_target_evs_sized`'s per-size EV target --
        deliberately independent of `opponent_bots.py`'s LIVE `decide_preflop`/`decide_postflop`
        (which carry the [BET-1] `VALUE_PRICE_SENSITIVITY` fix for live self-play only).

        Root cause this decoupling fixes (see versions/v23/SPECS.md, versions/v24/SPECS.md): V23
        called the live, price-sensitive decide_* functions directly here too. Making bots fold
        MORE to oversized bets doesn't just describe more realistic live play -- it ALSO
        mechanically inflates hero's own ALLIN training target, since `p_all_fold * pot` is
        credited straight into that size's counterfactual EV. The two effects (opponents demand
        more to continue vs. hero gets more fold-equity credit for shoving) point in opposite
        directions for the shove-preference goal, and the wrong one won (action_diversity/
        deep_stack_ood_guard regressed). This function reverts the VALUE branch to the PRE-BET-1
        flat `need_for_value` for the target computation specifically, while keeping the original
        P1b `continue_bar` price-sensitivity (validated, predates BET-1, not the regression's
        cause) -- live opponents still play with the BET-1 fix, hero's target just doesn't inherit
        its inflation.

        [V24] "Show of strength": for non-all-in raises, with probability `1.0 -
        bot.current_bluff_perc`, the opponent "respects" the raise as committed value and folds
        MORE than raw price alone would justify. Implemented as a boost to `continue_bar`
        specifically (NOT `need_for_value`/the value bar) -- direct EV-arithmetic calibration
        found `need_for_value` never actually gates the fold-vs-continue decision at realistic
        raise-pot price levels (po~0.5); `continue_bar` (price + style_shift) is the ONLY thing
        that determines fold vs continue there, `need_for_value` only matters far up the equity
        range where continue_bar is independently already high (i.e. shove-level pot odds). A
        "show of strength" bonus meant to matter at NORMAL raise sizes has to move the actual
        fold-gate, not a threshold that's moot until the price is already huge. All-in gets NONE
        of this bonus -- priced honestly on raw pot odds. Directly targets the no-middle-gear
        problem: fold-equity today scales monotonically with size (bigger bet -> more folds),
        which makes all-in dominate by construction; a raise-only, non-price-scaling fold-equity
        source breaks that monotonicity. `bot_bluff_perc` makes this personality-conditioned (a bot
        that bluffs a lot itself respects raises less), giving hero a genuine, learnable reason to
        condition on WHICH opponent is raising, not just a static color tag.

        Returns True (fold) / False (continue) -- only the binary outcome matters for
        `_mc_target_evs_sized`'s `p_all_fold` sampling, not the raise-vs-call split live play cares
        about.

        Non-FuzzyPlayerArchetype bots (e.g. `TieredLookupBot`, used by model_verify's
        `beats_offformula_stress` check) don't carry `current_value_threshold`/
        `current_fold_to_pressure`/`current_bluff_perc` at all -- fall back to calling that bot's
        OWN `decide_preflop`/`decide_postflop` directly. This is safe (not the V23 self-coupling
        problem) because only `FuzzyPlayerArchetype.decide_*` carries the BET-1 price-sensitivity
        fix this function exists to decouple from; a bot without these attributes was never part
        of that fix in the first place."""
        if not hasattr(bot, 'current_value_threshold'):
            d = (bot.decide_preflop(equity, pot_odds) if street_idx == 0
                 else bot.decide_postflop(equity, pot_odds, 0.0, 0.0, street_idx))
            return d == 'fold'

        street_map = {0: 'flop', 1: 'flop', 2: 'turn', 3: 'river'}
        street_str = street_map.get(street_idx, 'river')
        need_for_value = bot.current_value_threshold.get(street_str, 0.7)
        style_shift = (bot.current_fold_to_pressure - 0.5) * STYLE_SHIFT_SCALE
        continue_bar = min(0.95, max(0.02, pot_odds + style_shift))

        if (not is_allin) and random.random() < (1.0 - bot.current_bluff_perc):
            continue_bar = min(0.95, continue_bar + RAISE_RESPECT_BOOST)

        if equity >= need_for_value:
            return False   # continues (value)
        if equity >= continue_bar:
            return False   # continues (marginal)
        # Below the bar -> mostly fold, rare bluff-raise (still "not fold" for this purpose).
        if random.random() < bot.current_bluff_freq and random.random() < bot.current_agg_freq * 1.5:
            return False
        return True   # fold

    def _mc_target_evs_sized(self, hero_cards, pot, to_call, hero_stack, street_idx,
                             active_opponents, board_str, raise_fracs, range_aware_eq=None,
                             last_raiser=-1):
        """V14 P1a — PER-SIZE counterfactual EV target: EV of [fold, call, raise(frac_0), ...] for
        every raise size in `raise_fracs` (None = all-in). Each size's opponent fold-out is sampled
        via `_ev_target_fold_decision` -- a DECOUPLED fold model (see its own docstring), not the
        live `bot.decide_*` functions, since V23 found sharing them directly inflates hero's own
        ALLIN target whenever live opponent behavior gets more price-sensitive (see
        versions/v23/SPECS.md, versions/v24/SPECS.md). Still size-aware (a bigger bet earns more
        folds, P1b), plus [V24]'s raise-only "show of strength" bonus. Computed for EVERY size at
        EVERY decision regardless of what was played -> the counterfactual signal that lets the
        hero learn WHICH size to use (and to value overbet/all-in) without having to stumble into
        it. See SPECS.md P1/P1a.

        V16 [P4]: `true_equity` normally comes from the opponents' LITERAL dealt cards (oracle,
        via `specific_opponents=opp_hands`) -- correct postflop, where a still-active opponent's
        real cards already carry a true selection skew toward their style (they survived their
        own decide_postflop this hand). But at the PREFLOP entry decision (street_idx==0), no
        opponent has acted yet, so that oracle equity is statistically independent of opponent
        style -- a nit and a maniac holding the same real cards score identically, which is why
        the preflop CALL/FOLD target carried no tightness signal even though RAISE already did
        (via `p_all_fold` below). Fix: at street_idx==0, use the caller-supplied `range_aware_eq`
        (hero equity vs each opponent's VPIP-implied continuing RANGE, already computed once for
        the input features -- see `_calculate_range_aware_equity`) instead. Postflop is untouched.

        [2026-07-17 fix] `opp_bluff_prob`: previously `max_opp_equity < 0.33` -- ANY active
        opponent holding weak cards, regardless of whether anyone had actually acted aggressively.
        That fires just as often on a hand where a weak opponent folds immediately as one where
        they genuinely bluff, so it was really measuring "is someone at the table weak" (largely
        redundant with `opp_strength`), not "is my opponent bluffing me right now". Now gated on
        `last_raiser`: only meaningful when a specific OPPONENT (not hero, not nobody) is the last
        aggressor this street, and reads THAT opponent's own equity, not the field's weakest. See
        OFK known-shortcomings-backlog / versions/v21_auxhead/SPECS.md.
        """
        opp_hands = [opp['cards'] for opp in active_opponents]
        oracle_equity = (self._calculate_equity(hero_cards, board_str, len(active_opponents),
                                              specific_opponents=opp_hands) if opp_hands else 1.0)
        true_equity = range_aware_eq if (street_idx == 0 and range_aware_eq is not None) else oracle_equity
        opp_base_eq = [self._calculate_equity(opp['cards'], board_str, 1) for opp in active_opponents]
        max_opp_equity = max(opp_base_eq) if opp_base_eq else 0.0

        evs = [0.0, true_equity * (pot + to_call) - to_call]   # fold, call
        for frac in raise_fracs:
            raise_size = self._raise_size_for_fraction(frac, pot, to_call, hero_stack)
            raise_increment = raise_size - to_call
            new_pot = pot + raise_size + raise_increment
            size_pot_odds = raise_increment / max(1.0, new_pot)   # price the opponent faces (rises with size)
            is_allin = (frac is None)
            p_all_fold = 1.0
            for opp, oeq in zip(active_opponents, opp_base_eq):
                bot = opp['bot']
                f = 0
                for _ in range(10):
                    # [V24] Decoupled from live decide_preflop/decide_postflop -- see
                    # _ev_target_fold_decision's own docstring for why (V23's regression).
                    if self._ev_target_fold_decision(bot, oeq, size_pot_odds, street_idx, is_allin):
                        f += 1
                p_all_fold *= f / 10.0
            ev_if_called = true_equity * (pot + 2.0 * raise_size - to_call) - raise_size
            evs.append(p_all_fold * pot + (1.0 - p_all_fold) * ev_if_called)

        opp_bluff_prob = 0.0
        if last_raiser > 0:
            for opp, oeq in zip(active_opponents, opp_base_eq):
                if opp.get('seat') == last_raiser:
                    opp_bluff_prob = 1.0 if oeq < 0.33 else 0.0
                    break
        return evs, max_opp_equity, opp_bluff_prob

    def _hero_decide(self, equity, pot_size, call_amount, hero_stack, num_opponents, 
                     is_preflop, hand_cards=None, table_state_dict=None, model_state_history=None, hero_actions_history=None):
        """Decision logic for Hero (the active learning model) with hybrid exploration split."""
        # Verify mode: no epsilon-random and no heuristic anchor -> data is 100% the model's
        # own policy (pair with disable_bootstrap so alpha=0 makes model_share fully engage).
        eps = 0.0 if self.disable_exploration else 0.05
        # [V21] was 0.80 (leaving a permanent 15% steady-state heuristic-anchor floor past
        # bootstrap); now sums to 1.0 with eps, removing that floor. The heuristic chart is still
        # reached on a model-query EXCEPTION (see the `except` branch below) -- this only removes
        # its ROUTINE use. See SPECS.md item 2.
        model_share = 1.0 if self.disable_exploration else 0.95
        # 1. Pure Random Exploration to prevent off-policy data gaps
        roll = random.random()
        if roll < eps:
            # V14: sample across ALL raise sizes incl. all-in so exploration visits overbet/shove
            # states (the counterfactual targets already SCORE every size; this drives visitation).
            raise_opts = [f'raise_{k}' for k in range(len(self.raise_fracs))]
            if equity > 0.70:
                return random.choice(['call'] + raise_opts)
            return random.choice(['fold', 'call'] + raise_opts)

        # 2. Dynamic model vs heuristic split
        # Early: 90% Heuristic
        # RL Takeover: 80% Active Model, 10% Heuristic Anchor (100% model in verify mode)
        model_prob = (1.0 - self.bootstrap_alpha) * model_share

        if roll < eps + model_prob and self.hero_model is not None:
            try:
                decision = self._query_model_decide(self.hero_model, hand_cards, equity, pot_size, call_amount, hero_stack, num_opponents, table_state_dict, model_state_history, hero_actions_history)
                
                # Action Forcing for Hero Personality Training
                hero_vpip = self.seat_histories[0]['vpip_acts'] / max(1, self.seat_histories[0]['vpip_ops'])
                hero_agg = self.seat_histories[0]['agg_acts'] / max(1, self.seat_histories[0]['agg_ops'])
                
                if self.hero_personality == 'maniac':
                    if hero_agg < 0.60 and random.random() < 0.50:
                        return 'raise'
                    if hero_vpip < 0.65 and is_preflop and random.random() < 0.50:
                        return random.choice(['call', 'raise'])
                elif self.hero_personality == 'nit':
                    if hero_vpip > 0.15 and is_preflop and random.random() < 0.80:
                        return 'fold'
                elif self.hero_personality == 'sticky':
                    if hero_vpip < 0.50 and is_preflop and random.random() < 0.60:
                        return 'call'
                    if hero_agg > 0.20 and random.random() < 0.80 and decision.startswith('raise'):
                        return 'call'
                        
                return decision
            except Exception as e:
                self._note_query_error("_hero_decide", e)

        # 3. Heuristic Chart Anchor fallback
        if is_preflop:
            if self.hero_personality == 'maniac':
                return self.maniac_heuristic.decide_preflop(equity, call_amount / max(1, pot_size))
            elif self.hero_personality == 'nit':
                return self.nit_heuristic.decide_preflop(equity, call_amount / max(1, pot_size))
            elif self.hero_personality == 'sticky':
                return self.fish_heuristic.decide_preflop(equity, call_amount / max(1, pot_size))
            else:
                return self.tag_heuristic.decide_preflop(equity, call_amount / max(1, pot_size))
        else:
            pot_odds = call_amount / (pot_size + call_amount) if (pot_size + call_amount) > 0 else 0.0
            street_idx = table_state_dict.get('street', 1) if table_state_dict else 1
            if self.hero_personality == 'maniac':
                return self.maniac_heuristic.decide_postflop(equity, pot_odds, pot_size, hero_stack, street_idx)
            elif self.hero_personality == 'nit':
                return self.nit_heuristic.decide_postflop(equity, pot_odds, pot_size, hero_stack, street_idx)
            elif self.hero_personality == 'sticky':
                return self.fish_heuristic.decide_postflop(equity, pot_odds, pot_size, hero_stack, street_idx)
            else:
                return self.tag_heuristic.decide_postflop(equity, pot_odds, pot_size, hero_stack, street_idx)

    def _opponent_decide(self, seat_idx, opponent, equity, pot_odds, pot_size, stack, street_idx, table_state_dict=None, model_state_history=None, hero_actions_history=None):
        """Decision logic for Seats 1 to 5. [V18] Delegates to `opponent['agent']` (an
        `opponents.Opponent` -- HeuristicOpponent or NNOpponent), uniformly for every style. This
        method now only owns what's genuinely SIMULATOR-level: the 5% exploration mix, the
        bootstrap heuristic-anchor gate (preflop only, matching every prior version), and reading
        `self.seat_histories` to feed the agent's own `apply_forcing_*` (a no-op unless the agent
        was built with `forced=True`, see opponents.py)."""
        agent = opponent['agent']

        # 1. 5% Random Exploration for Opponent Bots (unchanged, applies uniformly regardless of
        # whether the seat is heuristic- or NN-driven).
        if random.random() < 0.05:
            if equity > 0.70:
                decision = random.choice(['call', 'raise'])
            else:
                decision = random.choice(['fold', 'call', 'raise'])
            if street_idx == 0:
                agent.recording_bot.record_preflop(decision)
            else:
                agent.recording_bot.record_postflop(decision)
            return decision

        is_preflop = (street_idx == 0)

        if is_preflop:
            # Bootstrap heuristic-anchoring gates PREFLOP only (matches every prior version --
            # postflop always queries the model if one loaded, no bootstrap gate there either).
            roll = random.random()
            force_heuristic = roll < self.bootstrap_alpha
            decision = agent.decide_preflop(
                equity, pot_odds, pot_size, stack, opponent['num_opps'], opponent['cards'],
                table_state_dict, model_state_history, hero_actions_history,
                force_heuristic=force_heuristic)

            if roll >= self.bootstrap_alpha:
                slot = STYLE_SLOT.get(agent.style, seat_idx)
                opp_vpip = self.seat_histories[slot]['vpip_acts'] / max(1, self.seat_histories[slot]['vpip_ops'])
                opp_agg = self.seat_histories[slot]['agg_acts'] / max(1, self.seat_histories[slot]['agg_ops'])
                decision = agent.apply_forcing_preflop(decision, opp_vpip, opp_agg)

            agent.recording_bot.record_preflop(decision)
        else:
            decision = agent.decide_postflop(
                equity, pot_odds, pot_size, stack, street_idx, opponent['num_opps'], opponent['cards'],
                table_state_dict, model_state_history, hero_actions_history, force_heuristic=False)

            slot = STYLE_SLOT.get(agent.style, seat_idx)
            opp_vpip = self.seat_histories[slot]['vpip_acts'] / max(1, self.seat_histories[slot]['vpip_ops'])
            opp_agg = self.seat_histories[slot]['agg_acts'] / max(1, self.seat_histories[slot]['agg_ops'])
            decision = agent.apply_forcing_postflop(decision, opp_vpip, opp_agg)

            agent.recording_bot.record_postflop(decision)

        return decision

    def simulate_hand(self, current_hand=0):
        """Simulate a single 6-Max NLH hand using V8 specifications."""
        self.hand_counter += 1
        
        # Fuzz the heuristic bots
        self.nit_heuristic.start_new_hand()
        self.fish_heuristic.start_new_hand()
        self.maniac_heuristic.start_new_hand()
        self.tag_heuristic.start_new_hand()
        
        # 1. Initialize Seats
        button_seat = random.randint(0, 5)
        
        # Dynamic curriculum stacks
        starting_stack_chips = self._get_starting_stack(current_hand)
        stacks = [starting_stack_chips for _ in range(6)]
        
        active = [True] * 6
        committed = [0.0] * 6
        # [V23] whole-hand raise counter -- "how many raises have occurred in this hand so far"
        # (any street, any actor), used to derive the `pot_type` context feature (limped/
        # single-raised/3-bet+). Reset once per hand alongside `committed`. See core/contract.py.
        raise_count = 0
        folded = [False] * 6
        model_state_histories = {s: [] for s in range(6)}
        hero_actions_histories = {s: [] for s in range(6)}
        
        # Live-player cap (diagnostic): deal in only Hero + (live_players-1) opponents.
        # Takes precedence over the curriculum's random pre-fold so the field size is a
        # controlled variable. Pre-folded seats never act (folded=True) and are masked out
        # of the model's opponent context, exactly like the curriculum pre-fold below.
        if 0 < self.live_players < 6:
            num_live_opps = self.live_players - 1
            live_opp_seats = set(random.sample(range(1, 6), num_live_opps))
            for s in range(1, 6):
                if s not in live_opp_seats:
                    active[s] = False
                    folded[s] = True
        # Phase 4: Dynamic Active Players (> 40,000 hands [V21] was 50,000)
        elif current_hand > 40000:
            # [V21] reweighted toward 3-5-handed starting tables (was [0.40,0.25,0.20,0.10,0.05]):
            # heavily biases away from full 6-max and heads-up as STARTING conditions (both still
            # occur via ordinary in-hand folding) so the range-aware equity call sees realistic
            # multiway-but-not-full fields more often. See SPECS.md item 3.
            num_to_fold = random.choices([0, 1, 2, 3, 4], weights=[0.10, 0.25, 0.30, 0.25, 0.10], k=1)[0]
            if num_to_fold > 0:
                fold_seats = random.sample(range(1, 6), num_to_fold)
                for s in fold_seats:
                    active[s] = False
                    folded[s] = True
        
        # Cards dealing
        deck = Deck()
        hands_ints = [deck.draw(2) for _ in range(6)]
        hands_str = [[Card.int_to_str(c) for c in hand] for hand in hands_ints]
        
        # Phase 5 Focus Archetype Populator
        focus_seats = []
        if self.focus_archetype is not None:
            num_focus = random.choice([3, 4])
            focus_seats = random.sample(range(1, 6), num_focus)
            
        # Assign each opponent seat a personality from the configured pool. Only the LIVE
        # opponent seats (post pre-fold) get a real draw; when the number of live opponents
        # exactly matches the pool size we assign a random permutation so every pool style is
        # present exactly once (e.g. a 3-handed table always fields one Nit + one TAG, with
        # which seat gets which randomized to avoid positional overfitting). Otherwise each
        # live seat samples independently by weight. Folded seats get a harmless placeholder.
        pool = self.opponent_pool_styles
        weights = self.opponent_pool_weights
        live_opp_seats = [s for s in range(1, 6) if not folded[s]]
        base_style_by_seat = {}
        if len(live_opp_seats) == len(pool):
            perm = random.sample(pool, len(pool))
            for seat, st in zip(live_opp_seats, perm):
                base_style_by_seat[seat] = st
        else:
            for seat in live_opp_seats:
                base_style_by_seat[seat] = random.choices(pool, weights=weights, k=1)[0]
        for s in range(1, 6):
            base_style_by_seat.setdefault(s, pool[0])

        # Map each table seat to its personality stat-bucket slot for this hand.
        # Seat 0 is always the Hero (slot 0); seats 1-5 depend on the shuffled/focus style.
        seat_slot = {0: STYLE_SLOT['hero']}

        # Compute VPIP / AGG from cumulative per-personality historic events
        opponents = []
        opponents_profiles = {}
        for s in range(1, 6):
            style = 'tag'
            if s in focus_seats:
                style = self.focus_archetype
            else:
                style = base_style_by_seat[s]

            slot = STYLE_SLOT[style]
            seat_slot[s] = slot

            # Pull this personality's accumulated tendencies for the model's opponent context.
            v_ops = self.seat_histories[slot]['vpip_ops']
            v_acts = self.seat_histories[slot]['vpip_acts']
            a_ops = self.seat_histories[slot]['agg_ops']
            a_acts = self.seat_histories[slot]['agg_acts']

            v_val = v_acts / v_ops if v_ops > 0 else 0.30
            a_val = a_acts / a_ops if a_ops > 0 else 0.40

            # [V18] Uniform pool lookup -- replaces the old style->model `elif` chain (the exact
            # shape that let a stray leftover line silently nullify v17_gauntlet's `tag` seat, see
            # versions/v17_gauntlet/SPECS.md "CORRECTION"). `self.opponent_pool` is built once per
            # worker by opponents.build_opponent_pool(...); a style missing from it (shouldn't
            # happen once config is wired, but no worse than every prior version's own fallback)
            # gets a bare TAG heuristic so a hand can never crash on a lookup miss.
            agent = self.opponent_pool.get(style) or HeuristicOpponent(style, self.tag_heuristic, forced=True)

            opponents.append({
                'seat': s,
                'agent': agent,
                'style': style,
                'cards': hands_str[s],
                'num_opps': 5
            })
            opponents_profiles[f"seat_{s}"] = {
                'vpip': v_val,
                'agg': a_val,
                'style': style
            }
            
        record = HandRecordV4(self.hand_counter, hands_str[0], opponents_profiles)
        
        # Blinds
        sb_seat = (button_seat + 1) % 6
        bb_seat = (button_seat + 2) % 6
        
        sb_amt = self.bb_size * 0.5
        bb_amt = self.bb_size
        
        committed[sb_seat] = min(sb_amt, stacks[sb_seat])
        stacks[sb_seat] -= committed[sb_seat]
        
        committed[bb_seat] = min(bb_amt, stacks[bb_seat])
        stacks[bb_seat] -= committed[bb_seat]
        
        pot = sum(committed)
        action_history = []
        board_cards_int = []
        board_str = []
        
        hero_position = (0 - button_seat) % 6
        
        streets = [
            ('preflop', 0, 0),
            ('flop', 1, 3),
            ('turn', 2, 1),
            ('river', 3, 1),
        ]
        
        step_counter = 0
        
        # VPIP Tracking variables
        vpip_this_hand = [False] * 6
        preflop_had_decision = [False] * 6
        
        for street_name, street_idx, num_cards in streets:
            active_count = sum(1 for i in range(6) if not folded[i])
            
            if street_idx == 1:
                self.global_metrics['flop_players'] += active_count
                self.global_metrics['flop_count'] += 1

            if active_count <= 1:
                break
                
            players_with_stacks = sum(1 for i in range(6) if not folded[i] and stacks[i] > 0)
            if players_with_stacks <= 1 and street_idx > 0:
                if num_cards > 0:
                    board_cards_int.extend(deck.draw(num_cards))
                continue
                
            if num_cards > 0:
                new_cards = deck.draw(num_cards)
                board_cards_int.extend(new_cards)
                board_str = [Card.int_to_str(c) for c in board_cards_int]
                
            street_committed = [0.0] * 6
            if street_idx == 0:
                street_committed[sb_seat] = committed[sb_seat]
                street_committed[bb_seat] = committed[bb_seat]

            # [V20_preflopEq Finding 2] Ground-truth "has this seat acted (and matched the
            # current bet) THIS betting round" tracking -- reset fresh each street, and again
            # whenever a raise reopens action (see the raise branches below). Distinct from
            # `street_committed[s] == highest_bet`, which would misclassify the blinds as
            # "already acted" before anyone has acted at all. Feeds the hero's front/after
            # opponent split for range-aware equity (front = guaranteed in, no fold-roll).
            acted_this_round = [False] * 6
                
            if street_idx == 0:
                current_actor = (button_seat + 3) % 6
                highest_bet = bb_amt
            else:
                current_actor = (button_seat + 1) % 6
                highest_bet = 0.0
                
            last_raiser = -1
            betting_ended = False
            first_round = True
            
            while not betting_ended:
                if not folded[current_actor] and stacks[current_actor] > 0:
                    to_call = highest_bet - street_committed[current_actor]
                    
                    if to_call == 0.0 and not first_round and last_raiser == -1:
                        break
                        
                    active_opps_count = sum(1 for i in range(6) if i != current_actor and not folded[i])
                    if self.range_aware_equity and current_actor == 0:
                        # Hero sees equity vs each active opponent's VPIP-color range (V13).
                        # [V20_preflopEq Finding 2] Split by whether the opponent has ALREADY
                        # acted+committed this betting round (front -- guaranteed in, no VPIP
                        # fold-roll) vs still to act (after -- unchanged roll). An all-in seat
                        # (stack==0) can never act again so it's always "front" too. Previously
                        # every active opponent got the identical roll regardless of this simulator's
                        # own (correct) ground-truth action order -- see SPECS.md.
                        front_colors = [
                            _vpip_to_color(opponents_profiles.get(f"seat_{s}", {}).get('vpip', 0.3))
                            for s in range(1, 6)
                            if not folded[s] and (stacks[s] == 0 or acted_this_round[s])
                        ]
                        after_colors = [
                            _vpip_to_color(opponents_profiles.get(f"seat_{s}", {}).get('vpip', 0.3))
                            for s in range(1, 6)
                            if not folded[s] and stacks[s] > 0 and not acted_this_round[s]
                        ]
                        eq = self._calculate_range_aware_equity(hands_str[0], board_str, after_colors,
                                                                 front_colors=front_colors)
                    else:
                        eq = self._calculate_equity(hands_str[current_actor], board_str, active_opps_count)

                    # [V20_preflopEq] Field-independent hand-quality signal for whichever seat is
                    # currently deciding -- computed once here (mirrors `eq`'s own pattern) and
                    # reused for both this decision's live query AND (hero only) the training record.
                    hs = self._hand_strength(hands_str[current_actor], board_str)

                    table_state = {
                        "board": board_str,
                        "street": street_idx,
                        "action_history": action_history,
                        "opponents_profiles": opponents_profiles,
                        # V19 [hero_position fix]: the CURRENT actor's own seat + the button seat,
                        # so _query_model_decide can compute THIS actor's real button-relative
                        # position for every query (hero's own turn AND every opponent NN query --
                        # both funnel through the same table_state_dict-carrying call chain).
                        # Previously absent -> BoardState.hero_position silently defaulted to 0
                        # (Button) for every single training-time query, hero and opponent alike.
                        "actor_seat": current_actor,
                        "button_seat": button_seat,
                        "hand_strength": hs,
                        # [V22] full-table committed-this-hand array (absolute seat index 0-5,
                        # hero==0), so _query_model_decide can read the CURRENT actor's own
                        # (hero_committed) and each opponent slot's (opp_committed) values -- same
                        # seat_key/idx mapping already used there for stack/vpip/agg.
                        "committed": list(committed),
                        # [V23] whole-hand raise count so far (any street, any actor) -- source for
                        # the `pot_type` context feature. See core/contract.py.
                        "raise_count": raise_count,
                    }
                    
                    if current_actor == 0:  # Hero
                        preflop_had_decision[0] = True
                        active_mask = [0] * 5
                        opp_stacks = [0.0] * 5
                        active_opps_list = []
                        for s in range(1, 6):
                            if not folded[s]:
                                active_mask[s - 1] = 1
                                opp_stacks[s - 1] = stacks[s]
                                active_opps_list.append({
                                    # [V18] .recording_bot: every Opponent (heuristic or NN) carries
                                    # a heuristic archetype bot for exactly this kind of fold-
                                    # probability ESTIMATE (target-EV computation, not a real
                                    # decision) -- equivalent to the old 'bot' key.
                                    'bot': [o for o in opponents if o['seat'] == s][0]['agent'].recording_bot,
                                    'stack': stacks[s],
                                    'cards': hands_str[s],
                                    'seat': s,   # [2026-07-17] needed to match against last_raiser
                                                 # for the opp_bluff_prob fix -- see _mc_target_evs_sized
                                })
                                
                        target_evs, opp_strength, opp_bluff_prob = self._mc_target_evs_sized(
                            hero_cards=hands_str[0], pot=pot, to_call=to_call, hero_stack=stacks[0],
                            street_idx=street_idx, active_opponents=active_opps_list, board_str=board_str,
                            raise_fracs=self.raise_fracs,
                            # V16 [P4]: reuse the range-aware `eq` already computed just above for
                            # the input features (zero extra MC cost) as the preflop CALL/FOLD
                            # target basis. Only valid when range_aware_equity is actually on --
                            # otherwise `eq` is the plain vs-random fallback, not style-aware.
                            range_aware_eq=(eq if self.range_aware_equity else None),
                            last_raiser=last_raiser,
                        )
                        # Tightest active opponent's VPIP colour -> jam-by-color adaptation telemetry.
                        _tv = min((o['bot'].base_vpip for o in active_opps_list), default=None)
                        opp_color = _vpip_to_color(_tv) if _tv is not None else None

                        record.add_decision(
                            opp_vpip_color=opp_color,
                            step=step_counter,
                            street=street_idx,
                            board=board_str,
                            hero_position=hero_position,
                            pot_size=pot,
                            big_blind=self.bb_size,
                            call_amount=to_call,
                            hero_stack=stacks[0],
                            active_opponents_mask=active_mask,
                            opponents_stacks=opp_stacks,
                            action_history=action_history,
                            equity=eq,
                            action_taken=-1,
                            chips_committed_before=committed[0],
                            target_evs=target_evs,
                            opp_strength=opp_strength,
                            opp_bluff_prob=opp_bluff_prob,
                            hand_strength=hs,
                            opponents_committed=committed[1:6],
                            raise_count_before=raise_count,
                        )

                        decision = self._hero_decide(
                            eq, pot, to_call, stacks[0], active_opps_count,
                            (street_idx == 0), hand_cards=hands_str[0],
                            table_state_dict=table_state,
                            model_state_history=model_state_histories[0],
                            hero_actions_history=hero_actions_histories[0]
                        )
                            
                        if decision == 'fold':
                            action_idx = 0
                            folded[0] = True
                            self.seat_histories[0]['folds'] += 1
                            action_history.append('f')
                            hero_actions_histories[0].append(7)
                        elif decision == 'call':
                            vpip_this_hand[0] = True
                            action_idx = 1
                            call_amt = min(to_call, stacks[0])
                            stacks[0] -= call_amt
                            if stacks[0] == 0:
                                self.seat_histories[0]['all_ins'] += 1
                            committed[0] += call_amt
                            street_committed[0] += call_amt
                            pot += call_amt
                            action_history.append('c')
                            hero_actions_histories[0].append(3)
                        else:  # raise (decision = 'raise_k' from model/eps, or bare 'raise' legacy)
                            vpip_this_hand[0] = True
                            if decision.startswith('raise_'):
                                k = min(int(decision.split('_', 1)[1]), len(self.raise_fracs) - 1)
                            else:
                                k = self.default_raise_bucket
                            action_idx = 2 + k
                            self.seat_histories[0]['raises'] += 1
                            # V14: size from the chosen bucket (None=all-in), min-raise floored, stack capped.
                            raise_size = self._raise_size_for_fraction(self.raise_fracs[k], pot, to_call, stacks[0])

                            stacks[0] -= raise_size
                            if stacks[0] == 0:
                                self.seat_histories[0]['all_ins'] += 1
                            committed[0] += raise_size
                            street_committed[0] += raise_size
                            pot += raise_size

                            highest_bet = street_committed[0]
                            last_raiser = 0
                            raise_count += 1
                            action_history.append('r')
                            hero_actions_histories[0].append(6)
                            # [V20_preflopEq] A raise reopens action: every other still-live seat
                            # must respond to the new bet level again, so they're no longer "front".
                            for s in range(6):
                                if s != 0:
                                    acted_this_round[s] = False

                        # Track Hero AGG
                        if street_idx > 0:
                            self.seat_histories[0]['agg_ops'] += 1
                            if decision.startswith('raise'):
                                self.seat_histories[0]['agg_acts'] += 1

                        acted_this_round[0] = True
                        record.decision_points[-1]['action'] = action_idx
                        record.decision_points[-1]['is_all_in'] = (stacks[0] == 0)
                        step_counter += 1
                        
                    else:  # Opponent bot
                        preflop_had_decision[current_actor] = True
                        cur_slot = seat_slot[current_actor]  # personality stat bucket for this seat
                        opp_bot_struct = [o for o in opponents if o['seat'] == current_actor][0]
                        opp_bot_struct['num_opps'] = active_opps_count
                        
                        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
                        decision = self._opponent_decide(current_actor, opp_bot_struct, eq, pot_odds, pot, stacks[current_actor], street_idx, table_state, model_state_history=model_state_histories[current_actor], hero_actions_history=hero_actions_histories[current_actor])
                        
                        if decision == 'fold':
                            folded[current_actor] = True
                            self.seat_histories[cur_slot]['folds'] += 1
                            hero_actions_histories[current_actor].append(7)
                        elif decision == 'call':
                            vpip_this_hand[current_actor] = True
                            call_amt = min(to_call, stacks[current_actor])
                            stacks[current_actor] -= call_amt
                            if stacks[current_actor] == 0:
                                self.seat_histories[cur_slot]['all_ins'] += 1
                            committed[current_actor] += call_amt
                            street_committed[current_actor] += call_amt
                            pot += call_amt
                            hero_actions_histories[current_actor].append(3)
                        else:  # raise
                            vpip_this_hand[current_actor] = True
                            self.seat_histories[cur_slot]['raises'] += 1
                            raise_size = min(pot * 0.75, stacks[current_actor])
                            raise_size = max(raise_size, to_call + self.bb_size)
                            raise_size = min(raise_size, stacks[current_actor])

                            stacks[current_actor] -= raise_size
                            if stacks[current_actor] == 0:
                                self.seat_histories[cur_slot]['all_ins'] += 1
                            committed[current_actor] += raise_size
                            street_committed[current_actor] += raise_size
                            pot += raise_size
                            
                            highest_bet = street_committed[current_actor]
                            last_raiser = current_actor
                            raise_count += 1
                            hero_actions_histories[current_actor].append(6)
                            # [V20_preflopEq] A raise reopens action for everyone else still live.
                            for s in range(6):
                                if s != current_actor:
                                    acted_this_round[s] = False

                        # Track Opponent AGG. Exact '== raise' misses sized-model bucket strings
                        # ('raise_0'..'raise_3') -- a real NN opponent (e.g. the frozen 'past'
                        # seat) would show near-zero AGG despite raising constantly (the 'raises'
                        # counter above uses a catch-all else so it's unaffected -- that mismatch,
                        # large raise count vs ~0 AGG, is what exposed this). Match the hero's own
                        # tracking (line ~1114), which already uses startswith('raise').
                        if street_idx > 0:
                            self.seat_histories[cur_slot]['agg_ops'] += 1
                            if decision.startswith('raise'):
                                self.seat_histories[cur_slot]['agg_acts'] += 1

                        acted_this_round[current_actor] = True
                
                current_actor = (current_actor + 1) % 6
                
                active_players = [i for i in range(6) if not folded[i]]
                if len(active_players) <= 1:
                    betting_ended = True
                    break
                    
                all_matched = True
                for p in active_players:
                    if stacks[p] > 0 and street_committed[p] != highest_bet:
                        all_matched = False
                        break
                        
                if all_matched and (last_raiser == -1 or current_actor == last_raiser):
                    betting_ended = True
                    
                first_round = False
                
        # Showdown & Profit Allocation
        active_players = [i for i in range(6) if not folded[i]]
        win_shares = [0.0] * 6
        
        if len(active_players) == 1:
            winner = active_players[0]
            win_shares[winner] = pot
        else:
            scores = []
            board_ints = list(board_cards_int)
            while len(board_ints) < 5:
                c = deck.draw(1)
                board_ints.extend(c)
            for p in active_players:
                score = _treys_evaluator.evaluate(board_ints, hands_ints[p])
                scores.append((p, score))
                
            # Side pot resolution
            eligible_players = list(active_players)
            unique_commits = sorted(list(set([committed[p] for p in eligible_players if committed[p] > 0])))
            
            previous_commit = 0.0
            for current_commit in unique_commits:
                slice_amount = current_commit - previous_commit
                if slice_amount <= 0:
                    continue
                    
                slice_pot = 0.0
                for p in range(6):
                    if committed[p] > previous_commit:
                        contribution = min(committed[p] - previous_commit, slice_amount)
                        slice_pot += contribution
                
                slice_eligible = [p for p in eligible_players if committed[p] >= current_commit]
                
                if slice_pot > 0 and slice_eligible:
                    best_score = min(score for p, score in scores if p in slice_eligible)
                    winners = [p for p, score in scores if p in slice_eligible and score == best_score]
                    share = slice_pot / len(winners)
                    for w in winners:
                        win_shares[w] += share
                        
                previous_commit = current_commit
                
            total_distributed = sum(win_shares)
            leftover = pot - total_distributed
            if leftover > 1e-5:
                highest_active_commit = max([committed[p] for p in active_players])
                highest_bettors = [p for p in active_players if committed[p] == highest_active_commit]
                for hb in highest_bettors:
                    win_shares[hb] += leftover / len(highest_bettors)
                
        for p in range(6):
            slot = seat_slot[p]
            if preflop_had_decision[p]:
                self.seat_histories[slot]['vpip_ops'] += 1
                if vpip_this_hand[p]:
                    self.seat_histories[slot]['vpip_acts'] += 1
            self.seat_histories[slot]['profit'] += win_shares[p] - committed[p]

        # Calculate pairwise exchange for Exploitation Scoreboard (indexed by personality slot)
        net_profits = [win_shares[p] - committed[p] for p in range(6)]
        total_gains = sum(np for np in net_profits if np > 0)

        if total_gains > 0:
            for p_lose in range(6):
                if net_profits[p_lose] < 0:
                    for p_win in range(6):
                        if net_profits[p_win] > 0:
                            transfer = abs(net_profits[p_lose]) * (net_profits[p_win] / total_gains)
                            win_slot, lose_slot = seat_slot[p_win], seat_slot[p_lose]
                            self.global_exploitation_net[win_slot][lose_slot] += transfer
                            self.global_exploitation_net[lose_slot][win_slot] -= transfer
                                
        record.final_hero_profit = win_shares[0] - committed[0]
        return record

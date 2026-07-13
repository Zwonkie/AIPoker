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
from versions.v12d.self_play.opponent_bots import TAG, LAG, NIT, CALLING_STATION

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
                     target_evs, opp_strength, opp_bluff_prob):
        """Record a single decision point snapshot."""
        self.decision_points.append({
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
            'opp_bluff_prob': opp_bluff_prob
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


class SixMaxSimulator:
    """
    Headless 6-max NLH simulator for V8.
    Hero occupies Seat 0.
    Seats 1-5 are populated with league personalities (Maniac, Nit, Calling Station, Past Self, TAG),
    whose table seats are reshuffled each hand. Cumulative stats are bucketed per personality
    (see STYLE_SLOT), so telemetry attribution is stable regardless of seating.
    """
    
    def __init__(self, bb_size=10.0, equity_sims=200, hero_personality='main',
                 hero_model=None, maniac_model=None, nit_model=None, sticky_model=None, past_model=None,
                 bootstrap_alpha=0.0):
        self.bb_size = bb_size
        self.equity_sims = equity_sims
        self.hero_personality = hero_personality
        self.hero_model = hero_model
        
        # Opponent personality NNs
        self.maniac_model = maniac_model
        self.nit_model = nit_model
        self.sticky_model = sticky_model
        self.past_model = past_model
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

    def _query_model_decide(self, model, hand_cards, equity, pot_size, call_amount, hero_stack, num_opponents, table_state_dict=None, model_state_history=None, hero_actions_history=None, sample=True):
        """Decide an action from the model's ACTOR (policy) head.

        V12: the action is drawn from the policy distribution `softmax(policy_logits)`
        (sampled during self-play for exploration; argmax when `sample=False` for
        deterministic eval). This replaces the V11 `argmax(q_vals)`, which let one
        over-estimated Q head capture every decision (raise-/call-everything collapse).
        Falls back to Q-argmax only for a legacy checkpoint with no policy head.
        """
        from core.board_state import BoardState, SeatState, HUDStats
        from versions.v12d.core.contract import ContractV12
        
        board_cards = table_state_dict.get('board', []) if table_state_dict else []
        street_idx = table_state_dict.get('street', 0) if table_state_dict else 0
        opp_profiles = table_state_dict.get('opponents_profiles', {}) if table_state_dict else {}
        street_map = {0: "Preflop", 1: "Flop", 2: "Turn", 3: "River"}
        street_str = street_map.get(street_idx, "Preflop")
        
        board_state = BoardState(
            community_cards=board_cards,
            hero_cards=hand_cards,
            pot_size=pot_size,
            hero_stack=hero_stack,
            street=street_str,
            big_blind=self.bb_size,
            call_amount=call_amount,
            equity=equity
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
            
            board_state.seats[seat_key] = SeatState(
                name=f"Opponent {idx+1}",
                is_active=is_active,
                stack=hero_stack if is_active else 0.0,
                hud=HUDStats(
                    vpip_color=vpip_col,
                    agg_color=agg_col
                )
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

        actions = ['fold', 'call', 'raise']

        # V12 ACTOR path: choose from the policy distribution.
        if isinstance(preds, dict) and 'policy_logits' in preds:
            logits = preds['policy_logits'].squeeze(0)[-1]
            probs = torch.softmax(logits, dim=-1)
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

    def _hero_decide(self, equity, pot_size, call_amount, hero_stack, num_opponents, 
                     is_preflop, hand_cards=None, table_state_dict=None, model_state_history=None, hero_actions_history=None):
        """Decision logic for Hero (the active learning model) with hybrid exploration split."""
        # 1. 5% Pure Random Exploration to prevent off-policy data gaps
        roll = random.random()
        if roll < 0.05:
            if equity > 0.70:
                return random.choice(['call', 'raise'])
            return random.choice(['fold', 'call', 'raise'])
            
        # 2. Dynamic model vs heuristic split
        # Early: 90% Heuristic
        # RL Takeover: 80% Active Model, 10% Heuristic Anchor
        model_prob = (1.0 - self.bootstrap_alpha) * 0.80
        
        if roll < 0.05 + model_prob and self.hero_model is not None:
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
                    if hero_agg > 0.20 and random.random() < 0.80 and decision == 'raise':
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
        """Decision logic for Seats 1 to 5, querying personality NNs or heuristics."""
        # 1. 5% Random Exploration for Opponent Bots
        if random.random() < 0.05:
            if equity > 0.70:
                decision = random.choice(['call', 'raise'])
            else:
                decision = random.choice(['fold', 'call', 'raise'])
            if street_idx == 0:
                opponent['bot'].record_preflop(decision)
            else:
                opponent['bot'].record_postflop(decision)
            return decision

        is_preflop = (street_idx == 0)
        
        # Map seat index to corresponding NN model and heuristic fallback
        model = opponent.get('model')
        heuristic_bot = opponent.get('bot', self.tag_heuristic)
        
        if is_preflop:
            roll = random.random()
            if roll < self.bootstrap_alpha or model is None:
                decision = heuristic_bot.decide_preflop(equity, pot_odds)
            else:
                try:
                    decision = self._query_model_decide(model, opponent['cards'], equity, pot_size, pot_odds * pot_size, stack, opponent['num_opps'], table_state_dict, model_state_history, hero_actions_history)
                except Exception as e:
                    self._note_query_error("_opponent_decide/preflop", e)
                    decision = heuristic_bot.decide_preflop(equity, pot_odds)
            
            # Action Forcing for Opponent personalities, NN or heuristic
            # (reads this personality's own accumulated stats).
            if roll >= self.bootstrap_alpha:
                style = opponent.get('style')
                slot = STYLE_SLOT.get(style, seat_idx)
                opp_vpip = self.seat_histories[slot]['vpip_acts'] / max(1, self.seat_histories[slot]['vpip_ops'])
                opp_agg = self.seat_histories[slot]['agg_acts'] / max(1, self.seat_histories[slot]['agg_ops'])

                if style == 'maniac':
                    if opp_agg < 0.60 and random.random() < 0.50: decision = 'raise'
                    if opp_vpip < 0.65 and random.random() < 0.50: decision = random.choice(['call', 'raise'])
                elif style == 'nit':
                    if opp_vpip > 0.15 and random.random() < 0.80: decision = 'fold'
                elif style == 'fish':
                    if opp_vpip < 0.50 and random.random() < 0.60: decision = 'call'
                    if opp_agg > 0.20 and random.random() < 0.80 and decision == 'raise': decision = 'call'
            
            # Record VPIP stats on temporary bot profiles for HUD mapping
            opponent['bot'].record_preflop(decision)
        else:
            if model is None:
                decision = heuristic_bot.decide_postflop(equity, pot_odds, pot_size, stack, street_idx)
            else:
                try:
                    decision = self._query_model_decide(model, opponent['cards'], equity, pot_size, pot_odds * pot_size, stack, opponent['num_opps'], table_state_dict, model_state_history, hero_actions_history)
                except Exception as e:
                    self._note_query_error("_opponent_decide/postflop", e)
                    decision = heuristic_bot.decide_postflop(equity, pot_odds, pot_size, stack, street_idx)
            
            # Action Forcing for Opponent personalities, NN or heuristic
            # (reads this personality's own accumulated stats).
            style = opponent.get('style')
            slot = STYLE_SLOT.get(style, seat_idx)
            opp_vpip = self.seat_histories[slot]['vpip_acts'] / max(1, self.seat_histories[slot]['vpip_ops'])
            opp_agg = self.seat_histories[slot]['agg_acts'] / max(1, self.seat_histories[slot]['agg_ops'])

            if style == 'maniac':
                if opp_agg < 0.60 and random.random() < 0.50: decision = 'raise'
            elif style == 'fish':
                if opp_agg > 0.20 and random.random() < 0.80 and decision == 'raise': decision = 'call'
            
            opponent['bot'].record_postflop(decision)
            
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
        # Phase 4: Dynamic Active Players (> 50,000 hands)
        elif current_hand > 50000:
            num_to_fold = random.choices([0, 1, 2, 3, 4], weights=[0.40, 0.25, 0.20, 0.10, 0.05], k=1)[0]
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

            # Temporary opponent bots to track training dashboard VPIP/AGG averages
            if style == 'maniac':
                opp_bot = self.maniac_heuristic
                opp_model = self.maniac_model
            elif style == 'nit':
                opp_bot = self.nit_heuristic
                opp_model = self.nit_model
            elif style == 'fish':
                opp_bot = self.fish_heuristic
                opp_model = self.sticky_model
            elif style == 'past':
                # "Past Self": frozen former hero checkpoint (falls back to TAG heuristic).
                opp_bot = self.tag_heuristic
                opp_model = self.past_model
            else:  # 'tag' -> static TAG reference bot (pure heuristic, no NN)
                opp_bot = self.tag_heuristic
                opp_model = None

            opponents.append({
                'seat': s,
                'bot': opp_bot,
                'model': opp_model,
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
                    eq = self._calculate_equity(hands_str[current_actor], board_str, active_opps_count)
                    
                    table_state = {
                        "board": board_str,
                        "street": street_idx,
                        "action_history": action_history,
                        "opponents_profiles": opponents_profiles
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
                                    'bot': [o for o in opponents if o['seat'] == s][0]['bot'],
                                    'stack': stacks[s],
                                    'cards': hands_str[s]
                                })
                                
                        target_evs, opp_strength, opp_bluff_prob = self._calculate_mc_target_evs(
                            hero_cards=hands_str[0], pot=pot, to_call=to_call, hero_stack=stacks[0],
                            street_idx=street_idx, active_opponents=active_opps_list, board_str=board_str
                        )
                        
                        record.add_decision(
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
                            opp_bluff_prob=opp_bluff_prob
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
                        else:  # raise
                            vpip_this_hand[0] = True
                            action_idx = 2
                            self.seat_histories[0]['raises'] += 1
                            raise_size = min(pot * 0.75, stacks[0])
                            raise_size = max(raise_size, to_call + self.bb_size)
                            raise_size = min(raise_size, stacks[0])
                            
                            stacks[0] -= raise_size
                            if stacks[0] == 0:
                                self.seat_histories[0]['all_ins'] += 1
                            committed[0] += raise_size
                            street_committed[0] += raise_size
                            pot += raise_size
                            
                            highest_bet = street_committed[0]
                            last_raiser = 0
                            action_history.append('r')
                            hero_actions_histories[0].append(6)
                            
                        # Track Hero AGG
                        if street_idx > 0:
                            self.seat_histories[0]['agg_ops'] += 1
                            if decision == 'raise':
                                self.seat_histories[0]['agg_acts'] += 1
                            
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
                            hero_actions_histories[current_actor].append(6)
                        
                        # Track Opponent AGG
                        if street_idx > 0:
                            self.seat_histories[cur_slot]['agg_ops'] += 1
                            if decision == 'raise':
                                self.seat_histories[cur_slot]['agg_acts'] += 1
                
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

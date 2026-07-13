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
from tools.self_play.v11.opponent_bots_v11 import TAG, LAG, NIT, CALLING_STATION

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

class SixMaxSimulator:
    """
    Headless 6-max NLH simulator for V8.
    Hero occupies Seat 0.
    Seats 1-5 are populated with league personalities (Maniac, Nit, Calling Station, Past Self, TAG).
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
        
        # Instantiate bots for heuristics fallback
        import copy
        self.nit_heuristic = copy.deepcopy(NIT)
        self.fish_heuristic = copy.deepcopy(CALLING_STATION)  # used for Sticky Caller
        self.maniac_heuristic = copy.deepcopy(LAG)
        self.tag_heuristic = copy.deepcopy(TAG)
        
        # V9 Dynamic Opponent Profiling (last 50 hands history)
        # Track VPIP (voluntary preflop entry) and AGG (postflop bet/raise ratio)
        self.seat_histories = {s: {
            'vpip': [], 'agg': [], 'profit': 0.0,
            'raises': 0, 'folds': 0, 'all_ins': 0
        } for s in range(6)}
        self.global_metrics = {'flop_players': 0, 'flop_count': 0}
        self.global_exploitation_net = {i: {j: 0.0 for j in range(6)} for i in range(6)}

    def _get_starting_stack(self, current_hand):
        """Curriculum stack sizing logic based on hand count."""
        if current_hand < 10000:
            std_dev = 0.0
        elif current_hand < 30000:
            std_dev = 10.0
        else:
            std_dev = 90.0
            
        if std_dev == 0.0:
            return 100.0 * self.bb_size
            
        stack_bb = random.gauss(100.0, std_dev)
        if current_hand < 30000:
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

    def _query_model_decide(self, model, hand_cards, equity, pot_size, call_amount, hero_stack, num_opponents, table_state_dict=None):
        """Decide action using a specified neural network model."""
        from core.board_state import BoardState, SeatState, HUDStats
        from core.bridge.v11.contract_v11 import ContractV8V9 as ContractV11
        
        board_cards = table_state_dict.get('board', []) if table_state_dict else []
        street_idx = table_state_dict.get('street', 0) if table_state_dict else 0
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
            
            if is_active and hasattr(self, 'seat_histories') and (idx+1) in self.seat_histories:
                v_hist = self.seat_histories[idx+1]['vpip']
                a_hist = self.seat_histories[idx+1]['agg']
                v_val = sum(v_hist) / len(v_hist) if len(v_hist) > 0 else 0.30
                a_val = sum(a_hist) / len(a_hist) if len(a_hist) > 0 else 0.40
                
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
            
        bridge = ContractV11()
        h_t, b_t, c_t, a_t = bridge.to_tensors(board_state)
        
        device = model.device if hasattr(model, 'device') else next(model.parameters()).device
        
        with torch.no_grad():
            preds = model(h_t.to(device), b_t.to(device), c_t.to(device), a_t.to(device))
            if isinstance(preds, dict):
                q_vals = preds['q_vals'].squeeze(0)[-1]
            else:
                q_vals = preds.squeeze(0)[-1]
            
        ev_fold = q_vals[0].item()
        ev_call = q_vals[1].item()
        ev_raise = q_vals[2].item()
        
        available_evs = {'fold': ev_fold, 'call': ev_call, 'raise': ev_raise}
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
                     is_preflop, hand_cards=None, table_state_dict=None):
        """Decision logic for Hero (the active learning model) with hybrid exploration split."""
        # 1. 5% Pure Random Exploration to prevent off-policy data gaps
        roll = random.random()
        if roll < 0.05:
            return random.choice(['fold', 'call', 'raise'])
            
        # 2. Dynamic model vs heuristic split
        # Early: 90% Heuristic
        # RL Takeover: 80% Active Model, 10% Heuristic Anchor
        model_prob = (1.0 - self.bootstrap_alpha) * 0.80
        
        if roll < 0.05 + model_prob and self.hero_model is not None:
            try:
                decision = self._query_model_decide(self.hero_model, hand_cards, equity, pot_size, call_amount, hero_stack, num_opponents, table_state_dict)
                
                # Action Forcing for Hero Personality Training
                hero_vpip = sum(self.seat_histories[0]['vpip']) / max(1, len(self.seat_histories[0]['vpip']))
                hero_agg = sum(self.seat_histories[0]['agg']) / max(1, len(self.seat_histories[0]['agg']))
                
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
            except Exception:
                pass
                
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

    def _opponent_decide(self, seat_idx, opponent, equity, pot_odds, pot_size, stack, street_idx, table_state_dict=None):
        """Decision logic for Seats 1 to 5, querying personality NNs or heuristics."""
        # 1. 5% Random Exploration for Opponent Bots
        if random.random() < 0.05:
            decision = random.choice(['fold', 'call', 'raise'])
            if street_idx == 0:
                opponent['bot'].record_preflop(decision)
            else:
                opponent['bot'].record_postflop(decision)
            return decision

        is_preflop = (street_idx == 0)
        
        # Map seat index to corresponding NN model and heuristic fallback
        model = None
        heuristic_bot = self.tag_heuristic
        
        if seat_idx == 1:
            model = self.maniac_model
            heuristic_bot = self.maniac_heuristic
        elif seat_idx == 2:
            model = self.nit_model
            heuristic_bot = self.nit_heuristic
        elif seat_idx == 3:
            model = self.sticky_model
            heuristic_bot = self.fish_heuristic
        elif seat_idx == 4:
            model = self.past_model
            heuristic_bot = self.tag_heuristic
        
        if is_preflop:
            roll = random.random()
            if roll < self.bootstrap_alpha or model is None:
                decision = heuristic_bot.decide_preflop(equity, pot_odds)
            else:
                try:
                    decision = self._query_model_decide(model, opponent['cards'], equity, pot_size, pot_odds * pot_size, stack, opponent['num_opps'], table_state_dict)
                except Exception:
                    decision = heuristic_bot.decide_preflop(equity, pot_odds)
            
            # Action Forcing for Opponent NNs
            if roll >= self.bootstrap_alpha and model is not None:
                opp_vpip = sum(self.seat_histories[seat_idx]['vpip']) / max(1, len(self.seat_histories[seat_idx]['vpip']))
                opp_agg = sum(self.seat_histories[seat_idx]['agg']) / max(1, len(self.seat_histories[seat_idx]['agg']))
                
                if seat_idx == 1:  # maniac
                    if opp_agg < 0.60 and random.random() < 0.50: decision = 'raise'
                    if opp_vpip < 0.65 and random.random() < 0.50: decision = random.choice(['call', 'raise'])
                elif seat_idx == 2:  # nit
                    if opp_vpip > 0.15 and random.random() < 0.80: decision = 'fold'
                elif seat_idx == 3:  # sticky
                    if opp_vpip < 0.50 and random.random() < 0.60: decision = 'call'
                    if opp_agg > 0.20 and random.random() < 0.80 and decision == 'raise': decision = 'call'
            
            # Record VPIP stats on temporary bot profiles for HUD mapping
            opponent['bot'].record_preflop(decision)
        else:
            if model is None:
                decision = heuristic_bot.decide_postflop(equity, pot_odds, pot_size, stack, street_idx)
            else:
                try:
                    decision = self._query_model_decide(model, opponent['cards'], equity, pot_size, pot_odds * pot_size, stack, opponent['num_opps'], table_state_dict)
                except Exception:
                    decision = heuristic_bot.decide_postflop(equity, pot_odds, pot_size, stack, street_idx)
            
            # Action Forcing for Opponent NNs
            if model is not None:
                opp_vpip = sum(self.seat_histories[seat_idx]['vpip']) / max(1, len(self.seat_histories[seat_idx]['vpip']))
                opp_agg = sum(self.seat_histories[seat_idx]['agg']) / max(1, len(self.seat_histories[seat_idx]['agg']))
                
                if seat_idx == 1:  # maniac
                    if opp_agg < 0.60 and random.random() < 0.50: decision = 'raise'
                elif seat_idx == 3:  # sticky
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
        
        # Phase 4: Dynamic Active Players (> 50,000 hands)
        if current_hand > 50000:
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
            
        # Compute VPIP / AGG from the last 50 hands running history
        opponents = []
        opponents_profiles = {}
        for s in range(1, 6):
            v_hist = self.seat_histories[s]['vpip']
            a_hist = self.seat_histories[s]['agg']
            
            v_val = sum(v_hist) / len(v_hist) if len(v_hist) > 0 else 0.30
            a_val = sum(a_hist) / len(a_hist) if len(a_hist) > 0 else 0.40
            
            style = 'tag'
            if s in focus_seats:
                style = self.focus_archetype
            else:
                if s == 1:
                    style = 'maniac'
                elif s == 2:
                    style = 'nit'
                elif s == 3:
                    style = 'fish'
                
            # Temporary opponent bots to track training dashboard VPIP/AGG averages
            if style == 'maniac':
                opp_bot = self.maniac_heuristic
            elif style == 'nit':
                opp_bot = self.nit_heuristic
            elif style == 'fish':
                opp_bot = self.fish_heuristic
            else:
                opp_bot = self.tag_heuristic
                
            opponents.append({
                'seat': s,
                'bot': opp_bot,
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
                        "action_history": action_history
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
                            table_state_dict=table_state
                        )
                            
                        if decision == 'fold':
                            action_idx = 0
                            folded[0] = True
                            self.seat_histories[0]['folds'] += 1
                            action_history.append('f')
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
                            
                        # Track Hero AGG
                        if street_idx > 0:
                            self.seat_histories[0]['agg'].append(1.0 if decision == 'raise' else 0.0)
                            if len(self.seat_histories[0]['agg']) > 50:
                                self.seat_histories[0]['agg'].pop(0)
                            
                        record.decision_points[-1]['action'] = action_idx
                        record.decision_points[-1]['is_all_in'] = (stacks[0] == 0)
                        step_counter += 1
                        
                    else:  # Opponent bot
                        preflop_had_decision[current_actor] = True
                        opp_bot_struct = [o for o in opponents if o['seat'] == current_actor][0]
                        opp_bot_struct['num_opps'] = active_opps_count
                        
                        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 0.0
                        decision = self._opponent_decide(current_actor, opp_bot_struct, eq, pot_odds, pot, stacks[current_actor], street_idx, table_state)
                        
                        if decision == 'fold':
                            folded[current_actor] = True
                            self.seat_histories[current_actor]['folds'] += 1
                        elif decision == 'call':
                            vpip_this_hand[current_actor] = True
                            call_amt = min(to_call, stacks[current_actor])
                            stacks[current_actor] -= call_amt
                            if stacks[current_actor] == 0:
                                self.seat_histories[current_actor]['all_ins'] += 1
                            committed[current_actor] += call_amt
                            street_committed[current_actor] += call_amt
                            pot += call_amt
                        else:  # raise
                            vpip_this_hand[current_actor] = True
                            self.seat_histories[current_actor]['raises'] += 1
                            raise_size = min(pot * 0.75, stacks[current_actor])
                            raise_size = max(raise_size, to_call + self.bb_size)
                            raise_size = min(raise_size, stacks[current_actor])
                            
                            stacks[current_actor] -= raise_size
                            if stacks[current_actor] == 0:
                                self.seat_histories[current_actor]['all_ins'] += 1
                            committed[current_actor] += raise_size
                            street_committed[current_actor] += raise_size
                            pot += raise_size
                            
                            highest_bet = street_committed[current_actor]
                            last_raiser = current_actor
                        
                        # Track Opponent AGG
                        if street_idx > 0:
                            self.seat_histories[current_actor]['agg'].append(1.0 if decision == 'raise' else 0.0)
                            if len(self.seat_histories[current_actor]['agg']) > 50:
                                self.seat_histories[current_actor]['agg'].pop(0)
                
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
            board_ints = board_cards_int
            for p in active_players:
                temp_board = list(board_ints)
                while len(temp_board) < 5:
                    c = deck.draw(1)
                    temp_board.extend(c)
                score = _treys_evaluator.evaluate(temp_board, hands_ints[p])
                scores.append((p, score))
                
            best_score = min(score for p, score in scores)
            winners = [p for p, score in scores if score == best_score]
            
            share = pot / len(winners)
            for w in winners:
                win_shares[w] = share
                
        for p in range(6):
            if preflop_had_decision[p]:
                self.seat_histories[p]['vpip'].append(1.0 if vpip_this_hand[p] else 0.0)
                if len(self.seat_histories[p]['vpip']) > 50:
                    self.seat_histories[p]['vpip'].pop(0)
            self.seat_histories[p]['profit'] += win_shares[p] - committed[p]
            
        # Calculate pairwise exchange for Exploitation Scoreboard
        net_profits = [win_shares[p] - committed[p] for p in range(6)]
        total_gains = sum(np for np in net_profits if np > 0)
        
        if total_gains > 0:
            for p_lose in range(6):
                if net_profits[p_lose] < 0:
                    for p_win in range(6):
                        if net_profits[p_win] > 0:
                            transfer = abs(net_profits[p_lose]) * (net_profits[p_win] / total_gains)
                            self.global_exploitation_net[p_win][p_lose] += transfer
                            self.global_exploitation_net[p_lose][p_win] -= transfer
                                
        record.final_hero_profit = win_shares[0] - committed[0]
        return record

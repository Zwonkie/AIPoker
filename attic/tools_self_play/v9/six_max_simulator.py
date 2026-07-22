"""
Headless 6-max No-Limit Hold'em Poker Simulator for Pluribus V8.
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
from tools.self_play.opponent_bots import NitBot, FishBot, ManiacBot, TAGBot

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
                     action_history, equity, action_taken, chips_committed_before):
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
            'committed_before': chips_committed_before
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
        
        self.bootstrap_alpha = bootstrap_alpha
        self.hand_counter = 0
        
        # Instantiate bots for heuristics fallback
        self.nit_heuristic = NitBot()
        self.fish_heuristic = FishBot()  # used for Sticky Caller
        self.maniac_heuristic = ManiacBot()
        self.tag_heuristic = TAGBot()
        
        # V9 Dynamic Opponent Profiling (last 50 hands history)
        # Track VPIP (voluntary preflop entry) and AGG (postflop bet/raise ratio)
        self.seat_histories = {s: {
            'vpip': [], 'agg': [], 'profit': 0.0,
            'raises': 0, 'folds': 0, 'all_ins': 0
        } for s in range(6)}
        self.global_metrics = {'flop_players': 0, 'flop_count': 0}

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
        
    def _calculate_equity(self, hero_cards_str, board_str, num_opponents):
        """Calculate Hero's equity using MC simulations."""
        if _cuda_evaluator is not None:
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
            num_simulations=self.equity_sims
        )
        return round(eq * 100) / 100.0

    def _query_model_decide(self, model, hand_cards, equity, pot_size, call_amount, hero_stack, num_opponents, table_state_dict=None):
        """Decide action using a specified neural network model."""
        from core.board_state import BoardState, SeatState, HUDStats
        from core.bridge.contract_v8_v9 import ContractV8V9
        
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
            board_state.seats[seat_key] = SeatState(
                name=f"Opponent {idx+1}",
                is_active=is_active,
                stack=hero_stack if is_active else 0.0,
                hud=HUDStats(
                    vpip_color="Green" if is_active else "Blue",
                    agg_color="Green" if is_active else "Blue"
                )
            )
            
        bridge = ContractV8V9()
        h_t, b_t, c_t, a_t = bridge.to_tensors(board_state)
        
        device = model.device if hasattr(model, 'device') else next(model.parameters()).device
        
        with torch.no_grad():
            preds = model(h_t.to(device), b_t.to(device), c_t.to(device), a_t.to(device))
            q_vals = preds.squeeze(0)[-1]
            
        ev_fold = q_vals[0].item()
        ev_call = q_vals[1].item()
        ev_raise = q_vals[2].item()
        
        available_evs = {'fold': ev_fold, 'call': ev_call, 'raise': ev_raise}
        if call_amount == 0:
            available_evs['fold'] = -9999.0
            
        return max(available_evs, key=available_evs.get)

    # Analytical EVs removed in favor of True Monte Carlo Returns

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
                return self._query_model_decide(self.hero_model, hand_cards, equity, pot_size, call_amount, hero_stack, num_opponents, table_state_dict)
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
            
            opponent['bot'].record_postflop(decision)
            
        return decision

    def simulate_hand(self, current_hand=0):
        """Simulate a single 6-Max NLH hand using V8 specifications."""
        self.hand_counter += 1
        
        # 1. Initialize Seats
        button_seat = random.randint(0, 5)
        
        # Dynamic curriculum stacks
        starting_stack_chips = self._get_starting_stack(current_hand)
        stacks = [starting_stack_chips for _ in range(6)]
        
        active = [True] * 6
        committed = [0.0] * 6
        folded = [False] * 6
        
        # Cards dealing
        deck = Deck()
        hands_ints = [deck.draw(2) for _ in range(6)]
        hands_str = [[Card.int_to_str(c) for c in hand] for hand in hands_ints]
        
        # Compute VPIP / AGG from the last 50 hands running history
        opponents = []
        opponents_profiles = {}
        for s in range(1, 6):
            v_hist = self.seat_histories[s]['vpip']
            a_hist = self.seat_histories[s]['agg']
            
            v_val = sum(v_hist) / len(v_hist) if len(v_hist) > 0 else 0.30
            a_val = sum(a_hist) / len(a_hist) if len(a_hist) > 0 else 0.40
            
            style = 'tag'
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
                        for s in range(1, 6):
                            if not folded[s]:
                                active_mask[s - 1] = 1
                                opp_stacks[s - 1] = stacks[s]
                                
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
                            chips_committed_before=committed[0]
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
                
        # Record preflop VPIP entry histories and accumulate profits
        for p in range(6):
            if preflop_had_decision[p]:
                self.seat_histories[p]['vpip'].append(1.0 if vpip_this_hand[p] else 0.0)
                if len(self.seat_histories[p]['vpip']) > 50:
                    self.seat_histories[p]['vpip'].pop(0)
            self.seat_histories[p]['profit'] += win_shares[p] - committed[p]
                
        record.final_hero_profit = win_shares[0] - committed[0]
        return record

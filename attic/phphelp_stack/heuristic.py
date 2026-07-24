import random
from core.models.base import PokerModelInterface
from core.evaluator import PokerEvaluator

class HeuristicEngine(PokerModelInterface):
    def __init__(self):
        # Pre-flop charts: starting hand ranges based on equity or tier
        self.preflop_top_tier = {
            'AA', 'KK', 'QQ', 'JJ', 'TT', 'AKs', 'AKo', 'AQs', 'AQo', 'AJs', 'KQs'
        }
        self.preflop_playable = {
            'AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', 'AKs', 'AKo', 'AQs', 'AQo',
            'AJs', 'AJo', 'ATs', 'ATo', 'KQs', 'KQo', 'KJs', 'KTs', 'QJs', 'QTs', 'JTs',
            'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s', 'K9s', 'Q9s', 'J9s',
            'T9s', '98s', '87s', '76s', '65s', '54s'
        }
        # Reused across decisions: PokerEvaluator wraps treys.Evaluator, whose __init__ rebuilds
        # the full flush/straight/multiples lookup tables. Constructing one per decision (as the
        # dynamic-sizing branch of predict_action used to) threw that work away every call.
        self.evaluator = PokerEvaluator()

    def get_preflop_hand_string(self, hand: list) -> str:
        c1, c2 = hand[0], hand[1]
        r1, s1 = c1[0], c1[1]
        r2, s2 = c2[0], c2[1]
        
        ranks = "23456789TJQKA"
        idx1 = ranks.index(r1)
        idx2 = ranks.index(r2)
        
        if idx1 >= idx2:
            high_r, low_r = r1, r2
            s_high, s_low = s1, s2
        else:
            high_r, low_r = r2, r1
            s_high, s_low = s2, s1
            
        if high_r == low_r:
            return f"{high_r}{low_r}"
        
        suited_char = 's' if s_high == s_low else 'o'
        return f"{high_r}{low_r}{suited_char}"

    def detect_draws(self, board: list, hand: list):
        all_cards = board + hand
        if len(all_cards) < 5:
            return False, False
            
        suits = [c[1] for c in all_cards]
        suit_counts = {s: suits.count(s) for s in set(suits)}
        has_flush_draw = any(count == 4 for count in suit_counts.values())
        
        ranks = "23456789TJQKA"
        rank_indices = sorted(list(set([ranks.index(c[0]) for c in all_cards])))
        
        has_straight_draw = False
        for i in range(len(rank_indices) - 3):
            span = rank_indices[i+3] - rank_indices[i]
            if span == 3 or span == 4:
                has_straight_draw = True
                break
                
        if 12 in rank_indices:
            low_ranks = sorted(list(set([-1] + [x for x in rank_indices if x < 4])))
            if len(low_ranks) >= 4:
                for i in range(len(low_ranks) - 3):
                    if low_ranks[i+3] - low_ranks[i] in [3, 4]:
                        has_straight_draw = True
                        break
                        
        return has_flush_draw, has_straight_draw

    def get_preflop_probabilities(self, equity: float, call_amount: float, num_opponents: int, preflop_looseness: float = 0.0) -> tuple:
        # Positional / Opponents baseline thresholds (+5% higher for premium)
        if num_opponents <= 1:
            premium_thresh = 0.59
            playable_thresh = 0.45
            marginal_thresh = 0.40
        elif num_opponents == 2:
            premium_thresh = 0.62
            playable_thresh = 0.47
            marginal_thresh = 0.42
        else:
            premium_thresh = 0.65
            playable_thresh = 0.49
            marginal_thresh = 0.44
            
        import math
        adj_eq = equity + preflop_looseness
        
        # Means (peaks) for each category
        mu_prem = premium_thresh
        mu_play = playable_thresh
        mu_marg = marginal_thresh
        mu_weak = marginal_thresh - 0.15
        
        sigma = 0.05
        
        # Calculate raw Gaussian weights
        if adj_eq >= mu_prem:
            w_prem = 1.0
            w_play = 0.0
            w_marg = 0.0
            w_weak = 0.0
        elif adj_eq <= mu_weak:
            w_prem = 0.0
            w_play = 0.0
            w_marg = 0.0
            w_weak = 1.0
        else:
            w_prem = math.exp(-((adj_eq - mu_prem) ** 2) / (2 * (sigma ** 2)))
            w_play = math.exp(-((adj_eq - mu_play) ** 2) / (2 * (sigma ** 2)))
            w_marg = math.exp(-((adj_eq - mu_marg) ** 2) / (2 * (sigma ** 2)))
            w_weak = math.exp(-((adj_eq - mu_weak) ** 2) / (2 * (sigma ** 2)))
            
        # Normalize to probabilities
        sum_w = w_prem + w_play + w_marg + w_weak
        if sum_w > 0:
            p_prem = w_prem / sum_w
            p_play = w_play / sum_w
            p_marg = w_marg / sum_w
            p_weak = w_weak / sum_w
        else:
            p_prem, p_play, p_marg, p_weak = 0.0, 0.0, 0.0, 1.0
            
        # Compute action mix based on call_amount
        if call_amount == 0:
            # Unopened pot base distributions (Playable open-raise tuned down to 8% to prevent loose raises)
            p_raise = p_prem * 0.90 + p_play * 0.08 + p_marg * 0.00 + p_weak * 0.00
            p_call  = p_prem * 0.10 + p_play * 0.92 + p_marg * 0.15 + p_weak * 1.00
            p_fold  = p_prem * 0.00 + p_play * 0.00 + p_marg * 0.85 + p_weak * 0.00
        else:
            # Facing bet base distributions
            p_raise = p_prem * 0.80 + p_play * 0.15 + p_marg * 0.00 + p_weak * 0.00
            p_call  = p_prem * 0.20 + p_play * 0.80 + p_marg * 0.10 + p_weak * 0.00
            p_fold  = p_prem * 0.00 + p_play * 0.05 + p_marg * 0.90 + p_weak * 1.00
            
        return p_raise, p_call, p_fold

    def predict_action(self, board: list, hand: list, equity: float, pot_size: float, 
                       call_amount: float, hero_stack: float, num_opponents: int,
                       is_preflop: bool, use_preflop_chart: bool, use_math_engine: bool,
                       use_bluff_engine: bool, use_dynamic_sizing: bool,
                       bet_raise_available: bool, check_call_available: bool,
                       active_opponents: list = None, table_state_dict: dict = None,
                       preflop_looseness: float = 0.0) -> tuple:
        if active_opponents is None:
            active_opponents = []
        if is_preflop:
            action = 'CHECK'
            reason = ""
            bet_size = 0.0
            
            adj_eq = equity + preflop_looseness
            p_raise, p_call, p_fold = self.get_preflop_probabilities(equity, call_amount, num_opponents, preflop_looseness)

            # Add random +- 0.15 modifier to call percentage (GTO variance increased)
            if p_call > 0.0 and p_call < 1.0:
                delta = random.uniform(-0.15, 0.15)
                p_call_new = max(0.0, min(1.0, p_call + delta))
                
                # Distribute difference proportionally to raise and fold
                diff = p_call - p_call_new
                p_call = p_call_new
                other_sum = p_raise + p_fold
                if other_sum > 0:
                    p_raise += diff * (p_raise / other_sum)
                    p_fold += diff * (p_fold / other_sum)
                else:
                    p_fold += diff
                
                # Ensure all are normalized and clamp
                p_raise = max(0.0, min(1.0, p_raise))
                p_fold = max(0.0, min(1.0, p_fold))
                s = p_raise + p_call + p_fold
                if s > 0:
                    p_raise /= s
                    p_call /= s
                    p_fold /= s

            # Draw action from probability distribution
            rnd = random.random()
            if rnd < p_raise:
                if call_amount == 0:
                    action = 'BET_POT_70' if use_dynamic_sizing else 'BET'
                    bet_size = min(3.0 * pot_size if pot_size > 0 else 3.0, hero_stack)
                else:
                    action = 'RAISE_POT_100' if use_dynamic_sizing else 'RAISE'
                    bet_size = min(call_amount * 3.0, hero_stack)
                reason = f"Pre-flop (MC/Gaussian): Raising (Prob: {p_raise:.1%}, Eq: {equity:.1%}, Adj: {adj_eq:.1%}) (F: {p_fold:.1%}, C: {p_call:.1%}, R: {p_raise:.1%})"
            elif rnd < p_raise + p_call:
                if call_amount == 0:
                    action = 'CHECK'
                    bet_size = 0.0
                else:
                    pot_odds = call_amount / (pot_size + call_amount) if (pot_size + call_amount) > 0 else 0.0
                    if call_amount < 0.25 * hero_stack or equity >= pot_odds - 0.05:
                        action = 'CALL'
                        bet_size = call_amount
                    else:
                        action = 'FOLD'
                        bet_size = 0.0
                reason = f"Pre-flop (MC/Gaussian): Calling/Checking (Prob: {p_call:.1%}, Eq: {equity:.1%}, Adj: {adj_eq:.1%}) (F: {p_fold:.1%}, C: {p_call:.1%}, R: {p_raise:.1%})"
            else:
                if call_amount == 0:
                    action = 'CHECK'
                    bet_size = 0.0
                else:
                    action = 'FOLD'
                    bet_size = 0.0
                reason = f"Pre-flop (MC/Gaussian): Folding (Prob: {p_fold:.1%}, Eq: {equity:.1%}, Adj: {adj_eq:.1%}) (F: {p_fold:.1%}, C: {p_call:.1%}, R: {p_raise:.1%})"

            # Apply final pre-flop check/call availability redirections
            if not check_call_available and action == 'CALL':
                action = 'RAISE'
                reason = f"Call button unavailable. Cleft-click ALL-IN. (Original: {reason})"

            return action, reason, bet_size

        # Post-flop logic (Flop, Turn, River)
        has_flush_draw, has_straight_draw = self.detect_draws(board, hand)
        is_drawing = has_flush_draw or has_straight_draw

        # 1. Bluffing Layer
        is_bluffing = False
        bluff_reason = ""
        
        if use_bluff_engine:
            # Semi-bluff: drawing hand with moderate equity (30% to 48%)
            if is_drawing and 0.30 <= equity <= 0.48:
                if random.random() < 0.35:  # 35% frequency to raise/bet
                    is_bluffing = True
                    bluff_reason = "Semi-bluffing with draw"
            # Pure bluff: low equity, checked to us, turn or river, 15% frequency
            elif equity < 0.25 and call_amount == 0 and len(board) >= 4:
                if random.random() < 0.15:
                    is_bluffing = True
                    bluff_reason = "Pure bluff on checked board"

        action = 'CHECK'
        reason = ""

        # 2. Math Engine vs Flat Equity Thresholds
        if use_math_engine:
            total_pot = pot_size + call_amount
            pot_odds = call_amount / total_pot if total_pot > 0 else 0.0
            expected_value = equity - pot_odds
            
            # Dynamic equity thresholds scaling with active player count
            if num_opponents == 1:
                raise_threshold = 0.55
                bet_threshold = 0.55
                moderate_equity_threshold = 0.40
            elif num_opponents == 2:
                raise_threshold = 0.62
                bet_threshold = 0.60
                moderate_equity_threshold = 0.45
            else: # 3+ opponents
                raise_threshold = 0.70
                bet_threshold = 0.65
                moderate_equity_threshold = 0.50
                
            # Adjust for opponent aggression (AGG factor)
            agg_colors = [opp.get('agg_color') for opp in active_opponents if opp.get('agg_color')]
            if 'Red' in agg_colors:
                # Highly aggressive: play slightly tighter against bets, but we might trap more
                raise_threshold += 0.05
                moderate_equity_threshold += 0.05
            elif 'Yellow' in agg_colors:
                raise_threshold += 0.02
                moderate_equity_threshold += 0.02
            elif 'Green' in agg_colors:
                # Passive: we can value bet wider
                bet_threshold -= 0.03
                raise_threshold -= 0.03
            
            # Short-stack commitment adjustment (pot-committed / survival threshold shifts)
            stack_pot_ratio = hero_stack / pot_size if pot_size > 0 else 999.0
            is_short_stacked = (stack_pot_ratio < 2.5) or (hero_stack < 60.0)
            
            if is_short_stacked:
                # Lower equity requirements by 12% to defend/shove medium hands
                raise_threshold -= 0.12
                bet_threshold -= 0.12
                moderate_equity_threshold -= 0.12
            
            if call_amount == 0:
                if equity > bet_threshold or is_bluffing:
                    action = 'BET'
                    reason = bluff_reason if is_bluffing else f"Post-flop (EV): Strong equity ({equity:.1%})"
                elif equity > moderate_equity_threshold:
                    action = 'CHECK'
                    reason = f"Post-flop (EV): Moderate equity ({equity:.1%}). Checking."
                else:
                    action = 'CHECK'
                    reason = f"Post-flop (EV): Low equity ({equity:.1%}). Checking."
            else:
                if expected_value > 0.15 and (equity > raise_threshold or is_bluffing):
                    action = 'RAISE'
                    reason = bluff_reason if is_bluffing else f"Post-flop (EV): High EV ({expected_value:+.2f}) & equity ({equity:.1%})"
                elif expected_value >= 0.0:
                    action = 'CALL'
                    reason = f"Post-flop (EV): Positive EV ({expected_value:+.2f}) & equity ({equity:.1%})"
                else:
                    # Draw-based marginal calling
                    if equity > pot_odds - 0.05 and call_amount < 0.10 * hero_stack:
                        action = 'CALL'
                        reason = f"Post-flop (EV): Marginal negative EV but close odds with deep stacks."
                    else:
                        action = 'FOLD'
                        reason = f"Post-flop (EV): Negative EV ({expected_value:+.2f}) & low equity ({equity:.1%})"
        else:
            # Flat Equity Thresholds
            if call_amount == 0:
                if equity > 0.55 or is_bluffing:
                    action = 'BET'
                    reason = bluff_reason if is_bluffing else f"Post-flop (Flat): Good equity ({equity:.1%})"
                else:
                    action = 'CHECK'
                    reason = f"Post-flop (Flat): Low equity ({equity:.1%}). Checking."
            else:
                if equity > 0.65 or (is_bluffing and equity > 0.35):
                    action = 'RAISE'
                    reason = bluff_reason if is_bluffing else f"Post-flop (Flat): Strong equity ({equity:.1%})"
                elif equity > 0.35:
                    action = 'CALL'
                    reason = f"Post-flop (Flat): Playable equity ({equity:.1%})"
                else:
                    action = 'FOLD'
                    reason = f"Post-flop (Flat): Low equity ({equity:.1%}). Folding."

        # 3. Dynamic Sizing Layer (Apply Bet Sizing Shortcuts/Slider if Action is BET or RAISE)
        bet_size = 0.0
        if action in ['BET', 'RAISE']:
            if use_dynamic_sizing:
                # Use slider for all streets to remain immune to button label changes (BB multiples vs Pot %)
                # Post-flop only: every preflop path returned above, so dynamic sizing is
                # always board-texture (wetness) based.
                texture = self.evaluator.analyze_board_texture(board)
                wetness = texture['wetness']

                if is_bluffing:
                    bet_pct = 0.35  # Cheap bluff
                    reason += f" (Bluff on board wetness={wetness:.1f})"
                else:
                    if wetness >= 0.5:
                        bet_pct = 0.80  # Large bet on wet boards
                        reason += f" (Wet board sizing, wetness={wetness:.1f})"
                    else:
                        bet_pct = 0.40  # Small bet on dry boards
                        reason += f" (Dry board sizing, wetness={wetness:.1f})"

                target_bet = pot_size * bet_pct

                min_bet = 2.0 * call_amount if call_amount > 0 else 20.0
                clamped_bet = max(min_bet, min(target_bet, hero_stack))
                
                # The slider represents 1 Small Blind (10.0 chips) to max stacksize (hero_stack)
                min_slider_val = 10.0
                max_slider_val = hero_stack
                
                if max_slider_val > min_slider_val:
                    slider_fraction = (clamped_bet - min_slider_val) / (max_slider_val - min_slider_val)
                    slider_fraction = max(0.0, min(1.0, slider_fraction))
                    action = f"{action}_SLIDER_{slider_fraction:.2f}"
                    bet_size = clamped_bet
                else:
                    action = f"{action}_SLIDER_1.0"
                    bet_size = hero_stack
            else:
                # Default Sizing
                if action == 'BET':
                    bet_size = min(max(pot_size * 0.5, 2.0), hero_stack)
                else:
                    bet_size = min(call_amount + max(pot_size * 0.5, call_amount * 2.0), hero_stack)

        # Check if the decided action requires the Bet/Raise button when it's unavailable
        if not bet_raise_available and (action.startswith('BET') or action.startswith('RAISE')):
            if call_amount == 0:
                action = 'CHECK'
                reason = f"Bet/Raise button unavailable. Checking instead. (Original: {reason})"
                bet_size = 0.0
            else:
                # Determine if calling is reasonable
                can_call = False
                if use_math_engine:
                    total_pot = pot_size + call_amount
                    pot_odds = call_amount / total_pot if total_pot > 0 else 0.0
                    expected_value = equity - pot_odds
                    if expected_value >= 0.0 or (equity > pot_odds - 0.05 and call_amount < 0.10 * hero_stack):
                        can_call = True
                else:
                    if equity > 0.35:
                        can_call = True
                
                if can_call:
                    action = 'CALL'
                    reason = f"Bet/Raise button unavailable. Calling instead. (Original: {reason})"
                    bet_size = call_amount
                else:
                    action = 'FOLD'
                    reason = f"Bet/Raise button unavailable. Folding due to low EV. (Original: {reason})"
                    bet_size = 0.0

        # Check if call button is unavailable (All-In situation)
        if not check_call_available and action == 'CALL':
            action = 'RAISE'  # Clicks the third button (which turns into ALL-IN)
            reason = f"Call button unavailable. Cleft-click ALL-IN. (Original: {reason})"

        return action, reason, bet_size

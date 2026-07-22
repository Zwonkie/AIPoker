import random
import math

class HeuristicTrainingEngine:
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

    def get_preflop_probabilities(self, equity: float, call_amount: float, num_opponents: int, preflop_looseness: float = 0.0) -> tuple:
        # Positional / Opponents baseline thresholds
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
            p_raise = p_prem * 0.90 + p_play * 0.08 + p_marg * 0.00 + p_weak * 0.00
            p_call  = p_prem * 0.10 + p_play * 0.92 + p_marg * 0.15 + p_weak * 1.00
            p_fold  = p_prem * 0.00 + p_play * 0.00 + p_marg * 0.85 + p_weak * 0.00
        else:
            p_raise = p_prem * 0.80 + p_play * 0.15 + p_marg * 0.00 + p_weak * 0.00
            p_call  = p_prem * 0.20 + p_play * 0.80 + p_marg * 0.10 + p_weak * 0.00
            p_fold  = p_prem * 0.00 + p_play * 0.05 + p_marg * 0.90 + p_weak * 1.00
            
        return p_raise, p_call, p_fold

    def decide_preflop(self, hand: list, equity: float, call_amount: float, pot_size: float, 
                       hero_stack: float, num_opponents: int, preflop_looseness: float = 0.0) -> str:
        """
        Calculates preflop action probabilities and draws an action.
        Returns 'fold', 'call', or 'raise'.
        """
        p_raise, p_call, p_fold = self.get_preflop_probabilities(equity, call_amount, num_opponents, preflop_looseness)

        # Add GTO variance modifier
        if 0.0 < p_call < 1.0:
            delta = random.uniform(-0.15, 0.15)
            p_call_new = max(0.0, min(1.0, p_call + delta))
            diff = p_call - p_call_new
            p_call = p_call_new
            other_sum = p_raise + p_fold
            if other_sum > 0:
                p_raise += diff * (p_raise / other_sum)
                p_fold += diff * (p_fold / other_sum)
            else:
                p_fold += diff
            
            p_raise = max(0.0, min(1.0, p_raise))
            p_fold = max(0.0, min(1.0, p_fold))
            s = p_raise + p_call + p_fold
            if s > 0:
                p_raise /= s
                p_call /= s
                p_fold /= s

        # Draw action
        rnd = random.random()
        if rnd < p_raise:
            return 'raise'
        elif rnd < p_raise + p_call:
            if call_amount == 0:
                return 'call' # check/limp
            else:
                pot_odds = call_amount / (pot_size + call_amount) if (pot_size + call_amount) > 0 else 0.0
                if call_amount < 0.25 * hero_stack or equity >= pot_odds - 0.05:
                    return 'call'
                else:
                    return 'fold'
        else:
            if call_amount == 0:
                return 'call' # check
            else:
                return 'fold'

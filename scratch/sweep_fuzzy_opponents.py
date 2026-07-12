import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.self_play.v11.opponent_bots_v11 import TAG, LAG, NIT, CALLING_STATION

def generate_random_equity():
    return max(0.01, min(0.99, random.gauss(0.3, 0.15)))

def sweep_bot(bot, num_hands=10000):
    bot.hands_played = 0
    bot.vpip_count = 0
    bot.pfr_count = 0
    bot.agg_bets = 0
    bot.agg_calls = 0
    bot.agg_folds = 0
    
    for _ in range(num_hands):
        bot.start_new_hand()
        
        # PREFLOP
        equity = generate_random_equity()
        pot_odds = random.uniform(0.1, 0.4)
        is_blind = random.choice([True, False])
        
        action = bot.decide_preflop(equity, pot_odds, is_blind)
        bot.record_preflop(action)
        
        # POSTFLOP
        if action != 'fold':
            equity_pf = random.gauss(0.4, 0.2)
            pot_odds_pf = random.uniform(0.0, 0.5)
            pot_size = random.uniform(10, 100)
            stack = 1000
            street_idx = random.choice([1, 2, 3])
            action_pf = bot.decide_postflop(equity_pf, pot_odds_pf, pot_size, stack, street_idx)
            bot.record_postflop(action_pf)
            
    print(f"==================================================")
    print(f"Archetype: {bot.name}")
    print(f"==================================================")
    print(f"Target VPIP:  {bot.base_vpip:.2f} | Simulated VPIP:  {bot.vpip:.2f}")
    print(f"Target AGG:   {bot.base_agg_freq:.2f} | Simulated AGG:   {bot.agg_frequency:.2f}")
    print(f"PFR Proxy:    {bot.pfr_count / max(1, bot.hands_played):.2f}")
    print(f"Target Bluff: {bot.base_bluff_freq:.2f}")
    print(f"Postflop Actions -> Bets/Raises: {bot.agg_bets}, Calls: {bot.agg_calls}, Folds: {bot.agg_folds}")
    print(f"Total Hands:  {bot.hands_played}\n")

if __name__ == '__main__':
    random.seed(42)
    sweep_bot(TAG, 10000)
    sweep_bot(LAG, 10000)
    sweep_bot(NIT, 10000)
    sweep_bot(CALLING_STATION, 10000)

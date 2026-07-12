import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.models.pluribus_engine import PluribusEngine

def run_test():
    engine = PluribusEngine(game_type='NLH', expert_name='expert_nlh_combined.pth')
    
    # Mock 1 parameters (we manually force EV to raise)
    # Say EV of raising is 3.64 BB
    hand = ['Js', '5s']
    board = ['Kd', 'Ac', '3s']
    equity = 0.126
    pot_size = 80.0
    call_amount = 0.0
    hero_stack = 760.0
    num_opponents = 3
    is_preflop = False
    
    table_state_dict = {
        'community_cards': board,
        'hero_cards': hand,
        'pot_size': pot_size,
        'hero_stack': hero_stack,
        'action_history': [],
        'big_blind': 25.0
    }
    
    # We mock _get_q_values to return a high raise EV
    engine._get_q_values = lambda state: (0.1, 0.2, 3.64)
    
    print("Testing PluribusEngine with high EV(Raise) = 3.64 BB:")
    best_action, reason, bet_size, ev_dict = engine.predict_action(
        board, hand, equity, pot_size, call_amount, hero_stack, num_opponents,
        is_preflop, use_preflop_chart=False, use_math_engine=True,
        use_bluff_engine=True, use_dynamic_sizing=True,
        bet_raise_available=True, check_call_available=True,
        active_opponents=[],
        table_state_dict=table_state_dict
    )
    
    print("\nResults:")
    print("Action:", best_action)
    print("Reason:", reason)
    print("Bet Size:", bet_size)
    print("EV Dict:", ev_dict)

if __name__ == '__main__':
    run_test()

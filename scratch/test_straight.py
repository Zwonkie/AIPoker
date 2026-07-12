import os
import sys
import torch

sys.path.insert(0, os.path.abspath('.'))
from core.models.pluribus_engine import PluribusEngine

def test_hand():
    model = PluribusEngine(game_type='NLH', expert_name='expert_ev_v2.pth', model_type='ev')
    
    # State dictionary mirroring the telemetry
    state_dict = {
        'hero_cards': ['Ks', '9s'],
        'community_cards': ['Ts', 'Qh', 'Jd'],
        'pot_size': 150.0,
        'hero_stack': 2950.0,
        'num_opponents': 1, # telemetry had 1 opponent in context
        'action_history': ['f', 'b', 'b', 'r', 'c'],
        'big_blind': 30.0,
        'call_amount': 0.0
    }
    
    # Test 1: Telemetry raw action history ['f', 'b', 'b', 'r', 'c']
    q_fold, q_call, q_raise = model._get_q_values(state_dict)
    print("Q-values (f, b, b, r, c):")
    print(f"Fold: {q_fold:.4f} | Call: {q_call:.4f} | Raise: {q_raise:.4f}")
    
    # Test 2: Empty action history (first to act on Flop)
    state_dict['action_history'] = []
    q_fold, q_call, q_raise = model._get_q_values(state_dict)
    print("\nQ-values (Empty Action History):")
    print(f"Fold: {q_fold:.4f} | Call: {q_call:.4f} | Raise: {q_raise:.4f}")
    
    # Test 3: Facing a single bet ['b']
    state_dict['action_history'] = ['b']
    state_dict['call_amount'] = 30.0 # facing 1 BB bet
    q_fold, q_call, q_raise = model._get_q_values(state_dict)
    print("\nQ-values (Facing a bet ['b']):")
    print(f"Fold: {q_fold:.4f} | Call: {q_call:.4f} | Raise: {q_raise:.4f}")

    # Test 4: Pure Check-Check ['k', 'k']
    state_dict['action_history'] = ['k', 'k']
    state_dict['call_amount'] = 0.0
    q_fold, q_call, q_raise = model._get_q_values(state_dict)
    print("\nQ-values (Check-Check ['k', 'k']):")
    print(f"Fold: {q_fold:.4f} | Call: {q_call:.4f} | Raise: {q_raise:.4f}")

if __name__ == '__main__':
    test_hand()

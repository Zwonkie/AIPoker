import torch
import sys
import os

# Add project root to sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.pluribus_engine import PluribusEngine

def test_predictions():
    # Load V4 engine
    engine = PluribusEngine(expert_name="expert_v4_selfplay.pth", model_type="v4")
    
    # We will test two states:
    # 1. The erroneous OOD state (pot = 90)
    # 2. The corrected state (pot = 510)
    
    base_state = {
        "community_cards": [],
        "hero_cards": ["2s", "Jd"],
        "pot_size": 90, # erroneous
        "hero_stack": 690,
        "opponents": {
            "seat_1": {"name": "spoongreen827", "stack": 780, "is_active": False, "state": "Folded"},
            "seat_2": {"name": "Tid: 11", "stack": 740, "is_active": True, "state": "Active"},
            "seat_3": {"name": "VFR750R", "stack": 800, "is_active": False, "state": "Folded"},
            "seat_4": {"name": "AllaSun777", "stack": 290, "is_active": True, "state": "Active"},
            "seat_5": {"name": "Heronman", "stack": 800, "is_active": False, "state": "Folded"}
        },
        "num_active_players": 3,
        "active_buttons": ["button_fold"],
        "street": "Preflop",
        "dealer_name": "Heronman",
        "dealer_idx": 5,
        "hero_position": 1,
        "action_history": [],
        "opp_vpip_norm": 0.3,
        "opp_agg_norm": 0.4,
        "big_blind": 20.0,
        "call_amount": 470.0,
        "equity": 0.1845
    }
    
    print("--- 1. Erroneous State (pot = 90) ---")
    ev_fold, ev_call, ev_raise = engine._get_q_values(base_state)
    print(f"EVs -> Fold: {ev_fold:.4f}, Call: {ev_call:.4f}, Raise: {ev_raise:.4f}")
    
    print("\n--- 2. Corrected State (pot = 510) ---")
    corrected_state = base_state.copy()
    corrected_state["pot_size"] = 510
    ev_fold_c, ev_call_c, ev_raise_c = engine._get_q_values(corrected_state)
    print(f"EVs -> Fold: {ev_fold_c:.4f}, Call: {ev_call_c:.4f}, Raise: {ev_raise_c:.4f}")

if __name__ == '__main__':
    test_predictions()

import torch
import sys
import os

# Add project root to sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.pluribus_engine import PluribusEngine

def check_folds():
    engine = PluribusEngine(expert_name="expert_v4_selfplay.pth", model_type="v4")
    
    board = ["As", "Td", "4c"] # Flop
    
    # We will test 6 different hands:
    # 1. AA (Top Set)
    # 2. AK (Top Pair)
    # 3. AT (Two Pair)
    # 4. QJ (Gutshot Draw)
    # 5. 72o (Garbage)
    # 6. 32o (Garbage)
    # Facing a massive bet (call_amount = 400, pot_size = 500)
    
    test_hands = {
        "AA": ["Ah", "Ac"],
        "AK": ["Ah", "Kd"],
        "AT": ["Ah", "Ts"],
        "QJ": ["Qh", "Jd"],
        "72o": ["7s", "2c"],
        "32o": ["3h", "2d"]
    }
    
    print("## Postflop EV Predictions facing 20BB Bet (pot = 500, call = 400)\n")
    print(f"| Hand | EV (Fold) | EV (Call) | EV (Raise) | Prob Fold | Prob Call | Prob Raise |")
    print(f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for name, cards in test_hands.items():
        state = {
            "community_cards": board,
            "hero_cards": cards,
            "pot_size": 500.0,
            "hero_stack": 1500.0,
            "opponents": {},
            "num_active_players": 2,
            "active_buttons": ["button_fold", "button_call", "button_raise"],
            "street": "Flop",
            "dealer_idx": 0,
            "hero_position": 0,
            "action_history": ["b"],
            "opp_vpip_norm": 0.3,
            "opp_agg_norm": 0.4,
            "big_blind": 20.0,
            "call_amount": 400.0,
            "equity": 0.5 # dummy
        }
        for i in range(5):
            seat_key = f"seat_{i+1}"
            is_active = (i == 0)
            state["opponents"][seat_key] = {
                "name": f"Player_{i+1}",
                "stack": 1500.0 if is_active else 0.0,
                "is_active": is_active,
                "state": "Active" if is_active else "Folded"
            }
            
        ev_f, ev_c, ev_r = engine._get_q_values(state)
        
        # Calculate probabilities matching PHPHelp.py
        available_evs = {'FOLD': ev_f, 'CALL': ev_c, 'RAISE': ev_r}
        positive_evs = {k: v for k, v in available_evs.items() if v > 0}
        
        prob_dict = {k: 0.0 for k in ['FOLD', 'CALL', 'RAISE']}
        if not positive_evs:
            best_action = max(available_evs, key=available_evs.get)
            prob_dict[best_action] = 1.0
        else:
            total_pos = sum(positive_evs.values())
            for k, v in positive_evs.items():
                prob_dict[k] = v / total_pos
                
        print(f"| {name} | {ev_f:.2f} | {ev_c:.2f} | {ev_r:.2f} | {prob_dict['FOLD']:.1%} | {prob_dict['CALL']:.1%} | {prob_dict['RAISE']:.1%} |")

if __name__ == '__main__':
    check_folds()

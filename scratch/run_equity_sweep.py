import torch
import sys
import os
import numpy as np

# Add project root to sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.pluribus_engine import PluribusEngine

def run_equity_sweep():
    # Load V4 engine
    engine = PluribusEngine(expert_name="expert_v4_selfplay.pth", model_type="v4")
    
    # 1. Sweep 1: Preflop Equity Sweep
    # Cards: AA (so the card embeddings represent a strong hand, but we sweep context equity)
    print("## SWEEP 1: Preflop Equity Sweep (Hero on Button, Unopened Pot)\n")
    print(f"| Equity | EV (Fold) | EV (Call) | EV (Raise) | Best Action |")
    print(f"| :---: | :---: | :---: | :---: | :---: |")
    
    for eq in np.linspace(0.0, 1.0, 21):
        state = {
            "community_cards": [],
            "hero_cards": ["Ah", "As"],
            "pot_size": 30.0,
            "hero_stack": 2000.0,
            "opponents": {},
            "num_active_players": 4,
            "active_buttons": ["button_fold", "button_call", "button_raise"],
            "street": "Preflop",
            "dealer_idx": 0,
            "hero_position": 0,
            "action_history": [],
            "opp_vpip_norm": 0.3,
            "opp_agg_norm": 0.4,
            "big_blind": 20.0,
            "call_amount": 0.0,
            "equity": float(eq)
        }
        for i in range(5):
            seat_key = f"seat_{i+1}"
            is_active = (i < 3)
            state["opponents"][seat_key] = {
                "name": f"Player_{i+1}",
                "stack": 2000.0 if is_active else 0.0,
                "is_active": is_active,
                "state": "Active" if is_active else "Folded"
            }
            
        ev_f, ev_c, ev_r = engine._get_q_values(state)
        evs = {"FOLD": ev_f, "CALL": ev_c, "RAISE": ev_r}
        best_act = max(evs, key=evs.get)
        print(f"| {eq:.2f} | {ev_f:.2f} | {ev_c:.2f} | {ev_r:.2f} | **{best_act}** |")
        
    # 2. Sweep 2: Postflop Equity Sweep
    # Board: As Td 4c, Cards: AK, facing 3BB bet
    print("\n## SWEEP 2: Postflop Equity Sweep on Flop (Hero on Button, Facing 3BB Bet)\n")
    print(f"| Equity | EV (Fold) | EV (Call) | EV (Raise) | Best Action |")
    print(f"| :---: | :---: | :---: | :---: | :---: |")
    
    for eq in np.linspace(0.0, 1.0, 21):
        state = {
            "community_cards": ["As", "Td", "4c"],
            "hero_cards": ["Ah", "Kd"],
            "pot_size": 180.0,
            "hero_stack": 1880.0,
            "opponents": {},
            "num_active_players": 4,
            "active_buttons": ["button_fold", "button_call", "button_raise"],
            "street": "Flop",
            "dealer_idx": 0,
            "hero_position": 0,
            "action_history": ["b"],
            "opp_vpip_norm": 0.3,
            "opp_agg_norm": 0.4,
            "big_blind": 20.0,
            "call_amount": 60.0,
            "equity": float(eq)
        }
        for i in range(5):
            seat_key = f"seat_{i+1}"
            is_active = (i < 3)
            state["opponents"][seat_key] = {
                "name": f"Player_{i+1}",
                "stack": 1880.0 if is_active else 0.0,
                "is_active": is_active,
                "state": "Active" if is_active else "Folded"
            }
            
        ev_f, ev_c, ev_r = engine._get_q_values(state)
        evs = {"FOLD": ev_f, "CALL": ev_c, "RAISE": ev_r}
        best_act = max(evs, key=evs.get)
        print(f"| {eq:.2f} | {ev_f:.2f} | {ev_c:.2f} | {ev_r:.2f} | **{best_act}** |")

if __name__ == '__main__':
    run_equity_sweep()

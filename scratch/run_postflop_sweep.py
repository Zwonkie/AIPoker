import torch
import sys
import os
import numpy as np

# Add project root to sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.pluribus_engine import PluribusEngine

def run_sweep():
    # Load V4 engine
    engine = PluribusEngine(expert_name="expert_v4_selfplay.pth", model_type="v4")
    
    board = ["As", "Td", "4c"] # Flop
    
    hands = {
        "Top Pair (AK)": (["Ah", "Kd"], [0.85, 0.72, 0.62, 0.53, 0.45]),
        "7-High Garbage (72o)": (["2c", "7d"], [0.05, 0.02, 0.01, 0.005, 0.002])
    }
    
    # 1. Sweep 1: Positions (Flop, Unopened pot = 120, 3 active opponents)
    print("## SWEEP 1: Table Positions on Flop (Unopened Pot, 3 Active Opponents)\n")
    print(f"| Hand | Position | GTO Pos | EV (Fold) | EV (Call) | EV (Raise) | Best Action |")
    print(f"| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    
    pos_names = {0: "Button", 1: "Small Blind", 2: "Big Blind", 3: "UTG", 4: "Middle Pos", 5: "Cut Off"}
    
    for hand_name, (cards, equities) in hands.items():
        equity = equities[2]
        for pos in range(6):
            state = {
                "community_cards": board,
                "hero_cards": cards,
                "pot_size": 120.0,
                "hero_stack": 1880.0, # stack after posting/paying preflop
                "opponents": {},
                "num_active_players": 4,
                "active_buttons": ["button_fold", "button_call", "button_raise"],
                "street": "Flop",
                "dealer_idx": (0 - pos) % 6,
                "hero_position": pos,
                "action_history": ["k", "k", "k"], # checked to Hero
                "opp_vpip_norm": 0.3,
                "opp_agg_norm": 0.4,
                "big_blind": 20.0,
                "call_amount": 0.0,
                "equity": equity
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
            print(f"| {hand_name} | {pos_names[pos]} | {pos} | {ev_f:.2f} | {ev_c:.2f} | {ev_r:.2f} | **{best_act}** |")
            
    # 2. Sweep 2: Action Faced (Check, Half Pot Bet, Overbet, All-In)
    # Position: Button (pos=0), Active opponents: 3
    print("\n## SWEEP 2: Actions Faced on Flop (Hero on Button, 3 Active Opponents)\n")
    print(f"| Hand | Action Faced | Pot Size | Call Amt | EV (Fold) | EV (Call) | EV (Raise) | Best Action |")
    print(f"| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    actions = [
        ("Check (Unopened)", 120.0, 0.0),
        ("Half Pot Bet (3BB)", 180.0, 60.0),
        ("Double Pot Overbet", 360.0, 240.0),
        ("All-In Shove", 590.0, 470.0)
    ]
    
    for hand_name, (cards, equities) in hands.items():
        equity = equities[2]
        for act_name, pot_sz, call_amt in actions:
            state = {
                "community_cards": board,
                "hero_cards": cards,
                "pot_size": pot_sz,
                "hero_stack": 1880.0,
                "opponents": {},
                "num_active_players": 4,
                "active_buttons": ["button_fold", "button_call", "button_raise"],
                "street": "Flop",
                "dealer_idx": 0,
                "hero_position": 0,
                "action_history": ["k", "k", "b"] if call_amt > 0 else ["k", "k", "k"],
                "opp_vpip_norm": 0.3,
                "opp_agg_norm": 0.4,
                "big_blind": 20.0,
                "call_amount": call_amt,
                "equity": equity
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
            print(f"| {hand_name} | {act_name} | {pot_sz:.0f} | {call_amt:.0f} | {ev_f:.2f} | {ev_c:.2f} | {ev_r:.2f} | **{best_act}** |")

    # 3. Sweep 3: Active Opponents (1 to 5)
    # Position: Button (pos=0), Action: Half Pot Bet faced (call_amt=60, pot_size=180)
    print("\n## SWEEP 3: Number of Active Opponents on Flop (Hero on Button, Facing Half Pot Bet)\n")
    print(f"| Hand | Opponents | Equity | EV (Fold) | EV (Call) | EV (Raise) | Best Action |")
    print(f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for hand_name, (cards, equities) in hands.items():
        for opps_count in range(1, 6):
            equity = equities[opps_count - 1]
            state = {
                "community_cards": board,
                "hero_cards": cards,
                "pot_size": 180.0,
                "hero_stack": 1880.0,
                "opponents": {},
                "num_active_players": opps_count + 1,
                "active_buttons": ["button_fold", "button_call", "button_raise"],
                "street": "Flop",
                "dealer_idx": 0,
                "hero_position": 0,
                "action_history": ["b"],
                "opp_vpip_norm": 0.3,
                "opp_agg_norm": 0.4,
                "big_blind": 20.0,
                "call_amount": 60.0,
                "equity": equity
            }
            for i in range(5):
                seat_key = f"seat_{i+1}"
                is_active = (i < opps_count)
                state["opponents"][seat_key] = {
                    "name": f"Player_{i+1}",
                    "stack": 1880.0 if is_active else 0.0,
                    "is_active": is_active,
                    "state": "Active" if is_active else "Folded"
                }
                
            ev_f, ev_c, ev_r = engine._get_q_values(state)
            evs = {"FOLD": ev_f, "CALL": ev_c, "RAISE": ev_r}
            best_act = max(evs, key=evs.get)
            print(f"| {hand_name} | {opps_count} | {equity:.2f} | {ev_f:.2f} | {ev_c:.2f} | {ev_r:.2f} | **{best_act}** |")

if __name__ == '__main__':
    run_sweep()

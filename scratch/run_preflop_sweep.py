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
    
    hands = {
        "Pocket Aces (AA)": (["Ah", "As"], [0.85, 0.73, 0.64, 0.57, 0.50]),
        "72 Offsuit (72o)": (["2c", "7d"], [0.35, 0.23, 0.17, 0.13, 0.11])
    }
    
    # 1. Sweep 1: Positions (Dealer Button moves through all positions)
    # Positions: 0 (BU), 1 (SB), 2 (BB), 3 (UTG), 4 (MP), 5 (CO)
    # Active opponents: 3 (so 3 opponents + Hero = 4 active players)
    # Action faced: Unopened pot (call_amount=0, pot_size=30)
    print("## SWEEP 1: Table Positions (Unopened Pot, 3 Active Opponents)\n")
    print(f"| Hand | Position | GTO Pos | EV (Fold) | EV (Call) | EV (Raise) | Best Action |")
    print(f"| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    
    pos_names = {0: "Button", 1: "Small Blind", 2: "Big Blind", 3: "UTG", 4: "Middle Pos", 5: "Cut Off"}
    
    for hand_name, (cards, equities) in hands.items():
        equity = equities[2] # 3 opponents -> index 2 (active opponents = 3)
        for pos in range(6):
            state = {
                "community_cards": [],
                "hero_cards": cards,
                "pot_size": 30.0,
                "hero_stack": 2000.0, # 100 BB stack
                "opponents": {},
                "num_active_players": 4,
                "active_buttons": ["button_fold", "button_call", "button_raise"],
                "street": "Preflop",
                "dealer_idx": (0 - pos) % 6,
                "hero_position": pos,
                "action_history": [],
                "opp_vpip_norm": 0.3,
                "opp_agg_norm": 0.4,
                "big_blind": 20.0,
                "call_amount": 0.0,
                "equity": equity
            }
            
            # Setup 5 opponents, first 3 active, last 2 folded
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
            print(f"| {hand_name} | {pos_names[pos]} | {pos} | {ev_f:.2f} | {ev_c:.2f} | {ev_r:.2f} | **{best_act}** |")
            
    # 2. Sweep 2: Action Faced (Check, Call, 3BB Raise, Shove)
    # Position: Button (pos=0), Active opponents: 3
    print("\n## SWEEP 2: Actions Faced (Hero on Button, 3 Active Opponents)\n")
    print(f"| Hand | Action Faced | Pot Size | Call Amt | EV (Fold) | EV (Call) | EV (Raise) | Best Action |")
    print(f"| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    actions = [
        ("Check (Unopened)", 30.0, 0.0),
        ("Limp (Call 1BB)", 50.0, 20.0),
        ("Raise (3BB)", 90.0, 60.0),
        ("All-In (23.5BB)", 510.0, 470.0)
    ]
    
    for hand_name, (cards, equities) in hands.items():
        equity = equities[2]
        for act_name, pot_sz, call_amt in actions:
            state = {
                "community_cards": [],
                "hero_cards": cards,
                "pot_size": pot_sz,
                "hero_stack": 2000.0,
                "opponents": {},
                "num_active_players": 4,
                "active_buttons": ["button_fold", "button_call", "button_raise"],
                "street": "Preflop",
                "dealer_idx": 0,
                "hero_position": 0,
                "action_history": [] if call_amt == 0 else ["r"] if call_amt > 20 else ["c"],
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
                    "stack": 2000.0 if is_active else 0.0,
                    "is_active": is_active,
                    "state": "Active" if is_active else "Folded"
                }
                
            ev_f, ev_c, ev_r = engine._get_q_values(state)
            evs = {"FOLD": ev_f, "CALL": ev_c, "RAISE": ev_r}
            best_act = max(evs, key=evs.get)
            print(f"| {hand_name} | {act_name} | {pot_sz:.0f} | {call_amt:.0f} | {ev_f:.2f} | {ev_c:.2f} | {ev_r:.2f} | **{best_act}** |")

    # 3. Sweep 3: Active Opponents (1 to 5)
    # Position: Button (pos=0), Action: 3BB Raise faced (call_amt=60, pot_size=90)
    print("\n## SWEEP 3: Number of Active Opponents (Hero on Button, Facing 3BB Raise)\n")
    print(f"| Hand | Opponents | Equity | EV (Fold) | EV (Call) | EV (Raise) | Best Action |")
    print(f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for hand_name, (cards, equities) in hands.items():
        for opps_count in range(1, 6):
            equity = equities[opps_count - 1]
            state = {
                "community_cards": [],
                "hero_cards": cards,
                "pot_size": 90.0,
                "hero_stack": 2000.0,
                "opponents": {},
                "num_active_players": opps_count + 1,
                "active_buttons": ["button_fold", "button_call", "button_raise"],
                "street": "Preflop",
                "dealer_idx": 0,
                "hero_position": 0,
                "action_history": ["r"],
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
                    "stack": 2000.0 if is_active else 0.0,
                    "is_active": is_active,
                    "state": "Active" if is_active else "Folded"
                }
                
            ev_f, ev_c, ev_r = engine._get_q_values(state)
            evs = {"FOLD": ev_f, "CALL": ev_c, "RAISE": ev_r}
            best_act = max(evs, key=evs.get)
            print(f"| {hand_name} | {opps_count} | {equity:.2f} | {ev_f:.2f} | {ev_c:.2f} | {ev_r:.2f} | **{best_act}** |")

if __name__ == '__main__':
    run_sweep()

import torch
import sys
import os

# Add project root to sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.pluribus_engine import PluribusEngine

def run_hud_shove_sweep():
    # Load V4 engine
    engine = PluribusEngine(expert_name="expert_v4_selfplay.pth", model_type="v4")
    
    # Configuration:
    # - Hero Hand: 72o (7s 2c)
    # - Board cards for postflop streets:
    #   - Flop: As Td 4c
    #   - Turn: As Td 4c Kh
    #   - River: As Td 4c Kh 9d
    # - Facing a 23.5 BB all-in (pot_size = 510, call_amount = 470)
    # - Opponent 1 (Seat 1) is the one who went all-in (stack = 0, is_active = True)
    
    streets = ["Preflop", "Flop", "Turn", "River"]
    boards = {
        "Preflop": [],
        "Flop": ["As", "Td", "4c"],
        "Turn": ["As", "Td", "4c", "Kh"],
        "River": ["As", "Td", "4c", "Kh", "9d"]
    }
    
    opp_profiles = {
        "Tight / Nit (Blue VPIP, Blue AGG)": {"vpip_color": "Blue", "agg_color": "Blue"},
        "Maniac (Red VPIP, Red AGG)": {"vpip_color": "Red", "agg_color": "Red"}
    }
    
    print("## SWEEP: EV Sensitivity to Opponent HUD Profile Facing Shove with 72o\n")
    print(f"| Street | Opponent Type | EV (Fold) | EV (Call) | EV (Raise) | Best Action |")
    print(f"| :--- | :--- | :---: | :---: | :---: | :---: |")
    
    for street in streets:
        board = boards[street]
        for name, profile in opp_profiles.items():
            state = {
                "community_cards": board,
                "hero_cards": ["7s", "2c"],
                "pot_size": 510.0,
                "hero_stack": 690.0,
                "opponents": {},
                "num_active_players": 2,
                "active_buttons": ["button_fold", "button_call", "button_raise"],
                "street": street,
                "dealer_idx": 0,
                "hero_position": 0,
                "action_history": ["r"],
                "opp_vpip_norm": 0.3,
                "opp_agg_norm": 0.4,
                "big_blind": 20.0,
                "call_amount": 470.0,
                "equity": 0.05 # dummy
            }
            
            # Setup 5 opponents. Seat 1 is the one who went all-in.
            # Seat 1 is active, stack=0 (due to going all-in)
            state["opponents"]["seat_1"] = {
                "name": "Opponent_1",
                "stack": 0.0,
                "is_active": True,
                "state": "Active",
                "vpip_color": profile["vpip_color"],
                "agg_color": profile["agg_color"]
            }
            # Rest are folded
            for i in range(1, 5):
                seat_key = f"seat_{i+1}"
                state["opponents"][seat_key] = {
                    "name": f"Player_{i+1}",
                    "stack": 0.0,
                    "is_active": False,
                    "state": "Folded"
                }
                
            ev_f, ev_c, ev_r = engine._get_q_values(state)
            evs = {"FOLD": ev_f, "CALL": ev_c, "RAISE": ev_r}
            best_act = max(evs, key=evs.get)
            print(f"| {street} | {name} | {ev_f:.2f} | {ev_c:.2f} | {ev_r:.2f} | **{best_act}** |")

if __name__ == '__main__':
    run_hud_shove_sweep()

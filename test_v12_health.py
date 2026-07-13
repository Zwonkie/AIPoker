import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from versions.v12.core.model import PokerEVModelV4
from versions.v12.core.contract import ContractV12
from core.board_state import BoardState, SeatState, HUDStats

def create_mock_board_state(street, hero_cards, community_cards, pot, call, stack, equity, num_opps, opp_profile="Standard"):
    seats = {}
    
    # Opponents
    for i in range(1, num_opps + 1):
        vpip_color, agg_color = 'Green', 'Green'
        if opp_profile == "Nit":
            vpip_color, agg_color = 'Blue', 'Blue'
        elif opp_profile == "Maniac":
            vpip_color, agg_color = 'Red', 'Red'
        elif opp_profile == "Calling Station":
            vpip_color, agg_color = 'Yellow', 'Blue'
            
        seats[f'seat_{i}'] = SeatState(
            is_active=True,
            stack=stack,
            hud=HUDStats(vpip_color=vpip_color, agg_color=agg_color)
        )
        
    return BoardState(
        hero_cards=hero_cards,
        community_cards=community_cards,
        pot_size=pot,
        hero_stack=stack,
        call_amount=call,
        seats=seats,
        big_blind=10.0,
        street=street,
        equity=equity,
        hero_position=0
    )

def run_diagnostics(checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("==================================================")
    print(f"           MODEL HEALTH DIAGNOSTIC TOOL (V12)   ")
    print(f"           Target: {checkpoint_path}                 ")
    print("==================================================")
    
    model = PokerEVModelV4().to(device)
    from versions.v12.core.manifest import MANIFEST
    from shared.manifest import load_state_dict
    state_dict = load_state_dict(checkpoint_path, MANIFEST)
    model.load_state_dict(state_dict)
    model.eval()
    
    bridge = ContractV12(max_seq_len=20)
    
    scenarios = [
        ("River Pure Air (First to Act)", "River", ['2h', '3d'], ['As', 'Ks', 'Qs', 'Js', '9c'], 100.0, 0.0, 1000.0, 0.0, 1, "Standard"),
        ("River Pure Air (Facing Bet)", "River", ['2h', '3d'], ['As', 'Ks', 'Qs', 'Js', '9c'], 150.0, 50.0, 1000.0, 0.0, 1, "Standard"),
        ("River The Nuts (Facing Bet)", "River", ['Ts', 'Th'], ['As', 'Ks', 'Qs', 'Js', '9c'], 150.0, 50.0, 1000.0, 1.0, 1, "Standard"),
        ("River The Nuts (Calling Station)", "River", ['Ts', 'Th'], ['As', 'Ks', 'Qs', 'Js', '9c'], 100.0, 0.0, 1000.0, 1.0, 1, "Calling Station"),
        ("Preflop AA vs Nit (Deep)", "Preflop", ['Ah', 'Ad'], [], 15.0, 10.0, 4000.0, 0.85, 1, "Nit"),
        ("Preflop AA vs Maniac", "Preflop", ['Ah', 'Ad'], [], 15.0, 10.0, 1000.0, 0.85, 1, "Maniac"),
        ("Flop TPTK Multi-Way (4-way pot)", "Flop", ['As', 'Kh'], ['Ks', '7d', '2c'], 100.0, 0.0, 1000.0, 0.50, 3, "Standard"),
        ("Turn Flush Draw vs Bet", "Turn", ['Jh', 'Th'], ['9h', '8h', '2c', '4s'], 150.0, 50.0, 1000.0, 0.35, 1, "Standard"),
    ]
    
    print(f"{'Scenario'.ljust(35)} | {'Prob (Fold)'.ljust(15)} | {'Prob (Call)'.ljust(15)} | {'Prob (Raise)'.ljust(15)} | {'Final Action'}")
    print("-" * 115)
    
    def get_action(bs):
        h_t, b_t, c_t, a_t = bridge.to_tensors(bs, hero_actions=[6])
        with torch.no_grad():
            preds = model(h_t.to(device), b_t.to(device), c_t.to(device), a_t.to(device))
            # Extract logits and apply softmax to get probabilities
            logits = preds['policy_logits'].squeeze(0)[0]
            probs = torch.softmax(logits, dim=-1)
        
        p_f, p_c, p_r = probs[0].item(), probs[1].item(), probs[2].item()
        
        # Use simple greedy max over actor probs for "Raw EV" printout to match format
        ev_dict = {'FOLD': p_f, 'CALL': p_c, 'RAISE': p_r}
        action = "RAISE" if p_r >= p_c and p_r >= p_f else "CALL" if p_c >= p_f else "FOLD"
        return action, p_f, p_c, p_r

    for name, street, hand, board, pot, call, stack, eq, opps, prof in scenarios:
        bs = create_mock_board_state(street, hand, board, pot, call, stack, eq, opps, prof)
        action, p_f, p_c, p_r = get_action(bs)
        print(f"{name.ljust(35)} | {p_f:<15.2f} | {p_c:<15.2f} | {p_r:<15.2f} | {action}")

    print("\n==================================================")
    print("           PREFLOP EQUITY SWEEP                 ")
    print("==================================================")
    print(f"{'Eq Group'.ljust(15)} | {'Opps'.ljust(5)} | {'Prob (Fold)'.ljust(15)} | {'Prob (Call)'.ljust(15)} | {'Prob (Raise)'.ljust(15)} | {'Final Action'}")
    print("-" * 85)
    
    eq_groups = [
        ("<20% (Air)", 0.10, ['7d', '2c']),
        ("20-40% (Weak)", 0.30, ['Jc', '3d']),
        ("40-60% (Marg)", 0.50, ['9s', '8s']),
        ("60-80% (Strg)", 0.70, ['As', 'Qd']),
        (">80% (Nuts)", 0.90, ['Ah', 'As'])
    ]
    
    for opps in [1, 3, 5]:
        for label, eq, hand in eq_groups:
            bs = create_mock_board_state("Preflop", hand, [], 15.0, 10.0, 1000.0, eq, opps, "Standard")
            action, p_f, p_c, p_r = get_action(bs)
            print(f"{label.ljust(15)} | {opps:<5} | {p_f:<15.2f} | {p_c:<15.2f} | {p_r:<15.2f} | {action}")
        print("-" * 85)

    print("\n[Analysis Complete]")

if __name__ == '__main__':
    run_diagnostics("versions/v12/weights/expert_main.pth")

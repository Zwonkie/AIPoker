import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.decision import PokerDecisionEngine
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
        equity=equity
    )

def run_diagnostics(model_name='Pluribus (v9 Main)'):
    print("==================================================")
    print(f"           MODEL HEALTH DIAGNOSTIC TOOL         ")
    print(f"           Target: {model_name}                 ")
    print("==================================================")
    
    engine = PokerDecisionEngine()
    engine.set_active_model(model_name)
    
    scenarios = [
        # (Name, Street, Hero Cards, Board, Pot, Call, Stack, Equity, Opps, Opp Profile)
        ("River Pure Air (First to Act)", "River", ['2h', '3d'], ['As', 'Ks', 'Qs', 'Js', '9c'], 100.0, 0.0, 1000.0, 0.0, 1, "Standard"),
        ("River Pure Air (Facing Bet)", "River", ['2h', '3d'], ['As', 'Ks', 'Qs', 'Js', '9c'], 150.0, 50.0, 1000.0, 0.0, 1, "Standard"),
        ("River The Nuts (Facing Bet)", "River", ['Ts', 'Th'], ['As', 'Ks', 'Qs', 'Js', '9c'], 150.0, 50.0, 1000.0, 1.0, 1, "Standard"),
        ("River The Nuts (Calling Station)", "River", ['Ts', 'Th'], ['As', 'Ks', 'Qs', 'Js', '9c'], 100.0, 0.0, 1000.0, 1.0, 1, "Calling Station"),
        ("Preflop AA vs Nit (Deep)", "Preflop", ['Ah', 'Ad'], [], 15.0, 10.0, 4000.0, 0.85, 1, "Nit"),
        ("Preflop AA vs Maniac", "Preflop", ['Ah', 'Ad'], [], 15.0, 10.0, 1000.0, 0.85, 1, "Maniac"),
        ("Flop TPTK Multi-Way (4-way pot)", "Flop", ['As', 'Kh'], ['Ks', '7d', '2c'], 100.0, 0.0, 1000.0, 0.50, 3, "Standard"),
        ("Turn Flush Draw vs Bet", "Turn", ['Jh', 'Th'], ['9h', '8h', '2c', '4s'], 150.0, 50.0, 1000.0, 0.35, 1, "Standard"),
    ]
    
    print(f"{'Scenario'.ljust(35)} | {'Raw EV (Fold)'.ljust(15)} | {'Raw EV (Call)'.ljust(15)} | {'Raw EV (Raise)'.ljust(15)} | {'Final Action'}")
    print("-" * 115)
    
    for name, street, hand, board, pot, call, stack, eq, opps, prof in scenarios:
        bs = create_mock_board_state(street, hand, board, pot, call, stack, eq, opps, prof)
        
        # Turn OFF math guardrail so we can see the pure model output behavior
        action, reason, bet_size, ev_dict = engine.make_decision(
            bs, use_math_engine=False
        )
        
        ev_f = ev_dict.get('FOLD', 0.0)
        ev_c = ev_dict.get('CALL', 0.0)
        ev_r = ev_dict.get('RAISE', 0.0)
        
        print(f"{name.ljust(35)} | {ev_f:<15.2f} | {ev_c:<15.2f} | {ev_r:<15.2f} | {action}")

    print("\n[Analysis Complete]")

if __name__ == '__main__':
    target = 'Pluribus (v9 Main)'
    if len(sys.argv) > 1:
        target = sys.argv[1]
    run_diagnostics(target)

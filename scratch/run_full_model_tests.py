import sys
import os
sys.path.append(os.getcwd())

from core.decision import PokerDecisionEngine
from core.board_state import BoardState, SeatState, HUDStats

def create_mock_board_state(street, hero_cards, community_cards, pot, call, stack, equity, num_opps, opp_profile="Standard"):
    seats = {}
    
    for i in range(1, num_opps + 1):
        vpip_color, agg_color = 'Green', 'Green'
        if opp_profile == "Nit":
            vpip_color, agg_color = 'Blue', 'Blue'
        elif opp_profile == "Maniac":
            vpip_color, agg_color = 'Red', 'Red'
        elif opp_profile == "Calling Station" or opp_profile == "Fish":
            vpip_color, agg_color = 'Yellow', 'Blue'
        elif opp_profile == "TAG":
            vpip_color, agg_color = 'Green', 'Green'
            
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

def run_tests():
    engine = PokerDecisionEngine()
    engine.set_active_model('herocules_v11_fuzzyHeuristicsOpp.pth')
    
    print("==================================================")
    print("   SCENARIO A: 72o Facing Shove (Extrapolation)   ")
    print("==================================================")
    streets = [("Preflop", []), ("Flop", ['Th', '7c', '2d']), ("Turn", ['Th', '7c', '2d', '5s']), ("River", ['Th', '7c', '2d', '5s', '9c'])]
    print(f"{'Street'.ljust(10)} | {'Opp Profile'.ljust(15)} | {'EV(Fold)'.ljust(10)} | {'EV(Call)'.ljust(10)} | {'EV(Raise)'.ljust(10)}")
    print("-" * 65)
    for street, board in streets:
        for profile in ["Nit", "Maniac"]:
            bs = create_mock_board_state(street, ['7d', '2s'], board, 510.0, 470.0, 1000.0, 0.05, 1, profile)
            action, reason, bet_size, ev_dict = engine.make_decision(bs, use_math_engine=False)
            print(f"{street.ljust(10)} | {profile.ljust(15)} | {ev_dict.get('FOLD', 0.0):<10.2f} | {ev_dict.get('CALL', 0.0):<10.2f} | {ev_dict.get('RAISE', 0.0):<10.2f}")
            
    print("\n==================================================")
    print(" SCENARIO B: Opponent Personality Exploitation    ")
    print("==================================================")
    hands = [
        ("Strong (KK)", ['Ks', 'Kd'], 0.80),
        ("Medium (98s)", ['9s', '8s'], 0.40),
        ("Weak (J3o)", ['Jc', '3d'], 0.15)
    ]
    for label, hand, eq in hands:
        print(f"\n--- {label} ---")
        print(f"{'Opp Profile'.ljust(15)} | {'EV(Fold)'.ljust(10)} | {'EV(Call)'.ljust(10)} | {'EV(Raise)'.ljust(10)}")
        print("-" * 55)
        for profile in ["Nit", "TAG", "Fish", "Maniac"]:
            bs = create_mock_board_state("Flop", hand, ['Th', '7c', '2d'], 50.0, 10.0, 1000.0, eq, 1, profile)
            action, reason, bet_size, ev_dict = engine.make_decision(bs, use_math_engine=False)
            print(f"{profile.ljust(15)} | {ev_dict.get('FOLD', 0.0):<10.2f} | {ev_dict.get('CALL', 0.0):<10.2f} | {ev_dict.get('RAISE', 0.0):<10.2f}")

    print("\n==================================================")
    print(" SCENARIO C: Pot & Stack Size Scaling (Geometry)  ")
    print("==================================================")
    print("--- Pot Sweep (Stack=1000) ---")
    print(f"{'Pot (BB)'.ljust(10)} | {'EV(Fold)'.ljust(10)} | {'EV(Call)'.ljust(10)} | {'EV(Raise)'.ljust(10)}")
    for pot_bb in [10, 50, 100, 150]:
        bs = create_mock_board_state("Flop", ['Ac', 'Kc'], ['Ah', 'Qs', '5d'], pot_bb * 10.0, 10.0, 1000.0, 0.85, 1, "Standard")
        action, reason, bet_size, ev_dict = engine.make_decision(bs, use_math_engine=False)
        print(f"{str(pot_bb).ljust(10)} | {ev_dict.get('FOLD', 0.0):<10.2f} | {ev_dict.get('CALL', 0.0):<10.2f} | {ev_dict.get('RAISE', 0.0):<10.2f}")
        
    print("\n--- Stack Sweep (Pot=500, wait, Pot=50BB) ---")
    print(f"{'Stack (BB)'.ljust(10)} | {'EV(Fold)'.ljust(10)} | {'EV(Call)'.ljust(10)} | {'EV(Raise)'.ljust(10)}")
    for stack_bb in [10, 50, 100, 200]:
        bs = create_mock_board_state("Flop", ['Ac', 'Kc'], ['Ah', 'Qs', '5d'], 500.0, 10.0, stack_bb * 10.0, 0.85, 1, "Standard")
        action, reason, bet_size, ev_dict = engine.make_decision(bs, use_math_engine=False)
        print(f"{str(stack_bb).ljust(10)} | {ev_dict.get('FOLD', 0.0):<10.2f} | {ev_dict.get('CALL', 0.0):<10.2f} | {ev_dict.get('RAISE', 0.0):<10.2f}")

    print("\n==================================================")
    print(" SCENARIO D: Active Opponents Sensitivity (Multi-Way)")
    print("==================================================")
    print(f"{'Opps'.ljust(10)} | {'EV(Fold)'.ljust(10)} | {'EV(Call)'.ljust(10)} | {'EV(Raise)'.ljust(10)}")
    print("-" * 45)
    for opps in [1, 2, 3, 4, 5]:
        # Equity roughly decreases as active opponents increase
        eq = 0.80 - (opps * 0.10)
        eq = max(eq, 0.15)
        bs = create_mock_board_state("Flop", ['Js', 'Jh'], ['9d', '5c', '2h'], 100.0, 20.0, 1000.0, eq, opps, "Standard")
        action, reason, bet_size, ev_dict = engine.make_decision(bs, use_math_engine=False)
        print(f"{str(opps).ljust(10)} | {ev_dict.get('FOLD', 0.0):<10.2f} | {ev_dict.get('CALL', 0.0):<10.2f} | {ev_dict.get('RAISE', 0.0):<10.2f}")

if __name__ == '__main__':
    run_tests()

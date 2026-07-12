import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from core.models.pluribus_engine import PluribusEngine

def run_analysis(model_type='original'):
    print(f"# Standard Sensitivity Analysis Results ({model_type.upper()})\n")
    print(f"Evaluating `Pluribus ({model_type})`...\n")
    
    expert_map = {
        'original': ('expert_nlh_combined.pth', 'original'),
        'policy': ('expert_policy.pth', 'policy'),
        'ev': ('expert_ev_v2.pth', 'ev'),
        'v3': ('expert_v3_selfplay.pth', 'v3')
    }
    expert_file, m_type = expert_map.get(model_type, ('expert_nlh_combined.pth', 'original'))
    
    try:
        model = PluribusEngine(game_type='NLH', expert_name=expert_file, model_type=m_type)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    def eval_scenario(equity, pot_odds, spr, num_opps, is_preflop):
        # Calculate derived values
        pot_size = 2.0 if is_preflop else 20.0
        if pot_odds >= 1.0:
            call_amount = pot_size * 2
        else:
            call_amount = (pot_odds * pot_size) / (1.0 - pot_odds) if pot_odds > 0 else 0.0
            
        hero_stack = spr * pot_size
        
        hand = ['Ac', 'Ks']
        board = [] if is_preflop else ['2d', '7h', '8c']
        active_opps = [{'name': f'Opp{i}', 'stack': hero_stack, 'is_active': True} for i in range(num_opps)]
        
        ts_dict = {
            'action_history': [],
            'big_blind': 1.0,
            'call_amount': call_amount,
            'num_opponents': num_opps
        }
        
        check_call_avail = True
        bet_raise_avail = True
        
        _, _, _, ev_dict = model.predict_action(
            board, hand, equity, pot_size, call_amount, hero_stack, num_opps,
            is_preflop, False, True, True, True, bet_raise_avail, check_call_avail, active_opps, ts_dict
        )
        return ev_dict

    def print_res(res, label=""):
        pf = res.get('prob_fold', 0) * 100
        pc = res.get('prob_call', 0) * 100
        pr = res.get('prob_raise', 0) * 100
        rf = res.get('raw_fold', 0)
        rc = res.get('raw_call', 0)
        rr = res.get('raw_raise', 0)
        print(f"| {label} | {pf:.1f}% | {pc:.1f}% | {pr:.1f}% | {rf:.2f} | {rc:.2f} | {rr:.2f} |")

    def print_header():
        print("| Configuration | FOLD | CALL/CHECK | RAISE/BET | EV(Fold) | EV(Call) | EV(Raise) |")
        print("|---|---|---|---|---|---|---|")

    # A: Pure Value Bet
    print("## Situation A: The Pure Value Bet (Monster Hand)")
    print("*Post-flop, Eq 85%, SPR 10.0, Opp 1, Pot Odds 0% (Checked to Hero)*\n")
    print_header()
    res = eval_scenario(equity=0.85, pot_odds=0.0, spr=10.0, num_opps=1, is_preflop=False)
    print_res(res, "Base")
    print("\n")

    # B: Mathematical Draw
    print("## Situation B: The Mathematical Draw (Pot Odds Test)")
    print("*Post-flop, Eq 35%, SPR 10.0, Opp 1*\n")
    print_header()
    for po in [0.10, 0.25, 0.33, 0.50]:
        res = eval_scenario(equity=0.35, pot_odds=po, spr=10.0, num_opps=1, is_preflop=False)
        print_res(res, f"Pot Odds {po*100:.0f}%")
    print("\n")

    # C: Short-Stacked Commitment
    print("## Situation C: Short-Stacked Commitment")
    print("*Post-flop, Eq 60%, SPR 1.0, Opp 1*\n")
    print_header()
    for po in [0.0, 0.33]:
        res = eval_scenario(equity=0.60, pot_odds=po, spr=1.0, num_opps=1, is_preflop=False)
        print_res(res, f"Pot Odds {po*100:.0f}%")
    print("\n")

    # D: Deep-Stacked Multi-way Caution
    print("## Situation D: Deep-Stacked Multi-way Caution")
    print("*Post-flop, Eq 50%, SPR 15.0, Opp 4*\n")
    print_header()
    res = eval_scenario(equity=0.50, pot_odds=0.25, spr=15.0, num_opps=4, is_preflop=False)
    print_res(res, "Pot Odds 25%")
    print("\n")

    # E: Pure Air / Bluff Opportunity
    print("## Situation E: The Pure Air / Bluff Opportunity")
    print("*Post-flop, Eq 15%, SPR 5.0, Opp 1*\n")
    print_header()
    for po in [0.0, 0.25]:
        res = eval_scenario(equity=0.15, pot_odds=po, spr=5.0, num_opps=1, is_preflop=False)
        print_res(res, f"Pot Odds {po*100:.0f}%")
    print("\n")

    # F: Pre-Flop Marginal Defend
    print("## Situation F: Pre-Flop Marginal Defend")
    print("*Pre-flop, Eq 45%, SPR 50.0, Opp 1*\n")
    print_header()
    res = eval_scenario(equity=0.45, pot_odds=0.20, spr=50.0, num_opps=1, is_preflop=True)
    print_res(res, "Pot Odds 20%")
    print("\n")

    # G: Stack-to-BB Ratio Sensitivity (Stack Depth Test)
    print("## Situation G: Stack-to-BB Ratio Sensitivity (Stack Depth Test)")
    print("*Post-flop, Eq 55%, Pot 20.0 BB, facing 10.0 BB bet (Pot Odds 25%), Opp 1*")
    print("*Varying Hero Stack from 10 BB (Committed) to 400 BB (Deep)*\n")
    print_header()
    for stack_bb in [10, 40, 100, 250, 400]:
        res = eval_scenario(equity=0.55, pot_odds=0.25, spr=stack_bb/20.0, num_opps=1, is_preflop=False)
        print_res(res, f"Stack = {stack_bb} BB")
    print("\n")

    # H: 2D Stack-to-BB vs. Equity Sensitivity (Implied Leverage Test)
    print("## Situation H: 2D Stack-to-BB vs. Equity Sensitivity (Implied Leverage Test)")
    print("*Post-flop, Pot 20.0 BB, facing 10.0 BB bet (Pot Odds 25%), Opp 1*")
    print("*Evaluating 2D Grid: Stacks [10, 40, 100] BB vs Equities [30%, 50%, 70%]*\n")
    for eq in [0.30, 0.50, 0.70]:
        print(f"### Equity = {eq*100:.0f}%")
        print_header()
        for stack_bb in [10, 40, 100]:
            res = eval_scenario(equity=eq, pot_odds=0.25, spr=stack_bb/20.0, num_opps=1, is_preflop=False)
            print_res(res, f"Stack = {stack_bb} BB")
        print("\n")

    # I: Pre-Flop Monster Hand (Premium Squeezing/Raising)
    print("## Situation I: Pre-Flop Monster Hand (Premium Squeezing/Raising)")
    print("*Pre-flop, Eq 85%, SPR 50.0, Opp 1*")
    print("*Evaluating raw Pluribus network output on pocket Aces pre-flop (facing different pot odds)*\n")
    print_header()
    for po in [0.0, 0.10, 0.25, 0.50]:
        res = eval_scenario(equity=0.85, pot_odds=po, spr=50.0, num_opps=1, is_preflop=True)
        print_res(res, f"Pot Odds {po*100:.0f}%")
    print("\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='original', choices=['original', 'policy', 'ev', 'v3'])
    args = parser.parse_args()
    run_analysis(args.model_type)

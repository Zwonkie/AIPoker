import os
import sys
import torch
import math

sys.path.insert(0, os.path.abspath('.'))
from core.decision import PokerDecisionEngine

def run_sweep():
    # Load Decision Engine and set active model
    try:
        engine = PokerDecisionEngine(game_type='nlh')
        engine.set_active_model('Pluribus (EV v2 7-Feature)')
    except Exception as e:
        print(f"Failed to load engine: {e}")
        return

    # Post-flop parameters
    pot_base = 10.0 # 10 BB pot size
    hero_stack = 100.0 # 100 BB stack
    num_opps = 1
    is_preflop = False
    
    # Facing bet sizes (in BB)
    bet_scenarios = {
        'A (1 BB bet)': 1.0,
        'B (3 BB bet)': 3.0,
        'C (10 BB bet)': 10.0,
        'D (50 BB bet)': 50.0
    }
    
    # Equity levels to sweep
    equity_levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    
    report = []
    report.append("# Sweep Analysis: Facing Various Bet Sizes Post-Flop")
    report.append("Evaluating the **Pluribus (EV v2 7-Feature)** model facing different bet sizes across a range of equities.")
    report.append("### Baseline Scenario:")
    report.append(f"* **Pot Size**: {pot_base} BB")
    report.append(f"* **Hero Stack**: {hero_stack} BB")
    report.append(f"* **Street**: Flop (Street Level = 0.33)")
    report.append(f"* **Opponents**: {num_opps} active opponent\n")
    
    for name, call_amount in bet_scenarios.items():
        pot_odds = call_amount / (pot_base + call_amount)
        report.append(f"## Scenario {name}")
        report.append(f"Facing a bet of **{call_amount} BB** (Pot Size = {pot_base} BB, Pot Odds = **{pot_odds*100:.1f}%**)\n")
        report.append("| Equity | FOLD Prob | CALL Prob | RAISE Prob | EV(Fold) | EV(Call) | EV(Raise) | Chosen Action |")
        report.append("|---|---|---|---|---|---|---|---|")
        
        for eq in equity_levels:
            # Build state dict
            ts_dict = {
                'action_history': [],
                'big_blind': 1.0,
                'call_amount': call_amount,
                'num_opponents': num_opps,
                'community_cards': ['2d', '7h', '8c'],
                'hero_cards': ['Ac', 'Ks']
            }
            
            # Predict
            best_action, reason, bet_size, ev_dict = engine.make_decision(
                board=['2d', '7h', '8c'],
                hand=['Ac', 'Ks'],
                equity=eq,
                pot_size=pot_base,
                call_amount=call_amount,
                hero_stack=hero_stack,
                num_opponents=num_opps,
                is_preflop=is_preflop,
                use_preflop_chart=False,
                use_math_engine=True,
                use_bluff_engine=True,
                use_dynamic_sizing=True,
                bet_raise_available=True,
                check_call_available=True,
                active_opponents=[{'name': 'Opp1', 'stack': hero_stack, 'is_active': True}],
                table_state_dict=ts_dict
            )
            
            pf = ev_dict.get('prob_fold', 0) * 100
            pc = ev_dict.get('prob_call', 0) * 100
            pr = ev_dict.get('prob_raise', 0) * 100
            
            ev_f = ev_dict.get('raw_fold', 0)
            ev_c = ev_dict.get('raw_call', 0)
            ev_r = ev_dict.get('raw_raise', 0)
            
            chosen = best_action
            if 'RAISE_SLIDER' in chosen:
                chosen = f"RAISE ({bet_size:.1f} BB)"
                
            report.append(f"| {eq*100:.0f}% | {pf:.1f}% | {pc:.1f}% | {pr:.1f}% | {ev_f:.2f} | {ev_c:.2f} | {ev_r:.2f} | **{chosen}** |")
        report.append("\n")
        
    os.makedirs('C:/Users/zwonk/.gemini/antigravity-ide/brain/c68a647c-2540-4757-8bda-15d48c1088de', exist_ok=True)
    with open('C:/Users/zwonk/.gemini/antigravity-ide/brain/c68a647c-2540-4757-8bda-15d48c1088de/sweep_pot_odds_analysis.md', 'w') as f:
        f.write("\n".join(report))
        
    print("Sweep complete! Saved to C:/Users/zwonk/.gemini/antigravity-ide/brain/c68a647c-2540-4757-8bda-15d48c1088de/sweep_pot_odds_analysis.md")

if __name__ == '__main__':
    run_sweep()

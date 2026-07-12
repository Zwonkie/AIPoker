import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from core.models.heuristic import HeuristicEngine

def run_simulation():
    model = HeuristicEngine()
    
    # Test equities representing different regions:
    # 0.35 (Weak), 0.42 (Marginal), 0.47 (Playable), 0.52 (Middle-Playable), 0.58 (High-Playable), 0.65 (Premium), 0.75 (Strong Premium)
    test_equities = [0.35, 0.42, 0.47, 0.52, 0.58, 0.65, 0.75]
    
    print("=" * 70)
    print("PRE-FLOP DETERMINISTIC EXACT ACTION PROBABILITIES (GTO Variance Disabled)")
    print("Active Opponents: 1 (Heads-up) | Looseness Offset: +0.0%")
    print("=" * 70)
    
    for call_amt in [0.0, 20.0]:
        scenario = "UNOPENED POT (Call Amount: 0)" if call_amt == 0 else f"FACING A 3 BB BET (Call Amount: {call_amt})"
        print(f"\nScenario: {scenario}")
        print("-" * 70)
        print(f"{'Equity':<8} | {'RAISE %':<15} | {'CALL/CHECK %':<15} | {'FOLD %':<15}")
        print("-" * 70)
        
        for eq in test_equities:
            p_raise, p_call, p_fold = model.get_preflop_probabilities(
                equity=eq,
                call_amount=call_amt,
                num_opponents=1,
                preflop_looseness=0.0
            )
            
            print(f"{eq:<8.1%} | {p_raise:<15.1%} | {p_call:<15.1%} | {p_fold:<15.1%}")

if __name__ == '__main__':
    run_simulation()

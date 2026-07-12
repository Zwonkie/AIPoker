# V9 River Bluffing Collapse

**Date Recorded**: 2026-07-12
**Related Files**: 
- [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v9/six_max_simulator.py)
- [decision.py](file:///c:/REPO/Antigravity/AIPoker/core/decision.py)

## Context
During live play testing of the V9 model (`expert_v9_main.pth`), it was observed that the model plays conservatively and correctly until the River. On the River, the model heavily over-bluffs by raising or going all-in, even with 0.0 equity (pure air), and continues to jam even when facing a re-raise. 

This behavior is a classic Reinforcement Learning artifact known as **Terminal State Value Inflation** or a **Bluffing Collapse**. 

The root cause was isolated to the `_calculate_analytical_target_evs` function in `six_max_simulator.py`. The formula used to estimate opponent fold probability (`p_fold_single = 0.70 * fold_factor`) was designed for *preflop* profiles but was applied indiscriminately across all streets. Because River pots are large, the model mathematically assumed a constant baseline fold equity (e.g., ~35% on pot-sized shoves). The neural network learned that blindly shoving the river maximized its analytical Target EV because the inflated fold equity outweighed the loss of the bluff.

## Resolution / Guidelines
Since V9 is already trained with this behavior baked into its weights, there are two paths forward:

1. **Short-Term Fix (Live Play Guardrail)**: 
   Implement a "Math Override" in `core/decision.py` specifically for V9. If `street == 'River'` (or postflop streets) and `equity < 0.35`, programmatically intercept the Neural Network and override any "Raise" or "All-in" decisions to a "Check" or "Fold".
   
2. **Long-Term Fix (V10 Training)**:
   Rewrite `_calculate_analytical_target_evs` in the simulator to be **Street-Aware**. 
   - Preflop: Maintain current fold equity formulas.
   - Flop/Turn: Decay fold equity based on board texture and pot commitment.
   - River: Plunge `p_fold_single` to near-zero if Hero's equity is 0.0, acknowledging that opponents who have called down to the River are highly unlikely to fold to an air shove. Retrain a new model (V10) using these corrected targets.

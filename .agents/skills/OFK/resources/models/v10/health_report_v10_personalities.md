# Model Health Report: V10 Personalities

**Target Models:** `V10 Maniac`, `V10 Nit`, `V10 Sticky`
**Overall Grade:** > [!CRITICAL] CRITICAL FAIL (Complete Mode Collapse)

## Executive Summary
All three newly trained V10 personality models (`maniac`, `nit`, `sticky`) suffered catastrophic mode collapse during their training runs. 

The Dynamic PID Controller we introduced applied such massive, constant EV penalties to enforce the VPIP/AGG targets that the neural networks completely stopped looking at the cards. Instead, they overfit to the penalty gradients, learning to statically output the exact action that avoids the penalty regardless of equity, opponents, or board state.

### 1. V10 Maniac (Hyper-Aggressive Target)
- **Status:** **CRITICAL FAIL**
- **Symptom:** 100% Raise Rate. The model outputs `EV(Raise) = 2.07`, `EV(Call) = -0.64`, `EV(Fold) = -0.75` for literally every possible scenario, including 0% equity air.
- **Cause:** The heavy penalty on Folding and boosting of Raising caused the network weights to explode in the direction of raising.

### 2. V10 Nit (Hyper-Tight Target)
- **Status:** **CRITICAL FAIL**
- **Symptom:** 100% Fold Rate. The model outputs `EV(Fold) = 29.06`, `EV(Call) = -1.64`, `EV(Raise) = -10.53` for every scenario, including having Pocket Aces (AA) preflop.
- **Cause:** The penalty on calling/raising pushed the model to find safety in folding absolutely everything.

### 3. V10 Sticky (Calling Station Target)
- **Status:** **CRITICAL FAIL**
- **Symptom:** 100% Call Rate. The model outputs `EV(Call) = -1.31`, `EV(Fold) = -1.40`, `EV(Raise) = -4.32` for every scenario.
- **Cause:** It correctly learned that folding and raising were heavily penalized, so it settled into a local minimum of calling every single bet.

## Proposed Resolution

The `max(0, error) * 3.0` PID controller approach on the target Q-values is too mathematically violent. We cannot modify the true Monte Carlo returns (the ground truth labels) directly like this without causing the network's value approximations to diverge from reality.

If we want to train personalities safely, we should instead:
1. **Remove EV Penalties:** Keep the target EV strictly mapped to the true Monte Carlo returns.
2. **Train via Action Masking / Sampling:** Instead of penalizing the EV, we can force the model to explore certain actions during training (e.g. force the maniac to raise 60% of the time) while still teaching it the true EV of those actions.
3. **Train via Asymmetric Loss:** Apply a scaling factor to the loss *after* the EV error is calculated, so it learns faster when it makes aggressive moves but doesn't hallucinate fake EVs.

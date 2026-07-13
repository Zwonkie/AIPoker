# V11 EV Scaling Mismatch (Maniac Collapse)

**Date Recorded**: 2026-07-12
**Related Files**: 
- [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/train_selfplay.py)
- [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/six_max_simulator.py)

## Context
During the evaluation of `expert_v11_main.pth` using `evaluate-model-health`, the model demonstrated a catastrophic "Maniac Collapse". It would exclusively output `RAISE` in all scenarios, even with pure air preflop or postflop. However, the telemetry dashboard reported a conservative 33% VPIP and 0% Aggression. This discrepancy was because the training simulator's `bootstrap` logic enforced an 80% manual fold rate and `math_engine` guardrails suppressed the model's maniac tendencies during evaluation.

## Root Cause
The Q-learning target values generated during `train_selfplay.py` vectorization were mixed in two different scales:
1. The untaken actions' target EVs were returned from `_calculate_mc_target_evs` in **RAW CHIPS** (e.g. +13.5 chips).
2. The taken action's target EV was overridden by the actual profit `mc_return` in **BIG BLINDS** (e.g. +1.35 BBs).

Since 1 Big Blind = 10 chips, the model observed that the untaken actions (which often included Raise) were intrinsically ~10x more valuable than the action it actually took. Over thousands of hands, the Q-values for `Raise` hyper-inflated, leading the model to always select it. 

## Resolution
Modified `train_selfplay.py` line 116 to scale the `target_evs` from raw chips to Big Blinds before overriding the taken action:
```python
# These come back in RAW CHIPS. We must scale them to BIG BLINDS!
t_evs = [ev / bb for ev in list(dp.get('target_evs', [0.0, 0.0, 0.0]))]
```
This forces all target EVs (Fold, Call, Raise) to exist in the same mathematical space (BBs), ensuring the neural network gradients are structurally sound.

**Next Steps:** Retrain V11 from scratch, monitoring the raw EV outputs using `evaluate-model-health` directly after early epochs to ensure no runaway values.

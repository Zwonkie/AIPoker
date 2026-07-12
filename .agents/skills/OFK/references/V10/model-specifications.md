# V10 Model Specifications & Training Improvements

**Date Recorded**: 2026-07-12
**Related Files**: 
- [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v9/six_max_simulator.py)
- [opponent_bots.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/opponent_bots.py)

## Context
Following the V9 River Bluffing Collapse, it was determined that the V9 model learned a degenerate "final step" exploit. Because the simulated opponent bots in `opponent_bots.py` played too rigidly (e.g., they almost always folded to a shove on the River unless they had the nuts), the Hero model learned that blindly jamming the River with pure air was the optimal mathematical solution against the entire population.

V10 must resolve this rigid opponent exploitability and introduce granular telemetry to detect collapses early in training.

## Resolution / V10 Guidelines

### 1. Opponent Pool Randomization (Fixing Rigid Opponents)
To prevent the Hero from learning a "one-size-fits-all" River shove exploit, the opponent bots must be made less predictable:
- **Sticky Calling Stations**: Introduce a percentage of opponents who simply will not fold on the River to any bet size, forcing the Hero to learn when to give up with air.
- **Dynamic Fold Equity**: Instead of a hardcoded `p_fold`, opponent call/fold logic must factor in their own simulated hand strength (e.g., if the opponent bot simulates holding Top Pair, its fold probability to a shove should be significantly lower).
- **Maniac Re-raisers**: Opponents should occasionally trap and re-raise on the River to punish Hero bluffs.

### 2. Required Training Telemetry
V10 training logs must include the following metrics to catch edge-case collapses before deploying the model:
- **The Bluff Matrix (Action by Equity Tier)**: Track action distributions specifically when Hero equity is `< 20%`. If Raise % exceeds a logical threshold (e.g., 30%), trigger a collapse warning.
- **Showdown vs. Non-Showdown Win Rate**: Monitor the ratio. If Non-Showdown win rate skyrockets while Showdown win rate plummets, the model is over-bluffing.
- **Fold Equity by Street**: Log the average `p_fold` the simulator assumes. Confirm that River `p_fold` is appropriately low compared to Preflop.
- **Action Entropy**: Track policy confidence. If entropy drops to near 0 uniformly across complex boards, the model has fallen into a local minimum.

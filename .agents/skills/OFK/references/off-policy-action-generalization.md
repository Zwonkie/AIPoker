# Off-Policy Action Generalization and Exploration Issue

**Date Recorded**: 2026-07-11
**Related Files**: [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/train_selfplay.py), [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/six_max_simulator.py), [pluribus_engine.py](file:///c:/REPO/Antigravity/AIPoker/core/models/pluribus_engine.py)

## Context
During sensitivity sweeps of the trained Pluribus V4 model, we observed that while premium hands (AA, AK) show high raise EVs, weak hands like 72o also show positive raise EVs (e.g., EV(Raise) = 6.70 BB). Because the decision selection uses a probabilistic model proportional to positive EVs, this causes the agent to bluff-raise too frequently with garbage hands.

## Resolution / Guidelines

### 1. Root Cause: Masked Action Training (Off-Policy Generalization)
* In the headless self-play simulator, Hero's decisions are generated using a bootstrap heuristic (`self._hero_decide`).
* The model's loss is calculated using `criterion(pred_ev, Y_ev) * mask`, where the target EV is only gathered for the *action actually taken* by the heuristic.
* For weak hands like `72o` preflop, the heuristic always selects FOLD.
* Consequently, the neural network only receives loss gradients for the FOLD action of `72o` (target EV = 0.0). The CALL and RAISE actions for `72o` are never supervised.
* Due to standard neural network generalization, the model predicts positive values for these untrained actions, inheriting weights/representations from hands where raises *were* taken (e.g., AA/AK).

### 2. Resolution Strategy & Play Guardrails
To prevent the model from executing garbage bluffs based on unconstrained EV predictions:
1. **Mathematical / Preflop Chart Guardrails**:
   * The play engine wraps the model predictions with a preflop chart (`use_preflop_chart=True`). This overrides the raw EV neural net for preflop action selection unless the hand falls in a valid range.
2. **Bluff Engine Integration**:
   * Use the bluff engine (`use_bluff_engine=True`) and math engine to guard post-flop decisions, filtering out actions where the card equity does not warrant raising/calling.
3. **Exploration during Self-Play**:
   * In future self-play iterations, implementing an $\epsilon$-greedy exploration policy for Hero inside `six_max_simulator.py` (i.e. letting the neural network play and occasionally take random exploratory actions) will populate the dataset with real rollout outcomes for CALL and RAISE across all hand strengths, forcing the network to learn their true negative EVs.

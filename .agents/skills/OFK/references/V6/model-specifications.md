# Pluribus V6 Model Specifications & Improvements

**Date Recorded**: 2026-07-11
**Related Files**: 
*   [poker_transformer.py](file:///c:/REPO/Antigravity/AIPoker/core/models/poker_transformer.py)
*   [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/six_max_simulator.py)
*   [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/train_selfplay.py)

## Context
Following the deployment of Pluribus V5, several core limitations in target EV variance, opponent style modeling, and action representations were identified. This document specifies the architectural and training improvements to be implemented for **Pluribus V6**.

## Proposed Improvements for V6

### 1. Showdown Baseline Subtraction (Variance Reduction)
*   **Problem**: Target EVs are computed using raw, single-runout chip wins/losses. Runout luck (e.g., AA getting cracked by 72o) introduces massive variance, requiring a large volume of simulated hands (100k+) for the network to filter out the noise.
*   **V6 Specification**: Implement showdown baseline subtraction. When a hand goes to showdown (or players are all-in before the river), instead of assigning targets based on who won the pot, compute the exact mathematical equity of each hand using the card evaluator and assign targets as `equity * pot - chips_committed`. This eliminates all runout-related luck variance, reducing gradient noise by over 80% and speeding up training convergence.

### 2. Multi-Agent Reinforcement Learning (MARL) Pool
*   **Problem**: V5 trains exclusively against hardcoded rule-based bots (Nit, TAG, Maniac, Fish). This makes the policy susceptible to exploiting rule-based heuristics rather than learning generalized game theory.
*   **V6 Specification**: Expand the simulator's opponent pool to include historical neural network checkpoints (e.g., V4 and V5 models) with varying degrees of action temperature (exploration noise). This forces V6 to learn robust strategies against adaptive neural network opponents.

### 3. Structured Action Size Tokens
*   **Problem**: The model's action history only tokenizes character identifiers (`c`, `f`, `r`). The network does not know whether a raise was a minimum raise (2x) or a pot-sized raise (1.0x).
*   **V6 Specification**: Expand the action vocabulary to include quantized raise sizes relative to the pot:
    *   `VOCAB = {'F': 0, 'C': 1, 'R_min': 2, 'R_half': 3, 'R_pot': 4, 'R_allin': 5}`
    *   Map incoming simulator actions to the closest quantized token size, providing the transformer with explicit betting geometry.

### 4. Dynamic Opponent Profiling (Online HUD Adaptation)
*   **Problem**: Opponent VPIP/AGG profiles are passed as static global parameters at the start of a hand. The model cannot detect if an opponent is playing tighter or looser in a specific session.
*   **V6 Specification**: Feed a sliding window of the opponent's recent hands in the active session to compute dynamic, running VPIP/AGG values inside the context tensor, allowing the transformer attention heads to adapt to changing opponent dynamics in real-time.

### 5. Prioritized Experience Replay (PER)
*   **Problem**: Training batches are sampled uniformly, meaning the network spends equal capacity on simple folds and complex postflop decision points.
*   **V6 Specification**: Sample training batches weighted by the Huber loss prediction error. This forces more frequent updates on difficult, high-stakes decisions where the model's EV estimates are highly inaccurate.

### 6. Action-Sizing Optimization (Discrete Betting Heads)
*   **Problem**: In V5, the model only outputs a single generic "Raise" action. The simulator then applies a hardcoded sizing formula. Since the model does not control the size of its bets/raises, it cannot learn to optimize pot size or select small sizes (e.g. 2-3 BB preflop) to lure opponents into the pot.
*   **V6 Specification**:
    1.  **Multi-Action EV Heads**: Expand the model output head from 3 actions to 6 actions:
        *   `Action 0`: Fold
        *   `Action 1`: Call / Check
        *   `Action 2`: Raise Small (2x - 3x BB preflop / 33% pot postflop)
        *   `Action 3`: Raise Medium (50% - 75% pot postflop)
        *   `Action 4`: Raise Pot (100% pot size bet)
        *   `Action 5`: All-In Shove
    2.  **Sizing-Sensitive Opponents**: Update opponent bot logic to compute folding probabilities based on the bet size relative to the pot. Opponents should call small bets (33% pot) with wide, marginal ranges, and fold to large bets (all-ins) except with premium hands.
    3.  **EV Sizing Comparison**: During training, the Decision Transformer will learn that shoving preflop with AA yields a low EV ($+1.5$ BB) because everyone folds, whereas raising 2.5 BB yields a much higher EV ($+15.0$ BB) by luring weaker hands into postflop streets. The model will naturally learn to optimize pot sizes by picking the action head with the highest predicted EV.
    4.  **Noisy Sizing Exploration (Exploration-Exploitation)**: To prevent the model from locking into a sub-optimal policy early (e.g. only shoving because it hasn't explored raising small enough times to map its true EV), implement:
        *   **5% Epsilon-Greedy Sizing Exploration**: During simulation, there is a 5% chance the model ignores the highest predicted EV and selects a random betting size (e.g. forcing a Call or Raise Small).
        *   **Boltzmann (Softmax) Sampling**: Instead of deterministic `argmax(EV)` selection, sample actions using a temperature-scaled Softmax over predicted EVs: $P(a_i) \propto \exp(EV_i / T)$. This guarantees the model frequently tests alternative sizes, compiling the empirical EV data needed to prove that luring opponents in is superior.

### 7. Graceful Interruption, Resuming, & Early Stopping
*   **Problem**: Currently, training always starts from scratch (`hands_done = 0`), and if interrupted, the state of the optimizer and progress counters is lost. There is no support for resuming an interrupted training session.
*   **V6 Specification**:
    1.  **Resume Capability (`--resume_path`)**: Add support to load a complete training checkpoint. Instead of just saving model weights, save a unified checkpoint dictionary containing:
        *   `model_state_dict`: The network weights.
        *   `optimizer_state_dict`: The optimizer momentum/history states.
        *   `scheduler_state_dict`: The learning rate scheduler state.
        *   `hands_done`: Current simulated hand counter.
        *   `opponent_pool_stats`: Cumulative bot VPIP/AGG histories.
    2.  **KeyboardInterrupt Catching (Ctrl+C)**: Wrap the master training loop in a `try...except KeyboardInterrupt:` block. If interrupted manually from the terminal, the script will catch the event, print a graceful shutdown notice, save the complete checkpoint (including optimizer and scheduler states) to `v6_interrupted_checkpoint.pth`, and exit cleanly.
    3.  **Early Stopping Criteria**: Monitor the running average of the `Validation Loss` and `Hero Win Rate (BB/100)`. If the validation loss has stabilized (fluctuations $< \pm 0.05$) and the win rate has plateaued (moving by less than 2 BB/100 over 5 consecutive batches), Prompt user if the traning should be stopped gracefully. If the user agrees, save the final weights, and terminate.
    4.  **End-of-Batch Checkpoints**: Save temporary checkpoints at the end of every simulation batch (`v6_checkpoint_latest.pth`) so that the active model can be recovered even in the event of hardware or system failure.

## V6 Training Completion & Results
* **Date Completed**: 2026-07-11
* **Hands Simulated**: 500,002
* **Training Samples**: 866,492
* **Total Elapsed Time**: 3h 16m 5s
* **Final Train Loss (MSE)**: 20.6190
* **Final Val Loss (MSE)**: 23.0164
* **Hero Win Rate**: **`+106.4 BB/100`** (Plateaued/converged, achieving +26.3 BB/100 over V5)
* **Final Opponent Profiling HUD Stats**:
  * Fish: VPIP 35.8%, AGG 28.7%
  * TAG: VPIP 4.2%, AGG 53.4%
  * Maniac: VPIP 59.9%, AGG 74.8%
  * Nit: VPIP 3.5%, AGG 33.3%
* **Saved Weight path**: [expert_v6_selfplay.pth](file:///c:/REPO/Antigravity/AIPoker/core/weights/expert_v6_selfplay.pth)


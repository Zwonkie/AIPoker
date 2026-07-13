---
name: check-complete-model-training
description: A comprehensive deep-dive checklist to identify data misunderstandings, bad practices, and learning issues in the RL pipeline.
---

# Deep Check: Model Training Health & Pipeline Integrity

> [!CAUTION]
> **Zero Trust Mindset Required**: Before diving into the specifics below, adopt a "Zero Trust" mindset. Look at the entire codebase with a fresh, extremely critical eye. Assume that the existing code, mathematical logic, and architectural assumptions might be fundamentally flawed, broken, or misaligned, no matter how mature the project seems. Do not take variable names or docstrings at face value—verify exactly what the code is doing mechanically. Question everything.

If tasked with finding the root cause of a model collapse (like the Nit folding the absolute nuts), hallucination, or learning plateau, I would conduct a forensic review across the following 4 domains.

## 1. Data Contract & Vectorization (The Input Alignment)
**Goal:** Ensure the model sees exactly what we think it sees, and that training inputs perfectly match inference inputs.
**Files to Check:**
*   `core/bridge/v11/contract_v11.py`
*   `core/models/poker_transformer.py`

**Specific Checks:**
*   **Vectorization Misalignment:** Does the order of features packed into the state vector sequence perfectly match the order the model expects? A single shifted index means the model might mistake "Pot Size" for "Hero Stack".
*   **Normalization Scale Mismatches:** If Pot Size is divided by `big_blind` (e.g., 15.0) in the simulator, but passed raw during inference, the model will hallucinate.
*   **Causal Masking Failure:** Does the attention mask accidentally allow the model to see future actions in the sequence? (e.g., If the sequence is `Check -> Bet -> Call`, can the model see the `Call` before it predicts the `Bet`?). If so, it cheats during training and collapses in live play.
*   **Profile Blindness (Regression):** Ensure the HUD stats (VPIP/AGG) for the active opponents are correctly calculated and not globally hardcoded to static values.

## 2. The Training Loop & Loss Calculation (The Gradients)
**Goal:** Ensure the feedback signal mathematically guides the model toward winning poker, rather than exploiting a bug in the loss function.
**Files to Check:**
*   `tools/self_play/v11/train_selfplay.py`

**Specific Checks:**
*   **Action Index Swapping:** In V11, we replace the predicted EV of the *taken action* with the true Monte Carlo return (`mc_return`). If the hero `Folded` (Action Index 0), but the code accidentally replaces the EV for `Call` (Action Index 1) with the `mc_return`, the model learns to associate the Fold's result with the Call action.
*   **Gradient Explosions / Scale Dominance:** `mc_return` can be huge (+300 BB for an all-in). The Auxiliary targets (Bluff %, Equity %) are strictly `0.0` to `1.0`. Is the MSE of the Q-values completely drowning out the Auxiliary heads, or vice versa?
*   **Loss Component Logging:** Are `loss_q`, `loss_bluff`, and `loss_equity` all decreasing? If `loss_q` plateaus but `loss_aux` drops to zero, the model has memorized the board state but refuses to learn action values.

## 3. Simulation Environment & Ground Truth (The Reality)
**Goal:** Ensure the environment isn't lying to the model about what actions are profitable.
**Files to Check:**
*   `tools/self_play/v11/six_max_simulator.py`
*   `core/cuda_evaluator.py`
*   `core/action_executor.py`

**Specific Checks:**
*   **Terminal Showdown Math:** When the hand goes to showdown, does the split pot logic perfectly distribute chips? If a side-pot is miscalculated, the model might learn that going all-in with the nuts loses money.
*   **Heuristic Poisoning:** During the `bootstrap_alpha` phase, the heuristics drive the action. If the Nit heuristic has a bug where it randomly folds AA preflop 10% of the time, the Neural Network will observe this and learn that folding AA is a standard, acceptable play.
*   **Seat/Position Overfitting:** Is the Hero always sitting in Seat 0? If Seat 0 is always the Small Blind, the model will never learn how to play the Button. 

## 4. Model Architecture & Extreme Behavior (The Brain)
**Goal:** Identify if the neural network is physically capable of representing the complexity of the game.
**Files to Check:**
*   `core/models/poker_transformer.py`
*   `core/decision.py`

**Specific Checks:**
*   **The "Monster-Under-The-Bed" Syndrome:** Why does the Nit model fold the nuts to a bet? I would check the variance in the training data. If the heuristic opponents *only* ever bet the river when they have the absolute nuts, the model learns `Opponent Bet = 100% Death`. We must check if the Maniacs/Fuzzy bots are actually bluffing the river often enough in the simulation data to teach the model to call.
*   **Padding Token Attention:** If the maximum sequence length is 20, but a preflop hand ends in 2 steps, the remaining 18 steps are padding. Is the model's attention mechanism correctly ignoring padding? If it attends to padding zeros, it correlates "lots of zeros = safe to fold".
*   **Temperature / Action Entropy:** Is the model's output distribution too sharp? If it predicts `[Fold: 10, Call: -50, Raise: -50]`, it has collapsed into a single state. We would need to inspect the action entropy telemetry.
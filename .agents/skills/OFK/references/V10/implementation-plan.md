# V10 Implementation Plan (Revised)

Based on your feedback, we will structure V10 with a clean `v10/` directory, extract the telemetry into a dedicated class, and completely replace the flawed analytical Target EV formula with a true Monte Carlo GTO evaluation.

## Proposed Changes

---

### 1. Opponent Bots & Environment

#### [MODIFY] `tools/self_play/opponent_bots.py`
We will make the heuristic opponent bots less predictable and more realistic.
*   **FishBot (Sticky Calling Station)**: Will almost *never* fold on the River (`< 5%` fold rate), punishing pure air bluffs.
*   **ManiacBot (The Trapper)**: Add a small chance (5-10%) on the River to trap with a strong hand or execute a massive check-raise bluff.
*   **Dynamic Hand-Strength Awareness**: Incorporate `equity` vs `pot_odds` more smoothly so bots don't fold nutted hands to random shoves.

---

### 2. External Training Telemetry

#### [NEW] `tools/self_play/v10/telemetry.py`
We will create an external `TrainingTelemetry` class to completely decouple metric tracking from the PyTorch training loop.
*   **Bluff Matrix Tracking**: Intercepts actions where `equity < 0.20` and records the distribution of Folds, Calls, and Raises.
*   **Action Entropy**: Calculates and tracks the entropy of the model's action distribution over time to detect local-minimum collapses.
*   **Output**: Handles formatting and presenting the data (e.g., to console, TensorBoard, or a log file) cleanly.

---

### 3. V10 Simulator & MC GTO

#### [NEW] `tools/self_play/v10/six_max_simulator.py`
We will duplicate the V9 simulator into the new `v10/` directory and replace the target EV engine.
*   **Replace Analytical Math with MC GTO**: We will completely remove `_calculate_analytical_target_evs`.
*   **Implement `_calculate_mc_target_evs`**: Instead of a hardcoded formula, the simulator will use Monte Carlo simulations to find the true EV of Fold, Call, and Raise. 
    *   **Fold EV**: $0$ (relative to current pot).
    *   **Call / Raise EV**: The simulator will branch the state, query the *actual opponent bots* (using their V10 logic) for how they respond to the Call or Raise, and then use `_cuda_evaluator.calculate_equity_batched` to rollout the rest of the board to showdown.
    *   This ensures the Target EVs are dynamically grounded in the actual opponent population logic, completely curing the River Bluff Collapse.

## Verification Plan
1. I will run the `run_model_diagnostics.py` script against the new MC GTO Target EVs to mathematically prove that pure air bluffs on the River generate negative EVs.
2. I will write a quick test script for the `TrainingTelemetry` class to verify it correctly intercepts and formats the Bluff Matrix.

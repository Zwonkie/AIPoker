# Pluribus V7 Model Specifications & Architectural Remediations

**Date Recorded**: 2026-07-11
**Related Files**: 
*   [poker_transformer.py](file:///c:/REPO/Antigravity/AIPoker/core/models/poker_transformer.py)
*   [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/six_max_simulator.py)
*   [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/train_selfplay.py)
*   [pluribus_engine.py](file:///c:/REPO/Antigravity/AIPoker/core/models/pluribus_engine.py)

## Context
Standard validation sweeps on Pluribus V5 and V6 revealed a complete model collapse due to sequence dilution and attention collapse in the Decision Transformer. The V7 specification outlines critical architectural, data contract, and simulation pipeline changes required to remediate this collapse and train a generalized poker neural network.

---

## Proposed Improvements for V7

### 1. Key Padding Mask & Attention Collapse Remediation
*   **Problem**: In V6, sequences are padded to length 20, with actual data present only at step 19. The model lacks a key padding mask, so self-attention softmax distributes weights uniformly (`0.05` per step) across all 20 steps, diluting active step gradients by $20^3 = 8000\times$.
*   **V7 Specification**:
    1.  **Implement `key_padding_mask`**: Pass a boolean mask (`src_key_padding_mask`) to the transformer encoder. Padding tokens (index 0 to 18) must be marked `True` (ignored), forcing the attention weights for step 19 to attend 100% to active steps.
    2.  **Dense Sequence Packing**: Instead of padding to sequence length 20, pack the sequence to include only actual historical decision points (e.g., sequence length is exactly 1 preflop, 2 flop, etc.), letting sequence length grow dynamically.

### 2. Multi-Action EV Target Generation (All-Action Loss)
*   **Problem**: The training loop only calculates loss on the action that was actually taken (`b_sa`), leaving predicted EVs for other actions unconstrained. This leads to massive extrapolation errors (e.g. shoving `72o` because the shove EV output is untrained and drifts to high positive values).
*   **V7 Specification**:
    1.  **Dense Action Loss**: When generating a training sample, simulate/calculate the expected values for **all** actions (Fold, Call, and Raise) at that decision point using MC GT (gametree) rollouts.
    2.  **Full-Head Loss Calculation**: Update the loss function to calculate MSE/Huber loss across all 3 action EV outputs simultaneously, constraining the entire EV prediction vector at every step:
        $$\mathcal{L} = \sum_{a \in \{F, C, R\}} \left( \hat{Q}(s, a) - Q^*(s, a) \right)^2$$

### 3. Closed-Loop Postflop Model Play
*   **Problem**: In V6, Hero's postflop decisions in the simulator bypass the neural network entirely and use hardcoded equity thresholds. This prevents the model from generating closed-loop reinforcement learning data postflop.
*   **V7 Specification**: Enable postflop model inference. Hero must query the active V7 model postflop to select actions (Fold, Call, Raise) based on predicted EVs, establishing a true RL feedback loop on all streets.

### 4. Direct Feed-Forward Baseline (MLP Head)
*   **Problem**: The model acts as a single-step decision maker. A full sequence model (Transformer) is highly prone to overfitting and collapse when sequence depth is shallow.
*   **V7 Specification**: Implement a direct feed-forward multi-layer perceptron (MLP) architecture option in `poker_transformer.py` (e.g. `PokerEVModelV7MLP`) as a baseline comparison. This completely eliminates sequence-based attention collapse.

### 5. Chronological Sequence Packing (V7 Data Contract)
*   **Problem**: The sequence is vector-padded with static context. The transformer cannot learn temporal dependencies.
*   **V7 Specification**: Update the sequence vectorizer to store step-by-step game history chronologically (e.g. step 0 = preflop raise, step 1 = call, step 2 = flop check). This allows the transformer to learn historical betting context and bluff frequencies.

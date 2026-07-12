# Pluribus V5 vs V6 Sensitivity Analysis & Model Collapse Diagnosis

**Date Recorded**: 2026-07-11
**Related Files**:
*   [poker_transformer.py](file:///c:/REPO/Antigravity/AIPoker/core/models/poker_transformer.py)
*   [pluribus_engine.py](file:///c:/REPO/Antigravity/AIPoker/core/models/pluribus_engine.py)
*   [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/train_selfplay.py)
*   [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/six_max_simulator.py)

## Context
We conducted standard validation sweeps (Scenarios A through E) comparing **Pluribus V5** (`expert_v5_selfplay.pth`) and **Pluribus V6** (`expert_v6_selfplay.pth`). The analysis revealed that both models produce nearly identical, flat Q-value predictions across all hands, stack sizes, pot sizes, and opponent profiles. We performed a layer-by-layer diagnostic to identify the root cause of this behavior.

## Resolution / Findings

### 1. Sweep Results Summary
The Q-values predicted by both models were found to be completely insensitive to variations in the state space (e.g. AA vs 72o, shove sizes, active opponents, or opponent HUD colors):
*   **V5 Outputs**: Fold EV ≈ `0.02`, Call EV ≈ `0.17`, Raise EV ≈ `0.35`
*   **V6 Outputs**: Fold EV ≈ `-0.00`, Call EV ≈ `0.26`, Raise EV ≈ `0.27`

This flat prediction profile causes the models to act as a static random chooser (with constant action weights) preflop and postflop.

### 2. Root Cause Analysis: Sequence Dilution and Attention Collapse
Our diagnostic traced the issue to a structural training-optimization mismatch in the Decision Transformer implementation:

1.  **Sparse Training Signals**: The training data sequences are formatted to length 20, but the active state features (cards, board, context) are placed **only at the final step (index 19)**. The preceding 19 steps are filled with padding zeros (or padding card index 52).
2.  **Vanishing Gradients**: Because the loss is calculated only at step 19, the gradients for the input features (card embeddings and context MLP) must backpropagate through 3 layers of self-attention.
3.  **Attention Collapse**:
    *   During initialization, self-attention weights are random/uniform.
    *   At step 19, the attention softmax distributes weights across all 20 steps (uniform weight `1/20 = 0.05`).
    *   This dilutes the active step 19's features by 20x at each attention layer. For a 3-layer transformer, the feature signal (and its gradient) is attenuated by $20^3 = 8000\times$.
    *   Due to this massive $8000\times$ gradient dilution, the card embedding and context projection layers learn extremely slowly or not at all.
4.  **Bias Dominance**: The output head's biases (`head.2.bias`), which receive direct gradients without attention dilution, quickly learn to output the global average EV of the training set (e.g. ~0.26 BB) to minimize loss. The model collapses to this trivial bias-only mapping.

### 3. Simulator Hardcoded Fallbacks
Although the ML models are collapsed, the self-play training successfully achieved positive win rates (e.g., `+106.4 BB/100` for V6) due to:
*   **Preflop Fallback**: The preflop decision loop falls back to the heuristic range chart 45% of the time.
*   **Postflop Hardcoding**: The postflop decision-making for Hero in `six_max_simulator.py` does **not** query the ML model; it uses strict, hardcoded equity thresholds (e.g., raising if equity > 75%, folding if equity < pot odds).

## Guidelines for Future Models
To train a functional neural network without collapse, the model architecture and data format should be modified:
1.  **Reduce Sequence Length**: Use a sequence length of 1 (or only active history steps) instead of padding 19 steps of zeros.
2.  **Dense Loss**: Train the model on all decision steps in the sequence rather than masking out all but the last.
3.  **Direct Feed-Forward Baseline**: For single-step decisions, replace the Transformer with a multi-layer perceptron (MLP) to eliminate attention dilution.

---

## Detailed Sensitivity Sweeps Outcome Values

## Scenario A: The '72o Facing Shove' Test (Extrapolation Guardrail)
Evaluating predicted EV for `7d 2s` facing a 470 chip shove (47 BB) into a 510 chip pot (51 BB). Stack = 1000 chips. Big Blind = 10.0.

| Street | Opponent | V5 Fold EV | V5 Call EV | V5 Raise EV | V6 Fold EV | V6 Call EV | V6 Raise EV |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Preflop | Nit | 0.02 | 0.18 | 0.35 | -0.00 | 0.26 | 0.27 |
| Preflop | Maniac | 0.02 | 0.18 | 0.35 | -0.00 | 0.26 | 0.27 |
| Flop | Nit | 0.02 | 0.18 | 0.35 | -0.00 | 0.26 | 0.27 |
| Flop | Maniac | 0.02 | 0.18 | 0.35 | -0.00 | 0.26 | 0.27 |
| Turn | Nit | 0.02 | 0.18 | 0.35 | -0.00 | 0.26 | 0.27 |
| Turn | Maniac | 0.02 | 0.18 | 0.35 | -0.00 | 0.26 | 0.27 |
| River | Nit | 0.02 | 0.17 | 0.35 | -0.00 | 0.26 | 0.27 |
| River | Maniac | 0.02 | 0.18 | 0.35 | -0.00 | 0.26 | 0.27 |

## Scenario B: Opponent Personality Exploitation Test (Profile Sensitivity)
Evaluating Flop (`Th 7c 2d`) with `Ks Kd` (Strong Overpair) facing a small bet (call amount = 10, pot = 50). Stack = 1000, Big Blind = 10.0.

| Opponent Profile | V5 Fold EV | V5 Call EV | V5 Raise EV | V6 Fold EV | V6 Call EV | V6 Raise EV | Best Action |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Nit (Blue/Blue) | 0.02 | 0.17 | 0.35 | -0.00 | 0.26 | 0.27 | **RAISE** |
| TAG (Green/Green) | 0.02 | 0.17 | 0.35 | -0.00 | 0.26 | 0.27 | **RAISE** |
| Fish (Yellow/Blue) | 0.02 | 0.17 | 0.35 | -0.00 | 0.26 | 0.27 | **RAISE** |
| Maniac (Red/Red) | 0.02 | 0.17 | 0.35 | -0.00 | 0.26 | 0.27 | **RAISE** |

## Scenario C: Pot & Stack Size Scaling Test (Geometry Check)
Evaluating Flop (`Ah Qs 5d`) with `Ac Kc` (Top Pair, Top Kicker). Big Blind = 10.0.

### 1. Pot Size Sweep (Stack fixed at 100 BB = 1000 chips)

| Pot Size | V5 Call EV | V5 Raise EV | V6 Call EV | V6 Raise EV |
| :---: | :---: | :---: | :---: | :---: |
| 10 BB | 0.17 | 0.35 | 0.26 | 0.27 |
| 30 BB | 0.17 | 0.35 | 0.26 | 0.27 |
| 50 BB | 0.17 | 0.35 | 0.26 | 0.27 |
| 70 BB | 0.17 | 0.35 | 0.26 | 0.27 |
| 90 BB | 0.17 | 0.35 | 0.26 | 0.27 |
| 110 BB | 0.17 | 0.35 | 0.26 | 0.27 |
| 130 BB | 0.17 | 0.35 | 0.26 | 0.27 |
| 150 BB | 0.17 | 0.35 | 0.26 | 0.27 |

### 2. Stack Size Sweep (Pot fixed at 50 BB = 500 chips)

| Hero Stack | V5 Call EV | V5 Raise EV | V6 Call EV | V6 Raise EV |
| :---: | :---: | :---: | :---: | :---: |
| 10 BB | 0.17 | 0.35 | 0.26 | 0.27 |
| 40 BB | 0.17 | 0.35 | 0.26 | 0.27 |
| 70 BB | 0.17 | 0.35 | 0.26 | 0.27 |
| 100 BB | 0.17 | 0.35 | 0.26 | 0.27 |
| 130 BB | 0.17 | 0.35 | 0.26 | 0.27 |
| 160 BB | 0.17 | 0.35 | 0.26 | 0.27 |
| 190 BB | 0.17 | 0.35 | 0.26 | 0.27 |

## Scenario D: Active Opponents Sensitivity (Multi-Way Check)
Evaluating Flop (`9d 5c 2h`) with `Js Jh` (Vulnerable Overpair) facing a small bet (pot = 100, call amount = 20). Stack = 1000, Big Blind = 10.0.

| Opponents | Equity | V5 Call EV | V5 Raise EV | V6 Call EV | V6 Raise EV | Best Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.80 | 0.18 | 0.35 | 0.26 | 0.27 | **RAISE** |
| 2 | 0.65 | 0.18 | 0.35 | 0.26 | 0.27 | **RAISE** |
| 3 | 0.52 | 0.18 | 0.35 | 0.26 | 0.27 | **RAISE** |
| 4 | 0.42 | 0.18 | 0.35 | 0.26 | 0.27 | **RAISE** |
| 5 | 0.35 | 0.18 | 0.35 | 0.26 | 0.27 | **RAISE** |

## Scenario E: Preflop Equity Sensitivity (Range Check)
Evaluating Preflop facing a standard raise (pot = 30, call amount = 20). Stack = 1000, Big Blind = 10.0.

| Hand | Equity | V5 Fold EV | V5 Call EV | V5 Raise EV | V6 Fold EV | V6 Call EV | V6 Raise EV | Best Action |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 7d 2s (Garbage) | 0.30 | 0.02 | 0.18 | 0.35 | -0.00 | 0.26 | 0.27 | **RAISE** |
| Jh Ts (Medium Speculative) | 0.46 | 0.02 | 0.18 | 0.35 | -0.00 | 0.26 | 0.27 | **RAISE** |
| Ad Qo (Strong Broadway) | 0.60 | 0.02 | 0.17 | 0.35 | -0.00 | 0.26 | 0.27 | **RAISE** |
| Qd Qs (Premium Monster) | 0.78 | 0.02 | 0.17 | 0.35 | -0.00 | 0.26 | 0.27 | **RAISE** |
| Ah As (Absolute Nuts) | 0.85 | 0.02 | 0.17 | 0.35 | -0.00 | 0.26 | 0.27 | **RAISE** |
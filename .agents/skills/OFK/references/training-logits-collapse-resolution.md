# Training Logits Collapse and High-Epoch Static Dataset Solution

**Date Recorded**: 2026-07-10
**Related Files**: [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/train_selfplay.py), [train_static_poc.py](file:///c:/REPO/Antigravity/AIPoker/scratch/train_static_poc.py)

## Context
During initial self-play POC runs of the V4 model (100k hands), predictions for all hands and board states collapsed to a global constant value: `EV(F,C,R) = [-0.05, 6.37, 73.22]`. This represented only the global dataset means, indicating the Transformer attention layers and card embeddings had not learned card representations or positional variations.

## Resolution / Guidelines

### 1. Root Cause Analysis
* **Underfitting / Low Updates**: The standard RL training loop performed only 3 epochs per batch of 10,000 hands. Because sequence Transformers are deep models with a large parameter space, a small number of updates makes it extremely easy for the optimizer to hit a plateau by simply matching the global bias (means of the MSE targets).
* **Card Embedding Gradient Flow**: The random initialization of card embeddings means the model cannot initially distinguish `AA` from `72o`. Learning these representations requires thousands of backpropagation updates, which does not happen in a handful of RL batches.

### 2. Resolution Strategy (Static Dataset High-Epoch Training)
To build a functional POC model:
1. **Simulate a Fixed Dataset**: Generate a fixed, high-quality simulated dataset of hands (e.g. 20,000 hands) via the headless 6-max simulator.
2. **Epoch-Heavy Training**: Instead of discarding hands quickly, load the entire dataset into memory on the GPU (or CPU) and train for **150+ epochs**.
3. **Outcome**: The extensive epoch count (11,000+ batch updates) allows card embeddings and attention weights to specialize, leading to high-variance, hand-sensitive EV predictions (e.g., proper positive EV for `AA` and negative EV for `72o` call/raise decisions) while bypassing RL on-policy constraints for POC verification.

### 3. Core Architectural and Optimization Fixes (Final Resolution)

The final resolution that unlocked full model convergence and highly dynamic, profile-sensitive EVs involved:

1. **Sequence Alignment Correction**:
   * *Problem*: Training populated sequences from the front (preflop at index 0, flop at index 1, etc.), whereas online inference placed the active state at the very end of the sequence (index 19) and evaluated predictions using `squeeze(0)[-1]`. positional embeddings at index 19 were completely untrained.
   * *Fix*: Each simulated hand is now split into separate training samples (one per decision point). In each sample, the active decision point is always placed at the **final step (index 19)**, pre-padded with card/context histories.
2. **Robust Loss Function (Huber Loss)**:
   * *Problem*: Target EVs have extremely high variance (up to $\pm 100$ BB from single-rollout hand results). MSE loss was dominated by outliers, forcing the network to predict the global mean class averages to minimize squared errors.
   * *Fix*: Replaced `MSELoss` with `HuberLoss(delta=2.0)`. Huber loss is linear for large errors, which dramatically dampens the impact of noisy outliers and stabilizes gradient descent.
3. **Gradient Norm Clipping**:
   * Added `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` to prevent gradient explosion and parameter saturation during batch updates.
4. **Stable Optimization Learning Rate**:
   * Lowered the learning rate from `1e-3` to a stable `5e-4` for smooth Transformer convergence.
5. **VPIP/AGG Feature Alignment**:
   * *Problem*: During training, raw simulation VPIP/AGG values were continuous floats, but during inference, the OCR visual HUD only extracts discrete colors (Red, Yellow, Green, Blue) mapped to midpoints.
   * *Fix*: Added mapping helpers (`map_vpip_to_midpoint` and `map_agg_to_midpoint`) to bin training HUD floats into the exact color midpoints seen during live play, eliminating feature shift.
6. **Execution Environment CUDA Upgrade**:
   * Replaced the CPU-only PyTorch build in the virtual environment with PyTorch `2.1.2+cu121` to run on CUDA (Nvidia RTX 4080), reducing training time for 1,000,000 hands from ~33 hours to under 3 hours.


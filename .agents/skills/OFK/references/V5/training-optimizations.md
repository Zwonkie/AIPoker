# Pluribus V5 Self-Play RL Training Optimizations

**Date Recorded**: 2026-07-11
**Related Files**: 
*   [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/six_max_simulator.py)
*   [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/train_selfplay.py)
*   [heuristic_training_engine.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/heuristic_training_engine.py)

## Context
To prevent policy collapse, action EV hallucinations, and CPU-to-GPU data bottlenecks during the Pluribus V5 self-play reinforcement learning training session, an optimized hybrid preflop decision architecture and a set of PyTorch CUDA optimization techniques were designed and implemented.

## Resolution / Guidelines

### 1. Hybrid Preflop Exploration Split
To train preflop decisions without causing policy deterioration or EV target collapse:
*   **5% Pure Random Exploration**: Ensures baseline exploration of weak/unorthodox hands.
*   **50% Active Model Inference**: Queries the active V5 model currently being optimized.
*   **45% Heuristic Anchoring**: Falls back to the preflop range charts of the `HeuristicTrainingEngine`.
*   **Opponent Exploration**: Opponent bots make preflop decisions with 5% epsilon-random exploration to push weak hands into postflop streets.

### 2. Multiprocessing CPU-GPU Execution Contract
To avoid CUDA inter-process communication overhead and GPU memory errors:
*   Hand simulations and Monte Carlo equity checks are run in parallel across CPU cores using a multiprocessing pool.
*   The active neural network model is loaded on the **CPU** inside the worker processes. CPU forward passes are only run for the 50% feedback decisions during preflop play (1-2 times per hand), maintaining a high simulation throughput.

### 3. GPU CUDA / Tensor Core Optimization (RTX 4080)
*   **DataLoader Memory Pinning**: `pin_memory=True` enables direct memory access transfers from CPU host to GPU device.
*   **Automatic Mixed Precision (AMP)**: Runs forward/backward passes in float16 via `torch.cuda.amp.autocast()`, utilizing RTX Tensor Cores for up to 3x speedups.
*   **Non-Blocking Transfers**: Set `non_blocking=True` on all `.to(device)` calls to overlap copy operations with CUDA kernel execution.
*   **Learning Rate Decay**: Use `CosineAnnealingLR` to smooth updates and stabilize Huber Loss optimization.

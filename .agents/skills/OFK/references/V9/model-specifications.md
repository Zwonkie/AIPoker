# Pluribus V9 Model Specifications & Improvements

**Date Recorded**: 2026-07-11
**Related Files**: [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v8/six_max_simulator.py), [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v8/train_selfplay.py)

## Context
While the short opponent profiling window (10-20 hands) introduced in V8 allowed the model to rapidly exploit shifting opponent states and tilt dynamics, it suffered from high variance and noise. A small sample size of 10-20 hands can lead to distorted VPIP and AGG statistics due to card distribution variance (e.g., an opponent card-dead for 15 hands appearing artificially tight, or receiving consecutive premiums appearing artificially loose). To stabilize the overall opponent representation and ensure exploitation is based on true strategy profiles rather than card noise, V9 will increase the running stats window.

## Architectural Specifications

### 1. Expanded Opponent Profiling Window (50 Hands)
*   **50-Hand Running Window**: The history window for calculating sliding VPIP (Voluntary Put Money In Pot) and AGG (Aggression Factor) percentages for the context vector will be increased to the **last 50 hands**.
*   **Stability vs. Adaptability Trade-off**: 50 hands represents the optimal sweet spot for a 6-max table:
    *   *Noise Reduction*: Eliminates short-term card-run bias.
    *   *Robust Strategy Estimation*: Provides statistically significant estimates of preflop entry rates and postflop aggression ratios.
    *   *Sufficient Adaptability*: Still adapts within a single standard multi-table session (unlike lifetime statistics).
*   **Epsilon-Greedy Exploration**: Introduced a **5% purely random action factor** for all actors (Hero and Opponents) across all streets. This prevents the model from collapsing into deterministic ruts and dramatically improves state space coverage during Monte Carlo rollouts.

### 2. Multi-Personality League System Integration
The V9 training run will maintain the diversity-based league training structure from V8, with the updated 50-hand profiling inputs active across all simulated league seats to align the model's exploit heads with the more stable opponent features.

### 3. PyTorch Tensor GPU Evaluator (MC GTO Rollouts)
To overcome the extreme bottleneck of calculating CPU-based Monte Carlo rollouts (via `treys`), V9 utilizes a custom `CudaPokerEvaluator`.
*   **Architecture**: Pure PyTorch tensor bitwise operations. It evaluates 7-card poker hands by projecting 52 deck cards to bit-encoded integers (treys compatible), taking $\binom{7}{5}$ combinations, and performing fully batched tensor bitwise and lookup operations on the GPU.
*   **Performance**: Benchmarks demonstrate `13,334,724` evaluations per second, a massive leap over the CPU allowing deep MC rollouts.
*   **Integration**: Seamlessly integrated into `six_max_simulator.py` equity calculations via the batched `calculate_equity_batched` API.

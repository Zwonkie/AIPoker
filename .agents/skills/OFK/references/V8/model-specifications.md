# Pluribus V8 Model Specifications & League Training

**Date Recorded**: 2026-07-11
**Related Files**: [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v7/six_max_simulator.py), [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v7/train_selfplay.py)

## Context
Pure self-play reinforcement learning is prone to local minima and policy collapse (e.g. agents becoming overly passive or passive-aggressive because they only play against copies of themselves). To achieve robust, generalizable, and unexploitable GTO (Game Theory Optimal) strategies, V8 will implement **Diversity-Based League Training** (inspired by Google DeepMind's AlphaStar league) and **Dynamic Opponent Profiling**.

## Architectural Specifications

### 1. Multi-Personality Neural Network League
Instead of training a single agent against its own past checkpoints, the V8 simulator will maintain a "league" of active neural networks with distinct behaviors trained via reward/loss shaping:

*   **Personality A: The Aggressive Attacker (Maniac NN)**:
    *   *Loss/Reward Modifier*: Penalizes passive actions (checking/folding) and rewards raising/betting.
    *   *Role*: Forces the active model to learn how to defend against high-frequency bluffs and wide ranges.
*   **Personality B: The Defensive Wall (Nit NN)**:
    *   *Loss/Reward Modifier*: Penalizes chip loss heavily (e.g., multiplier of `1.5x` on negative showdown rewards).
    *   *Role*: Forces the active model to learn range selectivity and how to fold marginal hands.
*   **Personality C: The Sticky Caller (Calling Station NN)**:
    *   *Loss/Reward Modifier*: Penalizes folding postflop, forcing a high showdown rate.
    *   *Role*: Forces the active model to eliminate pure bluffs and maximize thin value-betting.

### 2. Stack Size Curriculum Learning
To make the model robust to various table geometries (from short-stacked survival to deep-stacked implied odds play), starting stacks will evolve dynamically through three training phases:

*   **Phase 1 (0 to 10k hands - Static Baseline)**:
    *   *Starting Stack*: Exactly **100 BB** (1,000 chips) for all seats ($\sigma = 0$ BB).
    *   *Purpose*: Establishes a clean, stable baseline for standard cash game play.
*   **Phase 2 (10k to 30k hands - Moderate Variance)**:
    *   *Starting Stack*: Randomized around a mean of **100 BB** with a standard deviation of **10 BB** ($\sigma = 10$ BB), clipped to `[80, 120] BB`.
    *   *Purpose*: Introduces minor stack imbalances and teaches the network to adjust to slight coverage differences.
*   **Phase 3 (30k+ hands - Extreme Geometry)**:
    *   *Starting Stack*: Randomized around a mean of **100 BB** with a standard deviation of **90 BB** ($\sigma = 90$ BB), clipped to a minimum of **10 BB** and a maximum of **300 BB**.
    *   *Purpose*: Forces the model to master extreme stack dynamics:
        *   *Short-stacks (10-30 BB)*: Focuses on high-urgency, preflop/flop shoves, and direct pot odds.
        *   *Deep-stacks (150-300 BB)*: Focuses on implied odds, multi-street pot control, and massive river decisions.

### 3. Personality Pre-Training Strategy
To ensure the league personalities play coherent poker, they will be trained using a two-stage process:
1.  **Bootstrap Phase**: Personalities start by using simple preflop heuristics (e.g. `HeuristicTrainingEngine` rules matching their targets) to guide early preflop exploration.
2.  **RL Takeover**: As training progresses, the neural network weights gradually take over decision-making, using self-play and personality-specific reward modifiers to optimize postflop and preflop parameters.

### 4. Dynamic Opponent Profiling (Short running window)
Currently, opponent profiles (VPIP/AGG) in the simulator context features are calculated over lifetime hand statistics. To enable rapid exploitation and realistic live-play behavior, V8 will modify this:
*   **10-20 Hand Running Window**: Opponent VPIP and AGG statistics fed into the model's context vector will be computed dynamically using only the last 10 to 20 hands played against that opponent.
*   **Exploitation Benefit**: This allows the active model to quickly identify if an opponent has tilted, tightened up, or changed their strategy, enabling real-time adaptation within a single session.


### 5. Corrected Preflop Target EV Formula (Pre-training Calibration)
To ensure V8's pre-trained baseline model starts with correct relative preflop action EVs, the preflop analytical target EV formula will be rewritten to eliminate artificial bluffing bias:
1.  **Equity Independence for Opponent Folds**:
    Opponent fold probability ($p_{fold}$) must be calculated independently of Hero's private card equity:
    $$\text{p\_fold\_single} = \text{baseline\_fold\_prob} \times \text{fold\_factor}$$
    where $\text{baseline\_fold\_prob}$ is a static opponent-fold rate (e.g. 70%) determined by the opponent's VPIP profile.
2.  **Opponent Pot Odds Integration**:
    Scale down $\text{p\_fold\_single}$ when Hero min-raises or offers highly favorable pot odds to the opponent, forcing the simulator to predict negative EV for min-raising weak holdings:
    $$\text{pot\_odds} = \frac{\text{raise\_increment}}{\text{new\_pot}}$$
    $$\text{p\_fold\_single} = \text{p\_fold\_single} \times (1.0 - e^{-k \cdot \text{pot\_odds}})$$
3.  **Active Opponents Decay**:
    Account for multiple active players behind Hero. The probability that *all* opponents fold decays exponentially with the number of active opponents:
    $$\text{p\_fold} = \text{p\_fold\_single}^{\text{num\_opponents}}$$

### 6. CUDA-Accelerated Vectorized Simulation Engine (V8 Engine)
To make Monte Carlo Game-Tree (MC GT) rollouts computationally viable for V8 training without dropping the simulation speed (aiming for >100 hands/sec), the simulator will run entirely on the GPU:

*   **Batched Tensor Game States**: 
    The simulator will clone the decision points and run 100 parallel rollouts as a single PyTorch tensor batch of size `B = 100` directly on the GPU.
*   **Vectorized Dealing and Evaluation**: 
    Community cards are dealt using GPU-parallel random integers, and hand evaluations at showdown will use a vectorized CUDA card lookup table.
*   **Zero CPU-GPU Transfer Bottleneck**: 
    Because game state tracking and neural network predictions are both kept in GPU (CUDA) memory, there is zero latency from PCIe bus copy overhead during the rollout process.

### 7. Weight Initialization, Non-Stationarity, and Preflop Bias Purging
A key design consideration is ensuring that the preflop target EV flaws of Pluribus V7 (which predicted positive Raise EV for weak hands like `72o`) do **not** propagate into the V8 model or its opponent league personalities:

*   **Flawed Weight Initialization is Overwritten**:
    Initializing V8's personalities and main model from `expert_v7_selfplay.pth` transfers high-value representations (card embeddings, board textures, postflop play). However, the moment training begins, the simulator uses the **corrected V8 target EV formula** as the training label. The loss function gradients immediately force the preflop EV heads of the networks to align with these new targets, actively overwriting the incorrect V7 preflop beliefs.
*   **Purging during the Bootstrap Phase**:
    During the first 10,000 hands of each pre-training session, the preflop heuristic overlay forces the models to play realistic, structured preflop ranges while training their EV prediction heads on the corrected mathematical targets. This guarantees that the preflop EV outputs are completely recalibrated and the V7 bias is purged before the RL takeover begins.
*   **Sequential Pre-training vs. Active Learning**:
    To prevent **non-stationarity** (where multiple learning NNs chase each other's strategies in circles, causing policy collapse), the Maniac, Nit, and Sticky NNs are pre-trained individually in self-play and then **frozen** when loaded into the main training run. During their individual pre-training, they successfully learn and correct their preflop play. Consequently, they enter the main training run as corrected, stable, and unexploitative benchmarks.


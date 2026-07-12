# Pluribus V4 Transformer Architecture and Tensor Specifications

**Date Recorded**: 2026-07-10
**Related Files**: [poker_transformer.py](file:///c:/REPO/Antigravity/AIPoker/core/models/poker_transformer.py), [pluribus_engine.py](file:///c:/REPO/Antigravity/AIPoker/core/models/pluribus_engine.py), [ml_bridge.py](file:///c:/REPO/Antigravity/AIPoker/core/ml_bridge.py), [heuristic_training_engine.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/heuristic_training_engine.py)

## Context
The V4 neural network (`PokerEVModelV4`) transitions from flat MLP models to a sequence-based causal Transformer decoder. It embeds hand cards, community cards, and context features across a causal decision timeline to predict EV.

## Resolution / Guidelines

### 1. Context Tensor Features (31-dim)
The model takes a `31-dimensional` context vector per decision step, structured as:
* **Opponents HUD Stats Matrix** ($5 \times 5 = 25$ float values): For the 5 other seats (ordered relative to Hero, starting from left):
  1. `vpip_normalized` (VPIP color midpoint ratio, 0.0 to 1.0)
  2. `agg_normalized` (AFq color midpoint ratio, 0.0 to 1.0)
  3. `is_active` (0.0 or 1.0 mask indicating if seat is still in the hand)
  4. `relative_stack_size` (opponent stack divided by pot size)
  5. `relative_committed_chips` (opponent chips committed in the current street divided by big blind)
* **Hero-Specific Context** ($6$ float values):
  6. `hero_position` (float: SB=0, BB=1, UTG=2, MP=3, CO=4, BTN=5)
  7. `hero_stack_bb` (Hero stack in Big Blinds)
  8. `pot_size_bb` (Total pot size in Big Blinds)
  9. `call_amount_bb` (Chips required to match the highest bet in Big Blinds)
  10. `committed_chips_bb` (Hero chips committed in the current street in Big Blinds)
  11. `hand_equity` (Hero's win probability from Monte Carlo simulation, 0.0 to 1.0)

### 2. Card Embeddings & Sequence Length
* **Card IDs**: Integer card IDs are computed as `suit * 13 + rank` (suit: $c=0, d=1, h=2, s=3$; rank: 2..A as $0..12$). Card ID `52` is the `<PAD>` token.
* **Hole Cards**: Encoded as a tensor of shape `[batch, 2]` and embedded via a shared card embedding table. The hole embeddings are summed along the card dimension to produce a `64-dimensional` hole embedding per step.
* **Board Cards**: Encoded as a tensor of shape `[batch, seq_len, 5]`. Card embedding vectors for the board are summed at each step to produce a `64-dimensional` board embedding.
* **Sequence Alignment**: The model expects a sequence length of exactly `20`. During live play or single-decision evaluation, the context sequence is padded:
  * The first `19` steps are filled with `<PAD>` cards (ID 52) and zero context.
  * The current state is placed on step `20` (index `19`).
  * Slicing `[-1]` from the sequence predictions retrieves the EV outputs for the active decision.

---

## Known Limitations & Training Improvements

### 1. Off-Policy Action EV Extrapolation (Hallucination)
* **Problem**: Because self-play training policies correctly learn to fold weak/garbage hands preflop, the model's training dataset contains **zero** postflop examples of garbage hands. Consequently, when queried on postflop garbage hand states (OOD), the self-attention mechanism extrapolates the untrained `RAISE` and `CALL` actions to arbitrary positive expected values (e.g. +10 BB EV for raising with `72o` postflop).
* **Solutions to Implement in Future Training Runs**:
  1. **Conservative Q-Learning (CQL)**: Regularize the loss function during training to actively penalize Q-values for actions that deviate from the dataset's policy (forcing untrained/OOD actions to have massive negative EVs).
  2. **Offline Data Augmentation**: Inject synthetic postflop garbage hand records into the training set, assigning explicit negative EV targets to `CALL` and `RAISE` actions (e.g. `-call_amount / BB`).
  3. **Policy-Head Masking (Actor-Critic)**: Train an explicit policy head (actor) alongside the EV head. During live play, mask out any action with a policy probability of `0.0%` (e.g. FOLD = 100%, preventing the bot from looking at hallucinated EV values).
  4. **Heuristic Overrides**: Maintain the hybrid architecture, using the **Preflop Range Chart** and **Postflop Math Engine** as safety nets to fold garbage hands before they can query the raw model.
  5. **Noisy Exploration ($\epsilon$-Greedy Opponents)**: Introduce a small probability (e.g., $\epsilon = 5\%$) during self-play simulation for opponents to take random actions (or play against a dedicated random bot in the training pool). This naturally pushes garbage hands into postflop streets, providing the model with real training examples of postflop garbage and teaching it to fold them.
  6. **Preflop Random Play Exploration**: Allow Hero (and opponents) to occasionally play garbage hands preflop during training (by choosing random preflop actions with a small $\epsilon$ probability). This allows the neural network to experience the consequences of weak preflop plays first-hand, backpropagating negative target EVs for these actions and teaching the network why folding is superior.
  7. **Hybrid Preflop Exploration Split**: To optimize learning without policy collapse, use a 3-way split for simulator preflop decision-making:
      * **5% Pure Random Exploration**: Ensures continuous baseline exploration of weak hands and unorthodox plays.
      * **50% Model Inference (Current Policy)**: Leverages the active neural network's EV predictions to drive decision-making, establishing a reinforcement feedback loop.
      * **45% Heuristic Anchoring**: Retains a solid mathematical baseline utilizing the [HeuristicEngine](file:///c:/REPO/Antigravity/AIPoker/core/models/heuristic.py) preflop range charts (raising top-tier premium hands and calling playable hands dynamically adjusted to active opponent counts). This regularizes the training data, preventing the self-play loop from collapsing into degenerate local minima.

### 2. EV as the Optimal Training Fitness Metric
* **Why EV (Q-values) is superior to Policy-only (probabilities)**:
  * In poker, the size of the pot varies continuously. Optimization metrics like binary win/loss or cross-entropy probabilities cannot capture pot geometry.
  * **Expected Value (EV) in BB** acts as the true utility function, perfectly balancing equity, bet sizes, and risk.
  * Predicting the raw EV of each action allows the model to directly maximize chip accumulation (BB/100 hands), which is the ultimate goal of poker GTO.
  * The V5 training session will continue using **Huber Loss (on BB-normalized EV targets)** as the optimal fitness parameter.

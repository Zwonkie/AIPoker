# AIPoker Roadmap

This document tracks upcoming architectural changes, major versions, and features for the AIPoker training and simulation environment.

## 🚀 Upcoming: Version 11 (V11)

V11 will focus on enhancing the curriculum, improving opponent stability, and introducing mechanistic interpretability so the model's inner thoughts can be analyzed.

### 1. Weighted Personality Focus Rounds
- Introduce a new training phase (Phase 5) where a specific personality dominates the table seating (e.g., 3 Maniacs, 1 TAG, 1 Past Self).
- **Goal:** Provide highly concentrated signal exposure to specific player archetypes while preserving crucial multiway pot dynamics (position, implied odds, multi-way callers).

### 2. Heuristic Personality Variance (Fuzzy Opponents)
- Shift opponent personality bots from unpredictable Neural Networks to **Fuzzy Heuristics**.
- Apply a Gaussian distribution to action thresholds at the start of every hand (e.g., Maniac raise threshold = `mean 20% equity, std 5%`).
- **Goal:** Provide a mathematically stable baseline that won't suffer from RL collapse, while preventing the Hero from overfitting to exact deterministic trigger points. 

### 3. Interpretable Auxiliary Heads ("Subconscious" Telemetry)
- Modify the Transformer architecture to output additional diagnostic heads alongside the standard Q-values (`EV_Fold`, `EV_Call`, `EV_Raise`).
- New heads will predict: `Opponent_Bluff_Probability`, `Predicted_Opponent_Hand_Strength`, and `Self_Perceived_Equity`.
- Train these heads via an auxiliary Cross-Entropy loss using the simulator's absolute ground truth.
- **Goal:** Crack open the "black box." When the model makes a decision, we can immediately read its auxiliary outputs to understand exactly *why* it chose that action.

### 4. Adaptive Curriculum Triggering (Stretch)
- Dynamically monitor Hero counter-strategy win rates per personality in real-time.
- If the Hero struggles against a specific archetype, automatically shift the table seating to increase exposure to that personality until the win rate recovers.

---

*Note: For detailed implementation plans, refer to the [V11 Model Specifications](.agents/skills/OFK/references/V11/model-specifications.md).*

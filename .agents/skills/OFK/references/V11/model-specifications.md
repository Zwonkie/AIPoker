# V11 Model Specifications & Training Improvements

**Date Recorded**: 2026-07-12
**Status**: IMPLEMENTED
**Related Files**: 
- [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v10/six_max_simulator.py)
- [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v10/train_selfplay.py)
- [V10 Model Specs](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V10/model-specifications.md)

## Context
V10 introduced personality-based opponent training (Maniac, Nit, Sticky) alongside curriculum stack sizing and dynamic active players. While this successfully teaches the Hero to counter individual styles, all personality exposure currently occurs simultaneously at a fixed 6-max table. The Hero sees each personality once per hand in a static seating arrangement, giving it equal — and diluted — exposure to each style.

V11 should improve personality counter-strategy learning by introducing **Weighted Personality Focus Rounds** and refining the curriculum to give the Hero concentrated, high-signal exposure to specific styles without sacrificing multiway dynamics. V11 also aims to introduce **Mechanistic Interpretability** and **Heuristic Personality Variance**.

## V11 Proposed Changes

### 1. Weighted Personality Focus Rounds (New Curriculum Phase)

Instead of training heads-up against each personality (which would lose critical multiway dynamics like position, implied odds, and multi-caller pot geometry), V11 introduces **focus rounds** where a single personality dominates the table seating.

#### Why not Heads-Up first?
- HU poker and 6-max poker are fundamentally different games. A model trained HU would need to "unlearn" habits when transitioning to 6-max.
- The input encoding (`active_opponents_mask`, `opponents_stacks`, seat-aware action history) is designed for multiway. Training HU leaves most input channels at zero.
- Multiway dynamics (e.g., "the Maniac raised, but the Nit cold-called — what does that mean?") are the hard part and cannot be learned in HU.

#### Focus Round Design
Run 6-max sessions where one personality occupies **multiple seats**. Example configurations:

| Focus Target | Seat 0 | Seat 1 | Seat 2 | Seat 3 | Seat 4 | Seat 5 |
|:---|:---|:---|:---|:---|:---|:---|
| Maniac Focus | Hero | Maniac | Maniac | Maniac | Past Self | TAG |
| Nit Focus | Hero | Nit | Nit | Nit | Past Self | TAG |
| Sticky Focus | Hero | Sticky | Sticky | Sticky | Past Self | TAG |
| Mixed (current V10) | Hero | Maniac | Nit | Sticky | Past Self | TAG |

- Past Self and TAG Bot always remain for stability/baseline anchoring.
- Focus rounds give 3x the signal density for the target personality per hand.
- The Hero still learns multiway pot dynamics, position, and multi-caller ranges.

#### Proposed Curriculum Integration
```
Phase 1: 100BB Static (0 - 10k hands)         — Mixed seating (current V10)
Phase 2: Moderate Stacks (10k - 30k hands)     — Mixed seating (current V10)
Phase 3: Extreme Stacks (30k - 50k hands)      — Mixed seating (current V10)
Phase 4: Dynamic Active Players (50k - 70k)    — Mixed seating (current V10)
Phase 5: Personality Focus Rounds (70k - 100k)  — Rotate focus every 10k hands
```

Phase 5 cycles through: Maniac Focus → Nit Focus → Sticky Focus, spending ~10k hands on each. This gives the model a final "intensive study" phase after it already understands general 6-max play.

### 2. Adaptive Focus Triggering (Stretch Goal)
Rather than a fixed Phase 5, monitor the Hero's counter-strategy performance per personality in real-time. If the Hero's winrate against Maniacs drops below a threshold while it's crushing Nits, dynamically shift the seating to give it more Maniac exposure. This creates an adaptive curriculum that self-corrects weaknesses.

### 3. Interpretable Auxiliary Heads (The "Subconscious" Approach)
Currently, the model acts as a black box that just outputs `[EV_Fold, EV_Call, EV_Raise]`. In V11, we will modify the network architecture to output additional diagnostic heads alongside the Q-values to decode its "thinking state."
- **Auxiliary Outputs:** `Opponent_Bluff_Probability`, `Predicted_Opponent_Hand_Strength`, and `Self_Perceived_Equity`.
- **Training Method:** Since the simulator knows the ground truth (e.g., the opponent's actual hole cards and equity), we add an auxiliary loss function (Cross-Entropy) that trains these heads to predict the true state of the simulation.
- **Result:** During live play, we can read these auxiliary outputs to understand *why* the model made a decision (e.g. "I called because Opponent_Bluff_Probability is 85%").

### 4. Heuristic Personality Variance (Fuzzy Opponents)
Training against rigid NNs or static heuristics allows the Hero model to overfit to highly specific trigger points. In V11, the personality bots will be driven by **Fuzzy Heuristics**:
- **Gaussian Trait Distribution:** Instead of a Maniac always raising with exactly >20% equity, the trigger threshold is sampled from a normal distribution (`mean=20%, std=5%`) at the start of every hand.
- **Benefit:** This prevents the Hero from learning a brittle exploit (like "if Maniac bets exactly X, he has exactly Y"). It forces the Hero to learn robust, generalized counter-strategies against an *archetype* rather than a specific deterministic bot.
- **Reliability:** Heuristics provide a much more stable training baseline than potentially collapsing Neural Network opponents, ensuring the Hero is always training against mathematically sound (even if imperfect) poker concepts.

### 5. All-In Telemetry Fix
V10 had a bug where the Equity Matrix All-In column always showed 0.0% because the telemetry checked `hero_stack == 0` from the *pre-action* decision point snapshot (which is always > 0 before the action is applied). V11 must use the explicit `is_all_in` flag set *after* the action is resolved.

> [!NOTE]
> This fix has already been applied to the V10 codebase in `telemetry.py` and `six_max_simulator.py` as of 2026-07-12. V11 inherits this fix.

### 6. Unified Training Log (`active_training.log`)
All training runs, regardless of personality or model version, must output to `active_training.log`. The dashboard parser and HTML UI read exclusively from this file, eliminating the need to track per-personality log filenames.

> [!NOTE]
> This convention has been established in V10's `train_all.ps1` and `start_dashboard.ps1` as of 2026-07-12. V11 should maintain it.

### 7. Core Architectural Remediations (Fixing Mode Collapse)
V10 suffered a complete attention collapse (100% Calling Station) because the architecture proposed in V7 was never properly implemented. V11 must enforce the following structural overhauls to the neural network and training loop:

- **Multi-Action Target Generation (All-Action Loss):** The V10 training loop calculates MSE loss only on the action that was actually taken (`b_sa`). This allows unchosen actions to drift infinitely, causing the network to degenerate into a single action. V11 must calculate target Expected Values for *all three actions* (`Fold, Call, Raise`) using the math evaluator, constraining the entire EV prediction vector simultaneously.
- **Autoregressive Causal Masking:** The `_generate_causal_mask` in V10's `poker_transformer.py` uses `~torch.eye()`, which isolates every single step and completely destroys sequence memory. V11 must implement a standard causal mask (e.g., `torch.triu(..., diagonal=1)`) so the model can read past actions.
- **Dense Chronological Sequence Packing:** V10 pads sequences to `max_seq_len=20` but leaves steps 0-18 entirely blank, placing data only at index 19. V11's `vectorize_hand_samples` must pack sequences sequentially and dynamically to prevent the attention mechanism from diluting gradients with padding.

### 8. Profile Blindness Remediation (HUD Vectorization)
A critical logic bug was discovered during V11 training where the Neural Network was blind to opponent personality vectors (VPIP/AGG). It was unable to deviate its EV predictions based on opponent types.
- **The Simulator Bug**: `six_max_simulator.py` was hardcoding `Green/Green` (TAG) HUD stats for all active opponents during `_query_model_decide()`, forcing the Hero to play the entire rollout under the assumption that all opponents were TAGs, regardless of their actual archetype.
- **The Vectorization Bug**: `contract_v11.py` and `train_selfplay.py` were hardcoding `0.3` and `0.4` as the global `opp_vpip_norm` and `opp_agg_norm` inputs into the context vector sequence.
- **The Fix**: Both systems were patched on 2026-07-13. `six_max_simulator.py` now maps live seat histories to specific color bands dynamically during model inference. `contract_v11.py` and `train_selfplay.py` now calculate the active global norms dynamically by iterating over the `active_opponents_mask` and averaging the specific VPIP/AGG inputs for the active seats. This guarantees the model receives variance in its HUD inputs, allowing the linear `state_proj` layer to assign proper weight gradients.

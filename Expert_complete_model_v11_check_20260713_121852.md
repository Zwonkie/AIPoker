# Expert Complete Model Check Report: Herocules V11
**Date**: 2026-07-13 12:18:52

## 1. Data Contract & Vectorization (The Input Alignment)
*   **Vectorization Dimension Match**: The docstring in `contract_v11.py` indicates a 31-feature context, but the code actively generates and outputs 35 features (10 global + 5*5 opponent). Fortunately, the model architecture (`PokerEVModelV4`) correctly initializes with `context_dim=35`, avoiding a crash, but the documentation is stale.
*   **Causal Masking and Action Shifting**: Implemented correctly. The model shifts `actions` by 1 (`shifted_actions[:, 1:] = actions[:, :-1]`) and applies a causal upper-triangular mask, physically preventing the network from seeing future sequence actions during training. Target leakage is properly mitigated.
*   **Inactive Opponent Padding**: Inactive opponents are padded with `pos_val = -1.0` and static HUD defaults (0.3 VPIP / 0.4 AGG). The model could inappropriately anchor onto these static values if not heavily regularized.

## 2. The Training Loop & Loss Calculation (The Gradients)
*   **Loss Component Scaling**: The auxiliary loss is heavily weighted (`final_loss = final_loss_q + 10.0 * final_loss_aux`). Because `final_loss_q` operates on Huber loss for Q-values bounded `[-100, 100]`, and the auxiliary targets are MSE on `[0, 1]`, this scaling prevents the Q-values from completely washing out the auxiliary gradients.
*   **Target Alignment**: `train_selfplay.py` safely aligns the predicted action dimension (Fold=0, Call=1, Raise=2) with the `target_evs` arrays from the simulator.

## 3. Simulation Environment & Ground Truth (The Reality)
*   **CRITICAL FLAW - Unchosen Action Counterfactuals**: In `six_max_simulator.py`'s `_calculate_mc_target_evs`, the `ev_call` and `ev_raise` for unchosen actions are calculated as `true_equity * pot_after_call - to_call`. This is a "Showdown EV" formula that assumes no further betting will occur! For preflop and flop decisions, this ignores implied odds and future streets entirely. The chosen action receives a true Monte Carlo rollout EV, while the unchosen actions receive a naive, single-street Showdown EV. The model is being trained on two conflicting mathematical paradigms simultaneously, heavily distorting the policy (e.g. undervaluing preflop calls with suited connectors).
*   **Clairvoyant Target Generation**: The target EVs are generated using `true_equity` against the specific, exposed opponent hole cards rather than a probabilistic range. While off-policy Q-learning can converge in expectation, this introduces massive variance on a per-hand basis.

## 4. Model Architecture & Extreme Behavior (The Brain)
*   **Architecture Alignment**: `PokerEVModelV4` successfully represents the sequences, and the Transformer correctly avoids padding tokens using `key_padding_mask=(b_m == 0.0)`. State projection efficiently aggregates 159 dims down to 128.
*   **Guardrail Masking**: In `decision.py`, the `use_math_engine` forcefully overrides the model to `FOLD` if raw equity is lower than pot odds minus a small buffer. This means if the model has collapsed or learned bad calling habits (due to the flawed target EVs), the math guardrail actively hides the model's incompetence during live evaluation. The model's raw action entropy and choices must be evaluated without the math engine to see its true behavior.

## 5. Reward System & Target Generation (The Feedback Loop)
*   **Forward-Looking EVs**: The Q-value calculation for the chosen action (`mc_return = (final_profit + dp['committed_before']) / bb`) is mathematically brilliant. It perfectly removes sunk costs, ensuring that folding correctly resolves to an exact `0.0` EV target.
*   **Target Clipping**: The target EVs are properly clipped to `[-100, 100]`, safeguarding the network from catastrophic gradient explosions resulting from deep-stack all-ins.

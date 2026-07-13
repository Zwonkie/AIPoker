# Deep Dive: V10 Simulation Harness & Logic

The V10 self-play RL system uses a complex multi-process simulation harness (`SixMaxSimulator`) to generate training data. This document breaks down the entire lifecycle of a simulated hand, the environmental conditions, opponent behavior mappings, and how Monte Carlo (MC) returns are translated into gradients.

## 1. The Environment (SixMaxSimulator)
The environment simulates 6-Max No Limit Hold'em. 

### Seat Assignments & The "League"
To prevent the model from overfitting to a single playing style, we populate the table with a diverse "league" of opponents. 
In V11, we have shifted away from rigid neural network adversaries (V10) to a fully heuristic-driven **Fuzzy Opponent Pool**. These bots use gaussian noise to slightly randomize their VPIP, AGG, and Bluff frequencies at the start of every hand to prevent the Hero from memorizing static triggers.
- **Seat 0 (Hero)**: The active Herocules transformer model being trained.
- **Seat 1-5**: Populated each hand by sampling *independently* from a **weighted, disciplined-majority opponent pool** (`OPPONENT_POOL_STYLES`/`OPPONENT_POOL_WEIGHTS` in `six_max_simulator.py`): `tag 0.30 / past 0.25 / nit 0.20 / maniac 0.15 / fish 0.10`. Independent per-hand draws also break positional overfitting (the NN cannot assume "Seat 1 is always a Maniac" and must rely on HUD stats to classify opponents). The pool is deliberately ~75% disciplined / ~25% spew-fish so the Hero is **not** trained against a table that is half deliberately-bad players (which previously pulled its policy into a loose-collapse — see `V11/issues-and-fixes.md`).
  - **Maniac/LAG**: Hyper-aggressive, loose preflop, very high AGG frequency.
  - **Nit**: Extremely tight, only plays premium hands (VPIP ~11%).
  - **Calling Station / Fish (Sticky)**: Plays far too many hands (VPIP ~45-50%) but is extremely passive.
  - **TAG**: A baseline Tight-Aggressive opponent (VPIP ~22%, AGG ~45%). Static heuristic, no NN.
  - **Past Self**: A frozen *lagged snapshot* of the training Hero, re-saved every 5,000 hands to `v11_past_checkpoint.pth` and loaded back as an opponent. Falls back to the TAG heuristic until the first snapshot of the current run exists. This is the true self-play adversary.

> [!IMPORTANT]
> **Stats are bucketed per personality, not per table seat.** Because the style occupying a seat is resampled every hand, cumulative VPIP/AGG/profit/exploitation are keyed by a fixed `STYLE_SLOT` (`0 Hero | 1 Maniac | 2 Nit | 3 Sticky | 4 Past | 5 TAG`), which matches the hard-coded dashboard row labels 1:1. Each hand builds a `seat_slot` map and routes every stat increment through it. (Earlier V11 keyed stats by table seat, which averaged all archetypes together and made every seat's HUD read an identical ~25%/43% blob — see `V11/issues-and-fixes.md`.) The archetype **action-forcing** in `_opponent_decide` (which pins Maniac/Nit/Sticky to their target ranges) applies to heuristic opponents too — it is no longer gated on an opponent NN being present.

### The Curriculum Learning Stack Sizing
The environment slowly increases the difficulty by widening the stack depth variance across three phases:
- **Phase 1: 100BB Static (0 - 10,000 hands)**
  Every player gets exactly 100 Big Blinds. This provides a stable environment to learn basic preflop ranges and simple postflop geometry.
- **Phase 2: Moderate Stacks (10,000 - 30,000 hands)**
  Stacks are randomized using a Gaussian distribution (`mean=100, std=10`), clamped between 80 BB and 120 BB. Introduces slight SPR (Stack-to-Pot Ratio) variance.
- **Phase 3: Extreme Stacks (> 30,000 hands)**
  Stacks are randomized wildly (`std=90`), clamped between 10 BB (short-stacked shover) and 300 BB (deep stack). Forces the model to understand ICM/survival pressure and deep-stack implied odds.
- **Phase 4: Dynamic Active Players (> 50,000 hands)**
  Randomly folds 0 to 4 opponents pre-flop using a weighted distribution (favoring 0-2 folds). This forces the model to learn heads-up and short-handed dynamics rather than assuming 6-max every hand.
- **Phase 5: Focus Rounds (>= 75,000 hands)**
  At the beginning of each simulation batch, a specific archetype (Maniac, Nit, or Calling Station) is randomly selected. The simulator forcibly populates 3 or 4 of the 5 opponent seats with this archetype, plunging the model into an extreme meta-game environment to harden its counter-strategies.

## 2. Decision Logic & The Bootstrap Alpha
The `SixMaxSimulator` does not immediately hand full control over to the randomly initialized Neural Network. 

We use a **Heuristic Bootstrap Decay (`bootstrap_alpha`)**:
- **0 - 10,000 Hands**: `alpha = 1.0` (100% Heuristic Control). The NN acts as a silent observer, learning the mapping from states to actions without destroying its own training data by making random chaotic raises.
- **10,000 - 30,000 Hands**: `alpha` decays linearly from 1.0 to 0.0. The NN slowly takes over driving the car.
- **> 30,000 Hands**: `alpha = 0.0`. The NN is fully in charge.

> [!TIP]
> **5% Exploration Anchor**: Even when `alpha = 0.0`, the system forces a completely random action 5% of the time. This applies to **ALL** bots at the table. To prevent heuristic poisoning, the simulation protects premium hands (Equity > 0.70) from randomly folding, restricting the exploration strictly to calls and raises for those nodes.

## 3. The Data Structure (`HandRecordV4`)
As the hand plays out, the simulator creates a sequence of `DecisionPoints` for the Hero. Each point records:
- **Board/Hole Cards**
- **Pot Size & Call Amount** (Pot Odds)
- **Active Opponents Mask & Stacks**
- **Street & Action History sequence** (A chronologically ordered list of actions on the current street. Possible states: `"f"` for Fold, `"c"` for Call/Check, `"r"` for Raise/Bet. e.g., `["c", "r", "c"]` means Check, Bet, Call).
- **Opponent HUD Stats** (Rolling VPIP/AGG mapped to "Blue", "Green", "Yellow", "Red" categorical buckets)
- **Opponent Ground Truths** (The simulator looks at opponent hole cards and calculates the `opp_strength` (max opponent equity) and `opp_bluff_prob` (probability that the betting opponent has garbage equity < 33%)).

Crucially, it also computes the mathematical **Equity** of Hero's hand against random opponent ranges using a fast C++ Monte Carlo evaluator.

## 4. The Loss Function, Targets, & Subconscious Heads
In V11, the loss calculation was fundamentally overhauled to solve sequence destruction and mode collapse. 

### Q-Target Generation (All-Action Counterfactual + Realized Override)
*(Evolved over 2026-07-13 across two fixes — see `V11/issues-and-fixes.md`. First a loose-collapse ratchet was fixed by adding a Fold baseline; then the 200k model was found to **raise everything** (0-equity air included) because the Call/Raise heads had no signal on states where those actions weren't taken. The current scheme trains all three heads.)*

For each Hero decision point the target `[t_fold, t_call, t_raise]` and a **per-action loss weight** `[w_fold, w_call, w_raise]` (`target_w_seq`) are built as follows:
1. **Go-forward MC return** for the action actually taken: `mc_return = (final_profit + committed_before) / bb` (excludes sunk cost, in Big Blinds).
2. **Preflop tightness prior** (anti-ratchet): if the taken action is call/raise on the preflop and Hero equity is below the fair multiway share `1.15 × 1/(active_opps+1)`, `mc_return` is lowered by up to `TIGHTNESS_PENALTY_BB` (4 BB) in proportion to the equity shortfall.
3. **Clip** every target to ±`TARGET_CLIP_BB` (**±40 BB**, tightened from ±100) to damp the fat right tail (occasionally stacking a fish) that biased "enter" Q-values upward.
4. **Counterfactual EVs for the untaken actions**: the simulator's true-equity Monte Carlo (`_calculate_mc_target_evs`) supplies `[ev_fold≈0, ev_call, ev_raise]` in chips → BB. These are correctly **negative for weak hands** (e.g. `ev_call = equity·pot − cost = −cost` for 0-equity air), which is the "aggression on air is −EV" signal that was previously missing.
5. **Assemble targets & weights**:
   - **Fold head** → `0.0`, weight `1.0` (exact go-forward baseline).
   - **Taken action** → its realized `mc_return`, weight `1.0` (ground truth).
   - **Untaken actions** → the MC counterfactual EV, weight `COUNTERFACTUAL_WEIGHT` (**0.5** — estimates, so trusted less than realized/ground-truth targets).
6. **Weighted Huber Loss**: `loss_q = Huber(preds, targets) * (target_w_seq * loss_mask)`, normalized by the total weight. Every head now gets a grounded target at every step: a stable Fold=0 baseline, the realized outcome for what was actually done, and a pessimistic (often negative) counterfactual for the roads not taken — closing both the loose-collapse ratchet *and* the raise-everything hallucination.

The tuning constants (`TARGET_CLIP_BB`, `TIGHTNESS_PENALTY_BB`, `ENTRY_EQUITY_MARGIN`, `COUNTERFACTUAL_WEIGHT`) live at the top of `train_selfplay.py`.

### Showdown & Side Pot Resolution
To guarantee mathematically perfect `mc_return` targets, the simulator employs a highly robust **Side Pot Slicing Algorithm** during showdown. Additionally, for terminal states reached before the river (all-in scenarios), the simulator carefully evaluates all active players against a single, shared, deterministic board runout to prevent MC evaluation fragmentation.

### Interpretable Auxiliary Heads (The Subconscious)
To force the model to build an internal world model, the V11 architecture splits the final representation into three smaller auxiliary heads before the main Q-value head:
- `head_bluff`: Predicts `opp_bluff_prob`.
- `head_strength`: Predicts `opp_strength`.
- `head_equity`: Predicts the mathematical `equity`.

The Mean Squared Error (MSE) of these three heads is summed up to create the `loss_aux`.
Because the Huber Loss for Q-values dominates the gradients (scaled in Big Blinds), the auxiliary heads are artificially scaled up. The final backpropagated loss is: `Total_Loss = loss_q + 10.0 * loss_aux`.

> [!IMPORTANT]
> **Personality shaping is done via action-forcing, not loss-target interception.**
> For personality archetypes (`maniac`, `nit`, `sticky/fish`) the simulator reads that personality's *own* rolling VPIP/AGG (from its `STYLE_SLOT` bucket) inside `_hero_decide` / `_opponent_decide` and, when it is drifting off target, **overrides the action actually taken** (e.g. forces a Maniac to raise/enter when its VPIP is too low, forces a Nit to fold). This biases the training *data distribution* toward the archetype rather than editing Q-targets. With the fold-baseline redesign above (Fold target fixed at 0), there is no loss-side "Fold EV subtraction" — any earlier description of a target-interception "PID" mechanism is obsolete and was never present in the code.

# Deep Dive: V10 Simulation Harness & Logic

The V10 self-play RL system uses a complex multi-process simulation harness (`SixMaxSimulator`) to generate training data. This document breaks down the entire lifecycle of a simulated hand, the environmental conditions, opponent behavior mappings, and how Monte Carlo (MC) returns are translated into gradients.

## 1. The Environment (SixMaxSimulator)
The environment simulates 6-Max No Limit Hold'em. 

### Seat Assignments & The "League"
To prevent the model from overfitting to a single playing style, we populate the table with a diverse "league" of opponents. 
In V11, we have shifted away from rigid neural network adversaries (V10) to a fully heuristic-driven **Fuzzy Opponent Pool**. These bots use gaussian noise to slightly randomize their VPIP, AGG, and Bluff frequencies at the start of every hand to prevent the Hero from memorizing static triggers.
- **Seat 0 (Hero)**: The active Herocules transformer model being trained.
- **Seat 1-5**: Populated dynamically from the fuzzy opponent pool using weighted archetypes. **Crucially, in V11, the archetypes (Maniac, Nit, Calling Station, TAG) are randomly shuffled across Seats 1-5 at the start of every single hand.** This prevents the NN from overfitting to positional indices (e.g. assuming Seat 1 is always a Maniac) and forces it to strictly rely on HUD stats to classify opponents.
  - **Maniac/LAG**: Hyper-aggressive, loose preflop, very high AGG frequency.
  - **Nit**: Extremely tight, only plays premium hands (VPIP ~11%).
  - **Calling Station / Fish**: Plays far too many hands (VPIP ~45%) but is extremely passive.
  - **TAG**: A baseline Tight-Aggressive opponent (VPIP ~22%, AGG ~45%).

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

### Multi-Action Target Generation (All-Action Loss)
Instead of only penalizing the Q-value of the action the Hero *actually took*, the V11 simulator runs a real-time Monte Carlo tree search for the opponent models at every decision point. It estimates the true mathematical Expected Value of **Fold**, **Call**, and **Raise** independently.
1. The batch of chronologically packed sequences is passed through the Transformer model.
2. The model outputs Q-Values (Expected Value) for Fold, Call, and Raise for every step in the hand sequence.
3. The simulator's estimated EVs for Fold, Call, and Raise become the target tensors (clipped between -100 BB and +100 BB to prevent gradient explosions).
4. For the specific action the Hero *actually took*, the estimated EV is overridden with the actual `mc_return` (the true profit experienced at the end of the hand).
5. A Huber Loss is calculated across **all three actions simultaneously**.

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
> **The PID Target Mechanism**
> For personality bots (`maniac`, `nit`, `sticky`), we intercept the target Q-values right before calculating the loss. We read the Hero's rolling VPIP/AGG. If the Maniac is folding too much (VPIP < 65%), we dynamically subtract massive value from the `Fold EV` target. This acts like a gravitational pull, forcing the gradient to discourage folding until the VPIP threshold is breached.

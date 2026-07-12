# Project Notes & Memory

This file serves as a dedicated space for user-requested notes, memory, and custom instructions. Ask me to append or update anything in this file at any time.

---

## Notes
- The preflop starting hand equity dataset (`preflop_equities.csv` / `preflop_equities.md`) can be used as a pre-calculated lookup table during preflop training runs instead of calculating Monte Carlo equity on the fly.

---

## Pluribus V3 Model Specifications & Training Assumptions

### 1. Network Parameters & Architecture
The V3 model (`expert_v3_selfplay.pth`) uses a **9-dimensional context vector** instead of the legacy 4D or 7D vectors:
* **Feature 0: Position**: Normalized by dividing by 10.0.
* **Feature 1: Bankroll**: Player stack size converted to Big Blinds (BB) and normalized by dividing by 500.0.
* **Feature 2: Pot Size**: Normalized by dividing by 500.0.
* **Feature 3: Equity**: Monte Carlo win equity (0.0 to 1.0).
* **Feature 4: Pot Odds**: Facing bet ratio, calculated as $\frac{\text{Call Amount}}{\text{Pot Size} + \text{Call Amount}}$.
* **Feature 5: Num Opponents**: Number of active players remaining in the hand, normalized by dividing by 10.0.
* **Feature 6: Street Level**: Stage of play (0.0: Pre-flop, 0.33: Flop, 0.66: Turn, 1.0: River).
* **Feature 7: Opponent VPIP**: Maximum VPIP norm of the active opponent pool (0.15 to 0.45).
* **Feature 8: Opponent AGG**: Maximum Aggression norm of the active opponent pool (0.10 to 0.60).

**Card Embeddings:** Uses a 32-dimensional embedding layer for 2 hole cards and up to 5 community cards.
**Action Sequence Tracking:** Uses a Gated Recurrent Unit (GRU) with a 64-dimensional hidden layer tracking the action sequence (Vocabulary: Pad, Bet, bet, check, Check, raise, fold, All-in, Q-fold).

### 2. Self-Play RL Training Assumptions
* **Simulation Engine:** Trained headlessly via `HeadlessPokerSimulator` under No-Limit 6-Max rules. 
* **Opponent Bot Pool:** Rather than training strictly against itself, the agent trains against a weighted mixture of low-stakes bot profiles:
  * **Fish (Loose-Passive):** 30%
  * **TAG (Tight-Aggressive):** 25%
  * **Maniac (Loose-Aggressive):** 25%
  * **Nit (Tight-Passive):** 20%
* **EV Optimization Target:** Learns directly from *Sunk-Cost-Corrected Expected Value (EV)* targets (`target_ev`), meaning the neural network optimizes predicted EV for taking actions relative to future stack adjustments, rather than absolute chip win/loss.
* **Hyper-parameters:**
  * **Optimizer:** Adam
  * **Learning Rate:** `1e-3`
  * **Loss Function:** MSE (Mean Squared Error) between predicted EV of action taken and actual simulated target EV.
  * **Batch Size:** `256`
  * **Epochs per Batch:** `3`
  * **Simulation Batch Size:** `10,000` hands simulated per batch iteration.
  * **Simulator Equity Speedup:** Employs only `50` Monte Carlo runs per hand inside the simulator (compared to 10k runs used during live capture) to drastically accelerate learning speed.
  * **Stack Bounds:** Opponent stacks are randomly initialized between `10` BB and `400` BB.

---

## Pluribus V4 Model Specifications & Training Assumptions

### 1. Network Parameters & Architecture (Enhanced Capacity)
To support multi-player 6-Max and sequential board states, V4 moves to a **Decision Transformer** architecture and leverages the **Nvidia RTX 4080 GPU** (CUDA) to scale model capacity:
* **Sequence Representation:** $S_0, A_0, S_1, A_1, ..., S_t$. Tracks complete board state history snapshots.
* **State Vector $S$ (9D + BB Ratios + Seat HUDs):**
  * **9D Context features** (same as V3).
  * **Absolute BB Ratios:** $Pot / BB$ and $Bet / BB$ (helps differentiate absolute risk vs ratios).
  * **Full Seat HUD Matrix:** Seats 1-5 relative to Hero (VPIP, AGG, active mask, stack sizes).
* **CUDA Hardware Acceleration:** Shift device mapping to `torch.device('cuda')` to leverage RTX 4080 cores.
* **Model Capacity Scaling:**
  * **Card Embeddings:** Increased from 32-dim to **64-dim**.
  * **Transformer Embedding Dimension ($d_{model}$):** **128-dim** (replaces 64-dim GRU).
  * **Attention Mechanism:** 4 attention heads, 3 transformer layers.
  * **Feed-Forward Dimension (FFN):** **512-dim**.

### 2. Multi-Player Self-Play RL Training Assumptions
* **6-Max Simulation Engine:** Simulates full 6-max tables headlessly. 5 seats are populated by active bots chosen from the Opponent Pool (`TAG`, `Maniac`, `Fish`, `Nit`).
* **Preflop/Postflop Resolution:** 200 Monte Carlo simulations run on *all* streets (including Preflop, adjusted for the number of active players).
* **Simplified All-Ins:** No pot-splitting calculations. Stack risk is capped at the lowest active stack for simple all-in EV convergence.
* **Big Blind Normalization:** All input stack sizes, pot sizes, and call amounts are normalized relative to the current `big_blind` to map features into stable range spaces (e.g., stacking bounds $[0.0, 1.0]$ representing 0 to 400 BB). Furthermore, the **Target EV is divided by the big blind**, expressing expected value predictions in BB units rather than raw chips. This prevents gradient explosion and allows the model to generalise to any table stakes.

### 3. V4 Simulation Output JSON Schema
```json
{
  "hand_id": 10542,
  "hero_cards": ["Ah", "Ks"],
  "opponents_profiles": {
    "seat_1": {"vpip": 0.15, "agg": 0.10, "style": "nit"},
    "seat_2": {"vpip": 0.55, "agg": 0.20, "style": "fish"},
    "seat_3": {"vpip": 0.45, "agg": 0.60, "style": "maniac"},
    "seat_4": {"vpip": 0.25, "agg": 0.50, "style": "tag"},
    "seat_5": {"vpip": 0.25, "agg": 0.50, "style": "tag"}
  },
  "final_hero_profit": 15.0,
  "decision_points": [
    {
      "step": 0,
      "street": 0,
      "board": [],
      "hero_position": 0,
      "pot_size": 4.5,
      "big_blind": 1.0,
      "call_amount": 2.0,
      "hero_stack": 98.0,
      "active_opponents_mask": [1, 1, 1, 0, 1],
      "opponents_stacks": [100.0, 95.0, 80.0, 0.0, 110.0],
      "action_history": ["r", "c", "c"],
      "equity": 0.612,
      "action_taken": 2,
      "chips_committed_before": 1.0,
      "target_ev": 16.0
    }
  ]
}
```

*Note: The JSON schema represents raw chip outputs from the simulator. During the vectorization step, these parameters are dynamically divided by `big_blind` to produce normalized training tensors.*



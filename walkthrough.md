# Walkthrough: IRC Poker Dataset Parsing

I have successfully finished building the pipeline and parsing the raw IRC dataset! We now have three distinct, massive datasets mapped to skill tiers, perfectly formatted to train your mixture-of-experts model.

## What Was Accomplished

1. **Dataset Download**: Successfully downloaded the University of Alberta `IRCdata.tgz` archive (almost 1 GB of heavily compressed raw logs).
2. **Parser Construction**: Wrote a memory-efficient `irc_parser.py` that streams the tar archive without needing to unpack everything to disk.
3. **Data Generation**: Extracted exactly 25,000 complete hands for each skill tier, resulting in three large `jsonl` files.

## Verification

The generated datasets are incredibly rich in features. The parser successfully extracts all preflop and postflop variables, including relative positional data and bet-sizing contexts.

### Output Files
- `hands_tier1.jsonl`: ~38 MB (25,000 Beginner Hands)
- `hands_tier2.jsonl`: ~29 MB (25,000 Intermediate Hands)
- `hands_tier3.jsonl`: ~25 MB (25,000 Advanced Hands)

### Data Sample
Here is an example of what a parsed player state looks like within a single hand:
```json
"gunner": {
  "position": 9,
  "bankroll": 785,
  "pocket_cards": ["8h", "Ad"],
  "total_bet": 60,
  "total_win": 120,
  "bets": [
    {"stage": "p", "actions": "c"},
    {"stage": "f", "actions": "c"},
    {"stage": "t", "actions": "c"},
    {"stage": "r", "actions": "b"}
  ]
}
```

> [!NOTE]
> The `bets` array tracks the precise string of actions across all streets (`p`reflop, `f`lop, `t`urn, `r`iver). This is exactly the data your model needs to understand aggression and opponent modeling!

## Next Steps

Now that we have the raw, tiered data, the next major step is to convert these raw strings into **Feature Tensors** (vectors) that your ML models can ingest. We'll need to define a state representation that encodes:
- The Hero's 2-card combo
- Board community cards
- The action history array
- Positional offsets

## ML Architecture Decision

We have evaluated the tradeoffs between different ML paradigms for this problem:

### Engine Choice: Deep Neural Networks (PyTorch) vs. XGBoost
We have chosen to build a **Deep Neural Network in PyTorch** rather than a tree-based ensemble (like XGBoost). 
- **Why:** Poker contains complex sequential data (the exact order of bets, raises, and calls across multiple streets) and combinatorics (card interactions like flush draws and blockers). While XGBoost is extremely fast, it requires heavy manual feature engineering to understand sequences and cards. A PyTorch architecture (using Embeddings for cards and an RNN/Transformer for betting history) can natively learn the mechanics of the game and the "story" of the hand.

### Strategic Objective: Maximizing EV over Behavioral Cloning
We have explicitly rejected **Behavioral Cloning** for the lower tiers. 
- **Why:** Behavioral Cloning (training the bot to mimic the human actions in the dataset using Cross-Entropy Loss) would force our Tier 1 model to adopt the exact same bad habits as Tier 1 players (e.g., calling too widely, playing passively).
- **Our Goal:** We do not want to *emulate* low-tier players; we want to *exploit* them. Therefore, our model will be trained using **Expected Value (EV) Regression / Offline Reinforcement Learning**. The fitness parameter will be predicting the financial outcome (`total_win` minus `total_bet`). This teaches the bot to find the most profitable counter-strategy against the specific leaks inherent to each tier.

## Vectorization Strategy & Tensors

To train the PyTorch models, we must translate the raw JSON into normalized numerical tensors for a single decision point. We will use the following schema:

### 1. Target Variable (Y): Expected Value
- **Raw Data:** `total_win - total_bet`
- **Vectorization:** Normalized to **Big Blinds (BB)** to scale uniformly across different limit structures.
- **Shape:** `[1]` (Continuous scalar)

### 2. Card Representation (Embeddings)
- **Vectorization:** Cards (e.g., `"8h"`) are mapped to integers `(0-51)`. These integers are passed through a PyTorch `nn.Embedding` layer. This allows the network to natively learn rank distances, flush potentials, and blockers without manual feature engineering. Missing community cards are padded with `<PAD>` (index 52).
- **Shape:** `[2]` (Hole Cards) and `[5]` (Board Cards)

### 3. State & Context Features (Chips & Position)
- **Raw Data:** Pot sizes, bankrolls, amount to call, position.
- **Vectorization:** 
  - Positions are one-hot encoded `[0-9]`.
  - Chip counts (`bankroll`, `current_pot_size`, `amount_to_call`) are normalized by the Big Blind.
  - We explicitly provide a pre-calculated `pot_odds` ratio (`amount_to_call / (current_pot_size + amount_to_call)`) to accelerate convergence.
- **Shape:** `[~25]` (Continuous & Boolean vector)

### 4. Action Sequence (RNN/Transformer Input)
- **Raw Data:** String of betting actions (e.g., `"Bcr"` for Bet, Call, Raise).
- **Vectorization:** We use a **Tokenized Sequence Strategy**. We define a vocabulary (e.g., `B=1, c=2, r=3, f=4, PAD=0`). The action string is converted to an integer sequence `[1, 2, 3, 0, 0]`. This sequence is processed by an RNN or Transformer layer to capture the temporal "story" of the hand and the precise flow of aggression.
- **Shape:** `[Seq_Length]`

### Network Merging
The network will process these inputs in parallel:
1. The **Sequence RNN** processes the `action_history` into an `[Aggression_Vector]`.
2. The **Embeddings** process the hole/board cards into a `[Card_Vector]`.
3. The **Context Tensor** holds the chip/pot data in a `[Chip_Vector]`.

These three vectors are concatenated `(Aggression + Cards + Chips)` and passed into the final dense layers to output the predicted Expected Value (EV).

---

Would you like to start building the python script to vectorize these JSON files (`tools/train_data_builder/vectorize_hands.py`), or begin designing the PyTorch model class directly?

# Standard Sensitivity Analysis Protocol

This document defines the standard parameters, interesting situations, and expected outcomes for conducting a comprehensive sensitivity analysis of the Poker AI models. Whenever a "standard sensitivity analysis" is requested, the models should be evaluated against the scenarios defined in this document.

## 1. Parameters for Analysis

When performing a broad sensitivity analysis, the following parameters should be swept across their relevant ranges:

*   **Equity**: The probability of winning the hand. 
    *   *Sweep Range*: 10% to 90% (in 10% increments).
*   **Pot Odds**: The ratio of the call amount to the total pot (including the call). 
    *   *Sweep Range*: 0% (checked to us), 15% (small bet), 25% (half pot), 33% (pot sized bet), 50% (overbet).
*   **SPR (Stack-to-Pot Ratio)**: The ratio of the effective stack size to the pot size.
    *   *Sweep Range*: 0.5 (pot committed), 3.0 (medium depth), 15.0+ (deep stacked).
*   **Number of Opponents**: Active players remaining in the hand.
    *   *Sweep Range*: 1 (Heads Up), 3 (Multi-way), 5 (Crowded).
*   **Stage**:
    *   *Values*: Pre-flop, Post-flop.

---

## 2. Interesting Situations & Expected Outcomes

A standard sensitivity analysis must test the models against these specific archetypal poker situations to ensure theoretical soundness.

### Situation A: The Pure Value Bet (Monster Hand)
*   **Configuration**: Post-flop, Equity = 85%+, SPR = 10.0, Opponents = 1, Pot Odds = 0% (Checked to Hero).
*   **Expected Outcome**: The model should overwhelmingly favor **BET/RAISE**. With high equity and a deep stack, the primary objective is to build the pot and extract value. 

### Situation B: The Mathematical Draw (Pot Odds Test)
*   **Configuration**: Post-flop, Equity = 35% (e.g., an open-ended straight draw or flush draw).
*   **Sweep**: Vary the Pot Odds (10%, 25%, 33%, 50%).
*   **Expected Outcome**: 
    *   At 10% - 25% Pot Odds (favorable): The model should frequently **CALL**.
    *   At 50% Pot Odds (terrible): The model should heavily lean towards **FOLD**.

### Situation C: Short-Stacked Commitment
*   **Configuration**: Post-flop, Equity = 60%, SPR = 1.0 (very short stacked), Opponents = 1.
*   **Expected Outcome**: The model should aggressively choose to **RAISE / ALL-IN**. With an SPR of 1, the hero is pot-committed, and any action other than getting the rest of the chips in is theoretically suboptimal.

### Situation D: Deep-Stacked Multi-way Caution
*   **Configuration**: Post-flop, Equity = 50% (e.g., top pair, weak kicker), SPR = 15.0, Opponents = 4, Pot Odds = 25%.
*   **Expected Outcome**: Despite decent absolute equity, facing a bet in a multi-way pot deep-stacked should elicit caution. The model should heavily favor **FOLD** or **CALL** (passive), avoiding raises due to the risk of running into a monster.

### Situation E: The Pure Air / Bluff Opportunity
*   **Configuration**: Post-flop, Equity = 15%, SPR = 5.0, Opponents = 1.
*   **Expected Outcome**: 
    *   If facing a bet (Pot Odds > 0%): Almost 100% **FOLD**.
    *   If checked to (Pot Odds = 0%): A balanced mixture of **CHECK** (giving up) and occasional **BET** (bluffing), depending on the aggressiveness of the model.

### Situation F: Pre-Flop Marginal Defend
*   **Configuration**: Pre-flop, Equity = 45%, Pot Odds = 20%, Opponents = 1.
*   **Expected Outcome**: A polarized mix. The model should recognize it is slightly behind but getting decent odds, leaning towards **CALL** or **FOLD**, with raises being extremely rare.

---

## 3. Analysis Methodology

When running this analysis:
1.  **Iterate**: Loop through the configurations described above.
2.  **Static Values**: Use static inputs (bypass Monte Carlo simulations) to ensure stable and deterministic outputs for the neural network and heuristic layers.
3.  **Compare**: Output the probability distribution (Fold, Check, Call, Bet, Raise) for each scenario.
4.  **Evaluate**: Cross-reference the model's output distribution against the *Expected Outcomes* defined above to flag areas where the model is deviating from sound poker theory.

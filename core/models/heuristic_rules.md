# Heuristic Engine Decision Rules & Branches

This document details the heuristic rules and decision branches implemented in `HeuristicEngine` ([core/models/heuristic.py](file:///c:/REPO/Antigravity/AIPoker/core/models/heuristic.py)).

The decision model is split into two primary segments: **Pre-flop** and **Post-flop (Flop, Turn, River)**.

---

## I. Pre-flop Decision Tree

Pre-flop play is governed by the `use_preflop_chart` configuration flag:

### 1. Pre-flop Charts Enabled (`use_preflop_chart=True`)
* **Hand Selection**: Starting hands are converted to a standard rank string (e.g., `AA`, `AKs`, `AKo`) and parsed against pre-defined ranges.
* **Dynamic Range Scaling**: The hand ranges scale dynamically based on the number of active opponents remaining in the hand:
  * **Heads-up (1 Opponent)**: Opens up to play a wide **top 55%** of starting hands.
  * **3-handed (2 Opponents)**: Restricts playing range to the **top 30%** of hands.
  * **Full Ring (4+ Opponents)**: Plays tight, standard pre-flop charts.
* **Decision Branches**:
  * **Premium Hands** (e.g. `AA`, `KK`, `QQ`, `AKs`): Plays aggressively. If no bet is faced, it **bets (opens)**. If facing a bet, it **raises (3-bets)**.
  * **Playable Hands**: If checked to, it **checks**. If facing a bet, it **calls** only if the bet size is less than 15% of Hero's stack; otherwise, it **folds**.
  * **Weak Hands**: **Checks** if checked to; **folds** if facing any bet.

### 2. Charts Disabled (Monte Carlo Equity Fallback)
* Uses simulated Monte Carlo equity thresholds:
  * **Equity > 60%**: **Bet / Raise** (aggressive play).
  * **Equity 42% - 60%**: **Check / Call** (passive play for reasonable bets $< 15\%$ stack).
  * **Equity < 42%**: **Check / Fold**.

---

## II. Post-flop Decision Tree (Flop, Turn, River)

Post-flop decisions are layered across three components: **Bluffing**, **Main Decision Engine**, and **Dynamic Bet Sizing**.

### 1. Bluffing Layer (`use_bluff_engine=True`)
Before making standard decisions, the engine checks for bluffing opportunities:
* **Semi-bluff**: If Hero holds a straight/flush draw with moderate equity (30% to 48%), there is a **35% random frequency** to bet/raise.
* **Pure Bluff**: If Hero has low equity ($< 25\%$), the board is checked to us on the turn/river, there is a **15% random frequency** to bet.

### 2. Main Decision Engine
Decisions are split based on the `use_math_engine` flag:

#### Math Engine Enabled (`use_math_engine=True`)
* Calculates expected value ($EV = Equity - Pot\_Odds$) and scales equity thresholds dynamically:
  * **Opponent Count Scaling**: The equity required to bet/raise scales with the number of opponents:
    * **1 opponent**: Raise/Bet threshold is `55%`, moderate equity is `40%`.
    * **2 opponents**: Raise/Bet threshold is `60% - 62%`, moderate equity is `45%`.
    * **3+ opponents**: Raise/Bet threshold is `65% - 70%`, moderate equity is `50%`.
  * **Opponent Aggression Adjustment**: If active opponents have tracking metrics (VPIP/AGG colors from our HUD):
    * Against highly aggressive (**Red** or **Yellow** AGG) players, thresholds increase (plays tighter).
    * Against passive (**Green** AGG) players, thresholds decrease (value-bets wider).
  * **Short-stack Commitment**: If Hero is short-stacked (stack-to-pot ratio $< 2.5$ or stack $< 60$ chips), all equity thresholds are lowered by **`12%`** to defend or shove wider.
* **Branches**:
  * **No Bet Faced (`call_amount == 0`)**:
    * **Bet**: If Equity $>$ Bet Threshold (or bluffing).
    * **Check**: If Equity $\le$ Bet Threshold.
  * **Facing Bet (`call_amount > 0`)**:
    * **Raise**: If expected value is positive ($EV > 0.15$) and Equity $>$ Raise Threshold (or bluffing).
    * **Call**: If $EV \ge 0.0$ (or marginal draw calling with deep stacks).
    * **Fold**: If $EV < 0.0$ and odds are poor.

#### Math Engine Disabled (Flat Equity Fallback)
* Uses rigid equity thresholds:
  * **No Bet Faced**: **Bet** if Equity $> 55\%$; otherwise **Check**.
  * **Facing Bet**: **Raise** if Equity $> 65\%$, **Call** if Equity $> 35\%$, **Fold** if Equity $\le 35\%$.

### 3. Dynamic Sizing Layer (`use_dynamic_sizing=True`)
Adjusts bet sizing dynamically to remain immune to button label changes using the betting slider:
* **Pre-flop**: Sets a standard 3 BB open or a 3x raise sizing.
* **Post-flop**: Evaluates board texture ("wetness") to determine bet sizing:
  * **Bluffs**: Sizes down to **`35%` of the pot** (cheap bluff).
  * **Wet Boards** (draw-heavy, wetness $\ge 0.5$): Sizes up to **`80%` of the pot** to protect hands and charge draws.
  * **Dry Boards**: Sizes down to **`40%` of the pot** to extract value cheaply.

---

## III. Emergency Action Redirection
If the decision engine requests a `BET` or `RAISE` but the visual parser detects that the Bet/Raise button is unavailable (e.g. Hero is facing an all-in bet where raise is blocked), the engine gracefully falls back to:
* **Check**: If facing no bet.
* **Call**: If equity/expected value is high enough to warrant a call.
* **Fold**: If equity is too low.

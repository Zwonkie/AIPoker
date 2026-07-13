# Model Testing Suite

**Date Recorded**: 2026-07-11
**Related Files**: 
*   [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/train_selfplay.py)
*   [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/six_max_simulator.py)

## Context
During model training and iteration (V4, V5, V6), standard sensitivity analysis sweeps are executed to verify that predicted action EVs scale logically with cards, pot sizes, stack sizes, and opponent profiles. This document outlines the standard test scenarios used to evaluate model health.

---

## Standard Scenarios

### Scenario A: The "72o Facing Shove" Test (Extrapolation Guardrail)
*   **State Configuration**:
    *   **Street**: Preflop, Flop, Turn, and River
    *   **Hero Hand**: `7d 2s` (Garbage)
    *   **Board (Flop/Turn/River)**: Dry and disconnected (e.g., `Th 7c 2d 5s 9c`)
    *   **Pot Size**: 510 chips (Big Blind = 10.0)
    *   **Call Amount**: 470 chips (Facing a massive all-in shove)
*   **Evaluated Opponent Personalities**:
    *   **Nit** (Blue VPIP / Blue AGG)
    *   **Maniac** (Red VPIP / Red AGG)
*   **Expected Behavior**:
    *   The model should predict negative/zero EV for both `Call` and `Raise` against all profiles, making `Fold` (EV = 0.0) the clear optimal decision.
    *   Raise EV against a Nit must be significantly lower than against a Maniac.
    *   If `Raise EV > 0.0` for `72o` facing a shove, it indicates the model is suffering from **action EV extrapolation (hallucination)** due to off-policy data gaps.

### Scenario B: Opponent Personality Exploitation Test (Profile Sensitivity)
*   **State Configuration**:
    *   **Street**: Flop
    *   **Board**: `Th 7c 2d` (Slightly dry texture)
    *   **Pot Size**: 50 chips (5 BB)
    *   **Call Amount**: 10 chips (Facing a small bet)
    *   **Hero Stack**: 1000 chips (100 BB)
*   **Hero Hands to Test**:
    *   `Ks Kd` (Strong Overpair)
    *   `9s 8s` (Medium Draw/Air)
    *   `Jc 3d` (Weak Air)
*   **Evaluated Opponent Personalities**:
    1.  **Nit** (VPIP: ~12%, AGG: ~18% / Blue/Blue):
        *   *Expected Model Output*: Low `Raise` and `Call` EV for all hands. A Nit's bet indicates a very strong range, so Hero's hand has lower relative strength.
    2.  **TAG (Tight-Aggressive)** (VPIP: ~22%, AGG: ~46% / Green/Green):
        *   *Expected Model Output*: Balanced EV profile. Standard GTO-aligned outputs.
    3.  **Fish (Loose-Passive)** (VPIP: ~42%, AGG: ~25% / Yellow/Blue):
        *   *Expected Model Output*: Highly positive `Call` EV for strong hands. Fish over-call with weak holdings, so Hero wants to keep them in the pot. Weak/Medium hands might fold.
    4.  **Maniac (Loose-Aggressive)** (VPIP: ~62%, AGG: ~85% / Red/Red):
        *   *Expected Model Output*: Highly positive `Raise` or `Call` (trapping) EV for strong hands. Maniacs bet wide ranges with high bluff frequencies. For medium/weak hands, the model might find profitable floats or bluff raises depending on fold equity.

### Scenario C: Pot & Stack Size Scaling Test (Geometry Check)
*   **State Configuration**:
    *   **Street**: Flop
    *   **Hero Hand**: `Ac Kc` (Top Pair, Top Kicker / Strong Draw)
    *   **Board**: `Ah Qs 5d`
*   **Sweeps**:
    1.  **Pot Sweep**: Fix Hero Stack at 100 BB. Sweep Pot Size from 10 BB to 150 BB.
        *   *Expected Behavior*: EV of all positive actions should scale proportionally with the pot.
    2.  **Stack Sweep**: Fix Pot Size at 50 BB. Sweep Hero Stack from 10 BB to 200 BB.
        *   *Expected Behavior*: As stack size increases, the EV of raising with a strong-but-vulnerable hand should adjust to reflect the deeper implied odds.

### Scenario D: Active Opponents Sensitivity (Multi-Way Check)
*   **State Configuration**:
    *   **Street**: Flop
    *   **Hero Hand**: `Js Jh` (Vulnerable Overpair)
    *   **Board**: `9d 5c 2h` (Low Board)
    *   **Pot Size**: 100 chips (10 BB)
    *   **Call Amount**: 20 chips (Facing a small bet)
    *   **Hero Stack**: 1000 chips (100 BB)
*   **Sweep**:
    *   Sweep the count of active opponents from **1 opponent (heads-up)** to **5 opponents (full table multi-way)**.
*   **Expected Behavior**:
    *   As the active opponent count increases:
        *   The EV of `Call` and `Raise` with a vulnerable overpair (`JJ`) should steadily **decrease**.
        *   This is GTO-logical: the average hand strength required to win a showdown increases dramatically with each extra player in the pot, and the probability of someone holding a set, a two-pair, or a dominating draw increases.
        *   The model should favor folding or passive play as the pot becomes heavily multi-way.

### Scenario E: Preflop Equity Sensitivity (Range Check)
*   **State Configuration**:
    *   **Street**: Preflop
    *   **Board**: `[]` (Empty Board)
    *   **Pot Size**: 30 chips (3 BB)
    *   **Call Amount**: 20 chips (Facing a standard preflop raise)
    *   **Hero Stack**: 1000 chips (100 BB)
*   **Sweep**:
    *   Sweep across five representative starting hand tiers (card values & matching preflop equities):
        1.  `7d 2s` (Garbage - Equity $\approx 0.30$)
        2.  `Jh Ts` (Medium Speculative - Equity $\approx 0.46$)
        3.  `Ad Qo` (Strong Broadway - Equity $\approx 0.60$)
        4.  `Qd Qs` (Premium Monster - Equity $\approx 0.78$)
        5.  `Ah As` (Absolute Nuts - Equity $\approx 0.85$)
*   **Expected Behavior**:
    *   As preflop equity scales from 30% (72o) to 85% (AA):
        *   The predicted EVs for `Call` and `Raise` must scale **monotonically** upwards.
        *   For `7d 2s`, EVs for both active decisions should be negative, making `Fold` the only profitable action.
        *   For `Ah As`, EVs for `Raise` and `Call` should be heavily positive.
        *   This confirms the transformer's preflop card token embeddings have successfully mapped cards into correct relative equity representations during training.

### Scenario F: The River Pure Air Bluff (Bluff Collapse Check)
*   **State Configuration**:
    *   **Street**: River
    *   **Hero Hand**: `2h 3d` (Pure Air)
    *   **Board**: `As Ks Qs Js 9c`
    *   **Pot Size**: 100 or 150 chips
    *   **Call Amount**: 0 (First to act) or 50 (Facing bet)
    *   **Equity**: 0.0
*   **Expected Behavior**:
    *   The model MUST evaluate `Fold` (EV=0 or near 0) higher than `Call` (Calling with 0 equity is mathematically impossible to be profitable).
    *   If `Raise EV > Fold EV` when facing a bet with pure air, the model suffers from a Bluffing Collapse (hallucinated fold equity).

### Scenario G: The Nutted Trap
*   **State Configuration**:
    *   **Street**: River
    *   **Hero Hand**: `Ts Th` (The absolute nuts - Royal Flush)
    *   **Board**: `As Ks Qs Js 9c`
    *   **Opponent**: Calling Station
*   **Expected Behavior**:
    *   `Raise EV` should overwhelmingly dominate `Call EV`. If the model is checking or flat-calling the absolute nuts on the river against a loose player, it is missing massive value.


## Presenting
Show all test scenario output data and give general comments on each scenario.
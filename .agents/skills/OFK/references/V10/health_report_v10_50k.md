# Model Health Report

**Target Model:** v10_50k_main.pt
**Overall Grade:** > [!CAUTION]
> **FAIL (Bluff Collapse, Passive Bias, & Multi-Way Miscalibration)**

## Scenario Breakdown

| Scenario | Model Action | Result | Note |
| :--- | :--- | :--- | :--- |
| **River Pure Air (First to Act)** | `CALL` | **FAIL** | Hallucinated `0.25` EV for calling with 0% equity. |
| **River Pure Air (Facing Bet)** | `CALL` | **FAIL** | Hallucinated `0.11` EV for calling with 0% equity. |
| **River The Nuts (Facing Bet)** | `RAISE` | **PASS** | EV(Raise) dominates EV(Call) appropriately. |
| **River The Nuts (Calling Station)** | `RAISE` | **PASS** | Correctly value bets. |
| **Preflop AA vs Nit (Deep)** | `CALL` | **FAIL** | Too passive. EV(Call)=`1.92` > EV(Raise)=`1.52`. |
| **Preflop AA vs Maniac** | `CALL` | **FAIL** | Too passive. EV(Call)=`1.87` > EV(Raise)=`1.45`. |
| **Flop TPTK Multi-Way (4-way pot)** | `CALL` | **FAIL** | Failed to protect TPTK in a multi-way pot. |
| **Turn Flush Draw vs Bet** | `RAISE` | **FAIL** | Model chooses to raise instead of call with a naked flush draw facing a bet. |

---

## Preflop Equity Sweep (Player Count Scaling)

We tested the model's Preflop action across 5 equity brackets while facing 1, 3, and 5 active opponents.

| Eq Group | 1 Opponent | 3 Opponents | 5 Opponents |
| :--- | :--- | :--- | :--- |
| **<20% (Air)** | `CALL` (Fail) | `FOLD` (Pass) | `FOLD` (Pass) |
| **20-40% (Weak)** | `CALL` (Fail) | `FOLD` (Pass) | `FOLD` (Pass) |
| **40-60% (Marginal)**| `CALL` | `CALL` | `FOLD` (Pass) |
| **60-80% (Strong)** | `CALL` | `CALL` | `FOLD` (Fail) |
| **>80% (Nuts)** | `RAISE` (Pass) | `CALL` (Fail) | `FOLD` (Critical Fail)|

## Holes Discovered

> [!WARNING]
> **Severe Multi-Way Miscalibration**
> The model clearly understands that it needs to tighten its range as the number of active opponents increases (which is a great sign of learning). However, its calibration is wildly broken:

- **Heads-Up (1 Opp)**: Plays insanely loose. It believes calling with absolute preflop garbage (<20% equity) is +EV.
- **Mid-Ring (3 Opps)**: Starts folding trash, but becomes too passive with premiums (just calls the nuts).
- **Full-Ring (5 Opps)**: Total fear collapse. It believes that playing a 6-way pot is mathematically impossible to win, to the point where it explicitly `FOLDS` the absolute Nuts preflop.
- **Conclusion**: The 50k model has learned the *direction* of multi-way scaling, but the magnitude of the penalty it applies for extra players is way too massive, resulting in a completely passive/scared state in full ring games.

# Model Health Report

**Target Model**: `Pluribus (v9 Main)`  
**Overall Grade**: 🚨 **CRITICAL FAIL**  
**Date Executed**: 2026-07-12

---

## Executive Summary
The V9 model has failed critical logic checks on the River. While it demonstrates strong preflop and value-betting fundamentals (correctly identifying and raising the nuts), it suffers from a massive hallucination regarding fold equity and bluffing success rates. It mathematically believes that calling with pure air (0.0 equity) is profitable, indicating a total breakdown in terminal state processing.

> [!CAUTION]
> Do not deploy V9 to production without the math-engine guardrails active. The model will bleed chips on the River by calling/shoving with busted draws.

---

## Scenario Breakdown

| Scenario | Raw EV (Fold) | Raw EV (Call) | Raw EV (Raise) | Action Taken | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **River Pure Air (First to Act)** | 0.01 | 0.84 | 0.13 | `CALL` | ❌ **FAIL** |
| **River Pure Air (Facing Bet)** | -0.01 | 0.63 | -0.26 | `CALL` | ❌ **CRITICAL FAIL** |
| **River The Nuts (Facing Bet)** | -0.05 | 16.32 | 22.23 | `RAISE` | ✅ **PASS** |
| **River The Nuts (Calling Station)** | -0.01 | 13.79 | 18.69 | `RAISE` | ✅ **PASS** |
| **Preflop AA vs Nit (Deep)** | -0.04 | 4.09 | 5.29 | `RAISE` | ✅ **PASS** |
| **Preflop AA vs Maniac** | 0.01 | 3.59 | 4.55 | `RAISE` | ✅ **PASS** |
| **Flop TPTK (4-way pot)** | -0.03 | 2.45 | 3.04 | `RAISE` | ⚠️ **WARNING** |
| **Turn Flush Draw vs Bet** | -0.05 | 4.15 | 5.61 | `RAISE` | ⚠️ **WARNING** |

---

## Holes Discovered

*   **The "Pure Air" Hallucination (River Bluff Collapse)**
    *   **The Bug**: When facing a bet on the River with 0.0 equity (pure air), the model evaluated `Call` at +0.63 EV and `Fold` at -0.01 EV. 
    *   **The Implication**: It is mathematically impossible to have a positive EV when calling with 0 equity. The model has learned a degenerate exploit where it assumes opponents are *always* bluffing with worse air, or the target EVs during training were inflated by a constant.
*   **Multi-Way Over-Aggression**
    *   **The Bug**: The model raises Top Pair Top Kicker on the flop into 3 active opponents.
    *   **The Implication**: V9 does not sufficiently scale down its hand strength requirements in multi-way pots. It plays TPTK exactly as it would heads-up, exposing it to massive reverse-implied odds against sets or two-pairs.
*   **Draw Over-Aggression**
    *   **The Bug**: It strongly prefers raising a Turn flush draw rather than flat calling.
    *   **The Implication**: While raising draws (semi-bluffing) is good, doing it indiscriminately indicates the model over-estimates fold equity on the Turn.

# Model Health Report (V12)

**Date**: 2026-07-13
**Target Model**: `versions/v12/weights/expert_main.pth`
**Architecture**: V12 Policy Actor (Regret Matching Target)

## 🚨 Overall Grade: CRITICAL FAIL 🚨

The V12 actor network failed fundamental boundaries regarding 0-equity defense and dynamic opponent scaling. Despite achieving an exceptionally low loss during training, its realized behavior on-table has mathematical holes.

## Scenario Breakdown

| Scenario | Model Action | Prob (F / C / R) | Grade | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **River Pure Air (First to Act)** | **CALL** | 0.33 / 0.34 / 0.33 | ❌ FAIL | Model preferred calling with 0.0 equity over folding. |
| **River Pure Air (Facing Bet)** | **CALL** | 0.33 / 0.34 / 0.33 | ❌ FAIL | Critical bluff-collapse/station behavior. Calling with pure air facing a bet. |
| **River The Nuts (Facing Bet)** | **RAISE** | 0.12 / 0.30 / 0.58 | ✅ PASS | Value extracts effectively with the nuts. |
| **River The Nuts (Calling Station)** | **RAISE** | 0.12 / 0.30 / 0.58 | ✅ PASS | Correctly raises for maximum value. |
| **Preflop AA vs Nit (Deep)** | **RAISE** | 0.15 / 0.31 / 0.54 | ✅ PASS | Does not shy away from value preflop. |
| **Preflop AA vs Maniac** | **RAISE** | 0.15 / 0.31 / 0.54 | ✅ PASS | Standard value push. |
| **Flop TPTK Multi-Way (4-way pot)** | **RAISE** | 0.18 / 0.32 / 0.49 | ❌ FAIL | Overvalued vulnerable TPTK against 3 opponents instead of pot-controlling. |
| **Turn Flush Draw vs Bet** | **RAISE** | 0.22 / 0.33 / 0.45 | ⚠️ WARN | High aggression with draws. |

## Preflop Equity Sweep (Multi-Way Sensitivity)

| Eq Group | 1 Opponent | 3 Opponents | 5 Opponents | Grade |
| :--- | :--- | :--- | :--- | :--- |
| **<20% (Air)** | FOLD | FOLD | FOLD | ✅ PASS |
| **20-40% (Weak)** | RAISE | RAISE | RAISE | ❌ FAIL |
| **40-60% (Marg)** | RAISE | RAISE | RAISE | ❌ FAIL |
| **60-80% (Strg)** | RAISE | RAISE | RAISE | ❌ FAIL |
| **>80% (Nuts)** | RAISE | RAISE | RAISE | ✅ PASS |

## Holes Discovered
1. **Calling Station Collapse on Pure Air**: The model evaluated the probability of calling (0.34) slightly higher than folding (0.33) when facing a bet on the river with pure air (0.0 equity). This is mathematically impossible to be profitable and represents a critical hole.
2. **Opponent Count Blindness**: Across the preflop equity sweep, the model outputted *identical* probabilities for Fold/Call/Raise regardless of whether it was facing 1 opponent, 3 opponents, or 5 opponents. It failed to tighten its range in multi-way pots.
3. **Overvaluing Vulnerable Hands Multi-Way**: The model aggressively raises Top-Pair Top-Kicker (TPTK) in a 4-way pot, failing to recognize the dilution of relative hand strength against multiple opponents.

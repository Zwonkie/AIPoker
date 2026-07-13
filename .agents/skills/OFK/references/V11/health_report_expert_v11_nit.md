# Model Health Report: V11 Nit (`expert_v11_nit.pth`)

**Date Evaluated**: 2026-07-13
**Target Model**: `V11 Nit`

## Overall Grade: CRITICAL FAIL 🚨

The V11 Nit model correctly avoids the Pure Air Hallucinations (unlike V9) and demonstrates exceptionally strong scaling against multiple opponents preflop. However, it suffers from severe extreme-nit behavior (folding the absolute nuts on the river when facing a bet) and slight over-calling with garbage air when heads-up.

---

## Scenario Breakdown

| Scenario | Model Action | Expected Action | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **River Pure Air (First to Act)** | FOLD | FOLD | **PASS** | Fold (0.00) > Raise (-1.51). Model successfully resists bluffing with 0 equity air. |
| **River Pure Air (Facing Bet)** | FOLD | FOLD | **PASS** | Fold (0.00) > Call (-0.85). Correctly folds air to a bet. |
| **River The Nuts (Facing Bet)** | FOLD | RAISE/CALL | **CRITICAL FAIL** | Fold (0.00) > Call (-0.30) > Raise (-0.76). **The model folds a Royal Flush to a bet.** |
| **River The Nuts (Calling Station)** | RAISE | RAISE | **PASS** | Raise (1.53) > Call (1.18). Correctly value bets heavily against a calling station. |
| **Preflop AA vs Nit (Deep)** | RAISE | RAISE | **PASS** | Highly positive EV for raising (+3.40). |
| **Flop TPTK Multi-Way (4-way)** | RAISE | CALL/FOLD | **WARNING** | EV Raise (+2.20). Overvaluing TPTK slightly in a 4-way pot, but not egregious. |

---

## Preflop Equity Sweep (Scaling Check)

The model's ability to adjust to multi-way pots was tested across equity intervals:

| Opponents | Air (<20%) | Weak (20-40%) | Marginal (40-60%) | Strong (60-80%) | Nuts (>80%) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1 Opp** | CALL ⚠️ | RAISE | RAISE | RAISE | RAISE | **WARNING**: EV(Call) = +0.36 for Pure Air. Over-calling garbage heads-up. |
| **3 Opps** | FOLD | FOLD | CALL | RAISE | RAISE | **PASS**: Perfect logical scaling. |
| **5 Opps** | FOLD | FOLD | FOLD | CALL | RAISE | **PASS**: Tightens up beautifully in heavily multi-way pots. |

---

## Holes Discovered

1. **The "Monster-Under-The-Bed" Bug (River Nuts Fold)**: 
   When facing a standard bet on the river with the absolute nuts (Equity = 1.0), the model evaluated `Call (-0.30)` and `Raise (-0.76)`. It chose to fold. It seems the Nit personality has learned such extreme risk aversion to aggression that it hallucinates negative EV even with a 100% win rate. Interestingly, against a *Calling Station*, it correctly evaluates Raise as highly positive (+1.53). This implies its evaluation is heavily skewed by opponent profiles, completely ignoring its own absolute nut equity.
2. **Heads-Up Loose Calling**: 
   Despite being a "Nit", when facing exactly 1 opponent preflop with `<20% equity` (e.g. 72o), it evaluated `Call (+0.36)` over `Fold (0.00)`. It correctly folds this garbage against 3 or 5 opponents, but gets too loose heads-up.

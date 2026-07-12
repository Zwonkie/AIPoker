# Single Parameter Sensitivity Analysis (NLH)

This report analyzes how the No-Limit Hold'em model scales EV when isolating single parameters.

## 1. Proximity of the Raise
Testing how the model evaluates a CALL/RAISE when a bet comes from far away vs right in front of Hero. Hero has QQ Preflop.

| Scenario | Sequence | FOLD EV | CALL EV | RAISE EV |
|---|---|---|---|---|
| Raise far away (3 folds after) | `['r', 'f', 'f', 'f']` | 1.211 | 10.777 | 4.585 |
| Raise middle (1 fold before, 2 after) | `['f', 'r', 'f', 'f']` | 2.836 | 6.437 | -9.381 |
| Raise right in front (3 folds before) | `['f', 'f', 'f', 'r']` | 10.156 | -4.963 | 2.201 |
| Lots of action (raise, call, reraise) | `['r', 'c', 'r']` | -0.845 | -9.589 | 2.011 |

## 2. Preflop Hand Strength (Equity) Scaling
Testing various hands facing a single raise in front of them (`['f', 'r']`).

| Hand | FOLD EV | CALL EV | RAISE EV |
|---|---|---|---|
| 72o (Trash) (`['7h', '2c']`) | 0.178 | -8.047 | -3.047 |
| T5o (Weak) (`['Th', '5c']`) | -6.487 | 15.257 | 5.726 |
| JTo (Drawing) (`['Jh', 'Tc']`) | -4.232 | 2.007 | -4.517 |
| 88 (Mid Pair) (`['8h', '8s']`) | 3.194 | 6.796 | -5.777 |
| AKs (Premium) (`['As', 'Ks']`) | -1.341 | -4.991 | 11.878 |
| AA (Nuts) (`['Ah', 'As']`) | 2.510 | -3.505 | -1.297 |

## 3. Pot Size Scaling
Hero has AKs facing a raise `['r']`. Varying the pot size.

| Pot Size | FOLD EV | CALL EV | RAISE EV |
|---|---|---|---|
| 10.0 | -14.364 | -7.541 | 4.695 |
| 50.0 | 2.415 | -2.399 | -1.463 |
| 200.0 | -10.909 | 6.533 | 14.569 |
| 500.0 | -47.889 | -34.121 | -62.054 |
| 1000.0 | -213.076 | -242.560 | -223.091 |

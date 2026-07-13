# Model Health Report

**Target Model:** `herocules_v11_fuzzyHeuristicsOpp.pth`
**Overall Grade:** ✅ **PASS**

## Scenario Breakdown

| Scenario | Final Action | OFK Grade |
| :--- | :--- | :--- |
| River Pure Air (First to Act) | FOLD | ✅ PASS |
| River Pure Air (Facing Bet) | FOLD | ✅ PASS |
| River The Nuts (Facing Bet) | RAISE | ✅ PASS |
| River The Nuts (Calling Station) | RAISE | ✅ PASS |
| Preflop AA vs Nit (Deep) | RAISE | ✅ PASS |
| Preflop AA vs Maniac | RAISE | ✅ PASS |
| Flop TPTK Multi-Way (4-way pot) | RAISE | ⚠️ WARNING |
| Turn Flush Draw vs Bet | RAISE | ✅ PASS |
| Preflop Equity Sweep (<20% to >80%) | Dynamic Scaling | ✅ PASS |

## Passed Criteria

*   **Preflop Dynamic Scaling**: The V11 transformer architecture proves it accurately understands individual hole card embeddings. When given Air (e.g., `7d 2c`), it correctly evaluates `FOLD` as the highest EV. Furthermore, the model dynamically shifts its starting hand requirements based on the table structure: it plays looser against 1 opponent (willing to call/raise marginal hands) but correctly tightens up and folds out everything below `60% (Strong)` equity when facing 3 or 5 opponents. This is a masterclass in preflop play!
*   **No Bluff Collapse**: The model correctly evaluates `FOLD` as the most profitable action when holding pure air on the river (EVs for Call and Raise are strictly negative).
*   **No Trapping With Nuts**: The model correctly maximizes value when holding the absolute nuts on the river, evaluating `RAISE` significantly higher than `CALL`.

# MoE Sensitivity Analysis

This report isolates the pure (100%) evaluation of each Tier's PyTorch Neural Network, bypassing the blending Gating Network. Values shown are raw EV predictions.

## Preflop - High Equity (AA) - Facing All-in
- **Hero Cards:** `['Ah', 'As']`
- **Community Cards:** `[]`
- **Pot Size:** `200.0`
- **Facing Bet:** `1000.0`

| Action | Tier 1 (Beginner) EV | Tier 2 (Nit) EV | Tier 3 (Pro) EV |
|---|---|---|---|
| **FOLD** | 0.56 | 0.13 | 0.30 |
| **CALL** | 0.56 | -0.25 | -0.60 |
| **RAISE** | 0.56 | 0.14 | 1.16 |

## Preflop - Low Equity (72o) - Facing All-in
- **Hero Cards:** `['7h', '2c']`
- **Community Cards:** `[]`
- **Pot Size:** `200.0`
- **Facing Bet:** `1000.0`

| Action | Tier 1 (Beginner) EV | Tier 2 (Nit) EV | Tier 3 (Pro) EV |
|---|---|---|---|
| **FOLD** | 0.01 | 0.18 | 0.44 |
| **CALL** | 0.01 | -0.16 | -0.61 |
| **RAISE** | 0.01 | 0.18 | 1.67 |

## Turn - Medium Equity (Draw) - Facing Pot Bet
- **Hero Cards:** `['Jh', 'Th']`
- **Community Cards:** `['2h', '5h', 'Kc', '7d']`
- **Pot Size:** `100.0`
- **Facing Bet:** `100.0`

| Action | Tier 1 (Beginner) EV | Tier 2 (Nit) EV | Tier 3 (Pro) EV |
|---|---|---|---|
| **FOLD** | 0.07 | -0.17 | 0.31 |
| **CALL** | 0.07 | -0.49 | 0.16 |
| **RAISE** | 0.07 | -0.15 | 1.42 |

## Turn - High Equity (Set) - Facing Small Bet
- **Hero Cards:** `['7h', '7c']`
- **Community Cards:** `['2h', '5h', 'Kc', '7d']`
- **Pot Size:** `100.0`
- **Facing Bet:** `20.0`

| Action | Tier 1 (Beginner) EV | Tier 2 (Nit) EV | Tier 3 (Pro) EV |
|---|---|---|---|
| **FOLD** | -0.05 | -0.06 | -0.64 |
| **CALL** | -0.05 | -0.43 | -0.94 |
| **RAISE** | -0.05 | -0.05 | 1.08 |

## River - Low Equity (Missed Draw) - Facing All-in
- **Hero Cards:** `['Jh', 'Th']`
- **Community Cards:** `['2h', '5h', 'Kc', '7d', '2s']`
- **Pot Size:** `200.0`
- **Facing Bet:** `1000.0`

| Action | Tier 1 (Beginner) EV | Tier 2 (Nit) EV | Tier 3 (Pro) EV |
|---|---|---|---|
| **FOLD** | 0.19 | -0.05 | 0.15 |
| **CALL** | 0.19 | -0.42 | -0.03 |
| **RAISE** | 0.19 | -0.01 | 1.42 |

## River - High Equity (Nut Flush) - Facing Small Bet
- **Hero Cards:** `['Ah', 'Jh']`
- **Community Cards:** `['2h', '5h', 'Kc', '7h', '2s']`
- **Pot Size:** `200.0`
- **Facing Bet:** `20.0`

| Action | Tier 1 (Beginner) EV | Tier 2 (Nit) EV | Tier 3 (Pro) EV |
|---|---|---|---|
| **FOLD** | 0.43 | -0.32 | 0.40 |
| **CALL** | 0.43 | -0.51 | 0.15 |
| **RAISE** | 0.43 | -0.30 | 1.45 |

## River - Medium Equity (Top Pair) - Facing All-in
- **Hero Cards:** `['Kh', 'Qc']`
- **Community Cards:** `['2h', '5h', 'Kc', '7d', '2s']`
- **Pot Size:** `300.0`
- **Facing Bet:** `1000.0`

| Action | Tier 1 (Beginner) EV | Tier 2 (Nit) EV | Tier 3 (Pro) EV |
|---|---|---|---|
| **FOLD** | 0.13 | -0.00 | 0.07 |
| **CALL** | 0.13 | -0.31 | -0.48 |
| **RAISE** | 0.13 | 0.02 | 1.84 |


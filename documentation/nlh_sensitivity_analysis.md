# MoE Sensitivity Analysis

This report isolates the pure (100%) evaluation of each Tier's PyTorch Neural Network, bypassing the blending Gating Network. Values shown are raw EV predictions.

## Preflop - High Equity (AA) - Facing All-in
- **Hero Cards:** `['Ah', 'As']`
- **Community Cards:** `[]`
- **Pot Size:** `200.0`
- **Facing Bet:** `1000.0`

| Action | Tier 1 (Limit) EV | Tier 2 (Limit) EV | Tier 3 (Limit) EV | NLH Expert EV |
|---|---|---|---|---|
| **FOLD** | 0.22 | 0.18 | 0.07 | 0.00 |
| **CALL** | 0.22 | -0.11 | -0.75 | -0.02 |
| **RAISE** | 0.22 | 0.18 | 0.06 | -0.01 |

## Preflop - Low Equity (72o) - Facing All-in
- **Hero Cards:** `['7h', '2c']`
- **Community Cards:** `[]`
- **Pot Size:** `200.0`
- **Facing Bet:** `1000.0`

| Action | Tier 1 (Limit) EV | Tier 2 (Limit) EV | Tier 3 (Limit) EV | NLH Expert EV |
|---|---|---|---|---|
| **FOLD** | 0.22 | 0.18 | 0.07 | 0.00 |
| **CALL** | 0.22 | -0.11 | -0.75 | -0.01 |
| **RAISE** | 0.22 | 0.18 | 0.06 | -0.01 |

## Turn - Medium Equity (Draw) - Facing Pot Bet
- **Hero Cards:** `['Jh', 'Th']`
- **Community Cards:** `['2h', '5h', 'Kc', '7d']`
- **Pot Size:** `100.0`
- **Facing Bet:** `100.0`

| Action | Tier 1 (Limit) EV | Tier 2 (Limit) EV | Tier 3 (Limit) EV | NLH Expert EV |
|---|---|---|---|---|
| **FOLD** | 0.07 | 0.10 | -0.30 | 0.03 |
| **CALL** | 0.07 | -0.07 | -0.61 | 0.02 |
| **RAISE** | 0.07 | 0.16 | -0.39 | -0.01 |

## Turn - High Equity (Set) - Facing Small Bet
- **Hero Cards:** `['7h', '7c']`
- **Community Cards:** `['2h', '5h', 'Kc', '7d']`
- **Pot Size:** `100.0`
- **Facing Bet:** `20.0`

| Action | Tier 1 (Limit) EV | Tier 2 (Limit) EV | Tier 3 (Limit) EV | NLH Expert EV |
|---|---|---|---|---|
| **FOLD** | 0.07 | 0.10 | -0.30 | 0.03 |
| **CALL** | 0.07 | -0.07 | -0.61 | 0.03 |
| **RAISE** | 0.07 | 0.16 | -0.39 | 0.02 |

## River - Low Equity (Missed Draw) - Facing All-in
- **Hero Cards:** `['Jh', 'Th']`
- **Community Cards:** `['2h', '5h', 'Kc', '7d', '2s']`
- **Pot Size:** `200.0`
- **Facing Bet:** `1000.0`

| Action | Tier 1 (Limit) EV | Tier 2 (Limit) EV | Tier 3 (Limit) EV | NLH Expert EV |
|---|---|---|---|---|
| **FOLD** | 0.17 | 0.18 | -0.27 | 0.07 |
| **CALL** | 0.17 | -0.16 | -0.71 | 0.11 |
| **RAISE** | 0.17 | 0.36 | -0.44 | 0.00 |

## River - High Equity (Nut Flush) - Facing Small Bet
- **Hero Cards:** `['Ah', 'Jh']`
- **Community Cards:** `['2h', '5h', 'Kc', '7h', '2s']`
- **Pot Size:** `200.0`
- **Facing Bet:** `20.0`

| Action | Tier 1 (Limit) EV | Tier 2 (Limit) EV | Tier 3 (Limit) EV | NLH Expert EV |
|---|---|---|---|---|
| **FOLD** | 0.10 | 0.08 | -0.31 | 0.05 |
| **CALL** | 0.10 | -0.19 | -0.84 | 0.05 |
| **RAISE** | 0.10 | 0.23 | -0.43 | 0.02 |

## River - Medium Equity (Top Pair) - Facing All-in
- **Hero Cards:** `['Kh', 'Qc']`
- **Community Cards:** `['2h', '5h', 'Kc', '7d', '2s']`
- **Pot Size:** `300.0`
- **Facing Bet:** `1000.0`

| Action | Tier 1 (Limit) EV | Tier 2 (Limit) EV | Tier 3 (Limit) EV | NLH Expert EV |
|---|---|---|---|---|
| **FOLD** | 0.27 | 0.26 | -0.22 | 0.09 |
| **CALL** | 0.27 | -0.16 | -0.73 | 0.10 |
| **RAISE** | 0.27 | 0.46 | -0.36 | 0.07 |


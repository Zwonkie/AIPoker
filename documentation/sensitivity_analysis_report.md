# Poker Model Sensitivity & Distribution Analysis

This report shows how the **Pro Model** (`xgboost_poker.json`) and **Human Model** (`xgboost_human.json`) respond to different game parameters (Equity, Pot Odds, Stack-to-Pot Ratio) under both raw distributions ($T=1.0$) and temperature-softened distributions ($T=2.5$).

## Scenario: Post-flop Monster (High Equity)
**Inputs:** Pre-flop: `No` | Opponents: `2` | Equity: `85.0%` | Pot Odds: `0.0%` | SPR: `10.0`

| Model & Config | FOLD | CHECK | CALL | BET | RAISE |
| --- | --- | --- | --- | --- | --- |
| **Pro Model (Raw, T=1.0)** | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| **Pro Model (Softened, T=2.5)** | 1.8% | 90.7% | 2.5% | 1.8% | 3.2% |
| **Human Model (Raw, T=1.0)** | 0.0% | 99.9% | 0.0% | 0.0% | 0.1% |
| **Human Model (Softened, T=2.5)** | 2.0% | 88.6% | 2.6% | 2.0% | 4.8% |

## Scenario: Post-flop Weak Hand, No Bet faced
**Inputs:** Pre-flop: `No` | Opponents: `2` | Equity: `35.0%` | Pot Odds: `0.0%` | SPR: `10.0`

| Model & Config | FOLD | CHECK | CALL | BET | RAISE |
| --- | --- | --- | --- | --- | --- |
| **Pro Model (Raw, T=1.0)** | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| **Pro Model (Softened, T=2.5)** | 2.4% | 90.4% | 2.7% | 2.4% | 2.2% |
| **Human Model (Raw, T=1.0)** | 0.0% | 99.9% | 0.0% | 0.0% | 0.0% |
| **Human Model (Softened, T=2.5)** | 2.2% | 90.1% | 2.6% | 2.2% | 3.0% |

## Scenario: Post-flop Medium Hand, Facing Half-Pot Bet
**Inputs:** Pre-flop: `No` | Opponents: `2` | Equity: `55.0%` | Pot Odds: `25.0%` | SPR: `10.0`

| Model & Config | FOLD | CHECK | CALL | BET | RAISE |
| --- | --- | --- | --- | --- | --- |
| **Pro Model (Raw, T=1.0)** | 0.1% | 0.0% | 0.1% | 0.1% | 99.8% |
| **Pro Model (Softened, T=2.5)** | 4.2% | 2.8% | 5.4% | 4.2% | 83.5% |
| **Human Model (Raw, T=1.0)** | 0.3% | 0.3% | 31.6% | 0.3% | 67.5% |
| **Human Model (Softened, T=2.5)** | 5.5% | 5.5% | 35.4% | 5.5% | 48.0% |

## Scenario: Heads Up Bluff Opportunity (Low Equity)
**Inputs:** Pre-flop: `No` | Opponents: `1` | Equity: `25.0%` | Pot Odds: `0.0%` | SPR: `15.0`

| Model & Config | FOLD | CHECK | CALL | BET | RAISE |
| --- | --- | --- | --- | --- | --- |
| **Pro Model (Raw, T=1.0)** | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| **Pro Model (Softened, T=2.5)** | 1.0% | 94.4% | 1.2% | 1.0% | 2.4% |
| **Human Model (Raw, T=1.0)** | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| **Human Model (Softened, T=2.5)** | 1.4% | 92.7% | 1.8% | 1.4% | 2.8% |

## Scenario: Pre-flop Premium Hand
**Inputs:** Pre-flop: `Yes` | Opponents: `3` | Equity: `80.0%` | Pot Odds: `33.0%` | SPR: `25.0`

| Model & Config | FOLD | CHECK | CALL | BET | RAISE |
| --- | --- | --- | --- | --- | --- |
| **Pro Model (Raw, T=1.0)** | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| **Pro Model (Softened, T=2.5)** | 1.0% | 2.5% | 2.6% | 1.0% | 92.9% |
| **Human Model (Raw, T=1.0)** | 0.0% | 0.0% | 0.2% | 0.0% | 99.8% |
| **Human Model (Softened, T=2.5)** | 1.8% | 3.2% | 7.0% | 1.8% | 86.2% |

## Scenario: Pre-flop Marginal Hand
**Inputs:** Pre-flop: `Yes` | Opponents: `3` | Equity: `52.0%` | Pot Odds: `33.0%` | SPR: `25.0`

| Model & Config | FOLD | CHECK | CALL | BET | RAISE |
| --- | --- | --- | --- | --- | --- |
| **Pro Model (Raw, T=1.0)** | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| **Pro Model (Softened, T=2.5)** | 1.4% | 2.4% | 2.6% | 1.4% | 92.3% |
| **Human Model (Raw, T=1.0)** | 0.0% | 0.0% | 0.2% | 0.0% | 99.8% |
| **Human Model (Softened, T=2.5)** | 2.1% | 3.2% | 6.6% | 2.1% | 86.0% |

## Scenario: Pre-flop Weak Hand
**Inputs:** Pre-flop: `Yes` | Opponents: `3` | Equity: `38.0%` | Pot Odds: `33.0%` | SPR: `25.0`

| Model & Config | FOLD | CHECK | CALL | BET | RAISE |
| --- | --- | --- | --- | --- | --- |
| **Pro Model (Raw, T=1.0)** | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| **Pro Model (Softened, T=2.5)** | 1.0% | 2.4% | 2.6% | 1.0% | 93.0% |
| **Human Model (Raw, T=1.0)** | 0.0% | 0.0% | 0.1% | 0.0% | 99.8% |
| **Human Model (Softened, T=2.5)** | 1.7% | 2.9% | 6.2% | 1.7% | 87.5% |

## Scenario: Pre-flop Trash Hand
**Inputs:** Pre-flop: `Yes` | Opponents: `3` | Equity: `12.0%` | Pot Odds: `33.0%` | SPR: `25.0`

| Model & Config | FOLD | CHECK | CALL | BET | RAISE |
| --- | --- | --- | --- | --- | --- |
| **Pro Model (Raw, T=1.0)** | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| **Pro Model (Softened, T=2.5)** | 1.0% | 2.4% | 2.4% | 1.0% | 93.1% |
| **Human Model (Raw, T=1.0)** | 0.0% | 0.0% | 0.1% | 0.0% | 99.9% |
| **Human Model (Softened, T=2.5)** | 1.7% | 2.9% | 5.7% | 1.7% | 88.1% |

## Scenario: Post-flop Short-Stacked Commitment (High Equity)
**Inputs:** Pre-flop: `No` | Opponents: `2` | Equity: `75.0%` | Pot Odds: `33.0%` | SPR: `1.0`

| Model & Config | FOLD | CHECK | CALL | BET | RAISE |
| --- | --- | --- | --- | --- | --- |
| **Pro Model (Raw, T=1.0)** | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| **Pro Model (Softened, T=2.5)** | 1.1% | 2.5% | 2.1% | 1.1% | 93.3% |
| **Human Model (Raw, T=1.0)** | 0.0% | 0.0% | 0.1% | 0.0% | 99.9% |
| **Human Model (Softened, T=2.5)** | 1.7% | 3.2% | 5.1% | 1.7% | 88.3% |

## Scenario: Post-flop Deep-Stacked Facing Large Bet
**Inputs:** Pre-flop: `No` | Opponents: `2` | Equity: `20.0%` | Pot Odds: `40.0%` | SPR: `50.0`

| Model & Config | FOLD | CHECK | CALL | BET | RAISE |
| --- | --- | --- | --- | --- | --- |
| **Pro Model (Raw, T=1.0)** | 0.0% | 0.2% | 0.1% | 0.0% | 99.7% |
| **Pro Model (Softened, T=2.5)** | 2.6% | 6.2% | 4.6% | 2.6% | 83.8% |
| **Human Model (Raw, T=1.0)** | 0.0% | 0.0% | 0.1% | 0.0% | 99.9% |
| **Human Model (Softened, T=2.5)** | 1.6% | 2.8% | 5.1% | 1.6% | 89.0% |
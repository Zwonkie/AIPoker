# Poker Model Sensitivity & Distribution Analysis

This report details how the **Pro Model** (`xgboost_poker.json`) and the **Human Model** (`xgboost_human.json`) respond to different game scenarios across a range of input features, comparing the raw outputs ($T=1.0$) against temperature-softened outputs ($T=2.5$).

---

## 1. Post-flop Monster (High Equity)
- **Inputs**: Pre-flop: `No` | Opponents: `2` | Equity: `85.0%` | Pot Odds: `0.0%` | SPR: `10.0`

| Model & Config | FOLD | CHECK | CALL | BET | RAISE |
| --- | --- | --- | --- | --- | --- |
| **Pro Model (Raw, T=1.0)** | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| **Pro Model (Softened, T=2.5)** | 1.8% | 90.7% | 2.5% | 1.8% | 3.2% |
| **Human Model (Raw, T=1.0)** | 0.0% | 99.9% | 0.0% | 0.0% | 0.1% |
| **Human Model (Softened, T=2.5)** | 2.0% | 88.6% | 2.6% | 2.0% | 4.8% |

---

## 2. Post-flop Weak Hand, No Bet faced
- **Inputs**: Pre-flop: `No` | Opponents: `2` | Equity: `35.0%` | Pot Odds: `0.0%` | SPR: `10.0`

| Model & Config | FOLD | CHECK | CALL | BET | RAISE |
| --- | --- | --- | --- | --- | --- |
| **Pro Model (Raw, T=1.0)** | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| **Pro Model (Softened, T=2.5)** | 2.4% | 90.4% | 2.7% | 2.4% | 2.2% |
| **Human Model (Raw, T=1.0)** | 0.0% | 99.9% | 0.0% | 0.0% | 0.0% |
| **Human Model (Softened, T=2.5)** | 2.2% | 90.1% | 2.6% | 2.2% | 3.0% |

---

## 3. Post-flop Medium Hand, Facing Half-Pot Bet
- **Inputs**: Pre-flop: `No` | Opponents: `2` | Equity: `55.0%` | Pot Odds: `25.0%` | SPR: `10.0`

| Model & Config | FOLD | CHECK | CALL | BET | RAISE |
| --- | --- | --- | --- | --- | --- |
| **Pro Model (Raw, T=1.0)** | 0.1% | 0.0% | 0.1% | 0.1% | 99.8% |
| **Pro Model (Softened, T=2.5)** | 4.2% | 2.8% | 5.4% | 4.2% | 83.5% |
| **Human Model (Raw, T=1.0)** | 0.3% | 0.3% | **31.6%** | 0.3% | 67.5% |
| **Human Model (Softened, T=2.5)** | 5.5% | 5.5% | **35.4%** | 5.5% | 48.0% |

> [!NOTE]
> This scenario highlights the core distinction between the models. Facing a bet, the Pro model is highly polarizing and raises 99.8% of the time, whereas the Human model displays realistic, passive calling behavior 31.6% of the time.

---

## 4. Heads Up Bluff Opportunity (Low Equity)
- **Inputs**: Pre-flop: `No` | Opponents: `1` | Equity: `25.0%` | Pot Odds: `0.0%` | SPR: `15.0`

| Model & Config | FOLD | CHECK | CALL | BET | RAISE |
| --- | --- | --- | --- | --- | --- |
| **Pro Model (Raw, T=1.0)** | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| **Pro Model (Softened, T=2.5)** | 1.0% | 94.4% | 1.2% | 1.0% | 2.4% |
| **Human Model (Raw, T=1.0)** | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| **Human Model (Softened, T=2.5)** | 1.4% | 92.7% | 1.8% | 1.4% | 2.8% |

---

## 5. Pre-flop Premium Hand
- **Inputs**: Pre-flop: `Yes` | Opponents: `3` | Equity: `80.0%` | Pot Odds: `10.0%` | SPR: `50.0`

| Model & Config | FOLD | CHECK | CALL | BET | RAISE |
| --- | --- | --- | --- | --- | --- |
| **Pro Model (Raw, T=1.0)** | 0.0% | 0.1% | 99.5% | 0.0% | 0.3% |
| **Pro Model (Softened, T=2.5)** | 3.2% | 4.4% | 81.0% | 3.2% | 8.2% |
| **Human Model (Raw, T=1.0)** | 0.4% | 0.2% | 97.4% | 0.4% | 1.7% |
| **Human Model (Softened, T=2.5)** | 7.1% | 5.6% | 67.0% | 7.1% | 13.1% |

---

## 6. Pre-flop Marginal Hand
- **Inputs**: Pre-flop: `Yes` | Opponents: `3` | Equity: `52.0%` | Pot Odds: `10.0%` | SPR: `50.0`

| Model & Config | FOLD | CHECK | CALL | BET | RAISE |
| --- | --- | --- | --- | --- | --- |
| **Pro Model (Raw, T=1.0)** | 0.1% | 0.1% | 99.7% | 0.1% | 0.1% |
| **Pro Model (Softened, T=2.5)** | 4.2% | 4.2% | 82.4% | 4.2% | 5.0% |
| **Human Model (Raw, T=1.0)** | 0.4% | 0.1% | 98.0% | 0.4% | 1.0% |
| **Human Model (Softened, T=2.5)** | 7.6% | 5.1% | 68.6% | 7.6% | 11.1% |

---

## 7. Post-flop Short-Stacked Commitment (High Equity)
- **Inputs**: Pre-flop: `No` | Opponents: `2` | Equity: `75.0%` | Pot Odds: `33.0%` | SPR: `1.0`

| Model & Config | FOLD | CHECK | CALL | BET | RAISE |
| --- | --- | --- | --- | --- | --- |
| **Pro Model (Raw, T=1.0)** | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| **Pro Model (Softened, T=2.5)** | 1.1% | 2.5% | 2.1% | 1.1% | 93.3% |
| **Human Model (Raw, T=1.0)** | 0.0% | 0.0% | 0.1% | 0.0% | 99.9% |
| **Human Model (Softened, T=2.5)** | 1.7% | 3.2% | 5.1% | 1.7% | 88.3% |

---

## 8. Post-flop Deep-Stacked Facing Large Bet
- **Inputs**: Pre-flop: `No` | Opponents: `2` | Equity: `20.0%` | Pot Odds: `40.0%` | SPR: `50.0`

| Model & Config | FOLD | CHECK | CALL | BET | RAISE |
| --- | --- | --- | --- | --- | --- |
| **Pro Model (Raw, T=1.0)** | 0.0% | 0.2% | 0.1% | 0.0% | 99.7% |
| **Pro Model (Softened, T=2.5)** | 2.6% | 6.2% | 4.6% | 2.6% | 83.8% |
| **Human Model (Raw, T=1.0)** | 0.0% | 0.0% | 0.1% | 0.0% | 99.9% |
| **Human Model (Softened, T=2.5)** | 1.6% | 2.8% | 5.1% | 1.6% | 89.0% |

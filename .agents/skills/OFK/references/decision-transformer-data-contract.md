# Decision Transformer (v4) Data Contract

This document defines the formal data contract, feature mapping, normalization rules, and alignment verification between the **Self-Play Training Pipeline** (`train_selfplay.py`) and the **Live Production Bot** (`ml_bridge.py` / `PHPHelp.py`).

---

## 1. Input Tensor Specifications

The `PokerEVModelV4` model expects four PyTorch tensors of size `[batch_size, seq_len, ...]` representing the game history sequence.

| Tensor | Shape | Type | Values / Description |
| :--- | :--- | :--- | :--- |
| `hole` | `[B, 2]` | `torch.long` | Playing card integer representations of Hero's 2 pocket cards ($0$ to $51$). |
| `board` | `[B, seq_len, 5]` | `torch.long` | Playing card integer representations of the 5 community cards ($0$ to $51$). Padded with $52$ if not dealt. |
| `context` | `[B, seq_len, 31]` | `torch.float32` | 31-feature state context vector containing stack ratios, equity, pot odds, and opponent HUD stats. |
| `actions` | `[B, seq_len]` | `torch.long` | Action tokens representing historical moves in the sequence. Padded with $0$. |

---

## 2. Context Vector (31 Features) Specification

The context vector contains exactly 31 features, normalized to range within $[0.0, 1.0]$ where possible.

| Index | Feature | Unit / Formula | Value Range | Normalization Formula / Mapping |
| :---: | :--- | :--- | :---: | :--- |
| **0** | `hero_position` | Seat index ($0$ to $5$) | $[0.0, 1.0]$ | `float(position) / 5.0` |
| **1** | `hero_stack_ratio` | Hero Stack in BBs | $[0.0, 1.0]$ | `(hero_stack_cents / bb_cents) / 400.0` |
| **2** | `pot_size_ratio` | Total Pot in BBs | $[0.0, 1.0]$ | `(pot_size_cents / bb_cents) / 1000.0` |
| **3** | `win_equity` | Monte Carlo Win % | $[0.0, 1.0]$ | Raw float (computed against active players) |
| **4** | `pot_odds` | Ratio to call | $[0.0, 1.0]$ | `call_cents / (pot_cents + call_cents)` |
| **5** | `active_opponents` | Active count ($0$ to $5$) | $[0.0, 0.5]$ | `sum(active_mask) / 10.0` |
| **6** | `street_level` | Game stage ($0$ to $3$) | $[0.0, 1.0]$ | `street_level / 3.0` (0=PF, 1=Flop, 2=Turn, 3=River) |
| **7** | `global_opp_vpip` | Global VPIP stat | $[0.0, 1.0]$ | Defaults to `0.3` (fallback) |
| **8** | `global_opp_agg` | Global Aggression stat | $[0.0, 1.0]$ | Defaults to `0.4` (fallback) |
| **9** | `pot_size_bb` | Raw pot size in BBs | $[0.0, \infty)$ | `pot_size_cents / bb_cents` (unnormalized) |
| **10** | `call_amount_bb` | Raw call size in BBs | $[0.0, \infty)$ | `call_amount_cents / bb_cents` (unnormalized) |
| **11** | `seat_1_active` | Seat 1 active flag | $\{0.0, 1.0\}$ | `1.0` if active, `0.0` otherwise |
| **12** | `seat_1_stack` | Seat 1 stack in BBs | $[0.0, 1.0]$ | `(stack_cents / bb_cents) / 400.0` |
| **13** | `seat_1_vpip` | Seat 1 VPIP stat | $[0.0, 1.0]$ | HUD color map midpoint: Blue=0.10, Green=0.22, Yellow=0.30, Red=0.45 |
| **14** | `seat_1_agg` | Seat 1 Agg stat | $[0.0, 1.0]$ | HUD color map midpoint: Blue=0.18, Green=0.46, Yellow=0.63, Red=0.85 |
| **15** | `seat_2_active` | Seat 2 active flag | $\{0.0, 1.0\}$ | `1.0` if active, `0.0` otherwise |
| **16** | `seat_2_stack` | Seat 2 stack in BBs | $[0.0, 1.0]$ | `(stack_cents / bb_cents) / 400.0` |
| **17** | `seat_2_vpip` | Seat 2 VPIP stat | $[0.0, 1.0]$ | HUD color map midpoint: Blue=0.10, Green=0.22, Yellow=0.30, Red=0.45 |
| **18** | `seat_2_agg` | Seat 2 Agg stat | $[0.0, 1.0]$ | HUD color map midpoint: Blue=0.18, Green=0.46, Yellow=0.63, Red=0.85 |
| **19** | `seat_3_active` | Seat 3 active flag | $\{0.0, 1.0\}$ | `1.0` if active, `0.0` otherwise |
| **20** | `seat_3_stack` | Seat 3 stack in BBs | $[0.0, 1.0]$ | `(stack_cents / bb_cents) / 400.0` |
| **21** | `seat_3_vpip` | Seat 3 VPIP stat | $[0.0, 1.0]$ | HUD color map midpoint: Blue=0.10, Green=0.22, Yellow=0.30, Red=0.45 |
| **22** | `seat_3_agg` | Seat 3 Agg stat | $[0.0, 1.0]$ | HUD color map midpoint: Blue=0.18, Green=0.46, Yellow=0.63, Red=0.85 |
| **23** | `seat_4_active` | Seat 4 active flag | $\{0.0, 1.0\}$ | `1.0` if active, `0.0` otherwise |
| **24** | `seat_4_stack` | Seat 4 stack in BBs | $[0.0, 1.0]$ | `(stack_cents / bb_cents) / 400.0` |
| **25** | `seat_4_vpip` | Seat 4 VPIP stat | $[0.0, 1.0]$ | HUD color map midpoint: Blue=0.10, Green=0.22, Yellow=0.30, Red=0.45 |
| **26** | `seat_4_agg` | Seat 4 Agg stat | $[0.0, 1.0]$ | HUD color map midpoint: Blue=0.18, Green=0.46, Yellow=0.63, Red=0.85 |
| **27** | `seat_5_active` | Seat 5 active flag | $\{0.0, 1.0\}$ | `1.0` if active, `0.0` otherwise |
| **28** | `seat_5_stack` | Seat 5 stack in BBs | $[0.0, 1.0]$ | `(stack_cents / bb_cents) / 400.0` |
| **29** | `seat_5_vpip` | Seat 5 VPIP stat | $[0.0, 1.0]$ | HUD color map midpoint: Blue=0.10, Green=0.22, Yellow=0.30, Red=0.45 |
| **30** | `seat_5_agg` | Seat 5 Agg stat | $[0.0, 1.0]$ | HUD color map midpoint: Blue=0.18, Green=0.46, Yellow=0.63, Red=0.85 |

---

## 3. Core Alignment Validations

To ensure zero divergence between the off-line training environment and live execution, the following data pipelines are strictly aligned:

### A. Monetary Units (Cents-Basis)
*   **XML Parser**: Always extracts chip values, blinds, bets, and wins on a **cents-basis** (converting €1.00 cash values to integer `100` cents).
*   **Stakes Parser**: Window title parser extracts SB and BB on a cents-basis (e.g. `10/20` or `€0.10/€0.20` -> `sb = 10, bb = 20`).
*   **Pot & Call Amount**: OCR reads cents values directly from target windows.
*   **Verification**: All ratios (`stack / bb`, `pot_size / bb`, `call_amount / bb`) are computed using cents in both numerator and denominator, ensuring unit cancellation matches training exactly.

### B. Card Integers Mapping
Both training and live evaluation map card strings to integers using the formula:
$$\text{card\_int} = (\text{suit\_idx} \times 13) + \text{rank\_idx}$$
*   **Ranks**: `23456789TJQKA` (Indices $0$ to $12$)
*   **Suits**: `cdhs` (Indices $0$ to $3$)
*   **Pad Value**: `52`

### C. Action Tokenizer Vocab
The sequence action tokens are mapped to the same action vocabulary:
*   `f` (Fold): token `7`
*   `c` (Call): token `3`
*   `r` (Raise): token `6`
*   `<PAD>`: token `0`

---

## 4. Maintenance Guidelines

1.  **Do not modify scaling coefficients** (`400.0` stack scale, `1000.0` pot scale, `10.0` active opponent scale) in either `ml_bridge.py` or `train_selfplay.py` without synchronized updates across both files.
2.  Any additions of new context features MUST be appended to the end of the context vector sequence to preserve backwards compatibility, and the model's configuration input dimension (`context_dim=31`) must be updated in `poker_transformer.py`.

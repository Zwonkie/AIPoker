# Pluribus V8 Sensitivity Report

**Date Recorded**: 2026-07-11
**Related Files**: [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v8/train_selfplay.py), [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v8/six_max_simulator.py)

## Context
This report documents the standard sensitivity analysis sweeps (Scenarios A through E) executed on the Pluribus V8 League models. We evaluate V7 Base, V8 Maniac, V8 Nit, V8 Sticky, and the final V8 Hero (Main) model to analyze how preflop and postflop EV predictions respond to cards, pots, stacks, and opponent profiles.

## Scenario A: The '72o Facing Shove' Test (Extrapolation Guardrail)
Evaluating predicted EV for `7d 2s` (Garbage) facing a 470 chip shove (47 BB) into a 510 chip pot (51 BB). Stack = 1000 chips. Big Blind = 10.0.

### Preflop (Equity: 0.17)
| Opponent | Model | Fold EV | Call EV | Raise EV | Best Action |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Nit | V7 Base | -0.01 | 2.36 | 4.53 | **RAISE** |
| Nit | V8 Maniac | -0.20 | -0.93 | -1.03 | **FOLD** |
| Nit | V8 Nit | 0.04 | -0.32 | -2.08 | **FOLD** |
| Nit | V8 Sticky | -1.05 | 0.03 | -1.52 | **CALL** |
| Nit | V8 Hero (Main) | 0.01 | -0.01 | -0.62 | **FOLD** |
| Maniac | V7 Base | -0.01 | 2.33 | 4.49 | **RAISE** |
| Maniac | V8 Maniac | -0.21 | -0.91 | -1.02 | **FOLD** |
| Maniac | V8 Nit | 0.04 | -0.32 | -2.08 | **FOLD** |
| Maniac | V8 Sticky | -1.05 | 0.03 | -1.52 | **CALL** |
| Maniac | V8 Hero (Main) | 0.01 | 0.00 | -0.61 | **FOLD** |

### Flop (Equity: 0.17)
| Opponent | Model | Fold EV | Call EV | Raise EV | Best Action |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Nit | V7 Base | 0.03 | 2.98 | 5.55 | **RAISE** |
| Nit | V8 Maniac | -0.19 | -1.02 | -1.18 | **FOLD** |
| Nit | V8 Nit | 0.04 | -0.28 | -2.06 | **FOLD** |
| Nit | V8 Sticky | -1.05 | 0.07 | -1.51 | **CALL** |
| Nit | V8 Hero (Main) | 0.01 | 0.09 | -0.54 | **CALL** |
| Maniac | V7 Base | 0.03 | 2.92 | 5.43 | **RAISE** |
| Maniac | V8 Maniac | -0.19 | -1.01 | -1.18 | **FOLD** |
| Maniac | V8 Nit | 0.04 | -0.28 | -2.05 | **FOLD** |
| Maniac | V8 Sticky | -1.05 | 0.07 | -1.51 | **CALL** |
| Maniac | V8 Hero (Main) | 0.01 | 0.10 | -0.53 | **CALL** |

### Turn (Equity: 0.17)
| Opponent | Model | Fold EV | Call EV | Raise EV | Best Action |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Nit | V7 Base | 0.01 | 3.24 | 6.02 | **RAISE** |
| Nit | V8 Maniac | -0.18 | -1.08 | -1.27 | **FOLD** |
| Nit | V8 Nit | 0.05 | -0.29 | -2.07 | **FOLD** |
| Nit | V8 Sticky | -1.05 | 0.06 | -1.52 | **CALL** |
| Nit | V8 Hero (Main) | 0.00 | 0.16 | -0.48 | **CALL** |
| Maniac | V7 Base | 0.02 | 3.15 | 5.86 | **RAISE** |
| Maniac | V8 Maniac | -0.18 | -1.07 | -1.27 | **FOLD** |
| Maniac | V8 Nit | 0.05 | -0.29 | -2.07 | **FOLD** |
| Maniac | V8 Sticky | -1.05 | 0.07 | -1.52 | **CALL** |
| Maniac | V8 Hero (Main) | 0.00 | 0.17 | -0.47 | **CALL** |

### River (Equity: 0.17)
| Opponent | Model | Fold EV | Call EV | Raise EV | Best Action |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Nit | V7 Base | -0.03 | 3.79 | 6.92 | **RAISE** |
| Nit | V8 Maniac | -0.18 | -1.10 | -1.34 | **FOLD** |
| Nit | V8 Nit | 0.04 | -0.26 | -2.04 | **FOLD** |
| Nit | V8 Sticky | -1.05 | 0.07 | -1.52 | **CALL** |
| Nit | V8 Hero (Main) | -0.01 | 0.24 | -0.41 | **CALL** |
| Maniac | V7 Base | -0.02 | 3.65 | 6.69 | **RAISE** |
| Maniac | V8 Maniac | -0.18 | -1.10 | -1.34 | **FOLD** |
| Maniac | V8 Nit | 0.04 | -0.26 | -2.04 | **FOLD** |
| Maniac | V8 Sticky | -1.05 | 0.07 | -1.52 | **CALL** |
| Maniac | V8 Hero (Main) | -0.01 | 0.25 | -0.41 | **CALL** |

## Scenario B: Opponent Personality Exploitation Test (Profile Sensitivity)
Evaluating Flop (`Th 7c 2d`) with `Ks Kd` (Strong Overpair) facing a small bet (call amount = 10, pot = 50). Stack = 1000, Big Blind = 10.0.

| Opponent Profile | Model | Fold EV | Call EV | Raise EV | Best Action |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Nit (Blue/Blue) | V7 Base | 0.04 | 3.04 | 5.47 | **RAISE** |
| Nit (Blue/Blue) | V8 Maniac | -0.24 | 5.12 | 7.47 | **RAISE** |
| Nit (Blue/Blue) | V8 Nit | 0.00 | 9.10 | 10.36 | **RAISE** |
| Nit (Blue/Blue) | V8 Sticky | -1.06 | 5.93 | 7.87 | **RAISE** |
| Nit (Blue/Blue) | V8 Hero (Main) | 0.06 | 6.23 | 8.41 | **RAISE** |
| TAG (Green/Green) | V7 Base | 0.04 | 2.98 | 5.35 | **RAISE** |
| TAG (Green/Green) | V8 Maniac | -0.23 | 4.94 | 7.22 | **RAISE** |
| TAG (Green/Green) | V8 Nit | -0.00 | 8.84 | 10.03 | **RAISE** |
| TAG (Green/Green) | V8 Sticky | -1.05 | 5.77 | 7.59 | **RAISE** |
| TAG (Green/Green) | V8 Hero (Main) | 0.06 | 6.14 | 8.27 | **RAISE** |
| Fish (Yellow/Blue) | V7 Base | 0.04 | 3.00 | 5.38 | **RAISE** |
| Fish (Yellow/Blue) | V8 Maniac | -0.23 | 5.01 | 7.30 | **RAISE** |
| Fish (Yellow/Blue) | V8 Nit | -0.00 | 8.90 | 10.10 | **RAISE** |
| Fish (Yellow/Blue) | V8 Sticky | -1.06 | 5.83 | 7.70 | **RAISE** |
| Fish (Yellow/Blue) | V8 Hero (Main) | 0.06 | 6.17 | 8.31 | **RAISE** |
| Maniac (Red/Red) | V7 Base | 0.05 | 2.88 | 5.16 | **RAISE** |
| Maniac (Red/Red) | V8 Maniac | -0.21 | 4.64 | 6.83 | **RAISE** |
| Maniac (Red/Red) | V8 Nit | -0.01 | 8.39 | 9.48 | **RAISE** |
| Maniac (Red/Red) | V8 Sticky | -1.04 | 5.53 | 7.18 | **RAISE** |
| Maniac (Red/Red) | V8 Hero (Main) | 0.07 | 6.02 | 8.07 | **RAISE** |

## Scenario C: Pot & Stack Size Scaling Test (Geometry Check)
Evaluating Flop (`Ah Qs 5d`) with `Ac Kc` (Top Pair, Top Kicker). Big Blind = 10.0.

### 1. Pot Size Sweep (Stack fixed at 100 BB = 1000 chips)
| Pot Size | Model | Fold EV | Call EV | Raise EV | Best Action |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 10 BB | V7 Base | 0.03 | 9.84 | 16.41 | **RAISE** |
| 10 BB | V8 Hero (Main) | 0.01 | 9.66 | 13.23 | **RAISE** |
| 30 BB | V7 Base | 0.07 | 36.56 | 54.36 | **RAISE** |
| 30 BB | V8 Hero (Main) | -0.00 | 25.04 | 33.83 | **RAISE** |
| 50 BB | V7 Base | 0.03 | 68.84 | 92.99 | **RAISE** |
| 50 BB | V8 Hero (Main) | 0.11 | 44.06 | 57.42 | **RAISE** |
| 100 BB | V7 Base | 0.12 | 167.57 | 205.07 | **RAISE** |
| 100 BB | V8 Hero (Main) | 0.03 | 86.27 | 101.63 | **RAISE** |
| 150 BB | V7 Base | 0.08 | 244.59 | 285.75 | **RAISE** |
| 150 BB | V8 Hero (Main) | 0.04 | 126.46 | 140.51 | **RAISE** |

### 2. Stack Size Sweep (Pot fixed at 50 BB = 500 chips)
| Hero Stack | Model | Fold EV | Call EV | Raise EV | Best Action |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 10 BB | V7 Base | 0.03 | 68.52 | 92.61 | **RAISE** |
| 10 BB | V8 Hero (Main) | 0.11 | 44.42 | 57.84 | **RAISE** |
| 40 BB | V7 Base | 0.03 | 68.63 | 92.74 | **RAISE** |
| 40 BB | V8 Hero (Main) | 0.11 | 44.28 | 57.68 | **RAISE** |
| 100 BB | V7 Base | 0.03 | 68.84 | 92.99 | **RAISE** |
| 100 BB | V8 Hero (Main) | 0.11 | 44.06 | 57.42 | **RAISE** |
| 150 BB | V7 Base | 0.03 | 69.02 | 93.19 | **RAISE** |
| 150 BB | V8 Hero (Main) | 0.11 | 43.89 | 57.20 | **RAISE** |
| 200 BB | V7 Base | 0.03 | 69.18 | 93.37 | **RAISE** |
| 200 BB | V8 Hero (Main) | 0.10 | 43.70 | 56.97 | **RAISE** |

## Scenario D: Active Opponents Sensitivity (Multi-Way Check)
Evaluating Flop (`9d 5c 2h`) with `Js Jh` (Vulnerable Overpair) facing a small bet (pot = 100, call amount = 20). Stack = 1000, Big Blind = 10.0.

| Opponents | Equity | Model | Fold EV | Call EV | Raise EV | Best Action |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | 0.80 | V7 Base | 0.04 | 7.62 | 12.80 | **RAISE** |
| 1 | 0.80 | V8 Hero (Main) | -0.03 | 8.34 | 11.41 | **RAISE** |
| 2 | 0.65 | V7 Base | 0.00 | 3.65 | 6.50 | **RAISE** |
| 2 | 0.65 | V8 Hero (Main) | -0.00 | 3.68 | 4.49 | **RAISE** |
| 3 | 0.52 | V7 Base | -0.03 | 2.24 | 4.18 | **RAISE** |
| 3 | 0.52 | V8 Hero (Main) | 0.01 | 1.55 | 1.31 | **CALL** |
| 4 | 0.42 | V7 Base | 0.02 | 1.28 | 2.85 | **RAISE** |
| 4 | 0.42 | V8 Hero (Main) | -0.02 | 0.55 | -0.14 | **CALL** |
| 5 | 0.35 | V7 Base | -0.01 | 0.52 | 1.86 | **RAISE** |
| 5 | 0.35 | V8 Hero (Main) | -0.00 | 0.26 | -0.40 | **CALL** |

## Scenario E: Preflop Equity Sensitivity (Range Check)
Evaluating Preflop facing a standard raise (pot = 30, call amount = 20). Stack = 1000, Big Blind = 10.0.

| Hand | Equity | Model | Fold EV | Call EV | Raise EV | Best Action |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| 7d 2s (Garbage) | 0.30 | V7 Base | -0.01 | -0.49 | 0.99 | **RAISE** |
| 7d 2s (Garbage) | 0.30 | V8 Maniac | -0.20 | 0.01 | 0.31 | **RAISE** |
| 7d 2s (Garbage) | 0.30 | V8 Nit | 0.02 | -0.12 | -1.85 | **FOLD** |
| 7d 2s (Garbage) | 0.30 | V8 Sticky | -1.03 | 0.94 | -0.90 | **CALL** |
| 7d 2s (Garbage) | 0.30 | V8 Hero (Main) | -0.01 | 0.27 | -0.39 | **CALL** |
| Jh Ts (Medium) | 0.46 | V7 Base | -0.01 | -0.41 | 1.06 | **RAISE** |
| Jh Ts (Medium) | 0.46 | V8 Maniac | -0.14 | 1.01 | 2.26 | **RAISE** |
| Jh Ts (Medium) | 0.46 | V8 Nit | 0.01 | 0.84 | -0.77 | **CALL** |
| Jh Ts (Medium) | 0.46 | V8 Sticky | -0.95 | 2.35 | 1.52 | **CALL** |
| Jh Ts (Medium) | 0.46 | V8 Hero (Main) | 0.01 | 1.79 | 1.69 | **CALL** |
| Ad Qo (Strong) | 0.60 | V7 Base | 0.01 | 1.04 | 2.49 | **RAISE** |
| Ad Qo (Strong) | 0.60 | V8 Maniac | -0.17 | 2.85 | 4.67 | **RAISE** |
| Ad Qo (Strong) | 0.60 | V8 Nit | 0.00 | 3.35 | 3.13 | **CALL** |
| Ad Qo (Strong) | 0.60 | V8 Sticky | -1.01 | 3.12 | 2.79 | **CALL** |
| Ad Qo (Strong) | 0.60 | V8 Hero (Main) | -0.04 | 3.27 | 3.93 | **RAISE** |
| Qd Qs (Premium) | 0.78 | V7 Base | -0.05 | 1.69 | 3.46 | **RAISE** |
| Qd Qs (Premium) | 0.78 | V8 Maniac | -0.16 | 2.88 | 4.83 | **RAISE** |
| Qd Qs (Premium) | 0.78 | V8 Nit | -0.02 | 4.39 | 4.44 | **RAISE** |
| Qd Qs (Premium) | 0.78 | V8 Sticky | -0.96 | 3.95 | 4.28 | **RAISE** |
| Qd Qs (Premium) | 0.78 | V8 Hero (Main) | -0.03 | 3.31 | 3.99 | **RAISE** |
| Ah As (Nuts) | 0.85 | V7 Base | -0.04 | 1.67 | 3.38 | **RAISE** |
| Ah As (Nuts) | 0.85 | V8 Maniac | -0.17 | 3.28 | 5.34 | **RAISE** |
| Ah As (Nuts) | 0.85 | V8 Nit | -0.05 | 5.89 | 6.38 | **RAISE** |
| Ah As (Nuts) | 0.85 | V8 Sticky | -0.98 | 3.74 | 3.86 | **RAISE** |
| Ah As (Nuts) | 0.85 | V8 Hero (Main) | 0.04 | 6.78 | 9.20 | **RAISE** |

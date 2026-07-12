# Model Health Report

**Target Model:** V10 100k Final (`expert_v8_main.pth`)
**Date:** 2026-07-12
**Overall Grade:** 🔴 **CRITICAL FAIL** (Complete Mode Collapse)

## Executive Summary
The V10 Main Model (trained for 250,000 hands) has suffered a catastrophic representation collapse. It has entirely stopped processing input state variables and has degenerated into a pure Calling Station, blindly outputting identical Q-values for every possible scenario regardless of hole cards, board texture, or opponent action.

## Scenario Breakdown

| Scenario | Raw EV (Fold) | Raw EV (Call) | Raw EV (Raise) | Final Action | OFK Criteria |
| :--- | :--- | :--- | :--- | :--- | :--- |
| River Pure Air (First to Act) | -0.00 | 0.21 | -1.51 | CALL | ❌ FAIL |
| River Pure Air (Facing Bet) | -0.00 | 0.21 | -1.51 | CALL | ❌ FAIL |
| River The Nuts (Facing Bet) | -0.00 | 0.21 | -1.51 | CALL | ❌ FAIL |
| River The Nuts (Calling Station)| -0.00 | 0.21 | -1.51 | CALL | ❌ FAIL |
| Preflop AA vs Nit (Deep) | -0.00 | 0.21 | -1.51 | CALL | ❌ FAIL |
| Preflop AA vs Maniac | -0.00 | 0.21 | -1.51 | CALL | ❌ FAIL |
| Flop TPTK Multi-Way (4-way) | -0.00 | 0.21 | -1.51 | CALL | ❌ FAIL |
| Turn Flush Draw vs Bet | -0.00 | 0.21 | -1.51 | CALL | ❌ FAIL |

## Preflop Equity Sweep Breakdown
*The model outputs the exact same EVs across all equity buckets (<20% to >80%) and against all opponent counts (1, 3, and 5).*

| Eq Group | Opps | Raw EV (Fold) | Raw EV (Call) | Raw EV (Raise) | Final Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| All Groups | All | -0.00 | 0.21 | -1.51 | CALL |

## Holes Discovered
*   **Total Attention / Mode Collapse:** The neural network outputs `Fold: -0.00, Call: 0.21, Raise: -1.51` regardless of the inputs. This indicates the gradients either exploded (destroying the weights) or the network found a local minimum where it ignores all inputs and outputs a constant scalar vector.
*   **100% Calling Station:** It attempts to call a bet on the River with Pure Air (0% equity), and refuses to Raise with the absolute Nuts. 
*   **Zero Multiway Adaptation:** It plays exactly the same facing 5 opponents as it does facing 1 opponent.

This model's weights are completely broken and cannot be used in production.

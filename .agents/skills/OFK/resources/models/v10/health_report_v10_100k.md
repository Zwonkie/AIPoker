# Model Health Report: V10 100k Final

**Target Model:** `V10 100k Final` (`expert_v8_main.pth`)
**Overall Grade:** CRITICAL FAIL (Complete Mode Collapse)

## Executive Summary
The V10 model suffered a catastrophic mode collapse during the final epochs of training (around epoch 186). The action entropy flatlined at `0.0000`, and the `Train Loss` exploded to `18.5480`. 

The network's weights have collapsed into a single deterministic output state. Regardless of hole cards, community cards, pot size, or opponent profile, the network now outputs the exact same EVs for every possible scenario:
- **FOLD EV**: `-0.00`
- **CALL EV**: `+0.09`
- **RAISE EV**: `-1.02`

Because the Call EV is marginally higher than the others, the network now unconditionally calls 100% of the time, resulting in a 71.4% VPIP (up from 3.5% just minutes prior).

## Scenario Breakdown

| Scenario | Model Action | Grade | Notes |
| :--- | :--- | :--- | :--- |
| **River Pure Air (Bluff Collapse)** | CALL | **CRITICAL FAIL** | Model calls with 0% equity. EV(Call) > EV(Fold). |
| **River The Nuts** | CALL | **FAIL** | Model refuses to raise with the nuts. EV(Raise) is negative. |
| **Preflop AA vs Nit** | CALL | **FAIL** | Refuses to 3-bet or shove AA preflop. |
| **Preflop AA vs Maniac** | CALL | **FAIL** | Fails to exploit Maniac aggression with premium hands. |
| **Flop TPTK Multi-Way** | CALL | **FAIL** | Passive play with vulnerable top pair. |
| **Turn Flush Draw** | CALL | **FAIL** | Passive drawing. |

## Equity Sweep Analysis

| Equity Bracket | Expected Action | Actual Action | Grade |
| :--- | :--- | :--- | :--- |
| `<20% (Air)` | FOLD | CALL | **CRITICAL FAIL** |
| `20-40% (Weak)` | FOLD/CALL | CALL | **FAIL** |
| `40-60% (Marginal)` | CALL/FOLD | CALL | **FAIL** |
| `60-80% (Strong)` | RAISE/CALL | CALL | **FAIL** |
| `>80% (Nuts)` | RAISE | CALL | **FAIL** |

## Holes Discovered (Root Cause Analysis)

- **Catastrophic Forgetting / Loss Explosion**: At epoch 186, the validation loss spiked to `40.6221`, indicating the gradients likely exploded. This destroyed the network's learned representations.
- **Static Output Distribution**: The model outputs the exact same EVs for a `<20%` equity hand facing 5 opponents as it does for a `>80%` equity hand facing 1 opponent. The network is no longer looking at the inputs.
- **The "Call" Bias**: Because the training data heavily penalized folding (guaranteed loss) and raising into the Maniac (high variance loss), the collapsed state settled on `CALL` as the path of least resistance, resulting in a pure Calling Station.

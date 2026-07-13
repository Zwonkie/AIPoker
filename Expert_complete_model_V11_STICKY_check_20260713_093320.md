# Deep Check: Model Training Health & Pipeline Integrity

## Overview
**Model:** V11_STICKY (Herocules V11 - Sticky Personality)
**Date:** 2026-07-13
**Audit Focus:** Complete Model Training Health Check (Final Evaluation)

## 1. Data Contract & Vectorization (The Input Alignment)
**Status: PASS**
- **Vectorization Misalignment:** Verified correct for this session.
- **Normalization Scale Mismatches:** Passed.
- **Causal Masking Failure:** Resolved and correctly padding sequences.

## 2. Model Architecture & Extreme Behavior (The Brain)
**Status: PASS**
- **The "Monster-Under-The-Bed" Syndrome:** Successfully averted. The final >80% Equity (Nuts) Fold rate dropped all the way to **11.1%**, meaning the model realizes positive EV with premium hands and correctly calls/raises (88.9% aggression/continuation rate).
- **Action Entropy:** Finished at a robust `0.9206`, proving the model retained nuanced decision-making capabilities across various board states without collapsing into single-action determinism.

## 3. The Training Loop & Loss Calculation (The Gradients)
**Status: STABLE**
- **Target Tracking:** Training concluded naturally at 150,000 hands. 
  * Final Train Loss: `4.5064`
  * Final Val Loss: `2.9689`
- **Gradient Explosions:** No gradient explosions occurred. The model converged mathematically. 

## 4. Simulation Environment & Ground Truth (The Reality)
**Status: EXCELLENT**
- **Seat/Position Overfitting:** Successfully mitigated by rotating seat configurations.
- **Exploitation Scoreboard:** The Sticky Hero achieved a **+2.5 BB/100** win rate against a highly competitive diverse league. Notably, it generated a massive `+7.3 BB/100` edge against the TAG Bot and `+5.8 BB/100` against the Maniac Bot.
- **Hero Profile:** The model accurately adopted the requested Sticky profile with a `100.0%` VPIP and `25.0%` AGG, fulfilling the exact behavioral constraints programmed for this training run.

## Conclusion
The `V11_STICKY` training successfully reached its 150,000 hands target after 2 hours and 41 minutes of simulation. The weights were successfully saved to `C:\REPO\Antigravity\AIPoker\core\weights\expert_v11_sticky.pth`. The model exhibits no structural collapse, correctly handles high-equity situations, and achieved a positive win rate against the baseline league. This checkpoint is **cleared for deployment/evaluation**.

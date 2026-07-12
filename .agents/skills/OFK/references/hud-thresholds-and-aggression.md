# iPoker/bet365 HUD Thresholds and Aggression Frequency

**Date Recorded**: 2026-07-10
**Related Files**: [PHPHelp.py](file:///c:/REPO/Antigravity/AIPoker/PHPHelp.py), [opponent_bots.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/opponent_bots.py)

## Context
The vision and self-play system needs to align with the standard bet365/iPoker HUD thresholds and calculations. These thresholds categorize opponent profiles into colors (Blue, Green, Yellow, Red) based on VPIP and Aggression Frequency (AFq) stats.

## Resolution / Guidelines

### 1. Statistics Definition and Formula
* **VPIP**: Voluntarily Put Money in Pot.
* **AGG (AFq - Aggression Frequency)**: Defined by the bet365 Danish translation formula:
  $$\text{AFq} = 100 \times \frac{\text{Bets} + \text{Raises}}{\text{Bets} + \text{Raises} + \text{Calls} + \text{Folds}}$$
  Unlike the traditional Aggression Factor (AF ratio: (Bets+Raises)/Calls), AFq represents a true percentage between 0% and 100% and accounts for folds.

### 2. HUD Thresholds & Color Coding
The iPoker HUD partitions VPIP and AFq into these ranges:
* **VPIP**:
  * Tight (Blue): `0% - 18%`
  * Normal (Green): `18% - 26%`
  * Loose (Yellow): `26% - 35%`
  * Maniac (Red): `> 35%`
* **AGG (AFq)**:
  * Passive (Blue): `0% - 36%`
  * Normal (Green): `36% - 56%`
  * Aggressive (Yellow): `56% - 71%`
  * Maniac (Red): `> 71%`

### 3. Model Vectorization Map (Range Midpoints)
When processing OCR-detected visual color labels (e.g. `Blue`, `Green`) on opponents, the vision engine maps the color to the true midpoint of the corresponding percentage range for model context feeding:
* **VPIP Midpoints**:
  * `Blue` $\to 0.10$
  * `Green` $\to 0.22$
  * `Yellow` $\to 0.30$
  * `Red` $\to 0.45$
* **AGG Midpoints**:
  * `Blue` $\to 0.18$
  * `Green` $\to 0.46$
  * `Yellow` $\to 0.63$
  * `Red` $\to 0.85$

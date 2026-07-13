# Model Health Report — Deep Check

**Target Model:** `Herocules (v11 Main)` → `expert_v11_main.pth`
**Date:** 2026-07-13
**Overall Grade:** 🔴 **CRITICAL FAIL — PIPELINE BROKEN**

> [!CAUTION]
> The V11 model pipeline is **non-functional**. No V11 model can load weights, and a critical training-data vectorizer bug silently corrupts all multi-step hand sequences. The diagnostic outputs below are **random garbage** from an uninitialized network — they should not be interpreted as model behavior.

---

## 1. Pipeline Integrity Audit (Zero-Trust)

### 🔴 CRITICAL: Weight File Missing
The default active model `Herocules (v11 Main)` references weight file `expert_v11_main.pth`, but **this file does not exist** in [weights/](file:///c:/REPO/Antigravity/AIPoker/core/weights).

Available V11 weight files:
| File | Size |
|:---|:---|
| `expert_v11_maniac.pth` | 2.81 MB |
| `expert_v11_nit.pth` | 2.81 MB |
| `expert_v11_sticky.pth` | 2.81 MB |
| `herocules_v11_fuzzyHeuristicsOpp.pth` | 2.81 MB |
| `v11_past_checkpoint.pth` | 2.81 MB |

### 🔴 CRITICAL: Architecture ↔ Weight Shape Mismatch
**Every** V11 weight file fails to load with:
```
size mismatch for state_proj.0.weight: 
  copying a param with shape torch.Size([256, 159]) from checkpoint, 
  the shape in current model is torch.Size([256, 163]).
```

**Root Cause**: The saved weights were trained when `context_dim=31` (producing `state_proj` input = `64 + 64 + 31 = 159`). The current [poker_transformer_v11.py](file:///c:/REPO/Antigravity/AIPoker/core/models/v11/poker_transformer_v11.py#L10) declares `context_dim=35` (producing `state_proj` input = `64 + 64 + 35 = 163`). The contract was upgraded from 31→35 features **without retraining** the models.

**Feature dimension breakdown:**

| Component | V8/V9 Contract (31) | V11 Contract (35) | Delta |
|:---|:---:|:---:|:---|
| Base global features | 11 (incl. raw `pot_bb`, `call_bb`) | 10 (scaled `call_bb/400`) | V11 dropped raw `pot_bb`, changed scaling |
| Per-opponent features | 4 × 5 = 20 | 5 × 5 = 25 | V11 added `opp_position` per seat |
| **Total** | **31** | **35** | **+4 new features** |

### 🔴 CRITICAL: Training Vectorizer Variable Shadowing Bug
In [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/train_selfplay.py#L84-L155), the `idx` variable is **fatally shadowed** by inner loops:

```diff
 for i, dp in enumerate(dps):          # Outer loop: per decision point
     idx = start_idx + i                # ✅ Correct dp index (0, 1, 2, ...)
     ...
-    for idx in range(5):               # ⚠️ Line 97: SHADOWS idx → ends at 4
+    for j in range(5):                 # Fix: use 'j'
         if dp['active_opponents_mask'][idx] == 1.0:
             ...
     ...
-    for idx in range(5):               # ⚠️ Line 121: SHADOWS idx AGAIN → ends at 4
+    for j in range(5):                 # Fix: use 'j'
         ...
     
     context_seq[idx] = ctx             # ❌ Always writes to index 4!
     action_seq[idx] = ...              # ❌ Always writes to index 4!
     target_evs_seq[idx] = ...          # ❌ Always writes to index 4!
     loss_mask[idx] = 1.0               # ❌ Always writes to index 4!
```

**Impact**: For any hand with multiple decision points (e.g., Preflop → Flop → Turn → River), **only the last decision point survives** and it's always written to sequence position 4 regardless of the hand's actual length. All earlier decision points are silently discarded. The model effectively trains on **single-step samples with corrupted positional alignment**, destroying the transformer's sequential learning capability.

---

## 2. Diagnostic Scenario Output (Random Weights)

> [!WARNING]
> These outputs are from an **uninitialized random network** since no weights could be loaded. They confirm the pipeline is broken but should NOT be used to evaluate model quality.

### Core Scenarios

| Scenario | EV(Fold) | EV(Call) | EV(Raise) | Action | OFK Grade |
|:---|:---:|:---:|:---:|:---:|:---|
| River Pure Air (1st to Act) | 0.03 | 0.34 | 0.18 | CALL | 🔴 CRITICAL FAIL (random) |
| River Pure Air (Facing Bet) | 0.12 | 0.18 | 0.08 | CALL | 🔴 CRITICAL FAIL (random) |
| River The Nuts (Facing Bet) | 0.39 | 0.09 | 0.28 | FOLD | 🔴 CRITICAL FAIL (random) |
| River The Nuts (Calling Station) | 0.33 | 0.03 | 0.14 | FOLD | 🔴 CRITICAL FAIL (random) |
| Preflop AA vs Nit (Deep) | 0.22 | 0.27 | 0.13 | CALL | 🔴 FAIL (random) |
| Preflop AA vs Maniac | 0.24 | 0.11 | 0.13 | FOLD | 🔴 FAIL (random) |
| Flop TPTK Multi-Way (4-way) | 0.30 | -0.00 | 0.07 | FOLD | 🔴 FAIL (random) |
| Turn Flush Draw vs Bet | 0.55 | 0.00 | 0.38 | FOLD | 🔴 FAIL (random) |

### Preflop Equity Sweep

| Eq Group | Opps | EV(Fold) | EV(Call) | EV(Raise) | Action | OFK Grade |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| <20% (Air) | 1 | 0.01 | 0.29 | 0.26 | CALL | 🔴 FAIL |
| 20-40% (Weak) | 1 | 0.24 | 0.21 | 0.20 | FOLD | ❓ Inconclusive |
| 40-60% (Marg) | 1 | 0.21 | -0.08 | 0.09 | FOLD | 🔴 FAIL |
| 60-80% (Strg) | 1 | 0.46 | -0.03 | -0.02 | FOLD | 🔴 FAIL |
| >80% (Nuts) | 1 | 0.28 | -0.03 | -0.02 | FOLD | 🔴 CRITICAL FAIL |
| <20% (Air) | 3 | -0.00 | 0.29 | 0.24 | CALL | 🔴 FAIL |
| 20-40% (Weak) | 3 | 0.23 | 0.19 | 0.18 | FOLD | ❓ Inconclusive |
| 40-60% (Marg) | 3 | -0.07 | -0.25 | 0.17 | RAISE | 🔴 FAIL |
| 60-80% (Strg) | 3 | 0.23 | 0.02 | -0.05 | FOLD | 🔴 FAIL |
| >80% (Nuts) | 3 | 0.31 | -0.04 | -0.11 | FOLD | 🔴 CRITICAL FAIL |
| <20% (Air) | 5 | -0.03 | 0.22 | 0.16 | CALL | 🔴 FAIL |
| 20-40% (Weak) | 5 | 0.26 | 0.25 | 0.16 | FOLD | ❓ Inconclusive |
| 40-60% (Marg) | 5 | 0.21 | -0.08 | 0.20 | FOLD | 🔴 FAIL |
| 60-80% (Strg) | 5 | 0.48 | -0.05 | 0.04 | FOLD | 🔴 FAIL |
| >80% (Nuts) | 5 | 0.31 | -0.13 | -0.14 | FOLD | 🔴 CRITICAL FAIL |

**No monotonic scaling observed** — expected since these are random network outputs.

---

## 3. Holes Discovered

### Hole 1: No Functional V11 Model Exists
- `expert_v11_main.pth` is missing entirely.
- All other V11 weights (`maniac`, `nit`, `sticky`, `fuzzyHeuristicsOpp`) fail to load due to the 159→163 shape mismatch.
- **The V11 poker bot is running on a randomly initialized neural network.**

### Hole 2: Architecture Upgraded Without Retraining
The V11 inference contract ([contract_v11.py](file:///c:/REPO/Antigravity/AIPoker/core/bridge/v11/contract_v11.py)) was expanded from 31 to 35 context features (adding per-opponent position encoding), but the existing weight checkpoints were trained on the 31-feature contract. This breaks `state_dict` loading and makes the weights incompatible.

### Hole 3: Training Data Corruption via Variable Shadowing
The `vectorize_hand_samples()` function in [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/train_selfplay.py#L97) uses `idx` as both the outer decision-point index and the inner opponent-iteration variable. After the inner loops, `idx` is always `4`, causing:
- All context vectors to be written to position `[4]` instead of their true sequence position
- All action/target/mask data to be written to position `[4]`
- Multi-decision hands to silently lose all but the last decision point
- The transformer to receive corrupted positional data, preventing it from learning sequential patterns

### Hole 4: Stale Comment in V11 Model
[poker_transformer_v11.py](file:///c:/REPO/Antigravity/AIPoker/core/models/v11/poker_transformer_v11.py#L23) line 23 says `31-dim` but the actual `context_dim=35`. Minor cosmetic issue but indicates the code was changed piecemeal.

---

## 4. Previous V11 Report Comparison

The earlier report for `herocules_v11_fuzzyHeuristicsOpp.pth` ([existing report](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/resources/models/v11/health_report_herocules_v11_fuzzyHeuristicsOpp.md)) graded the model as ✅ **PASS**. This is now invalidated — that model also suffers from the same `[256, 159] → [256, 163]` weight loading failure and would have been running on random weights at the time of that report as well.

---

## 5. Recommended Fix Path

> [!IMPORTANT]
> These fixes must be applied **before** any new training run.

1. **Fix the variable shadowing bug** in `vectorize_hand_samples()`:
   - Rename inner loop variables from `idx` to `j` on lines 97 and 121
   
2. **Decide on architecture version**:
   - **Option A**: Revert `context_dim` to 31 in the model and contract → existing weights work again
   - **Option B**: Keep 35-feature contract → requires full retraining of all V11 models
   
3. **Retrain** all V11 personality models after fixing the vectorizer

4. **Re-run this diagnostic** on the newly trained model to get a real Pass/Fail grade

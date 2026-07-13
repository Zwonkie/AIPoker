# Model Health Report — Herocules (v11 Main), 200k

**Date**: 2026-07-13
**Checkpoint**: `core/weights/expert_v11_main.pth` (200,000-hand self-play, post loose-collapse fixes)
**Runner**: `scripts/math/run_model_diagnostics.py "Herocules (v11 Main)"` + direct inference probe
**Overall Grade**: 🔴 **CRITICAL FAIL** — Bluff Collapse / Action-EV Extrapolation

---

## Executive Summary

The model **raises literally everything**. Across every canonical test scenario — including a 0-equity pure-air river bluff and 72o — the predicted `Raise` EV is positive and the argmax action is `RAISE`. `Fold` is correctly pinned at ~0.0 (the new fold-baseline training working), but that only makes the problem starker: since `Raise` EV never drops below 0, `argmax(Q)` never folds.

I verified this is a **genuine model defect, not a harness bug**: the behavior is identical across all inference conventions (right-pad/left-pad, mask/no-mask, index `[0]`/`[seq_len-1]`/`[-1]`). The in-game decision function is a pure `argmax` of these same Q-values, so the raw EVs are what govern play.

The model *did* learn relative equity ordering (EV rises monotonically with equity), but not the **absolute break-even** for aggression — its `Raise` baseline sits far above `Fold=0` even for air.

---

## Scenario Breakdown

| Scenario | Fold | Call | Raise | Action | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|
| River Pure Air — first to act (eq 0.0) | −0.02 | 0.74 | **1.88** | RAISE | 🔴 CRITICAL FAIL (bluff collapse) |
| River Pure Air — facing bet (eq 0.0) | −0.02 | 0.74 | **1.88** | RAISE | 🔴 CRITICAL FAIL |
| River The Nuts — facing bet | −0.01 | 1.12 | **2.89** | RAISE | 🟢 PASS (values raise > call) |
| River The Nuts — vs Calling Station | −0.01 | 1.07 | **2.80** | RAISE | 🟢 PASS |
| Preflop AA vs Nit (deep) | −0.01 | 1.20 | **3.03** | RAISE | 🟡 WEAK (raises AA ✓, but no Nit/Maniac differentiation) |
| Preflop AA vs Maniac | −0.01 | 1.21 | **3.04** | RAISE | 🟡 WEAK |
| Flop TPTK multi-way (4-way) | −0.01 | 1.31 | **3.25** | RAISE | 🔴 FAIL (no multi-way caution) |
| Turn Flush Draw vs Bet | −0.01 | 1.24 | **3.09** | RAISE | 🟡 borderline |

### Preflop Equity Sweep (1 / 3 / 5 opponents)
| Eq tier | 1-opp Raise | 3-opp Raise | 5-opp Raise | Expected |
|:---|:---:|:---:|:---:|:---|
| <20% (Air) | 2.05 | 2.25 | 2.48 | should FOLD (esp. multi-way) → 🔴 RAISE |
| 20-40% | 2.35 | 2.42 | 2.54 | mostly fold → 🔴 RAISE |
| 40-60% | 2.58 | 2.60 | 2.73 | marginal → RAISE |
| 60-80% | 3.72 | 3.84 | 4.18 | raise → 🟢 RAISE |
| >80% (Nuts) | 3.76 | 3.83 | 3.66 | raise → 🟢 RAISE |

Two failures visible here: (1) **air never folds** — `Raise` EV ≈ 2.0+ even at <20% equity; (2) EV *increases* with opponent count (1→5) instead of tightening — the exact opposite of the GTO expectation that multi-way pots demand stronger hands.

---

## Holes Discovered

1. **Action-EV extrapolation (the core hole).** `Call`/`Raise` Q-heads are trained **only on states where those actions were actually taken** (taken-action masking + the fold baseline). In-game the model rarely voluntarily raised trash, so the raise-head **never received a negative signal for raising air** and extrapolates a positive EV to every unseen state. This is precisely the "off-policy data gap" hallucination the OFK suite is designed to catch (Scenario A/F).

2. **No absolute aggression break-even.** The model ranks hands correctly by equity (monotonic EV) but its `Raise` intercept is ~+2 BB for everything, so `argmax` always raises. It learned *ordering*, not *thresholds*.

3. **Inverted multi-way scaling.** `Raise` EV rises with more opponents instead of falling — the model treats extra players as more value, not more risk.

4. **Fold baseline is the one bright spot.** `Fold ≈ 0` across the board confirms the 2026-07-13 fold-baseline fix took hold; the remaining defect is entirely on the un-anchored Call/Raise heads.

---

## Why the training dashboard looked healthier

The self-play equity matrix showed weak hands folding ~90-95%. That reflects **in-distribution, multi-step, richly-contextualized** game states plus the 5% random / (earlier) heuristic-anchor mix. The health suite deliberately probes **canonical single-state spots**, where the off-policy extrapolation surfaces. Both are true: the model is passable in-distribution but degenerate on the canonical guardrail scenarios — and since play is pure `argmax(Q)`, the degeneracy is a real risk, consistent with the loose VPIP (~52%) seen in the final training window.

---

## Recommended Remediation (not yet applied)

The fold-baseline (Fix 1) anchored Fold but left Call/Raise uncalibrated. To close the extrapolation hole, give the **untaken aggressive actions a pessimistic counterfactual target** in fold/weak spots — e.g.:
- Train `Raise`/`Call` heads toward a **model-free EV estimate** (e.g. `equity·pot − cost`, already computed in `_calculate_mc_target_evs`) for *all* three actions, not just the taken one — with the realized `mc_return` still overriding the taken action.
- Or add an explicit **negative target for raising below-break-even-equity** hands (mirror of the preflop tightness prior, but on the target itself rather than only the taken-action penalty).

This directly supplies the "raising air is −EV" signal the current pipeline omits.

---

## Harness Notes (do not affect the verdict)

- `run_model_diagnostics.py` emits `159 vs 163` load warnings for the stale `expert_v11_{maniac,nit,sticky}.pth` (pre-35-feature) and "not found" for pruned v8/v9/v10 files. The **v11 Main target loads cleanly** (163-dim, strict).
- `core/models/engine.py:predict_ev` queries the model **without `key_padding_mask`** and reads index `[-1]`, differing from the simulator's masked `[seq_len-1]` — a real inconsistency, but the probe confirmed it does **not** change this verdict (all conventions → RAISE). Worth fixing separately since the **live production bot uses this same path**.

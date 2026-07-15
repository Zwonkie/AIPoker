# V13 — Validated Findings (Range-Aware Equity / Opponent Adaptation)

**Status:** 🟡 PARTIAL. Range-aware equity works, but the live-deployment fold-weight fix (§6)
flattened the field adaptation. Primary bugs fixed & behavior validated-sharp; adaptation is a
KNOWN open regression. Recorded 2026-07-14.
**Inherits:** all of [../v12_validated/VALIDATED_FINDINGS.md](../v12_validated/VALIDATED_FINDINGS.md) §1 (those fixes are unchanged and still locked). V13 **adds one thing**: range-aware equity.

> **READ §6 FIRST** — it supersedes the §3 winrate table for the *currently deployed* weights.
> §1–§5 describe the earlier range-aware model (before the live preflop-equity bug was found).

---

## 1. What V13 added — range-aware equity (opponent adaptation)

**Problem it fixed:** v12_validated played one fixed style and lost **−48 BB/100** to a pure
Nit/TAG field because its input equity was computed vs a *random* range — it couldn't sense a
nit's strong range and paid them off.

**The fix** (`config: range_aware_equity: true`; `simulator._calculate_range_aware_equity`):
the hero's INPUT equity is computed vs each active opponent's **VPIP-color-implied range**, not
random. Blue(nit)→top 10%, Green(TAG)→top 20%, Yellow(LAG)→top 35%, Red(loose/station)→top 55%+.
The equity-primary model then auto-adapts: vs a nit its equity drops (~0.45→~0.28) so it folds;
vs a station its equity is high so it plays. **No model/contract change** — only the equity
computation. Multiway is handled correctly (each opponent draws from its own range).

- **Bucket noise** (`range_equity_noise`, default on): the range percentile is sampled with a
  truncated-uniform draw *within* the color bucket each hand — because live play only reveals the
  color, not the exact VPIP, and an early-session HUD estimate is itself noisy. It can never leak
  into another color. This is robustness/augmentation, not bias.
- **Preflop ranking:** all 1326 combos sorted by equity-vs-random, cached to
  `self_play/preflop_ranking.json` (built once in the main process before workers spawn).
- **Perf:** ~150 CPU treys sims per hero decision (independent of the fast CUDA `equity_sims`).

**⚠️ TRAIN/SERVE CONSISTENCY (critical):** the model is trained on range-aware equity, so
`eval_pure_policy.py` and `inspect_policy_vs_target.py` were fixed to set `sim.range_aware_equity`
from config. If an evaluator/live path feeds vs-random equity to a range-aware-trained model, it
silently mismatches and looks broken. Any live-play bridge MUST compute range-aware equity too
(the live HUD color → range → equity), or you reintroduce the exact "trains fine, plays badly" gap.

## 2. Calibration finding — reduce the realization discount

Range-aware equity **compounds** with the v12 realization discount (`policy_tightness_bb`): both
lower equity vs tight opponents. At the inherited 5.0 the model was over-tight (lost to stations).
**Reduced to `policy_tightness_bb: 2.0`** — range-aware equity now does the tightness adaptation
precisely, so the crude global discount only needs to cover residual multi-street realization.

## 3. Validated deployed winrate (eval_pure_policy, mature 70k) — DOMINATES v12_validated

| Field | v12_validated | **v13 (this)** |
|-------|--------------:|---------------:|
| Loose (fish/tag/nit) live4 | −0.9 | **+29.4** |
| Tight (nit/tag) live3 | −48.3 | **−15.0** |
| Calling stations live3 | +2.7 | **+7.5** |

Better on **every** field. Wins vs loose & stations; the −15 vs a *pure*-nit table is inherently
near-unbeatable (nits only stack you with the nuts) and is a huge improvement from −48. VPIP now
**diverges by field** (adapts). Behavior unchanged-good: Part C P(fold) 0.99(air)→0.04(nuts),
Part B all 6 postflop spots correct.

## 4. LIVE DEPLOYMENT (done 2026-07-14)

V13 is the **active live model** ("Herocules (v13 Range-Aware)"). Wiring:
- `core/models/v13_engine.py` — `V13ModelEngine` loads the v13 equity-primary architecture +
  `versions/v13/weights/expert_main.pth` (via the v13 manifest) and returns the **ACTOR policy**
  (softmax(policy_logits)), not q_vals. Decision engine argmaxes it unchanged.
- `core/decision.py` — added the v13 entry, `bridge_v13 = ContractV12`, dispatches v13 through
  it, **bypasses the math-engine guardrail** for v13 (like v11), and sets it as `active_model_name`.
- `PHPHelp.py` (the live GUI/dashboard) — the equity computation now calls
  `compute_range_aware_equity(hero, board, opp_colors)` when v13 is active (opp_colors from each
  active opponent's `vpip_color`), falling back to vs-random otherwise. Model dropdown lists v13
  and **defaults to it**. This is the mandatory train/serve match (§1 warning): live equity is now
  range-aware, exactly like training.
- Shared logic: `compute_range_aware_equity` is a module function in
  `versions/v13/self_play/simulator.py` used by BOTH the training simulator and the live path.

**Verified headless:** engine loads, contract vectorizes, policy-head decisions correct
(AA→RAISE, 72o→FOLD), range-aware equity monotonic (nit < station). **Needs a live smoke test**
(the GUI vision loop can't run headless): confirm OCR'd card strings are treys-format ('As','Th')
and that decisions look right in-app.

## 5. Still open (future — not blocking)

- **AGG axis (learned aggression) — NOT implemented.** V13 did the VPIP/tightness axis only.
  Bluff/value-bet adaptation vs opponent AGG remains: feed **per-opponent** AGG (not the global
  ctx[8] average) to the decision and re-enable the `bluff`/`strength` aux heads. Validate via
  bluff frequency dropping vs stations / rising vs nits. See [SPECS.md](SPECS.md) §1b, §2.
- **Eval robustness:** the winrate numbers are 4000-hand single-seed evals (±~10-30 BB/100 noise).
  For final sign-off use ≥8000 hands / 2-3 seeds. The *direction* (all fields improved, tight-field
  −48→−15) is well outside the noise band.

## 6. LIVE PREFLOP-EQUITY BUG → FOLD-WEIGHT FIX → ADAPTATION FLATTENED (2026-07-14, later)

**The bug (found live):** the deployed range-aware equity read **0.17** for a decent hand vs 4
tight seats preflop. Cause: it counted *every yet-to-act* opponent as already-in-with-their-range
(full multiway over-count), compounded by training being 4-handed vs live 6-max. Multiway equity
vs 4-5 ranges collapses toward zero → the live model played almost nothing correctly, and a first
naive patch (scoring all-fold sims as a hero win = **bundling fold-equity into every hand**) blew
up the other way: all hands compressed to ~0.5–0.9 → no discrimination → **VPIP 95.6%, −249 BB/100**.

**The fix (shared sim+live `compute_range_aware_equity`):** PREFLOP each opponent is IN only with
prob = its color VPIP (`_COLOR_TO_VPIP` Blue.10/Green.22/Yellow.30/Red.45); **all-fold sims are
SKIPPED, not counted** → equity = showdown strength *conditional on being called*, over a realistic
(fold-weighted, ~heads-up) field. Config `live_players 4→6`. Fold-equity itself is learned from the
sim TARGETS (outcomes already include folds), never crammed into the equity feature.

**Validated (retrain to 71.7k, 6-max):**
- 95% blowup GONE — VPIP 33% (early peek at 11k already sane at 27%, not 95%).
- Behavior textbook-sharp: **Part C** equity sweep steep & monotonic (P(fold) 0.91→0.03, P(raise)
  0.04→0.64); **Part B** all 6 spots correct. The model maps equity→action perfectly.
- Original **tight-field bleed FIXED**: −15.0 → **+8.3** (this was the whole v13 point).

**The cost (KNOWN, root-caused):** fold-weighting is an opponent-**COUNT** effect that runs opposite
to the range-**STRENGTH** effect, and for these fields they roughly cancel → **VPIP no longer
diverges by field (flat ~32–36%)** → the adaptation goal is not met, and 2 fields regressed:

| Field | §3 range-aware (pre-bug) | **Deployed now (fold-weight, 6-max)** |
|-------|-------------------------:|--------------------------------------:|
| Loose (fish/tag/nit) 6max | +29.4 | **−5.4** |
| Tight (nit/tag) 6max | −15.0 | **+8.3** |
| Calling stations 6max | +7.5 | **−1.9** |

Net went from +7.3 avg (high variance, exploitable by tight players) to +0.3 avg (balanced, flat).
The regression is entirely in the equity **signal**, not the policy (Part B/C prove the model is
fine). vs [Y,G,B,Y]: AA 0.80 / KJo 0.43 (was 0.17) / T9s 0.35 / 72o 0.26 — discriminating but
field-flat.

**DECISION (user, 2026-07-14): STOP & DOCUMENT.** Do NOT burn another 100BB retrain to restore
adaptation — the live target is a short-stack DoN (~5–14 BB) where equity is mostly preflop all-in
/ few-way and this 100BB-flat eval gets reshaped anyway. The un-done fix (for whenever we revisit
adaptation, likely *in* the short-stack domain): **decouple range-strength from opponent-count** —
compute the strength signal heads-up-vs-aggregate-continuing-range (so nit-range vs station-range
clearly diverges) and represent multiway separately, rather than letting the multiway count wash
out the strength divergence. Current weights stand as a validated-sharp, safely-balanced baseline.

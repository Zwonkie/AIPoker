# V13 — Specification & Validation Plan

**Status:** 📋 PROPOSED. Recorded 2026-07-14.
**Baseline:** Build V13 by `cp -r versions/v12_validated versions/v13`, clear
`versions/v13/weights`, set `manifest.version_id="v13"` and `weights_dir="versions/v13/weights"`.
**Inherit unchanged:** every fix in [v12_validated/VALIDATED_FINDINGS.md](../v12_validated/VALIDATED_FINDINGS.md)
§1 (target clip, counterfactual policy target, equity-primary architecture, realization
discount, postflop-data field). Do not re-litigate those; extend from them.

---

## 1. Primary goal — opponent adaptation (fix the one known limitation)

V12_validated plays a **fixed style** and loses −48 BB/100 to pure Nit/TAG because it can't
sense tight ranges. **This is the #1 V13 objective.**

### 1a. Range-aware equity (recommended, principled)
Currently `simulator._calculate_equity(hero, board, n_opps)` deals **random** opponent hands.
Make the hero's INPUT equity reflect each opponent's style-implied range:
- Define a range per style (nit ≈ top 10% of hands, tag ≈ top 20%, station ≈ top 50%, maniac ≈ top 60%).
- In the MC, sample opponent hands and **reject those outside that opponent's range** (or weight them).
- Apply to the **input equity** (`simulator.py` L~737) so the equity-primary model auto-tightens
  vs nits (its equity there drops from ~40% vs-random to ~28% vs the nit range).
- **Bonus:** this also shrinks the input(vs-random) vs target(perfect-info) mismatch we found.

**Risks / cautions:**
- `_calculate_equity` is used in several places — change the HERO INPUT path deliberately; decide
  explicitly whether the counterfactual target keeps perfect-info equity or also moves to range-based
  (keeping them *consistent* is the point — test both).
- Verify no train/serve skew: the same range-aware equity must be computable at live-play time
  (the live HUD gives opponent style → range), or you reintroduce a mismatch.

### 1b. Fallback if 1a underperforms
Opponent-HUD-in-the-base-head was tried in v12_validated and **backfired** (worse everywhere) —
do not simply retry it. If range-aware equity is insufficient, consider a small dedicated
opponent-encoder that outputs a per-opponent tightness embedding feeding the base head, trained
with the aux `strength`/`bluff` heads re-enabled.

---

## 2. Secondary improvements (each behind its own test)

- **Value extraction:** the model raises the nuts only ~58% (mixed). Consider re-adding the
  equity aux head (`aux_loss_weight` > 0, was 0 in verify mode) to strengthen the trunk's
  strength encoding, and/or an actor sharpening on the *now-correct* target (temperature < 1 on
  the counterfactual regret target — earlier it hurt only because the target was loose).
- **Tune `policy_tightness_bb`** on a *mature* model with a *robust* eval (see §3), not single
  30k runs. 5.0 was best of {3,5}; sweep {4,5,6} × 2 seeds at 70k.
- **Re-introduce curriculum realism** once adaptation works: turn `disable_extreme_stacks` off
  and `fixed_stack_bb` to null (stack depth variety), then test vs the FULL league
  (`pool [tag,past,nit,maniac,fish]`, `live_players 6`, `disable_past_self/focus_rounds: false`).
  Re-validate that behavior (Part B/C) survives the harder distribution.

---

## 3. Required validation tests (gate every change)

1. **`overfit_sanity.py`** must pass (loop wired) after any model.py change.
2. **`inspect_policy_vs_target.py`**: Part C Q(equity) steep (Δ≥~1), Part B all 6 spots correct,
   Part A KL 0.2–0.8.
3. **`eval_pure_policy.py`** — the adaptation win condition for V13:
   - VPIP must **diverge by field**: tighter vs Nit/TAG (target ≤ ~25%), looser vs stations.
   - Winrate **non-negative across all three fields** (pure-nit target ≈ break-even, not big +).
4. **Robustness upgrade (do this):** raise `eval_pure_policy` to ≥ 8000 hands and average **2–3
   seeds** — 4000-hand single-seed evals had winrate variance of ±30 BB/100 and misled tuning.
5. **Maturity:** judge at ≥ 70k hands. All V12 diagnosis was at ~31k (barely post-bootstrap) and
   was noisier; the mature read changed conclusions (loose-field −14 → −0.9).

---

## 4. Open research questions (not blocking, worth investigating)

- **Perfect-info vs range equity in the target:** the counterfactual target uses perfect-info
  equity (god-mode) the model can't have at inference. Does moving the *target* to range-based
  equity (consistent with a range-aware input) improve learnability? A/B it.
- **Actor vs critic at play time:** the actor (regret-matching) is what plays; the critic Q is
  well-calibrated. Would argmax/softmax over the *critic* (now that the clip stabilizes it) play
  better than the soft actor? Cheap to test via a play-time flag.
- **Multi-street realization:** the realization discount (#4) is a static equity-pivot proxy. A
  better target would use actual multi-street outcomes without the fold-equity survivorship bias
  (e.g. a debiased realized return, or a short lookahead). Research-grade; only if §1 plateaus.

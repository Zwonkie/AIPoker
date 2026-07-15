# V12_VALIDATED — Validated Training Foundation

**Status:** 🟢 AUTHORITATIVE FOUNDATION. Recorded 2026-07-14.
**Purpose:** This version is the frozen, test-verified V12 foundation. Future versions (v13+)
should `cp -r versions/v12_validated versions/vN` and build from here.

> **Binding rule for AI agents & humans:** The fixes in §1 are each backed by a reproducible
> test (§3). **Do not change, remove, or "simplify" any of them except through the testing
> workflow in §4** — i.e. change one thing, re-run the two validators, and keep the change
> only if the metrics hold or improve. Every one of these was reached by ruling out a
> *different* plausible hypothesis; reverting one silently reintroduces a bug we already paid
> for. When in doubt, run `overfit_sanity.py`, `inspect_policy_vs_target.py`, and
> `eval_pure_policy.py` before and after.

---

## 1. Locked-in fixes (DO NOT TOUCH without re-testing)

Each row is a root cause we diagnosed and a fix we verified. "Symptom if reverted" is what
comes back if you undo it.

| # | Fix | Where | Why it's load-bearing | Symptom if reverted |
|---|-----|-------|-----------------------|---------------------|
| 1 | **Target clip = 40 bb** (`TARGET_CLIP_BB`, config `target_clip_bb`) | train.py | Unclipped realized returns are fat-tailed (±100bb when a stack goes in); the same state then gets wildly varying targets → critic diverges | Q-loss 10–20 (vs ~6), policy collapses to **uniform**, VPIP 73%. This is VARIANCE control, **not** a bias hack — never bucket it with the shaping priors. |
| 2 | **Counterfactual policy target** (`policy_target_source: counterfactual`) | train.py `vectorize_hand_samples` | The realized-return override reinforced weak hands that *won* uncontested vs folders (fold-equity survivorship) | Fold-equity ratchet: weak entries reinforced, VPIP climbs. Verified: realized target for weak-hand raises was +7.82bb vs correct counterfactual −4.04bb. |
| 3 | **Equity-primary architecture** — `Q = equity_base(strength+price scalars) + residual(transformer)`; card embed bottlenecked **64→16**; residual heads **zero-initialized** | model.py | With a 128-dim card embedding vs a 1-dim equity scalar, gradient descent takes the hole-card shortcut and **ignores equity** → postflop-blind (values a SET like a weak pair) | Model plays off hole-card ranks, ignores board+equity, spews (air all-in 8%), folds a set / raises air. Equity-response Q(0.1→0.9) was flat (Δ0.03); this made it steep. |
| 4 | **Realization discount** (`policy_tightness_bb: 5.0`, pivot 0.45) | train.py | The counterfactual uses **all-in equity with no multi-street realization**, so it overvalues speculative entries | Structurally-loose baseline: pure-policy VPIP ~60%, loses to tight fields. Discount → VPIP ~30%, tight-aggressive. |
| 5 | **Postflop data via a loose field + exploration + bootstrap** (`opponents.pool [fish,tag,nit]`, `disable_exploration/bootstrap: false`) | config.yaml | A tight field folds preflop → hands never reach a flop → postflop is starved (~39 river samples) → the model can't learn a steep Q(equity) | Postflop play is undertrained/inverted (nut flush folds); equity stays compressed. Loose callers make hands go to showdown. |

**Also validated as correct (don't "fix" these):**
- The **contract** (`contract.py`, 35-feature context, `contract_version=2`) is correct and feeds equity at `ctx[3]`. The postflop-blindness was NOT a plumbing bug — equity is fed correctly; the model was ignoring it (fix #3).
- The **counterfactual EV magnitudes** (`_calculate_mc_target_evs`) are well-calibrated by equity bucket (air raise ≈ −0.2, nuts raise ≈ +32). Don't rewrite the EV math to "fix looseness" — the looseness was the *policy target source* (#2) and *realization* (#4), not the EV ranking.

---

## 2. Verified behavior of the deliverable model (`weights/expert_main.pth`, mature ~72k)

- **Equity is load-bearing:** P(fold) sweeps **0.97 (air) → 0.10 (nuts)**, steep & monotonic.
- **Correct postflop reads:** air / weak pair / missed draw → FOLD; top pair / set / nut flush → RAISE (all 6 canonical spots correct).
- **Deployed winrate (pure policy, no exploration):** vs loose mixed field **−0.9 BB/100** (break-even), vs calling stations **+2.7**, vs pure Nit/TAG **−48.3** (see §5 KNOWN LIMITATION).
- Starting point for context: the broken model was −110 BB/100, VPIP 70%, postflop-blind.

---

## 3. The validators (run these to check the foundation still holds)

All under `self_play/`, run from repo root:

- **`overfit_sanity.py`** — proves the training LOOP is wired: memorizes a tiny fixed batch (critic + actor both fit; real targets learnable). If this fails, a plumbing/gradient bug was introduced.
- **`inspect_policy_vs_target.py`** — BEHAVIOR. Part A: model policy vs training target on live states (KL should be 0.2–0.8, not ~1.1). Part B: 6 canonical postflop spots (all should be correct). Part C: equity ablation — Q must swing steeply with equity (Δ≈1+, not flat).
- **`eval_pure_policy.py`** — DEPLOYED winrate across loose/tight/station fields. **This is the only honest winrate metric.**
- **`inspect_ev_targets.py`** — audits the counterfactual EV targets by equity bucket + the realization gap.

> ⚠️ **Never trust training-time BB/100 or VPIP** shown on the dashboard — they are masked by
> the exploration heuristic anchor (tight TAG), which made a 62%-VPIP model *look* like 43%.
> Always judge with `eval_pure_policy.py` (pure policy, no exploration).

---

## 4. How to change something safely

1. Change **one** knob/file.
2. Retrain to at least 30k (≥70k for a mature read — behavior stabilizes, winrate is less noisy).
3. Run `inspect_policy_vs_target.py` (behavior) **and** `eval_pure_policy.py` (winrate).
4. Keep the change only if Part C stays steep, Part B stays correct, and deployed winrate holds/improves.
5. Note: single 30k runs have real **run-to-run variance** (fresh random init) + eval noise (4000 hands). Don't chase a scalar on one noisy run — the *behavior* (Part B/C) is robust; the *winrate* needs maturity + multiple seeds to trust.

---

## 5. KNOWN LIMITATION (open, not a regression) → see `versions/v13/SPECS.md`

The model plays **one fixed style and does not adapt to opponent type** — it loses badly
(−48 BB/100) to a pure Nit/TAG field because it pays them off instead of tightening. Root:
the hero's **input equity is computed vs a RANDOM range** (`simulator.py` `_calculate_equity`
with no `specific_opponents`), so it cannot sense a nit's strong range; uniform tightness (#4)
and bolting opponent-HUD scalars onto the base head both FAILED to fix it. The principled fix
is **range-aware equity** (v13). Note pure-nit tables are inherently near-unbeatable, so the
target there is ~break-even, not a big win.

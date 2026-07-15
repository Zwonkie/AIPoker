# V17 — Roadmap / consolidated backlog

Planning doc only (2026-07-15) — no code yet. Synthesized by reading every V16-line spec/result
(`versions/v16/SPECS.md`, `versions/v16_foldregret/SPECS.md`, `versions/v16_vsNN/SPECS.md`, both
`tools/model_verify` result JSONs) end to end: what actually moved the needle vs what didn't,
scored by the model_verify results, not by how good a change looked on the training dashboard alone.

## Carry forward as the V17 baseline (validated, no material downside)

Start V17 as a copy of `versions/v16` (NOT `v16_foldregret` — see below), which already includes:
- **P2** stack-scaled live sampling temperature (`core/decision.py`, live-serve only).
- **P4** range-aware equity as the preflop CALL/FOLD target. `vpip_adapts_to_style` PASSES cleanly
  on V16 (+8.7pt short / +8.4pt deep, vs V15's near-flat gap); `bb100_vs_standard_fields` tight_deep
  went from V15's -42.6 to V16's -7.7. Real, validated, keep building on it.
- The AGG-tracking bug fix (`decision.startswith('raise')` — correct opponent-seat AGG feature for
  any sized-model NN opponent).
- Periodic 25k-hand restore-point checkpoints (infra).
- The live "Thinking" narrative (`core/decision.py` `_narrate_thinking` + `PHPHelp.py`) — UI only,
  carries over automatically, no version-specific wiring needed.
- **`tools/model_verify --full` as a mandatory gate before any deploy decision.** Caught two real
  regressions this line alone (V15's masked deep-stack OOD, foldregret's loose-deep collapse) that
  training-dashboard telemetry alone did not surface. Non-negotiable going forward.

**Do NOT carry forward `v16_foldregret`'s fold-relative regret baseline as-is.** It fixed
`air_folds_mostly` cleanly (0.62 -> 1.00) but flipped the model's style: tight_deep improved
(-7.7 -> +24.5) while loose_deep collapsed (+62.1 -> -11.6) against the actual loose/station-heavy
live population, and its raw (pre-mask) free-check fold-preference ballooned from 0.28 to 0.95 —
evidence the change induced a general fold bias, not a targeted correction. Superseded by the
actor-critic proposal below, which targets the same problem at its actual root.

---

## [P-actor-critic] Route the actor's target through the critic's own learned Q — NOT a blended baseline

**Why not the "blend two regret baselines" idea:** flagged as the fix for air/draws overcontinuation
after the model_verify results came in, but explicitly rejected — another tuned analytical knob
(`alpha`) layered on top of an already-patched formula, the same pattern already rejected once for
[P4] (tuned discount constants) in favor of a principled single substitution. It would also still be
choosing between two baselines that are both computed from the SAME noisy single-hand estimate, so
it doesn't address why that estimate misleads the policy in the first place.

**Root cause, traced in `versions/v16/self_play/train.py`:** the actor's training target
(`policy_target_seq`, line ~295) is built ENTIRELY at dataset-vectorization time, once per hand,
from `regret_match_policy(p_evs)` where `p_evs` is a copy of that single hand's simulator Monte
Carlo counterfactual EV vector (`mc_evs`) — a ONE-SHOT, noisy point estimate (2,000 MC playouts,
plus the range-aware opponent-response heuristic baked into `_mc_target_evs_sized`). This target is
computed completely independent of the model's own weights or how many hands it has already seen —
every training label is a fresh single-sample estimate, with zero accumulated signal from what the
network has already learned about the population-level value of that action in similar spots.

Meanwhile the **critic** (`q_vals`, `model.head`) is trained via MSE regression against exactly
these same noisy per-hand `mc_evs` labels, across every hand and epoch — which is precisely what
supervised regression asymptotically estimates: the smoothed, denoised, POPULATION-AVERAGE value of
each action given the state. That is the literal "you have done this many hands and you keep
losing, on average" signal the training loop is missing for the actor — and the network is already
computing an approximation of it, in a head that currently has zero influence on action selection
(`model.py`'s own comment: the critic is deliberately never argmax'd directly, "that was the V11
failure mode" — but that's about not picking actions by raw argmax over one uncalibrated head, not
a reason the actor's distributional regret-matching can't be informed by the critic's value).

**Proposed fix:** move `regret_match_policy` from dataset-build time into the TRAINING LOOP, and
compute the actor's regret-matching target from the model's own (detached, no backprop into the
critic) `q_vals` predictions instead of the raw single-hand `mc_evs` sample:
- Needs a batched/differentiable-shaped (but detached) torch reimplementation of
  `regret_match_policy` usable inside the loop (currently a plain per-decision Python function).
- `q_vals` stays trained exactly as now (MSE regression vs `mc_evs`, unbiased) — only the ACTOR's
  target source changes, not the critic's own loss.
- **Bootstrapping risk (the real risk here, needs real validation, not a hand-wave):** early in
  training the critic itself is poorly calibrated, so regret-matching against it would train the
  actor on garbage until the critic converges. Mitigation: reuse the EXISTING `bootstrap_alpha`
  decay machinery (already present, 1.0 -> 0.0 by 30k hands) rather than inventing a new tuned
  constant — extend its schedule to also gate this transition, so the actor uses the raw
  `mc_evs`-based target (today's behavior) while the critic is still unreliable, and progressively
  shifts to the critic-based target as training matures. This is scheduling on TRAINING PROGRESS
  (a thing the codebase already does), not tuning a blend between two value spaces.
- **Working hypothesis worth testing explicitly:** once the regret-matching input is a smoothed,
  denoised value rather than one noisy sample, the original mean-vs-fold baseline debate may become
  much less consequential — a denoised mean baseline might resolve the air/draws overcontinuation
  problem on its own, without needing v16_foldregret's fold-anchoring change at all. If true, this
  is a single structural fix that dissolves the whole tradeoff, not a new tuning surface.

**Validation plan:** this is a genuine architecture change (bigger than any single-variable V16-line
tweak) — needs its own isolated experiment, same discipline as v16_foldregret (one variable, normal
config otherwise, `overfit_sanity` first, then a short run with the same 50k sanity checkpoint
pattern, full `tools/model_verify --full` before any deploy consideration, explicitly re-checking
BOTH `air_folds_mostly` and `bb100_vs_standard_fields` across all 4 fields so a repeat of the
loose-deep-collapse-while-something-else-improves pattern can't hide again).

### Implementation (2026-07-15) — scaffolded, tested, launched as a test run

Scoped to exactly this experiment per discussion: **P-actor-critic + confirm P3** (no build needed
for P3 — `short_stack_polarization` already passes on both V16 and foldregret, so it's carried
forward and re-verified for free by the same `model_verify` pass). Everything else (P0 deep-stack
OOD, P5/P6, P7) stays parked pending this result, as agreed — "P-actor-critic will likely change
the bigger picture if it works."

`versions/v17/` scaffolded from `versions/v16` (refs rewired, manifest `version_id="v17"`,
`versions/v16/weights/expert_main.pth` copied to `versions/v17/weights/frozen_v16.pth` as the new
pinned past-self benchmark — V17 must beat frozen-V16). Config otherwise IDENTICAL to V16's main
recipe (station-heavy pool, DoN stack mixture, range-aware equity, `policy_tightness_bb: 2.0`) —
only `target_hands: 100000` differs (test-run budget, not the main line's 200k; matches
`v16_foldregret`'s precedent of a short directional check first).

**Discovered prerequisite, implemented:** `COUNTERFACTUAL_WEIGHT` (train.py) decoupled from
`disable_target_shaping` — it's now ALWAYS at its module default (0.5) regardless of that flag,
which still only zeros `TIGHTNESS_PENALTY_BB` (left out of scope). Without this the critic would
have had real gradient for only FOLD + the taken action per sample — too sparse a supervision to
trust as the actor's new value source. See `core/manifest.py`'s docstring for the full trace.

**Core mechanism:** `regret_match_policy_torch(action_values, equity=None)` — a batched torch
reimplementation of the existing scalar `regret_match_policy`, unit-tested to match it exactly on
identical inputs, plus reproducing the SAME `POLICY_TIGHTNESS_BB` realization discount the
dataset-time path applies (via the batch's own `ctx[3]` equity feature) so this is a like-for-like
swap of only the VALUE SOURCE, not a second simultaneous change to the shaping logic. Wired into
both the train and validation loops: for `hands_done < ACTOR_CRITIC_CUTOVER_HANDS` (30,000 — the
same milestone `bootstrap_alpha` already uses to finish its own decay, reused rather than a new
tuned constant), the actor's cross-entropy target stays the dataset's precomputed `b_pol` (today's
behavior, unchanged). Above that hand count, the target is computed fresh each batch from
`regret_match_policy_torch(preds.detach(), equity=...)` — the model's OWN critic prediction for
that exact batch, detached so the actor's loss cannot backprop into the critic's weights. Hard
cutover, not a blended/tuned mix of two value sources.

**Pre-launch checks, all passed:** `regret_match_policy_torch` unit-tested standalone (exact match
vs the scalar function on identical inputs; degenerate all-tie case falls back to uniform; equity
discount correctly zeroes weak-equity actions and leaves strong-equity ones untouched; batched
shapes/normalization hold across a random `[4,7,6]` tensor). `overfit_sanity` PASSES (synthetic
|Q-target| 0.45bb / actor KL 0.0003; real targets |Q-target| 2.21bb / KL 0.0230, learnable) — note
this only exercises the pre-cutover path (64 synthetic hands, far below 30k), so it validates the
unchanged wiring, not the new critic-routing branch directly; that branch's correctness rests on
the standalone unit tests plus live monitoring once the run crosses 30k hands.

**Launched** fresh (no warm-start, matching every prior version in this line):
`python -m versions.v17.self_play.train --personality main --num_hands 100000`.

**Validation plan for this run:** sanity checkpoint once past 30k hands (into the new critic-routed
regime) — compare the Equity Action Matrix's air/draws rows against V16's own numbers, watching
specifically whether `air_folds_mostly`-style overcontinuation improves WITHOUT the
`bb100_vs_standard_fields` style-flip foldregret showed. Full `tools/model_verify --full` at
completion against both V17 and a fresh V16 baseline before any deploy consideration, per the
lesson logged in `model-verification-suite.md`.

### Round 1 result (2026-07-15) — STOPPED at 55,296 hands, diagnosed, not a dead end

Stopped by request after the 50k sanity check showed the WRONG trend: air/draws Fold% *declining*
steadily since the 30k cutover (Pure Air 63.4%->56.3%, Draws 84.5%->74.3% between hands 29,357 and
55,296) and hero VPIP climbing 48.8%->64.5% with no sign of leveling off.

**Diagnosis (comparing the 25,668- and 51,579-hand checkpoints directly):** NOT a value-
overestimation feedback loop (the original worry going in) -- the critic itself is well-calibrated
and got MORE confident FOLD is correct at weak equity over training, not less. The actual bug: the
MEAN-baseline regret-matching formula dilutes FOLD's share regardless of how clean the input values
are, because one steeply-negative outlier action (ALLIN, e.g. Q=-8.3 at 25% equity) drags the shared
mean down far enough that CALL/small-raise -- objectively worse than FOLD in the critic's own
assessment -- still clears that diluted mean and keeps real probability mass. Confirmed directly:
feeding the SAME 51,579-hand checkpoint's critic Q-vector through a fold-relative baseline instead
of mean gave 100% FOLD vs mean-baseline's ~29% on the identical numbers. The actor-critic routing
itself works exactly as designed (the actor closely tracks regret-matching over its own critic by
51k hands) -- it just proved the formula, not the noise, was the persistent bug all along
(`v16_foldregret`'s fold-relative fix looked risky because it was tested on NOISY raw `mc_evs`,
which is a different combination than fold-relative on a DENOISED critic).

### Round 2 (2026-07-15) — fold-relative baseline on the critic-routed Q, launched

Single change from round 1: `regret_match_policy_torch` gained a `baseline_mode` param
(`'mean'` default, `'fold'` new). Post-cutover call sites (train + val loops) now pass
`baseline_mode='fold'`. Pre-cutover path is completely untouched (still the dataset's mean-baseline
`b_pol` over raw `mc_evs`, exactly matching V16 and every prior version) -- this isolates the
change to exactly one variable: which baseline the DENOISED critic-routed target uses, not a second
simultaneous change to the value source or the noisy-raw regime. Degenerate-tie fallback for
`'fold'` mode is fold-outright (matching `v16_foldregret`'s scalar version), not uniform.

Also bumped `equity_sims: 2000 -> 5000` (CUDA MC rollouts, user-flagged as low time cost) --
orthogonal measurement-precision knob, not a competing hypothesis, noted here in case results need
disentangling later.

Old (round-1, rejected) checkpoints deleted before relaunch (`weights/checkpoints/*`,
`temp_active_model_main.pth`) to avoid stale-artifact confusion at matching hand-count filenames.

**Pre-launch checks:** `regret_match_policy_torch('fold' mode)` unit-tested — exact match vs a
scalar fold-relative reference; degenerate case folds outright (not uniform); `'mean'` mode
regression-tested unchanged vs round 1; the exact diagnosed Q-vector reproduces clean >99% FOLD.
`overfit_sanity` critic check is noisy run-to-run (unseeded synthetic data: 0.44bb / 1.55bb / 0.76bb
across 3 runs, 2/3 clean) but the fold-relative code path isn't even exercised by its small
synthetic set (well under the 30k cutover) -- the FAIL run doesn't implicate this change.

**Launched** fresh (no warm-start): `python -m versions.v17.self_play.train --personality main
--num_hands 100000`. Validation plan unchanged from above, plus this round specifically watches
whether the deep-stack/style-adaptation regression `v16_foldregret` showed reappears (the risk this
round is testing is whether it doesn't, now that fold-anchoring applies to a denoised value).

### 50k-ish sanity check, round 2 (2026-07-15) — working, opposite trend from round 1

| Hands | Fold% Pure Air | Fold% Draws | Hero VPIP | Action Entropy |
|---|---:|---:|---:|---:|
| 29,378 (pre-cutover) | 64.5% | 84.6% | 48.5% | 1.60 |
| 40,509 (+11k post) | 64.1% | 84.7% | 39.3% | 1.26 |
| 53,465 (+23k post) | 64.6% | 86.1% | 34.9% | 1.03 |

VPIP declining and Draws-fold improving post-cutover -- the opposite of round 1's runaway
widening. Hero BB/100 also climbing (+18.2 -> +22.7). No NaN/crash, no sign yet of foldregret's
style-flip.

### Telemetry gap found + fixed while investigating a user question (2026-07-15)

User asked why the live dashboard's `<20%` bucket still showed ~23% Call / ~7% All-In despite the
fold-relative fix. Direct offline probe of the LIVE checkpoint (`temp_active_model_main.pth`) at
true air equity (0.03-0.18) facing a real bet: **folds 100% of the time, every scenario tested,
both loose and tight opponents.** So the mechanism itself is clean when actually facing money --
the dashboard number was measuring something else.

Root cause: `telemetry.py`'s `record_hand_terminal_state(equity, street, action, call_amount, ...)`
took `call_amount` as a parameter but never used it -- every decision landed in the equity bucket's
Fold/Call/raise tally regardless of whether the hero was actually facing a bet or just checking for
free. This contract has no separate CHECK action (checking with air when nothing is asked -- correct
and mandatory -- and calling a real bet with air both show up as "Call"), so a large, unknown
fraction of every bucket's "Call%" was free checks, not paid continuations. The chart could not
previously distinguish "hero is making a bad continuation decision" from "hero is doing the
required, zero-risk thing."

**Fixed:** `record_hand_terminal_state` now tallies `call_amount<=0` decisions into a separate
`free_checks` counter, excluded from `total`/`actions`/`net_chips`/`won_chips`/`lost_chips` -- the
existing Fold%/Call%/raise/All-In/Net-Chips columns now mean "given the hero FACED A BET", which is
what actually answers the continue-vs-fold quality question. Dashboard print header relabeled
"Equity Matrix (FACING A BET ONLY)" with a new `Free` column showing the excluded count per bucket.
`tools/training_monitor/parse_training_log.py` + `dashboard.html` updated to parse/render the new
column, backward-compatible with both older log formats (no N-Hands column at all, and N-Hands-but-
no-Free) via the same positional integer-cell detection used for the earlier N-Hands addition.
Unit-tested (telemetry split logic + all 3 parser format generations).

**Not retroactive to the currently-running round-2 job**: same caveat as every mid-run source edit
this session -- Python doesn't hot-reload into an already-running process, so the live dashboard
keeps showing the OLD conflated Call% until the next launch. Round 2's own remaining sanity checks
should be read with that in mind (the visible Call% still overstates paid continuations); the
All-In% column is unaffected by this bug (bluffing/shoving is never a free action) and remains a
reliable read.

### Round 2 COMPLETED + DEPLOYED LIVE (2026-07-15/16)

Trained to completion: 100,000 hands, 2h26m, zero NaN/crash. Terminal telemetry note: the very
last dashboard tick's VPIP (28.1%) is a tail-batch artifact (`batch_hands = min(sim_batch_size,
num_hands - hands_done)` shrinks the last few batches to 40/6/8 hands to land exactly on
`num_hands`, and VPIP is EMA-tracked at alpha=0.2 -- a 6-8 hand batch has huge single-batch
variance). The real converged number, read from the full 50k->99k trajectory, is **VPIP stable at
~36-40%** from ~80k hands onward -- moderate and plausible, not the runaway 64% of round 1 nor an
implausible over-tightened 28%. Action Entropy's decline (1.60->0.63) is genuine, not an artifact
(smooth and monotonic through the tail, no jump).

**`tools/model_verify --full`: 10 PASS, 1 WARN, 1 FAIL** (`results/v17__expert_main.pth.json`):
- `deep_stack_ood_guard` FAIL — same pre-existing carried defect every version in this line fails
  (V15/V16/foldregret/V17 all show the same eq≈0.55, 15-40bb, single-bet -> ALL-IN argmax
  signature). NOT a V17 regression; tracked as V18 [P0].
- `vpip_adapts_to_style` PASS: short +9.7pt / deep +5.8pt (both clear the 5pt bar; V16 was
  +8.7/+8.4, foldregret FAILED deep at +2.0pt). **No style-flip regression.**
- `bb100_vs_standard_fields`: loose_short +28.9, **loose_deep +90.3** (V16 +62.1, foldregret
  collapsed to -11.6 — V17 has NO collapse, in fact the best loose_deep number in the whole line),
  tight_short +18.4 (down from V16's +38.7 — the one real give-back), tight_deep +32.6 (V16 -7.7,
  fixed, matching foldregret's fix).
- `air_folds_mostly`: 1.00 (V16 was 0.62 — the original problem, cleanly fixed).
- `beats_frozen_predecessor`: +87.5 BB/100 vs frozen-V16 over 4000 hands — convincingly beats its
  own immediate parent.

**Verdict: confirms the round-1/round-2 diagnosis.** Fold-relative regret-matching was the right
fix for the air/draws overcontinuation problem all along -- it just needed to be anchored to a
DENOISED critic Q instead of a noisy single-hand `mc_evs` sample to avoid `v16_foldregret`'s
overcorrection. V17 gets the original fix's benefit (air_folds_mostly, draws-fold) without the
side effect (loose_deep collapse), and even improves on V16's own numbers in most fields.

**DEPLOYED LIVE** (user request): `core/models/v17_engine.py` (`V17ModelEngine`, `is_v17=True`) is
the ACTIVE model in `core/decision.py` as `Herocules (v17 Actor-Critic)`; V15/V14/V13 kept as
fallbacks. `is_sized_model` bridge condition extended to include `is_v17_model` (shares the
identical V14/V15 6-action contract, slider-sizing, and math-engine-bypass paths unchanged).
`PHPHelp.py` dropdown default -> V17, range-aware equity import extended to
`versions.v17.self_play.simulator.compute_range_aware_equity`. Smoke-tested clean (air at 6% eq
facing a bet -> FOLD 100%; 97% eq -> ALL-IN, slider 1.00; "Thinking" narrative renders correctly
for both).

**Remaining backlog pushed to `versions/v18/SPECS.md`** per user request: [P0] deep-stack OOD (now
elevated -- 4 versions running unaddressed), [P5]/[P6] input-contract gaps, [P7] opponent-pool NN
personalities, and the `equity_sims` budget analysis (see below).

### equity_sims 2000->5000 — analyzed, recommend reverting for V18

User asked whether the mid-experiment bump (justified as "CUDA, should be cheap") had a real
impact, and offered either a comparison training run or a direct performance analysis. Did the
direct analysis (cheaper and more conclusive than another training run) — full numbers and
verdict in `versions/v18/SPECS.md` ("MC equity_sims budget"). Short version: the assumption was
wrong for the path that matters (`_calculate_equity` with `specific_opponents` set — the ONLY path
`_mc_target_evs_sized` uses for every decision's target — explicitly skips the CUDA evaluator and
runs a pure-Python per-simulation loop). Measured 2.46x wall-clock cost per call at 5000 vs 2000
sims (matches training's observed ~20-50 -> ~11-12 hands/sec slowdown, compounded by 3-5+
`_calculate_equity` calls per decision in multi-way spots), for a measured ~0.3-percentage-point
reduction in equity-estimate noise (0.87%->0.55%, matching theoretical `1/sqrt(n)` scaling exactly).
Given the critic gets its own, much more powerful denoising for free via regression across
thousands of hands (the actual mechanism V17 validated), this modest per-hand noise reduction is
unlikely to have been load-bearing for the result. Recommend V18 revert to 2000.

---

## [P0] Deep-stack OOD trash-jam — highest-priority carried defect

`deep_stack_ood_guard` FAILS on V15, V16, AND v16_foldregret — three versions running with this
unaddressed as a side effect of unrelated work (each retrain fixed something else and just happened
to inherit this). V17 should give it a dedicated pass rather than hoping another change fixes it
incidentally again: revisit the critic's `target_clip_bb=40` behavior at depth (V15's own SPECS
flagged Q-loss running high at 30-50bb as a known-not-urgent gap — may be related), add explicit
deep-stack decision-quality curriculum weight, or a dedicated diagnostic sweep at the exact incident
conditions (43-55% equity, 15-40bb, facing a modest bet) as a first-class training signal, not just
an eval-time check.

## [P3] Preflop polarization — CONFIRM RESOLVED, close it out

Sitting as "deferred, watch if it self-resolves" since the P4 retrain. The data already answers
this: `short_stack_polarization` PASSES on both V16 (avg P(call) 0.09 in shove-or-fold spots) and
v16_foldregret (0.13) — comfortably polarized. Recommend explicitly closing this item in V17 specs
as confirmed-resolved-by-P4 rather than leaving it open indefinitely; re-open only if a future
model_verify run regresses it.

## [P5]/[P6] Input-contract gaps — promote over further target-formula tuning

Two versions running (P4, foldregret) have squeezed real-but-tradeoff-laden gains out of tuning the
regret-matching TARGET FORMULA. The model still has no encoding of who raised, how many opponents
raised, or bet-size patterns in the action history (`act` tensor is hero-only; history tokens are
size-blind). This is a genuinely new information channel, not another reweighting of existing
signal — likely a cleaner lever than continuing to iterate on the target/loss side. Candidate to
bundle into V17's contract bump alongside the actor-critic change above (both touch the training
pipeline; worth landing together if the actor-critic experiment validates).

## [P7] Opponent-pool NN personalities / "Yellow"-LAG gap — still backlog, lower priority

Checked empirically (V15): the live model's response across the untrained VPIP=0.26-0.35 gap is
smooth/monotonic, so not an urgent correctness bug. Mechanism to close it properly (dedicated NN
opponent personalities via the dormant `hero_personality` forcing path) is fully scoped in
`versions/v16/SPECS.md` [P7] if it becomes worth prioritizing later.

## Tooling: `model_verify` composite/weighted score

Diagnosing the foldregret regression required manually cross-referencing `bb100_vs_standard_fields`'s
4-way table against the known live-population mix (loose-heavy) to see the loose-deep collapse — the
single most decision-relevant number wasn't surfaced by the check's own PASS/FAIL summary (it PASSED
as "no baseline recorded"). Add a composite score weighted toward the actual live field mix so a
future comparison surfaces this kind of tradeoff automatically instead of requiring a manual re-read
of the raw JSON.

## v16_vsNN (reference only, not actionable)

Exploratory 3-seat NN-vs-NN-vs-NN side branch, stopped early for resource reasons at ~65% of its
75k-hand budget. Showed hero VPIP climbing rapidly (38.7% -> 63.5% over 8 minutes) against an
all-NN opponent population with no heuristic bots. Inconclusive (stopped too early to distinguish
a real finding from a transient), but worth noting as a caution if V17's opponent pool composition
changes: self-play against skilled-but-unconstrained NN opponents may drift VPIP fast without a
heuristic anchor in the mix. Not scheduled; revisit only if pool composition changes intentionally.

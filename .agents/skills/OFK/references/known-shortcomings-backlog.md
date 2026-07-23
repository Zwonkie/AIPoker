# Known Model Shortcomings — Tracked Backlog

**Date Recorded**: 2026-07-17
**Related Files**: spans the whole model line — see each entry's own references.

## Purpose

A standing, living catalog of identified weaknesses in the Herocules model line, independent of
which version is currently active. This is NOT a per-version snapshot (those live in
`references/V*/specs.md`) — it's the cross-version comparison point.

**How to use this doc:**
- Before diagnosing a new live-play or training-time issue, check whether it matches an existing
  entry here first — a "new" observation is very often a known item resurfacing.
- When investigating or shipping a new version, re-check each OPEN/PARTIAL item's status against
  that version's actual behavior (model_verify results, live sessions) and update the entry's
  **Last confirmed** field + status. Don't just append a new entry for the same underlying issue.
- New shortcomings get a new entry in the relevant category, following the same
  Simple / Technical / Suggestion structure.
- Move an item to **Resolved** only when a version has actually verified the fix (not just
  attempted it) — keep resolved items listed (briefly) so a future regression is recognized as a
  regression, not treated as a fresh discovery.

## Status legend
- 🔴 OPEN — actively present, unaddressed
- 🟡 PARTIAL — improved, tried-and-failed, or resolved-with-a-tradeoff
- 🟢 RESOLVED — fixed and verified; kept for history/regression pattern-matching
- ⚪ METHODOLOGY — a gap in how we validate/tool, not a model behavior bug

---

## Betting behavior

### [BET-1] No middle gear — shove-preference 🟢 RESOLVED (V29, 2026-07-20 — critic-consistency filter + higher risk-aversion)
**First identified**: V15 SPECS.md (`[P1]`, framed as a raise-fraction-floor artifact, believed
subsumed by V15's stack-range widening). **Reconfirmed**: V20_preflopEq / V20_preflopEq_AI
(2026-07-17) — NOT actually resolved by V15 as once believed. **Last confirmed**: V25 100k-hand
confirmatory run (2026-07-18) — MIXED result, see below (50k diagnostic was the best result of the
whole investigation; 100k confirmed the core `vpip_adapts_to_style` hypothesis even more strongly,
but the direct Q-gap widened back to 1.73-1.78x from 50k's 1.35-1.36x, echoing V24's own
more-training-hurts-not-helps pattern — full detail in `versions/v25/SPECS.md`, not yet resolved
which of the two readings to trust). V24's own calibrated
setting (2026-07-18) FAILED outright: allin-vs-next-best Q-value gap WIDENED (V23: 2.4-2.5x at
stack 15/25/40bb; V24: 2.75-3.18x at the SAME cells), `action_diversity` unchanged at
`{'fold':9,'allin':12}` (2 actions). V24_extreme (diagnostic) then showed the opponent-response
lever COULD move the needle at an extreme, non-production setting, at the cost of a new
`vpip_adapts_to_style` regression. V25 pivoted to a structurally different fix — see below — and
achieved a tighter Q-gap than V24_extreme WITHOUT that regression, at realistic settings.

**Simple**: We rarely bet a normal amount — decisions are mostly fold, call, or shove all-in;
medium-sized raises (33%/66%/pot) are almost never actually chosen even though the option exists.

**Technical**: `model_verify`'s `action_diversity` check shows RAISE_33/RAISE_66 essentially never
winning argmax across the whole equity×stack test grid, in every checkpoint of V20_preflopEq
through V22 (`{'fold':9,'allin':9-11,...}`, at most 1-5 raise-bucket cells out of 21). Root cause
traced directly in `opponent_bots.py`'s `FuzzyPlayerArchetype`: once a training opponent's equity
clears its `need_for_value` threshold, it calls/raises "regardless of price" (verbatim in the
code) — a min-raise and a shove extract identical zero-extra-fold-cost value from those hands, so
the critic learns ALLIN as the dominant action.

**Fix tried and RULED OUT (2026-07-17/18, V23)**: made the value-raise branch price-sensitive
(`VALUE_PRICE_SENSITIVITY=0.05`, calibrated via a standalone probe across all 4 archetypes,
carefully chosen to avoid overcorrection — see `versions/v23/self_play/calibrate_bet1.py` and
`versions/v23/SPECS.md` for the full derivation) in both `decide_postflop` and a restructured
`decide_preflop`, then retrained 150k hands fresh. Result: `action_diversity`/
`deep_stack_ood_guard` got WORSE, not better. Direct Q-value inspection at the same failing cell
(`eq=0.55, stack=40bb`) shows why: V22's Q-values there were `call=0.52, raise_33=0.63,
raise_66=0.64 (actual max), raise_pot=0.56, allin=0.49` (healthy, ALLIN not even top); V23's at
the IDENTICAL cell: `call=0.76, raise_33=1.03, raise_66=1.10, raise_pot=1.24, allin=2.99` — ALLIN
now more than double the next-best action.

**Root cause of the regression, found by code inspection (not speculation)**: `simulator.py`'s
`_mc_target_evs_sized` — the function computing HERO's OWN per-size training target — calls the
*exact same* `bot.decide_preflop`/`decide_postflop` functions the fix patched, to sample how often
opponents fold to each of hero's hypothetical raise sizes (`p_all_fold`), then credits
`p_all_fold * pot` straight into that size's counterfactual EV. Making bots fold MORE to oversized
bets doesn't just describe more realistic live-opponent behavior — it ALSO mechanically INFLATES
hero's own ALLIN target, since more folding = more free pot-wins credited to the shove option. The
fix's two effects (opponents demand more to continue vs. hero gets more fold-equity credit for
shoving) point in OPPOSITE directions for this goal, and the fold-equity-credit effect won,
re-inforcing the shove preference instead of reducing it.

**Fix 2 tried and RULED OUT (2026-07-18, V24)**: implemented exactly the revised suggestion below
— decoupled `_mc_target_evs_sized` from live `decide_*` (new `_ev_target_fold_decision`, reverting
the target's value branch to pre-BET-1 flat `need_for_value`, keeping the pre-existing P1b
`continue_bar`) — PLUS a new "show of strength" mechanism: a per-personality `bot_bluff_perc`
trait (`opponent_bots.py`) giving non-all-in raises a probabilistic, price-independent bonus fold
rate that all-in doesn't get (`RAISE_RESPECT_BOOST=0.10`, calibrated via direct EV-arithmetic
checks on the isolated fold-decision function — confirmed a real, non-degenerate
`P(fold|raise) - P(fold|allin)` gradient existed for all 4 archetypes BEFORE committing to a
retrain). A real implementation bug was caught during that same calibration (boosting
`need_for_value` has NO effect at realistic raise-pot price levels — only `continue_bar` gates
fold-vs-continue there; fixed by boosting `continue_bar` directly) — a good example of the
calibrate-before-retrain discipline catching an error cheaply.

Despite all that care, the FULL 150k retrain result was a clean negative: `action_diversity`
unchanged (still 2 actions), `deep_stack_ood_guard` still fails, and the allin-vs-next-best
Q-value gap WIDENED at every tested stack depth (see versions/v24/SPECS.md's full table) rather
than narrowing. Aggregate win-rate metrics (`bb100_vs_standard_fields`, `beats_offformula_stress`)
held up fine or slightly improved -- suggesting the training population still doesn't punish
over-shoving enough in EV terms for this lever (even with a real, confirmed-in-isolation
behavioral change) to overcome all-in's advantages. One NEW regression found: `committed_sensitivity`
dropped from V23's PASS (0.109) to WARN (0.011) -- not yet root-caused, plausibly downstream of
the policy being even MORE all-in-concentrated (sizing-relevant features matter less when the
policy barely uses graduated sizes).

**Diagnostic run (2026-07-18, V24_extreme)**: deliberately EXTREME, throwaway test -- `bot_bluff_
perc` pushed to near-0 for all 4 archetypes, `RAISE_RESPECT_BOOST` 0.10->0.40, 50k hands (not
150k), `stack_depth_mix` replaced with a single `fixed_stack_bb=35`, bootstrap warmup skipped
entirely. Result: **real, multi-dimensional movement for the first time in this whole
investigation**:
- Allin-vs-next-best Q-value gap roughly HALVED at every tested stack depth (e.g. 35bb: 3.18x ->
  1.68x; 40bb: 3.18x -> 1.60x).
- `action_diversity`: `{'fold':8,'raise_pot':2,'allin':11}` -- `raise_pot` won 2 argmax cells, the
  FIRST time any raise bucket has won a cell across the entire V20-V24 lineage.
- `stack_full_sweep`'s argmax path became a genuine multi-action transition by stack depth
  (`allin`x4 -> `raise_pot`x2 -> `fold`x3) instead of one action dominating all 9 points.
- Three long-flat features became load-bearing for the first time: `committed_sensitivity` (WARN
  0.011 -> PASS 0.075), `pot_type_sensitivity` (WARN 0.004 -> first-ever PASS 0.122),
  `opponent_style_sweep` ([OPP-5], WARN since V20_preflopEq_AI -> first-ever PASS 0.151).
- Real win-rate held up: `bb100_vs_standard_fields` PASS all 4 fields (+27.6 to +42.8 BB/100),
  `beats_offformula_stress` PASS -- still a genuinely profitable strategy, not diversity-for-its-
  -own-sake.
- **Cost / new regression**: `vpip_adapts_to_style` FAILED (short delta only +1.9pts, needs
  >=5pt) -- every prior version passed this. Hero now enters 61-75% of hands almost regardless of
  opponent tightness; opponents fold to raises so readily at this setting that a wide-and-
  aggressive default works nearly everywhere, at the cost of no longer adapting entry range to who
  is at the table. `deep_stack_ood_guard` still FAILs too (cell moved to 15bb).
- **Confound, by design**: FIVE things changed at once (the two bluff constants, fixed-vs-mixed
  stack depth, bootstrap on/off, 50k-vs-150k hands) -- this run cannot isolate how much of the
  improvement (or the new VPIP regression) is attributable to the bluff mechanism specifically vs.
  the simplified curriculum/faster convergence in general. See versions/v24_extreme/SPECS.md for
  the full numbers.

**Fix 3 — structural pivot, tried and currently the best result (2026-07-18, V25)**: rather than
another opponent-response calibration pass, attacked the root cause identified during the
V24_extreme write-up directly: `_mc_target_evs_sized`'s `ev_if_called = true_equity * (pot +
2*raise_size - to_call) - raise_size` treated a called (non-all-in) raise as a TERMINAL, right-now
showdown, with zero representation of the value a raise preserves by keeping future streets' betting
alive (implied odds when hero improves, continued fold equity later) vs. an all-in that forecloses
that entirely. User explicitly chose to prioritize a fix that produces EMERGENT/learned behavior
(discover the value of continued betting by simulating it) over more hand-tuned opponent-response
constants. Two designs were considered and put to the user as an explicit fork — TD-bootstrap via
the model's own in-training critic (maximally "emergent" but a real paradigm shift/instability risk
for a codebase that's never done self-referential targets) vs. a one-street-deep MC rollout with a
FIXED continuation policy (same MC/counterfactual paradigm already used everywhere in this codebase,
lower risk, directly calibratable). **User chose the MC rollout.**

New `SixMaxSimulator._rollout_continuation_ev` (`versions/v25/self_play/simulator.py`): for every
non-all-in raise size on a non-river street, deals the next card(s), recomputes a cheap MC equity,
applies hero's fixed c-bet policy (bet ~2/3 pot if new equity clears a threshold, else check --
deliberately NOT the live NN, to avoid bootstrapping the target off the model being trained), asks
the opponent's REAL `decide_postflop` whether it folds, and adds the realized delta vs. the old
single-street assumption as an ADDITIVE correction. River and all-in are untouched (correctly --
neither has a "next street"). Calibrated in isolation first (`calibrate_multistreet_ev.py`) --
caught and fixed a methodology bug in the calibration script itself (eyeballed `true_equity` instead
of computing it from the actual cards) before drawing any conclusion. Confirmed: exactly 0 at
river/all-in, meaningfully positive on a deep-stack flush draw (+2-11% of pot) and an already-strong
made hand (+1-19%, larger on average since a favorite gets paid nearly every trial vs. a draw only
benefiting on the trials that hit), shrinking toward 0 as the stack empties. Cost: ~2.4x slower
per hand (smoke-tested), from up to 3 sizes x 4 rollout trials x a 150-sim equity call each.

**Result (50k-hand fast diagnostic, realistic settings, no extreme parameters)**: allin-vs-next-best
Q-gap at the standard cells (eq=0.55, stack 15/25/35/40bb) narrowed to **1.35-1.36x** -- tighter than
V24_extreme's own extreme-parameter 1.60-1.76x -- AND Q-values are now cleanly monotonic by size
(fold<call<r33<r66<rPot<allin), a smooth curve rather than an erratic one. Critically,
**`vpip_adapts_to_style` still PASSES** (+8.1/+6.0pts) -- V24_extreme could only reach a comparable
Q-gap by breaking this same check. `model_verify --full`: 18 PASS/2 WARN/1 FAIL/1 SKIP.
`deep_stack_ood_guard` still FAILS but at the lowest all-in argmax confidence ever measured (0.31 vs
V24's 0.46/V24_extreme's 0.35) -- the one open failure left in this whole lineage. Win-rate checks
all PASS and strong (+29.6 to +80.6 BB/100). `action_diversity`'s own strict grid doesn't show a
raise bucket winning outright (unlike V24_extreme's 2-cell win) despite the tighter gap: its
synthetic cells apparently don't land where the gap narrowed most; `stack_full_sweep` DOES show one
`raise_pot` win in the argmax path though. 50k checkpoint backed up to
`versions/v25/weights/frozen_v25_50k_diag.pth`. A longer 100k-hand from-scratch confirmatory run
(user-authorized ceiling) is in progress to check this holds up with more training exposure. Full
detail: `versions/v25/SPECS.md`.

**Suggestion (four-times-revised)**: the structural target-EV fix (V25) looks like the strongest
lever found so far — it reaches an equal-or-better Q-gap than the best opponent-response attempt
(V24_extreme) WITHOUT that attempt's regression. Once the 100k confirmatory run lands: if it holds,
this is a strong deploy candidate pending `deep_stack_ood_guard`'s persistent failure (not yet
resolved by ANY version since V19 — may need its own dedicated investigation, separate from BET-1).
The opponent-response lever (V24's `bot_bluff_perc`/`RAISE_RESPECT_BOOST`) is still live/inherited in
V25's base and may be complementary — a future ablation isolating V25's rollout fix ALONE (reverting
V24's opponent-response tuning) would clarify how much each contributes, if worth the cycle.
**Also tried and failed** (2026-07-17): diversifying the opponent POOL toward real NN opponents
(V20_preflopEq_AI) — a promising early read at 35k hands fully faded by 140k/150k.

**V26/V27 diagnostic work (2026-07-18/19)**: built the FIRST permanent, reproducible Q-gap check
(`allin_vs_nextbest_qgap` in `tools/model_verify/checks.py`) — every prior number quoted above was
measured with a different one-off script each time. Ran against V26/V27 with a proper WORST-CELL
(not averaged) breakdown by stack depth and opponent archetype: confirmed the gap MONOTONICALLY
WORSENS with stack depth (V27: 15bb=+0.35 → 40bb=+0.61, fraction of pot) — backwards from correct
theory. Tested the `RAISE_RESPECT_BOOST` is_allin-asymmetry hypothesis directly (paired probe,
N=2000): real and large for TAG/LAG (fold rate facing all-in vs. an equally-priced raise: -66pts/
-37pts), but the archetype PATTERN didn't match the Q-gap's own pattern (worst at NIT, where the
boost asymmetry is ~zero since NIT is already fold-saturated) — concluded this mechanism is real
but not the primary driver. Investigated a "range narrows on a bigger bet" hypothesis and ruled it
out for training specifically (full-information simulator, oracle equity from real dealt cards, no
hidden range to narrow for a single opponent hand — verified the math is internally consistent).

**Fix 4 — risk-adjusted target, tried and WORKED (V28, 2026-07-19)**: the better-supported lead —
`_mc_target_evs_sized`'s target is a raw point-estimate EV with NO risk/variance penalty, and
because a bigger bet's `raise_size` scales with stack while its outcome variance scales too, the
same marginal equity edge produces a linearly bigger raw EV number at deeper stacks with nothing
counteracting all-in's much higher variance. Confirmed via code research: no variance-aware
mechanism exists anywhere in this codebase. Fix: closed-form `Var[X]` from the SAME three
quantities the EV blend already computes (fold/call-win/call-lose 3-point mixture, `E[X]` verified
to match the existing formula exactly via a 20-trial randomized unit test) — `risk_adjusted_ev =
raw_ev - RISK_AVERSION_COEFFICIENT * sqrt(Var[X])`, applied UNIFORMLY to every sized action, NOT an
`is_allin` special case (all-in's larger `raise_size` gives it naturally larger variance, so the
same coefficient penalizes it most on its own). Calibrated coefficient (0.10) via a standalone
script (averaged over 40 trials/cell to cut sampling noise) BEFORE training, confirming it flips
the worst diagnosed cell from strongly all-in-favoring to clearly disfavoring while barely touching
a legitimate value shove. Trained 100k hands fresh from V27.

**Result (V28): real, validated improvement, but partial**. `allin_vs_nextbest_qgap` shrank ~40-50%
at every cell AND the pathological worsening-with-stack-depth pattern is GONE (roughly flat
15bb-40bb instead of escalating to 0.61 at 40bb). `action_diversity` recovered from a 2-action
collapse to 4 actions (call/raise_33 win cells again); `stack_full_sweep`'s argmax path is now a
coherent call→raise_33 progression with zero all-in wins (was all-allin in V27); `position_sweep`
recovered from V27's WARN back to PASS (0.653, even better than V26's 0.378).
`beats_frozen_predecessor` PASSES (+29.7 BB/100 vs frozen V27). `deep_stack_ood_guard` STILL FAILS
(the one check no version since V19 has cleared) but at meaningfully lower confidence (0.33→0.24).
Full detail: `versions/v28/SPECS.md`.

**Fix 5 — critic-consistency filter + higher risk-aversion, RESOLVED (V29, 2026-07-20)**: calibrated
a candidate ALLIN-specific consistency filter against V28's own real Q-values across
`deep_stack_ood_guard`'s eq × stack grid BEFORE building it (honest finding, not assumed): at
eq=0.43, V28's critic already ranked ALLIN below RAISE_POT while ALLIN still cleared the fold
baseline (the exact spurious-weight case a consistency filter fixes) — but at eq=0.48/0.55 (the
check's OTHER 10 failing cells), ALLIN was the critic's OWN genuine Q-argmax, where a policy-side
filter is a correct no-op. Built BOTH levers together: `regret_match_policy_torch` gained
`critic_consistency_margin` (vetoes ALLIN's regret when any other action's Q beats it by more than
the margin — ALLIN-specific, not an all-pairs rule, since an all-pairs version tested against the
same calibration data collapsed legitimate raise-size mixing in 19-23/25 grid cells), AND
`risk_aversion_coefficient` bumped 0.10→0.15 to push on the eq=0.48/0.55 half the filter can't
reach. Trained 100k hands fresh from V28.

**Result: RESOLVED, and qualitatively different from every prior attempt**. `deep_stack_ood_guard`
PASSES — the first clean pass since V22, after V23-V28 all failed it. `allin_vs_nextbest_qgap`'s
worst-cell gap is NEGATIVE at every stack depth (15bb=-1.00 → 40bb=-1.97) and every archetype
(NIT/TAG/LAG=-0.43/-0.43/-0.44, CALLING_STATION=-0.47) — V28's own worst cells were still POSITIVE;
V29 doesn't just shrink the gap, it flips the sign everywhere tested. `action_diversity` stayed
healthy (fold/call/raise_pot all still win argmax cells), confirming the ALLIN-only filter scope
didn't collapse sizing mixing. The two mechanisms (filter + coefficient bump) weren't tested in
isolation, so their individual contributions aren't separately attributable — see
`versions/v29/SPECS.md` for the full derivation, calibration, and `model_verify --full` breakdown
(21 PASS/2 WARN/0 FAIL/1 SKIP, the cleanest scorecard in this whole lineage). Verified, NOT yet
deployed live (V28 stays active pending user evaluation).

### [BET-3] Multiway passivity — model collapses to call/fold with 3+ opponents ⚠️ REOPENED (entry side, V43, 2026-07-21)

> **REOPENED 2026-07-21 (V43, live).** The **aggression** side stays resolved (see V41 resolution
> below); the **entry** side was never fixed and is measurable on the live model. `equity_edge`
> exists precisely to say "this hand is strong *for this field size*" — and the model ignores it.
>
> AKs preflop, equity computed exactly as training computes it (range-aware vs N Yellow opponents,
> all still-to-act, VPIP fold-roll applied, all-fold samples skipped), `hand_strength` constant at
> 0.661 throughout:
>
> | opp | equity | equity_edge | P(FOLD) | P(raise) | chosen |
> |---|---|---|---|---|---|
> | 1 | 0.670 | 1.34 | 0.000 | **0.826** | RAISE_POT |
> | 2 | 0.610 | 1.83 | 0.006 | 0.723 | RAISE_POT |
> | 3 | 0.600 | 2.40 | 0.008 | 0.674 | CALL |
> | 4 | 0.570 | 2.85 | 0.120 | 0.339 | CALL |
> | 5 | 0.520 | **3.12** | **0.907** | **0.005** | **FOLD** |
>
> `equity_edge` climbs 1.34 → 3.12 exactly as designed while `P(raise)` collapses 0.826 → 0.005.
> **A top-5 starting hand is folded 91% of the time at the exact moment its edge feature peaks**,
> and the cliff sits between 4 and 5 opponents — a full ring.
>
> Threshold sweep confirms the model gates on near-constant ABSOLUTE equity, not edge: the entry
> switch sits at eq* ≈ 0.51 from 2 opponents up (0.393 / 0.492 / 0.509 / 0.516 / 0.514), so edge*
> rises linearly with field size (0.79 / 1.47 / 2.04 / 2.58 / 3.08). **If the model used the edge
> feature, edge\* would be flat.**
>
> **A design flaw in the feature itself, which likely explains why it was never learned.** `equity`
> is measured against the EFFECTIVE contested field (each still-to-act opponent is rolled at their
> VPIP and all-fold samples are skipped, so 5 Yellow opponents = only **1.80** expected contesting
> opponents, fair share 0.357), while `equity_edge = equity × (num_active + 1)` normalizes by the
> NOMINAL field (fair share 0.167). The two halves use different denominators, and the discrepancy
> grows with field size — so the feature is not the clean "equity vs fair share" ratio its docstring
> claims. Against the honest effective-field yardstick the model still tightens 0.79× → 1.44×
> across 1→5 opponents, i.e. the defect is real but ~1.8×, not the ~4× the naive reading suggests.
>
> **Deferred to the next trained version by user decision (2026-07-21)**, since it cannot be fixed
> live. Fix the denominator disagreement FIRST, then retrain. Note this also reframes
> [P4]/`vpip_adapts_to_style`: entry-range behaviour has been measured for many versions against a
> feature that could not do its job.
>
> **LIVE EVIDENCE RE-ATTRIBUTED HERE (2026-07-23, assembler adjudication).** The flagged JJ fold
> of 2026-07-22 (69.5bb, folded 0.92 at eq 0.38) was originally root-caused as a phantom-seat
> count inflation ("4th opponent at a 3-opponent table"). The new live2 assembler's ground-truth
> adjudication of that exact turn OVERTURNS the count claim: `Tid: 18` was the REAL occupant
> Paul6969 (dealt in, limped, bet the flop), and 4 active opponents was the TRUE count at hero's
> decision. The equity input was computed on correct occupancy — so the too-tight fold is this
> entry-side cliff (absolute-eq gate ≈0.51 at 4-5 opponents) compounded by the V42-round-2
> multiway equity depression, NOT a vision bug. The timer-name hotfixes remain valid (timer text
> polluted seat names ~170 times in that one session) but were not what folded JJ.
>
> #### PROPOSED FIX (user's, 2026-07-21) — reuse the effective contested field as the `n` in `n+1`
>
> Closed form, no MC, no added noise, using the same `_COLOR_TO_VPIP` the equity roll already uses:
>
> ```
> E[k | k>=1] = (|front| + sum(p_after)) / (1 - prod(1 - p_after))
> ```
>
> Front opponents are `p = 1`; if any front exists the denominator is 1 (someone is guaranteed in,
> so no conditioning is needed). **Postflop it degenerates to the nominal count** — there is no
> fold-roll postflop — so this is a PREFLOP-ONLY change and postflop semantics do not move.
>
> Validated on AKs (equity computed exactly as training computes it):
>
> | opp | equity | nominal | effective | edge NOW | edge NEW |
> |---|---|---|---|---|---|
> | 1 | 0.660 | 1 | 1.00 | 1.32 | **1.32** |
> | 3 | 0.580 | 3 | 1.37 | 2.32 | **1.37** |
> | 5 | 0.520 | 5 | 1.80 | 3.12 | **1.46** |
>
> A 2.4x field-size swing becomes flat (spread 0.19), and the feature STILL separates hands, which
> is the point of it: AA 1.70-2.16, AKs 1.30-1.49, JTs 0.88-1.01, 94o 0.64-0.67, 72o 0.59-0.62.
> The residual drift on AA is real signal (a monster's share does outgrow fair share as the field
> widens), not noise.
>
> **Compute it at the CALLER, not inside `equity_edge_feature`** — the contract only receives
> `num_active` and cannot know the front/after split, while the simulator already builds
> `front_colors`/`after_colors` immediately before the equity call (`simulator.py` ~L1592) and
> PHPHelp already has `colors_in_pot`/`colors_still_to_act`. Populate it onto `BoardState`, the
> same caller-populated pattern `equity` and `hand_strength` already use. **Keep `ctx[5]
> num_active` NOMINAL** — the model should still know how many players are seated; only the edge
> denominator changes.
>
> **Why this half of the fork and not the other**: dropping the fold-roll from `equity` instead
> would move `ctx[3]`, the most load-bearing feature in an equity-primary architecture, and would
> destroy the conditional-on-contested property that stops 72o and AA both reading ~0.9 (see
> `compute_range_aware_equity`'s own "everyone folded -> SKIP the sample" note). This touches only
> `ctx[35]`, a feature the model demonstrably ignores today, so there is almost nothing to unlearn.
> Still a contract change (ctx[35] semantics move) -> new `contract_version`, new slice, fresh
> weights per [VAL-5].
>
> The live half of this thread WAS fixed and needs no retrain — see Tier 7 of
> `fable-review-resolution-log.md` (`front_colors` awarded on seat position alone; `is_active` not
> monotonic within a hand). Both inflated the field size the model was told about, which pushed it
> further up the very curve measured above.

> **RESOLUTION (2026-07-21, V41 — DEPLOYED LIVE).** `multiway_shortstack_aggression` PASSES:
> 3-way aggression at eq 0.65 is **0.81**, flat from heads-up, where V29 gave ~0.01. Fixed across
> two versions, both driven by the Fable review:
> - **V40** — the root cause. A single check ended the betting round, so check-behind, check-raise,
>   "checked to me", delayed c-bet and BB-option nodes had **never appeared in any training sample**
>   (0 of 849 postflop checks in an instrumented run were followed by anyone acting). CALL was also
>   exempt from the variance penalty and continuation credit every sized raise received — a
>   structural anti-raise tilt that scaled with pot size, i.e. bit hardest exactly here. V40 fixed
>   3 of 6 collapsed cells.
> - **V41** — the simulation-realism package (dead blinds, NN opponents no longer a degraded self,
>   asymmetric stacks, min-raise floor, [OPP-7] tensor boundary) carried the remaining 3 cells.
>
> Do NOT read this as "multiway is solved". Postflop average active players in training is still
> **1.96**, and the eq-0.55 multiway cells still soften (0.81 → ~0.68). The last member of the
> reviewer's [BET-3] bundle is untouched: **every opponent raise is still exactly 0.75 pot**
> (review #6), so hero has never faced an open-jam, overbet or min-raise. That is the natural next
> version. Verify before trusting the resolution — see [VAL-5] and the false "RESOLVED (V22)" label
> this backlog previously carried on [BET-1].

**Original entry (V29, 2026-07-20) follows.**
**First identified**: 2026-07-20, triaging a real live complaint ("v29 is really bad — too tight,
almost not aggressive") against recorded Double-or-Nothing turns
(`history/Double_Or_Nothing_1171441087` & `_1171442571`, normal actor-policy serving temp~0.2-0.5).

**Simple**: Heads-up the model plays a correct, aggressive short-stack range. But as soon as 3+
opponents are in the hand it stops raising almost entirely — it becomes a call/fold machine, folding
clear short-stack jams (88 @5.75bb 3-way → FOLD 0.97; KQs @13.6bb → FOLD; K6s @7bb; JTs @9bb) and
calling instead of shoving. Live DoN play is almost all multiway short stacks, so this is what the
user experiences constantly.

**Technical (root cause ISOLATED, not guessed)**: reproduced the live FOLD by feeding the exact
recorded context through v29 via `build_ctx`/`run_policy` — so it is NOT a live-bridge/scale bug
(feeding the correct 5.75bb reproduces FOLD 1.00 vs recorded 0.97; opponent `is_active` counts match
`num_opponents` everywhere — no folded-seat over-count; V20-class scaling bug RULED OUT). The driver
is `num_active_opp`: holding the 88 @5.75bb spot fixed and sweeping only opponent count —
  - HU (opp=1): eq 0.50 → RAISE_POT 0.44, fold 0.09 (correct aggressive push/fold).
  - 3-way (opp=3): eq 0.50 → fold 0.68 / call 0.32 / **raise 0.00**; even eq **0.90** → call 0.46 /
    allin 0.25 (won't build the pot with the near-nuts multiway).
So the model learned an extreme multiway caution: some tightening is correct multiway, but this is
degenerate (no raising even at 90% equity). Compounded by low multiway range-aware equity and no ICM
([FMT-1]) in a DoN.

**Why model_verify missed it (a real suite blind spot)**: `nuts_aggressive`, `action_diversity`,
and even [VAL-1]'s Nash checks all run at 1-2 opponents with higher equities. NOTHING in the suite
exercises 3+-opponent short-stack aggression — the exact live condition. This is the concrete reason
"live doesn't match verify."

**Priority**: HIGH — real live money in the format actually played, and it SUBSUMES the [VAL-1]
BB-5-6bb lead (a facet of the same short-stack weakness). Outranks [OPP-8].

**Suggestion**: (1) FIRST add a multiway (3-5 opp) short-stack aggression check to model_verify so
it's measurable and regression-guarded (cheap, closes the blind spot). (2) The real fix is a trained
version: curriculum upweighting 3-5-way SHORT-stack spots where jamming for multiway fold equity is
rewarded (the current target/population evidently teaches multiway=call/fold); likely bundles with
[FMT-1] (DoN ICM) and subsumes [VAL-1]'s BB-5-6bb finding + [BET-2]. (3) Interim live mitigation
worth considering: a Nash push/fold override below ~8-10bb (the [VAL-1] solver already produced the
ranges) to bypass the model where it's provably weakest. See [[v30-val1-external-nash-axis]] and
[[ofk-known-shortcomings-backlog]].

**Update 2026-07-20 (Fable review → V40, BUILT NOT TRAINED)**: the 4-area V29 audit produced two
CODE-LEVEL mechanisms for this, both now fixed in `versions/v40` (clone of V29, no contract change):
1. **The betting round ended on any check.** Postflop `highest_bet` starts 0, so the round's
   `all_matched and last_raiser == -1` terminator was true from the street's first instant and only
   the opening seat ever acted — empirically 0 of 849 postflop checks were followed by anyone
   acting, and the BB never once got its limped-pot option. The model had ZERO training samples for
   check-behind, check-raise, "checked to me", delayed c-bet or BB-option nodes. Fixed using the
   already-maintained `acted_this_round`; verified postflop actions/hand 3.13 → 5.16 and BB-option
   decisions 0 → 259/750 hands.
2. **CALL was exempt from both the V28/V29 variance penalty and the V25 continuation credit**, while
   every raise carried both — and the penalty scales with pot size, i.e. it bit hardest in exactly
   the multiway/high-equity spots where the model refuses to raise. Both now applied to CALL (with a
   deliberate no-penalty carve-out at `to_call == 0`, so a free check is never pushed below fold).
Status stays 🔴 OPEN until a V40 training run + `model_verify --full` measures the effect; the
suggestion above (a multiway short-stack aggression check — the suite's blind spot) is still
unbuilt and should land first so the effect is measurable. See `versions/v40/SPECS.md` and
`.agents/skills/OFK/references/fable-review-resolution-log.md`.

### [BET-2] Short-stack polarization — residual flatting 🟡 LIKELY RESOLVED (V29, 2026-07-20 — unconfirmed side effect)
**First identified**: tracked since V15/V16 era as `[P3]`. **Last confirmed failing**: V22
(2026-07-17, 0.35 avg P(call) in shove-or-fold spots) — still present after the [STACK-1]/[STACK-2]
fix (deeper stack curriculum), similar magnitude to V20's own 0.25-0.35 range.

**Simple**: In clear shove-or-fold spots (short stack facing a raise sized near the stack), we
sometimes just call instead of shoving or folding — a real strong player would rarely flat here.

**Technical**: `short_stack_polarization` (WARN-gated, not a hard FAIL). Avg P(call) in a
shove-or-fold equity/stack grid should be near 0. V29 (2026-07-20): **PASS at 0.14** — well down
from V22's 0.35 and V20's 0.25-0.35 range. NOT a targeted fix this version ([BET-1]'s
critic-consistency filter + risk-aversion bump were aimed at the ALLIN side of the same equity/stack
neighborhood, not specifically at short-stack call-flatting) — reads as a plausible side effect of
the same mechanism (a sharper fold-or-shove signal at marginal equity/short stacks could naturally
squeeze out the dominated flat-call option too), but NOT confirmed via a dedicated test. Flagged
YELLOW rather than closed outright: watch this check across future versions before calling it
durably resolved, same discipline [STACK-1] itself needed (one clean pass at V22 didn't hold).

**Suggestion**: If a future version's `short_stack_polarization` regresses, revisit whether it
tracks with [BET-1]'s own mechanism specifically (both live in the same equity/stack neighborhood).

---

## Stack depth

### [STACK-1] Deep-stack OOD trash-jam ⚠️ REOPENED (V48, 2026-07-23 — held V29→V47, regressed under the measured joint curriculum)

> **REOPENED 2026-07-23 (V48, trained not deployed).** After holding PASS for the V29→V47 run
> (exactly the multi-version confirmation this entry demanded), V48 FAILs
> `deep_stack_ood_guard` again: eq=0.55 @ 15bb → ALL-IN argmax at 0.34 confidence. V48's only
> curriculum change is Change 2 — the MEASURED seat×depth joint mix, which concentrates mass
> at 4-6-handed short/mid depths and thins 30-100bb coverage relative to V47's uniform 5-50bb
> band. That mechanism-shape (coverage thinning → guard regression) matches the V22 history
> below: coverage buys or loses this gate even when the V29 target-formula fix stands.
> Candidate fix for V48.1: keep the measured joint mix but floor the deep-band mass (e.g.
> min 10% at 30bb+), re-measure. Note the regression arrived WITH real gains ([W1] opponent
> style 0.105 PASS, nash3 btn 79%) — trade-off, not a broken build.
**First identified**: live incident, V14/V15 era (K9o jammed 20bb into a single limper).
**Present in EVERY version from V14 through V21_auxhead** (V19, V20, V20_preflopEq,
V20_preflopEq_AI, V21, V21_auxhead all FAILed `deep_stack_ood_guard`, at slightly different
specific cells each time). V22 (2026-07-17) PASSED this gate for the first time ever -- but this
turned out to be a FALSE RESOLUTION: **every version since (V23, V24, V24_extreme, V25, V26, V27,
V28) FAILed this same check again**, at gradually improving-but-never-clearing confidence
(V25=lowest all-in argmax confidence ever at the time 0.31, V28=0.33→0.24) -- V22's depth-curriculum
fix reduced the problem's severity but did not durably close it, contradicting this entry's own
"RESOLVED (V22)" label for six versions running. Corrected here rather than left stale. **RE-
RESOLVED**: V29 (2026-07-20, [BET-1] Fix 5 -- critic-consistency filter + risk_aversion_coefficient
0.10→0.15) PASSED cleanly again, this time alongside `allin_vs_nextbest_qgap` flipping fully
negative at every cell (not just a narrower gap) -- a categorically stronger signal than V22's own
pass had. See [BET-1] for the full mechanism.

**Regression watch, upgraded to a HARD requirement given the V22 false-resolution history**: do NOT
mark this RESOLVED again on a single clean pass alone -- re-check on every future version for AT
LEAST 2-3 consecutive versions before trusting a "RESOLVED" label here, and if it regresses, update
this entry's status immediately rather than leaving a stale claim (as happened V23-V28).

**Simple**: At medium-deep stacks (15-40bb) facing a modest bet with a marginal hand (~45-55%
equity), we used to sometimes jam all-in when we should just call or fold. V22's deeper stack
curriculum (see [STACK-2]) narrowed this but did NOT durably fix it -- the real, lasting fix came
from V29's [BET-1] mechanism (a training-target change, not a training-DATA-coverage change).

**Technical**: `deep_stack_ood_guard` regression check. V22's ORIGINAL hypothesis (in hindsight,
INCOMPLETE): the model had never actually trained beyond 50bb (see [STACK-2]) and the live-serve
clamp created a hard extrapolation boundary right around the failure zone. V22 raised
`STACK_CEIL_BB` 50->100 and gave `stack_depth_mix` real training density up to 100bb (an
overlapping tail band, not a hard cutoff) -- training-DATA-coverage exposure alone bought one clean
pass, then the check failed again for six straight versions, showing coverage wasn't the whole
story. V29's fix instead targets the training TARGET FORMULA itself (risk-unaware raw EV +
policy/critic disagreement, see [BET-1]) -- a mechanism V22 never touched. The originally-suspected
shared mechanism with [BET-1] (shoving reading as "free"
value against a price-insensitive population) turned out NOT to be the (or at least not the only)
cause -- [BET-1] itself remains fully unresolved in V22 (`action_diversity` still shows no
raise_33/66/pot argmax anywhere), yet [STACK-1] cleared anyway. Keep this in mind if [BET-1] work
starts: the two issues are less coupled than assumed.

**Regression watch**: re-check this gate on every future version -- it's cleared now, but the
mechanism (deep-stack curriculum) is new and unproven over multiple versions yet.

### [STACK-2] Beyond-50bb extrapolation 🟢 RESOLVED (V22)
**First identified**: V20 pre-deployment smoke test (a flopped set of aces showed fold%
climbing 10%→32% from 45bb→150bb, a real OOD extrapolation artifact).
**RESOLVED**: V22 (2026-07-17) -- `STACK_CEIL_BB`/`POT_CEIL_BB`/`CALL_CEIL_BB` raised 50/100/50 ->
100/200/100 and `stack_depth_mix` widened to `[5-14bb:0.40, 14-30bb:0.30, 30-60bb:0.20,
10-100bb:0.10]` (the last band deliberately overlapping, not a disjoint bucket, to avoid a
training-density cliff at the seam). Training now genuinely covers up to 100bb instead of clamping
at 50bb. Also fixed [STACK-1] as a side effect (see above).

**Simple**: We used to never train beyond 50bb effective stacks; a live clamp kept things safe at
deeper tables but the model had zero deep-stack skill. Now trains with real exposure up to 100bb.

**Residual scope**: 100-150bb+ tables still extrapolate past the new ceiling (the live-serve clamp
logic still applies beyond 100bb) -- fully resolved for the trained 5-100bb range, not for
truly-deep tournament-style stacks past that. Extend further in a future version if that becomes a
priority.

### [STACK-3] 6-max short-stack open-jam under-aggression — actor folds commits its own critic prefers 🔴 OPEN (live observation, not yet quantified)

**V47 note (2026-07-22)**: the V47 gate wanted this probed; it remains UNQUANTIFIED (the
battery has no actor-vs-critic divergence check and no dedicated probe was run). Indirect
signals from the V47 battery point the RIGHT way: first literal-jam commits since V29 in the
Nash sweep (74/971 vs V44's 0/971, the [M9] collapse signature), all-in argmax cells in
`action_diversity`, and `allin_vs_nextbest_qgap` all-negative. Build the 7–14bb first-in
actor-vs-critic probe before V48 judges this item.
**First identified**: 2026-07-20, live V29 (Herocules) Double-or-Nothing board
`history/Double_Or_Nothing_1171442571` — user flagged the stack bleeding
74→48→27→13→6→5.2bb straight into a bust-bound final jam, over 45 recorded decisions.

**Simple**: At short 6-max stacks (~7–14bb) we fold a lot of hands standard push/fold play would
open-jam (KQs, K8s, JTs closing the BB for 0.5bb, K6s), and we only start actually jamming at ~5bb
— when we're already too short to have real fold equity. We blind down and die instead of jamming
to pick up the blinds/antes.

**Technical**: across the board, in **6 of the 8** preflop spots where V29's OWN critic Q ranks a
non-fold action above FOLD, the actor policy folded anyway — and in those spots the raw policy puts
ZERO mass on any RAISE bucket. Examples: turn 23 KQs@13.6bb 3-way, policy `{FOLD 0.56, CALL 0.44,
all raises 0.00}` (critic prefers a raise); turn 40 K8s@8.2bb CO `{FOLD 1.00}`; turn 41 K6s@7.1bb
HU `{FOLD 0.93}`; turn 37 JTs@9bb BB facing 0.5bb `{FOLD 0.54, CALL 0.45}`. The actor never
open-JAMS in the 7–14bb band at all — it only ever chooses between fold and flat-call — and the
first genuine jam in the whole session is at 5.2bb (turn 45, Ah8s, `RAISE_POT → slider 1.00`,
which the critic AND actor both prefer). `to_call` in the folded spots is ~0.5–1.0bb (limp/BB
completion spots), NOT a big raise being folded to, and Layer-1/Layer-2 (OCR/decoded input) are not
implicated — this is an actor/critic policy-EXTRACTION divergence in the short-stack region: the
chip-EV critic is right, the actor ignores it.

**Relationship to existing entries (logged as DISTINCT, not a duplicate)**:
- Distinct from **[FMT-1]** (ICM/myopia): there the critic itself lacks future-cost awareness; here
  the chip-EV critic already prefers the commit and the ACTOR diverges from it. Same DoN board
  family, different layer.
- Distinct from **[VAL-1] Finding (A)** (BB too tight CALLING jams at 5–6bb HU, range-conditioned):
  this is FIRST-IN / open-jam aggression at 7–14bb, 6-max, often multiway — a regime the HU Nash
  push/fold checks don't cover.
- OPPOSITE direction to **[BET-1]** (deep-stack ALLIN over-preference). **Primary hypothesis worth
  checking**: V29's [BET-1] Fix 5 (ALLIN-specific critic-consistency filter that vetoes ALLIN's
  regret when another action's Q beats it, + risk_aversion 0.10→0.15) is an anti-jam mechanism — it
  may have OVERCORRECTED into the short-stack band, suppressing legitimate open-jams the critic
  still wants. If so, the same lever that RESOLVED [BET-1]/[STACK-1] created this. Not yet tested.

**Suggestion**: Confirm systematicity cheaply BEFORE any retrain — this is single-board evidence and
the per-spot critic-Q margins may be small. Re-inspect the actor-vs-critic gap directly in the
7–14bb first-in band via the EXISTING checks (`short_stack_polarization`, `stack_full_sweep`, and
the already-built Nash `nash_pushfold_vs_chart`/`nash_bbcall_vs_jam` from [VAL-1] — **Nash push/fold
was already run this session, do NOT re-run it**): does the actor assign ~0 to raise/allin buckets
while the critic's best raise/allin Q beats fold? If confirmed a V29-filter overcorrection, scope
the critic-consistency veto so it can NEVER suppress an ALLIN when the only alternative that beats it
is FOLD (only veto when a SIZED raise or CALL dominates). If instead it's curriculum coverage, fold
into the same short-stack/ICM pass [VAL-1] Finding (A) already points to (upweight `stack_depth_mix`
density in the 5–14bb first-in / multiway region). Quantify before committing a version.

---

## Opponent modeling

### [OPP-1] Overfitting-to-deterministic-training-formula risk 🟡 PARTIAL
**First identified**: discussion 2026-07-15, motivated `check_beats_offformula_stress`.
**Last confirmed**: V20_preflopEq_AI 150k — PASSES (+34.7/+65.0 BB/100 short/deep vs
`TieredLookupBot`), but [BET-1] shows the model HAS learned something population-specific
(the shove-preference), so this isn't fully clean.

**Simple**: We've mostly practiced against predictable, formula-driven opponents. Overall winrate
against a structurally different opponent still holds up, but at least one specific behavior
(shove-preference, see [BET-1]) is a direct product of that specific training population, not
general poker skill.

**Technical**: `FuzzyPlayerArchetype`'s fold/continue decision is a deterministic threshold given
equity+price (only the raise-vs-call split among continuing hands has randomness). The
off-formula stress test (`TieredLookupBot`, a price-insensitive-by-street lookup table) is itself
still a fairly mechanical opponent shape — it doesn't prove robustness against a genuinely
human-like or solver-like opponent.

**Suggestion**: [BET-1]'s fix (price-sensitive value branch) is the direct lever. Separately,
consider a genuinely different opponent archetype family for stress-testing beyond
`TieredLookupBot`.

### [OPP-2] No per-opponent action attribution 🟢 RESOLVED, TRAINING + LIVE (V29, 2026-07-20)
**First identified**: V16 ROADMAP `[P6]`. **Status prior to V29**: unchanged since — no
architecture work done.

**Simple**: We didn't used to track who specifically did what during a hand — only "someone
raised," not which particular opponent (with their own known tendencies) did it. Fixed for
TRAINING at first pass; the live wiring (`core/table_state.py`) was completed the same day per
explicit user request ("deploy to live and make sure the live boardstate can provide all
informations needed").

**Technical**: Corrected framing found while implementing the fix -- there was never actually a
"coarse action-history token stream" in the current architecture (that description predates a
prior redesign); the sequence axis is hero's own successive decision points this hand, each with a
static per-timestep context snapshot. The real gap was narrower but still real: a specific
opponent seat's only signal was its STATIC, cross-hand VPIP/AGG HUD color plus a hand-level
`pot_type` aggregate ("someone raised, 3-bet+") -- no way to attribute in-hand aggression to one
specific seat. **Fix (V29)**: two new per-seat boolean context features, `raised_this_hand[seat]` /
`raised_this_street[seat]` (context_dim 44->54, contract_version 7->8), threaded through
`versions/v29/self_play/simulator.py`'s betting loop (both hero's and every opponent bot's raise
branches), `core/board_state.py`'s shared `SeatState` (additive, inert for every earlier version's
contract), `versions/v29/core/contract.py`, and `train.py`'s separate `vectorize_hand_samples`
context builder (kept in lockstep by hand -- this is the exact duplication that let V20's own
rescale drift, watched carefully here). Verified via unit tests plus real-simulated-hand
integration smoke tests (including the real TreeOpponent/lagged-self opponent pool) before
committing to the full retrain.

**Live fix (2026-07-20)**: `core/table_state.py`'s `_generate_timeline_actions` gained real per-seat
raise/call classification -- a stack-drop diff is only a RAISE if it exceeds the bet level that
specific seat actually faced (`street_bet_before`, captured once per tick, updated locally so a
rare multi-drop frame doesn't misattribute), not "any drop." Preflop is seeded to the big blind
(not 0) so a plain limp isn't misread as a raise -- previously `current_street_bet_level` had no
concept of the BB's fixed opening price. `raised_this_street` resets every street change (mirroring
`current_street_bet_level`'s own reset); `raised_this_hand` persists the whole hand. Threaded into
`to_board_state()`'s `SeatState`s.

**Also fixed as a direct byproduct**: the EXISTING `committed`/`hero_committed`/`pot_type` context
features (V22/V23) had ALSO been silently inert (always 0) in live serving for every version since
V22 -- `to_board_state()` never set them. Now sourced from real tracked state: `committed`/
`hero_committed` = each player's start-of-hand stack minus their current stack (lazily seeded the
first time a stack is observed each hand); `pot_type` = a live whole-hand raise-EVENT counter
(`self.raise_count`), bucketed 0/1/2+ exactly like the simulator's own `raise_count`.

**Also fixed, a second-order byproduct**: neither hero's nor an opponent's stack correctly updated
to a genuine 0 (all-in) -- both had a monotonic-decay guard (`if stack > 0:`) that (correctly)
rejects a bare 0 as a likely failed OCR read, but had no way to distinguish that from a REAL all-in.
`core/vision.py` already emitted a reliable `state='All-In'` signal for OPPONENTS via an explicit
'ALL'/'IN' text match (checked before digit-cleaning mangles the letters into garbage digits) that
`table_state.py` simply wasn't using -- now it is. HERO's own OCR region had no equivalent signal at
all (just bare digit parsing) -- added the same 'ALL'/'IN' text-match to vision.py's hero-stack
extraction (`hero_all_in` flag), mirroring the already-proven opponent pattern exactly. Verified
both the fix (all-in correctly reaches 0) and the regression case (an unconfirmed bare-0 misread
still correctly gets rejected as noise, preserving the original protection) via direct tests.

**Verification**: unit tests (raise-vs-call classification including a limp, a preflop raise, a
flop bet-then-call, an exact-chip all-in call) plus a full end-to-end test through
`core.decision.PokerDecisionEngine.make_decision` with V29 active, confirming no crash and correct
context values reaching the model. Not tested against real screen captures/OCR (no live table
available this session) -- the vision.py text-matching logic is a direct structural mirror of the
opponent-side pattern already running in production, not a new heuristic, but real-table
confirmation is still worth doing when next played live.

**Suggestion**: none outstanding for the core mechanism. Worth keeping an eye on `current_street_bet_level`'s
epsilon tolerance (1% of BB, floor 0.01) across different stake levels/currencies if this is ever
used at very small or very large blinds.

### [OPP-3] Size-blind action history 🔴 OPEN
**First identified**: V16 ROADMAP `[P5]` (generalized from an original call_amount-only framing).
**Status**: the CURRENT bet's size is fed via context features (and was rescaled in V20), but the
HISTORY of past bet sizes within a hand still isn't.

**Simple**: We don't fully track how big previous bets were earlier in the same hand — the action
history is somewhat size-blind, so unusual historical sizing from an opponent may not register.

**Technical**: `act_ints`/history tokens use a coarse vocabulary (fold→7, call→3, any-raise→6
regardless of size) — see `ContractV12.to_tensors`'s action sequence construction. The width was
never widened to carry size info alongside action type.

**Suggestion**: Same category as [OPP-2] — a sequence-encoding change, not addressed by any
contract iteration so far (V13→V20_preflopEq_AI all share this gap). **See [OPP-10]**: size-blindness
is one facet of a bigger gap — the sequence carries only hero's own decision points, so opponents'
actions aren't sequence events at all. If [OPP-10] is ever built, fixing this comes with it.

### [OPP-10] Training sequence is hero's decision points only — no full hand history 🔴 OPEN (idea, user-raised 2026-07-21)
**First identified**: 2026-07-21, user question — "is the full hand history fed into the hero, so it
has the whole record of what the preflop equity was and so on?" Investigated against V41's code
rather than answered from memory; the answer is *partly*, and the gap is worth tracking.

**Simple**: The model gets a step for every point where HERO had to act, and each step carries a
full snapshot (equity, board, pot, stack, position, opponent block). So yes — the preflop equity is
still visible at the river, because the preflop step is still in the sequence. What it does NOT get
is a record of what everyone ELSE did between those steps. Villain's whole line gets flattened into
a few flags by the time hero acts again.

**Technical**: The sequence is up to `max_seq_len=20` steps, left-padded, reset per hand, built from
`record.decision_points` (`vectorize_hand_samples`, train.py) on the gradient path and from
`model_state_histories[seat]` (`_query_model_decide` → `ContractV12.to_tensors`) on the rollout/live
path. One step == one hero decision. Per step the model sees its own 54-feature context and its own
previous action token (fold→7/call→3/raise→6, shifted by one inside `PokerEVModelV4.forward`); hole
cards are a single embedding broadcast across all steps. Consequences:
- **Opponent actions are not sequence events.** They survive only as derived per-step features:
  `pot_type` (limped/single-raised/3-bet+, a hand-level bucket), [OPP-2]'s per-seat
  `raised_this_hand`/`raised_this_street`, and `committed`. "Villain bet, I raised, villain 3-bet"
  and "villain raised once" can look identical at hero's next decision point.
- **ORDER and COUNT between hero's turns are lost** — three bets and one bet both set the same flag;
  who acted first is not recoverable.
- **Nodes where hero doesn't act contribute nothing.** (Directly related: before V40's betting-round
  fix, post-check nodes did not exist in training data AT ALL — see [BET-3].)
- Compounding gaps already tracked separately: [OPP-3] (raise size absent from the action token) and
  [OPP-6] (no adversarial exploiter in the pool).

**Suggestion**: Consider building a genuine full-hand action history during training — a token
stream of EVERY seat's actions in real order (actor seat, action type, size bucket, street), not
just hero's decision snapshots. That is a sequence-encoding + contract change (same class as
[OPP-2]/[OPP-3], bigger than either), so it wants its own version and a careful look at
`max_seq_len` — a 6-handed hand can easily exceed 20 total actions where it rarely exceeds 20 HERO
decisions, so the current window would need to grow or the truncation would start discarding
preflop. Worth weighing against [OPP-6] as the higher-leverage opponent-modelling investment: this
one gives the model the raw material to read a line, that one gives it something worth reading.
Note the live path can supply this — `core/table_state.py` already tracks per-seat actions for
[OPP-2]/`pot_type` — so a train/serve-consistent version is feasible, but the live bridge would have
to be extended in lockstep (the exact failure mode [OPP-7] hit).

### [OPP-4] Live front/after equity — reopened-action blindness 🔴 OPEN (live-only)
**First identified**: 2026-07-17, while wiring V20_preflopEq's Finding 2 fix into live serving.

**Simple**: If an opponent calls, then someone else raises, we might still treat that first caller
as "definitely staying in" even though they haven't actually faced the new raise yet and could
still fold to it.

**Technical**: `PHPHelp.py`'s `_classify_opponents_by_action_order` is a pure TABLE-POSITION
heuristic (button-relative rotation order) — it has no real per-seat action-state to check whether
a subsequent raise reopened action for an earlier caller. Training's own simulator tracks this
correctly (`acted_this_round`, explicitly reset on every raise) because it has ground-truth access
to the full betting sequence; the live path only has vision-derived table state, not true
per-seat action history. This was a known, documented limitation of the classifier when it was
DISPLAY-only; it's now load-bearing (feeds the actual live equity computation for V20_preflopEq/
V20_preflopEq_AI), so the gap matters more than it used to.

**Suggestion**: Would need live per-seat action-state tracking (not just position) mirroring
training's exact mechanism. Scoped conceptually, not started — the live pipeline currently has no
per-seat "have they acted since the last raise" tracking at all.

### [OPP-5] Opponent-style/VPIP-AGG-color read may not be load-bearing 🟡 PARTIAL (resolved in V24_extreme/V25, per the Suggestion below)
**First identified**: `model_verify`'s `opponent_style_sweep` check, added 2026-07-15 (first
version to carry it). **Confirmed**: V20_preflopEq_AI (spread 0.000, completely flat) and V21
(spread 0.001, then re-confirmed at 0.004 after widening the sweep from 3 to 5 equity points to
cover the actual fold/continue transition zone — ruling out "the two saturated endpoints just
happened to hide a real mid-curve difference"). **Update (2026-07-18)**: the Suggestion below
(fix alongside [BET-1]) played out as predicted — V24_extreme's opponent-response push produced
the first-ever PASS (spread 0.151), and V25's structural multi-street fix (realistic settings, no
extreme parameters) ALSO passes (spread 0.109). Confirms this was genuinely a training-population
flatness issue, not dead wiring, and both [BET-1] fix families move it. See [OPP-8] below, though,
for a narrower, still-open finding about how COARSE this same signal is even once load-bearing.

**Simple**: Facing the identical bet at the identical hand strength, we play essentially the same
way against a tight, disciplined opponent (a "nit") as against a loose, aggressive one (a
"maniac") — the model doesn't seem to actually use its read on who it's up against.

**Technical**: `check_opponent_style_sweep` holds equity/stack/pot/call fixed and only varies the
fed-in opponent VPIP/AGG archetype (Blue=nit through Red=maniac). Every version tested shows
P(fold) essentially flat across all four archetypes (spread ≤0.004), despite the model receiving
per-opponent VPIP-color/AGG-color context features since early versions. This is a DIFFERENT gap
from [OPP-2]/[OPP-3] (those are about the in-hand ACTION sequence not being per-opponent-attributed
or size-aware) — this is about whether the per-seat HUD-style CONTEXT features (present since early
versions, no architecture change needed to use them) are actually load-bearing at all, and the
answer looks like no.

**Root cause narrowed (2026-07-17)**: added `check_opponent_color_isolated_ablation` (FAST,
`tools/model_verify/checks.py`) — pushes VPIP/AGG to synthetic extremes (0.0 vs 1.0, well past any
realistic archetype) and tests the table-level scalar (ctx[7]/ctx[8]) and the per-seat block
separately. Run against V21_auxhead: table-scalar TV=0.026, per-seat-block TV=0.077 (comparable to
`hand_strength_sensitivity`'s own 0.117) — the network CAN read the per-seat VPIP/AGG input, and
does respond once pushed far enough. Since `opponent_style_sweep`'s realistic archetype range
(0.10-0.85) shows no response but this synthetic extreme does, this is a **training-population
artifact, not dead wiring**: the deterministic heuristic-bot population apparently never
differentiates outcomes by opponent style enough, within realistic bounds, for the network to
learn to condition on it there — the same population-level-flatness mechanism already suspected
for [BET-1].

**Suggestion**: Since wiring is confirmed intact, the fix lever moves to the training population
side — likely the same lever as [BET-1] (making the heuristic bots' response actually vary with
their own declared style more sharply, or ensuring the training distribution samples a wide enough
style spread that within-realistic-range differences carry real EV consequences). Not yet
attempted; worth testing together with [BET-1]'s opponent-price-sensitivity fix rather than as a
fully separate change, since both may share the same root population issue.

### [OPP-6] No adversarial exploiter opponent — pool lacks a best-response hunter 🔴 OPEN (idea,
not yet built)
**First identified**: conceptual discussion 2026-07-17, comparing self-play-only training against
CFR-style equilibrium approaches (see that session's CFR time-cost estimates if this is revisited:
full tabular/Cepheus-style solving is off the table for this project's hardware; a Deep-CFR-style
build was ballparked at multi-day-to-multi-week dedicated infra work, separate from the current
actor-critic pipeline).

**Simple**: Every opponent the hero currently trains against is either a fixed heuristic script or
a "relative" (its own past self / an earlier checkpoint of the same training lineage). Nothing in
the pool is actively hunting the hero's CURRENT weaknesses, so subtle exploitable habits (unbalanced
bluff frequency, thin-value gaps, missed indifference points — the "smaller intricate details" of
the game) can persist even while aggregate winrate against the existing pool looks fine.

**Technical**: Current pool (`versions/v20_preflopEq_AI/self_play/config.yaml`) is 60% NN (25%
lagged-self + 20%/15% frozen same-lineage checkpoints) / 40% heuristic (`tag`/`nit`). All NN
opponents share ancestry with the hero being trained — self-play-with-relatives — which is exactly
the setup known in imperfect-info game literature to plateau or cycle rather than converge toward a
hard-to-exploit strategy. This is the same underlying gap [OPP-1] already flags (overfitting to a
deterministic training population) and likely shares mechanism with [BET-1] (heuristic bots' value
branch isn't price-sensitive, so overbet/shove patterns go unpunished by anything in the pool).

**Suggestion**: Add a dedicated "exploiter" opponent to the pool — a bot trained specifically to
maximize exploitation of the CURRENT frozen hero checkpoint (an approximate best-response), and
refreshed periodically as the hero improves, rather than another lagged-self mirror. This is a
lighter-weight, NFSP-flavored middle ground: no tree-traversal/regret-accumulation infra needed, it
slots into the existing V18 `Opponent` interface as one more pool entry. Real cost: training the
exploiter is a full training run of its own (comparable to any other NN opponent-pool member), plus
ongoing re-training to stay a meaningful adversary as the hero moves — an ongoing cost, not a
one-off. Not yet scoped against the actual pipeline.

### [OPP-7] NN-opponent self-play queries are self-referential and hero-blind 🟢 FIXED (V27 in the dict, V41 at the tensor boundary — see the STATUS CORRECTION below)
**First identified**: 2026-07-17, while reviewing V22's training dashboard (Lagged-Self (NN)'s
-31.7 BB/100 prompted a closer look at how NN opponents perceive the table). Present since V18's
opponent refactor introduced NN opponents (lagged-self mirrors, frozen checkpoints) -- not
introduced by V22, though V22's new `opp_committed_this_hand_bb` feature (see [OPP-2]/[OPP-3]
history) inherited the same flaw by reusing the existing loop.

**Simple**: When a non-hero NN opponent (e.g. the lagged self-play mirror) makes its OWN decision
during training, it ends up seeing a slightly wrong picture of the table: it lists ITSELF as one
of its own opponents, and never sees the real hero as an opponent at all. This doesn't affect
hero's own live decisions (those are unaffected), only how realistic/well-calibrated the NN
opponents hero trains against actually are.

**Technical**: `simulator.py`'s `_query_model_decide` builds each query's opponent-seat block with
a fixed loop (`for idx in range(5): seat_key = f"seat_{idx+1}"`, `is_active = idx < num_opponents`),
always using ABSOLUTE table seats 1-5, and `opponents_profiles` (per-seat VPIP/AGG) is keyed the
same way, built once per hand and reused for every actor's query. This is correct ONLY when hero
(seat 0) is the querying actor -- hero's real opponents genuinely are seats 1-5. For any other
actor (e.g. Lagged-Self at seat 4), with `num_opponents` correctly counting live players excluding
self (so e.g. `num_opponents=5` at a full 6-max table), the loop marks ALL of seat_1..seat_5
active -- including seat_4 itself (a phantom self-as-opponent entry, using its own VPIP/AGG/
committed values) -- while seat_0 (real hero) is structurally never representable, since the loop
range is fixed to 1-5. Every per-seat context field built inside this function (VPIP/AGG color,
and now `committed`) inherits this same self-referential/hero-blind construction for non-hero
queries.

**STATUS CORRECTION 2026-07-21 (Fable review finding #11) — this was NOT actually fixed by V27.**
V27's remap was right in the `board_state` dict but **defeated at the tensor boundary**: it keyed
each slot by the ABSOLUTE seat number (`seat_{seat_id}`) while `ContractV12.to_tensors` reads only
`seat_1..seat_5`. For any non-hero actor `other_seats` contains 0, so the real hero was written to a
`seat_0` key the encoder never reads — hero stayed structurally invisible to every non-hero NN
query, i.e. the exact symptom this entry describes — AND the surviving slots were misaligned (for
`actor_seat=4` the code wrote seat_0/1/2/3/5, so the encoder's 4th slot looked up a missing `seat_4`
and fell back to an inactive default). V27's verification checked the dict, not what survived
encoding — the same "verified the wrong object" failure mode as the `beats_frozen_predecessor` bug
(review #4). Measured 2026-07-21: **V40 dropped the hero on 128 of 128 NN-opponent queries; V41
drops zero.**

Genuinely fixed in `versions/v41` by keying each slot by SLOT INDEX (`seat_{idx+1}`), which is what
the encoder addresses, with the real seat number retained in `name` and `opponents_profiles` looked
up by the absolute key. For `actor_seat == 0` the two indices coincide, so hero's own query is
byte-identical. Fixed alongside it (review #10, same block): `is_active` was `idx < num_opponents`
(marking the first N SLOTS rather than the seats actually live) and every opponent's `stack` was a
`hero_stack` placeholder — both now read ground truth threaded through `table_state`. See
`versions/v41/SPECS.md` and `.agents/skills/OFK/references/fable-review-resolution-log.md`.

**Lesson worth keeping**: a fix to a data structure that feeds an encoder is not verified until you
check what the ENCODER sees. Both this and review #4 passed their original verification.

**Suggestion**: Fix requires a genuine seat-relative remap inside `_query_model_decide` -- compute
the ACTUAL other-live-seats list relative to whichever seat is querying (excluding itself,
including hero when relevant) instead of assuming "opponents are always seats 1-5". Nontrivial
(touches shared simulator code used by every version's self-play), and any fix should be validated
via a fresh training run before trusting the resulting NN-opponent behavior change -- not a
same-run hotfix. Likely low severity in practice (opponent VPIP/AGG color reads are already only
weakly load-bearing per [OPP-5], so a self-referential value there may add noise more than it
changes behavior), but worth confirming rather than assuming.

**Fixed (V27, 2026-07-19)**: exactly the suggested remap -- `other_seats = [s for s in range(6) if
s != actor_seat]`, indexed by slot instead of the old hardcoded `idx+1`. Verified two ways before
any real training: (1) a synthetic `actor_seat=0` query reproduces the OLD seats-1-5 ordering and
values byte-for-byte (hero's own path provably unchanged); (2) a synthetic `actor_seat=4` query
confirms no self-referential "seat 4" entry exists and the real hero ("seat 0") now appears, with a
live VPIP/AGG read computed the same `acts/ops` way every other seat's profile already is.

**Real-world impact CONFIRMED, and the Suggestion's own "likely low severity... but worth
confirming rather than assuming" caveat was WRONG** -- this was not low severity. The 100k-hand
V27 run (which ALSO carries the unrelated [VAL-3] fix, so the two aren't cleanly isolated from each
other) shows `opponent_style_sweep` genuinely improved (0.041 -> 0.165, Lagged-Self now seeing a
real hero plausibly enriched the training population) but ALSO a cluster of real regressions
alongside it: VPIP roughly doubled (~16-26% -> ~40-48%), `action_diversity` narrowed (3 actions ->
2), `stack_full_sweep`'s argmax flipped from all-call to all-allin across the full stack range,
`position_sweep` newly WARNs (spread 0.378 -> 0.022, position barely matters anymore), and
[BET-1]'s own `allin_vs_nextbest_qgap` got WORSE at every stack depth and archetype. See
`versions/v27/SPECS.md` "Results" for the full number-by-number comparison. Net verdict: `
beats_frozen_predecessor` still PASSES (+40.2 BB/100 vs frozen V26) but the win looks like "wins via
higher aggression," not "wins via better play" -- the same pattern V24_extreme showed. Per explicit
user decision (2026-07-19): both [VAL-3] and this fix were verified correct in isolation via direct
unit tests before training (not guesses), so V27 -- not V26 -- remains the base going forward
despite this regression cluster; WHICH of the two fixes (or plain run-to-run variance) actually
caused the broader shift has not been isolated -- open question, not resolved.

### [OPP-8] Opponent fold-tendency signal is a coarse 4-way color, not the real underlying trait 🔴 OPEN (severity likely overstated below — see 2026-07-18 correction)

**V47 note (2026-07-22) — WATCH, moved the wrong way**: `opponent_style_sweep` fold-spread
collapsed 0.127 (V44) → 0.027 (V47) and the isolated ablation's table-scalar TV fell
0.750 → 0.019 (per-seat block TV 0.307 — wiring alive, response flat at realistic values).
Working hypothesis: V47's occupant-true fold models ([M4]) made training-time fold behavior
depend on what is ACTUALLY seated rather than archetype labels, homogenizing the archetype
signal the sweep perturbs — a realism↔exploitability tension. Also relevant: the new
hand-history opponent DB (tools/handhistory/, 2026-07-22) can supply REAL per-account
VPIP/AF/fold-to-pressure — the long-term fix for this item is exact stats in, not sharper
color response.
**First identified**: 2026-07-18, investigating why V25's own trained policy barely differentiates
all-in frequency by opponent archetype (0.239 vs NIT, 0.216 vs CALLING_STATION at the SAME
eq=0.55) despite the opponent bots' actual fold-to-a-shove behavior being wildly different (NIT
~98% fold, CALLING_STATION ~0% fold at the identical price, per a direct probe against
`_ev_target_fold_decision`) — user's own observation ("why hasn't hero learned to always shove
low-medium equity vs NIT if it folds 98% of the time") prompted the investigation.

**Important correction (2026-07-18, user's own follow-up reasoning)**: the "hero should have
learned to always shove low-medium equity vs NIT" framing overstates the real exploit. The 98%
fold-rate probe FIXED the opponent's equity at a flat 0.45 and asked "given NIT genuinely has 45%
equity here, does it fold" — an if-then answer, not a "how often does this actually happen"
answer. Since NIT only voluntarily plays premium hands (VPIP 0.11), by the time it reaches a
postflop decision its range is already selection-skewed strong — a genuine 45%-equity NIT spot is
RARE in real simulated hands, not the typical case the synthetic sweep implied. The real training
target (`_mc_target_evs_sized`'s postflop `oracle_equity`) already partially reflects this, since it
uses the LITERAL dealt cards of opponents who are actually still active in that hand — a real
selection effect the standalone diagnostic scripts (built to isolate the fold-decision mechanic in
controlled isolation) don't capture. Net effect: hero's relatively FLAT all-in frequency across
archetypes (the original finding below) may be closer to CORRECT, sophisticated behavior than a
gap — don't over-read the raw fold-rate numbers as "free money being left on the table" without
weighting by how often that equity level actually arises against each archetype's real range.

**Simple**: Hero can tell a tight opponent from a loose one, but has no direct way to know exactly
HOW likely that specific opponent is to fold to a big bet — it's reading a rough category (tight/
loose), not the actual "folds under pressure" dial, so it can't fully exploit an extreme folder the
way the training bots' own logic would justify.

**Technical**: each `FuzzyPlayerArchetype` has an independent `base_fold_to_pressure` trait
(TAG=0.60, LAG=0.45, NIT=0.85, CALLING_STATION=0.15 — see `opponent_bots.py`) that's the actual
driver of `continue_bar`'s `style_shift` and therefore of the huge fold-rate gap above. This trait
is NEVER exposed as a model input feature, in any form. The ONLY opponent-identity signal in
`ContractV12`'s per-seat block is `vpip_color`/`agg_color` (`core/contract.py` `VPIP_MAP`/
`AGG_MAP`), a 4-bucket categorical read derived purely from `current_vpip` (`simulator.py`'s
`_vpip_to_color`, thresholds at 0.18/0.26/0.35). In THIS specific 4-archetype roster, VPIP happens
to be roughly ordinally consistent with `fold_to_pressure` (NIT is both tightest AND most
fold-prone) — so the color bucket is a real, usable, directionally-correct proxy (which is why
[OPP-5]'s `opponent_style_sweep` now passes, see above) — but it's still a coarse 4-way label
standing in for a continuous, independently-fuzzed (`random.gauss(base, 0.05)` every hand) trait,
so the network can infer the DIRECTION of an exploit but has much less signal on its true
MAGNITUDE than the underlying simulator population actually has. Compounding factors, not yet
separated out: NIT is only 15% of the opponent pool's sampling weight, and 6-max hands usually have
multiple simultaneously-active opponents, diluting "this one seat is foldy" against the joint
all-fold probability the training target already computes across every active seat.

**Sharper framing (2026-07-18, user's own follow-up)**: the VPIP/AGG-color correlation with real
fold-to-pressure is a COINCIDENCE of this specific 4-archetype roster's design (NIT happens to be
both tightest AND most fold-prone; CALLING_STATION happens to be both loosest AND least fold-prone)
— VPIP measures preflop entry frequency and AGG measures general aggression, neither of which
*means* "folds to a big bet" on its own. If a future archetype ever decouples those (a tight
player who's secretly a calling station once in, or a loose player who still folds hard to real
pressure — both exist among real opponents), the model has NO way to tell, because the trait that
actually matters was never observable to begin with. This makes it a genuine information
bottleneck, not just an undertrained corner — more training hands can't teach the network to read
a signal it was never given.

**Suggestion (concrete fix, revised)**: add a THIRD per-opponent signal alongside `vpip_color`/
`agg_color` — e.g. `fold_pressure_color`, computed the exact same way. Cheap on the training side:
`simulator.py`'s HUD-building code already does an ORACLE read of each bot's own `current_vpip`/
`current_agg_freq` per hand (not real empirical tracking) to build the existing two colors — adding
`current_fold_to_pressure` as a third oracle-read value is the same shape of change, not a new
mechanism. Real cost: a genuine contract change (new per-seat context feature, contract_version
bump, budget like [OPP-2]/[OPP-3]'s history — needs an actual retrain to evaluate, not a cheap
warm-start check). **Separate, additional gap this exposes**: this feature would be "free" in
training (oracle read) but LIVE opponents don't ship with a `fold_to_pressure` label — making this
useful outside self-play needs genuine empirical HUD tracking (e.g. "% folded facing a bet >=66%
pot," accumulated over observed hands) added to PHPHelp.py's live HUD, which doesn't exist yet —
a real, separate lift, not just a training-side change. Also still worth doing as a first, cheap
check before any of the above: re-run `allin_exploits_opponent_foldiness` (new FAST check, added
2026-07-18) against the 100k/150k confirmatory checkpoint once available, to see how much more
training exposure alone moves the spread before concluding a feature change is required.

**V26 update (2026-07-18)**: ran that suggested cheap check against V26's 100k confirmatory
checkpoint (2 of 5 opponent seats swapped to real-data TreeOpponents — see `versions/v26/SPECS.md`).
`allin_exploits_opponent_foldiness` spread: **0.011**, still WARN, no meaningful improvement over
V25. Training exposure and real-data-opponent diversity alone did NOT move this — consistent with
the mechanism above being a genuine information bottleneck (no `fold_pressure_color`-equivalent
input), not an undertrained corner, and consistent with the source data's own honest caveat (the
Pluribus-fitted TreeOpponents don't express a wide foldiness spread either, since all of Pluribus's
humans were similarly-skilled pros). Separately, LIVE telemetry from this run surfaced a related but
distinct pattern worth tracking: hero's actual **jam% by opponent color** (not the fixed-equity
probe, the real in-training action rate) is Blue 9-11% → Green 16-17% → Yellow 22-23% → Red 19-20%,
stable across the whole run — NOT flat, but also not monotonically ordered by real fold-proneness
(Blue/NIT-like is tightest AND most fold-prone, yet gets jammed on LEAST). Two live explanations, not
yet distinguished: (a) the same underexploitation this entry describes, or (b) a genuine
range-selection effect (reaching postflop against a tight opponent at all is already a
stronger-than-average spot for hero, so a lower jam rate there may be appropriate, not a leak — see
the "Important correction" above, same logic). The suggested `fold_pressure_color` feature (or a
same-seed V25-vs-V26 ablation) remains the next real lever; simply adding more/different opponents
without that feature does not appear sufficient.

**Pre-build calibration (2026-07-20, computed before committing to any V30/OPP-8 retrain — scope
was HELD as a result).** Confirmed with the real v29 archetype traits that `fold_pressure_color`
alone is REDUNDANT: across the current 4-archetype roster, `vpip_color` and `base_fold_to_pressure`
are PERFECTLY rank-anti-correlated — Spearman −1.00 (NIT 0.11vpip/Blue & 0.85 fold; TAG
0.22/Green & 0.60; LAG 0.32/Yellow & 0.45; STATION 0.45/Red & 0.15). So adding the feature CANNOT
move `allin_exploits_opponent_foldiness` off 0.011 in any way the model can't already get from
`vpip_color`. The correct fix is therefore TWO coupled changes: (a) +2 trait-DECOUPLED archetypes
(e.g. a loose fit-or-fold fish: vpip~0.42/Red but fold~0.78; a trap-nit: vpip~0.14/Blue but
fold~0.20) — which drops the roster correlation to ~−0.26 so the trait carries new info AND
mitigates the range-selection confound — PLUS (b) the `fold_pressure_color` feature (contract_version
8→9, context_dim 54→59). **Decision: HELD, not built.** Reasons: (1) the 0.011 metric likely
OVERSTATES the real leak (the range-selection correction above); (2) reshaping the opponent
population is the historically highest-collateral change in this repo (V24_extreme broke
`vpip_adapts_to_style`; V27's opponent-perception fix doubled VPIP / narrowed action diversity) —
risky to bet a clean V29 against; (3) no LIVE payoff without a separate empirical-HUD-tracking
project (fold_pressure_color is an oracle read live can't provide); (4) V29 is clean/validated —
this is a speculative capability-add, not a fix. Revisit only if the goal shifts to explicitly
exploitative play AND live fold-to-pressure HUD tracking exists. Higher-ROI next moves identified:
live-validation of V29's untested OPP-2/all-in vision wiring ([VAL-4]), then the short-stack/ICM
curriculum pass ([FMT-1] + the [VAL-1] BB-5-6bb finding) as the next lower-risk trained version.

### [OPP-9] Live range-aware equity doesn't narrow with continued action 🔴 OPEN
**First identified**: 2026-07-19, user's own question while scoping V28 ("in postflops should we
still assume that remaining players' VPIP is real when calculating range-aware equity?").

**Simple**: When estimating a live opponent's hand range for the equity calculation, we treat them
the same way whether they just showed up this street or have already called/raised through two
previous streets — the range never gets narrower just because they've kept putting more money in.

**Technical**: `compute_range_aware_equity` (`versions/v26/self_play/simulator.py:229-316`) takes
no parameter for street count, per-opponent action count, or committed-this-hand amount anywhere in
its signature — only a per-call `opp_colors`/`front_colors` split (the current betting round's
positional order, via `PHPHelp.py`'s `_classify_opponents_by_action_order`). Worse: its VPIP
fold-roll (the mechanism that actually narrows the sampled range toward stronger hands) is gated
`if is_preflop` only (`simulator.py:254-257`) — its own comment states postflop is "unchanged, no
roll applies there either way." So postflop, this function currently does NO VPIP-based narrowing
at all, on any street, regardless of how many times the opponent has already continued. This is
DIFFERENT from [OPP-4] (reopened-action detection within the CURRENT round) — this is about
cross-street accumulation: an opponent who called a flop bet and a turn bet has, in reality, already
filtered themselves toward a stronger range than their raw preflop VPIP number implies, and nothing
in this pipeline reflects that.

Confirmed this is LIVE-SERVING-ONLY: training's own `_mc_target_evs_sized` never faces this
question, since it always uses oracle equity postflop (the opponent's literal real dealt cards,
`simulator.py:922-925`) — there's no range to narrow when you already know the true cards. The
`opp_committed_this_hand_bb` feature ([OPP-2]-era, `contract.py`) is fed to the model as a separate
input but is NEVER passed into the range-sampling function itself — the model might learn to
partially compensate via that raw feature, but the EQUITY NUMBER hero's decision is built around is
not itself corrected.

**Suggestion**: extend `compute_range_aware_equity` to accept each opponent's action count / total
committed-this-hand amount and apply a progressively tighter effective VPIP cutoff (or a partial
fold-roll) postflop, scaled by how much they've already put in — reusing the SAME
`opp_committed_this_hand_bb` value already computed and passed into `contract.py`, not a new
mechanism. A real, contained scoping pass, separate from the risk-adjusted-target work in V28 (which
lives in `_mc_target_evs_sized`/training, not `compute_range_aware_equity`/live serving) — not yet
scoped in detail.

---

## Format fit

### [FMT-1] No ICM awareness 🔴 OPEN
**First identified**: 2026-07-17, live session on a Double-or-Nothing board. **Reconfirmed**:
v21_auxhead live, 2026-07-17, turn 28 (`history/Double_Or_Nothing_1171134565/flagged/
turn_28_20260717_212121/`), user caught it live and stopped the fold.

**Simple**: In tournament formats (like Double or Nothing), we make decisions purely on chip
count, not on how much actually busting out costs you — so we may fold spots a human would call
wider in a real bubble/survival situation.

**Technical**: Trained purely on cash-game-style BB/100 profit; no ICM (tournament equity) model
anywhere in the pipeline (simulator payouts, EV targets, or evaluation). Confirmed live: folds
that are clean chip-EV-correct even at 2-3bb effective stack, where ICM-aware strategy might
differ. **Concrete instance (v21_auxhead, turn 28)**: KcJd, 1.2BB stack, facing a 1.0BB call
(effectively covered, 0.2BB left after) 3-way vs Blue/nit+Red/maniac+Red/maniac. equity=0.26 (range-
aware) vs pot_odds=0.345 needed → model FOLDs at policy mass 1.00, critic Q confirms CALL is
chip-EV-worse by ~0.19BB. Both Layer 1 (OCR) and Layer 2 (model's actual decoded input) checked out
clean — this is a real Layer 3 policy outcome, not a perception/bridge bug. The single-hand chip-EV
math is technically correct in isolation, but at ~1BB effective, standard push/fold theory shoves
extremely wide (often ~ATC) because folding isn't free either — the stack faces blind attrition again
next hand at an even thinner depth. The model's per-hand EV target has no notion of that future cost,
so it undervalues calling/pushing in exactly this ultra-short-stack band. This is a myopic
single-hand-horizon gap that compounds [FMT-1]'s core ICM gap, not a separate root cause.

**Suggestion**: Would need genuinely different training targets (ICM-weighted payouts instead of
raw chip profit) — a substantial, separate project, not a config tweak. Only relevant if
tournament/DoN formats are a priority over cash-style play. The turn-28 instance suggests a
cheaper partial mitigation worth considering separately: a stack-depth-aware push/fold prior or
floor below ~2BB effective (matching known Nash push/fold charts) rather than full ICM-weighted
retraining, if ultra-short-stack folds keep recurring in live play before the full ICM project is
prioritized.

---

## Validation & tooling (methodology gaps, not model behavior bugs)

### [VAL-1] No external GTO/solver ground truth 🟡 PARTIAL (external axis built + Tier B in-repo Nash solver, V30 2026-07-20)

**V47 note (2026-07-22)**: P0.3 re-scored the Nash axis with a LITERAL jam-vs-fold primary
metric (composite kept as secondary) and re-baselined the lineage: V41 82%, V43 65%, V44 71%
(composites 78/65/66). V47 scored 65/65 — literal down vs V44, composite flat; top
disagreement cells are IDENTICAL to V44's (92s–94s@5bb overplayed, T2s–T4s@5bb called vs
jams): a lineage-wide ≤5bb looseness. Finding (A) (BB overcalls jams) still reproduces.
V48 P48-0 adds the 3-max solver; judge future models on the literal metric.

**V48 note (2026-07-23) — 3-MAX AXIS LIVE**: `solve_nash_3max.py` completed (BTN jam → SB
call → BB call/overcall, smoothed stochastic FP, 4.26M-triple lazy MC cache, anchors OK) +
two new FAST checks. Frozen-V47 baseline: `nash3_btn_jam` 74% / `nash3_bb_call` 73% (both
WARN, bar 0.75). Trained V48: **btn_jam 79% PASS** (Change 1's true-3-handed geometry
learned), bb_call 73% flat — facing-a-jam calls remain the weak side on BOTH the HU and
3-max axes (finding (A) again: HU nash_bbcall 72%, same T2s–T4s@5bb overcall cells). V48
HU literal: 66% (vs V47 65%), composite-commits now expressed as literal ALLIN 769/971 →
Change 0's collapse fixed the jam-labeling semantics (was 74/971 literal on V47).
**Simple**: We've never checked our play against real game-theory-optimal solutions — all
validation is "do we beat our own training opponents," not "are we close to unexploitable play."

**Technical**: The entire `model_verify` suite (FAST + SLOW checks) was self-referential — it tests
against the project's own simulator, its own heuristic archetypes, and its own frozen
predecessors. No external solver (e.g. PioSOLVER-derived ranges) had ever been used to validate a
specific spot.

**Progress (V30, 2026-07-20 — first external axis, chosen as the V30 scope)**: added
`nash_pushfold_vs_chart` — the FIRST check that tests hero against an EXTERNAL game-theory answer.
Fully ENCAPSULATED plug-in under `tools/model_verify/nash/` (curated static chart + baked-in
equities + one FAST check); integration into existing code is only an import + one `FAST_CHECKS`
entry + one `CHECK_DOCS` entry in `checks.py`. Touches NOTHING in `versions/*`, the simulator, or
`train.py`. This is deliberately a TOOLING addition, NOT a retrain — run it BEFORE deciding whether
a trained V30 model is even warranted. Method: curated set of UNAMBIGUOUS heads-up Nash push/fold
reference cells (SB open-jam, 0.5/1, chip-EV) — premiums/pairs/aces shove short, bottom-tier
offsuit trash folds at 15bb — avoiding near-indifference boundary hands whose exact BB threshold
varies by source. Equity fed to the model is raw HU equity vs one random opponent (assumption-free;
the model supplies push/fold reasoning from stack geometry). WARN-only, never a deploy gate (a
6-max cash model isn't REQUIRED to match a HU subgame; HU/position is mildly OOD). **Key
methodology lesson learned building it**: the first cut compared ALLIN-vs-FOLD specifically and
mislabeled 9 premiums as "folds" — the model was actually choosing RAISE_POT. Corrected to
compare AGGRESSION(raise-family+allin)-vs-FOLD (Nash "shove" ≈ "commit" in a discretized sizing
action space), with the ALLIN-vs-raise sizing split reported separately.

**First results (V29, the current live model)**: 34/35 (97%) in-range direction agreement, PASS.
Two genuinely useful external findings self-play never surfaced: (1) ALL 23 committed premiums are
played as a sized RAISE, never a literal jam — V29's anti-jam [BET-1] fix confirmed on an
independent axis; the model has essentially no literal all-in in its short-stack repertoire now
(defensible at 10-15bb since a pot raise commits a big fraction, but a real, quantified divergence
from pure push/fold theory). (2) ONE candidate leak: `Q2o@15bb` — Nash folds it, model commits it
(61% aggression, argmax raise_pot), plausibly over-weighting Q-high's deceptively high raw HU equity
(0.47 vs random) without accounting for domination. Single boundary cell, mild — but exactly the
class of thing this axis exists to catch. Worth watching whether it recurs / widens in future
versions.

**Tier B DONE (2026-07-20, same session)**: instead of copying a published chart (transcription-risk),
SOLVED the HU push/fold Nash equilibrium IN-REPO — `tools/model_verify/nash/solve_nash_pushfold.py`
(offline): a 169x169 Monte-Carlo preflop all-in equity matrix (cached to `equity_matrix.json`) fed
into fictitious play over the shove-or-fold zero-sum game (SB jam vs BB call, 0.5/1, chip-EV), for
stacks 5-20bb. Output `nash_solved.json` (SB jam range + BB call range + BB-equity-vs-jam-range per
stack). VALIDATED against 12 famous non-controversial anchors (all pass) and the solved range SIZES
match canonical published Nash closely (10bb: SB jam 56.9% / BB call 36.7%; 15bb: 45% / 26%; 20bb:
40% / 20%). Documented approximations: canonical-suit-representative MC equities (card removal within
each matchup, not across range weighting) + MC noise. Two runtime checks now read this file (still
pure lookup + run_policy, zero solver/sim deps): `nash_pushfold_vs_chart` (SB, full 169xstacks) and
the NEW `nash_bbcall_vs_jam` (BB facing a jam — the cleaner binary spot, no cheap-limp confound,
model equity RANGE-CONDITIONED on SB's Nash jam range).

**Tier B results vs V29 (both PASS, WARN-gated)**: SB 1247/1498 (83%) over unambiguous cells; BB
1304/1507 (87%). By-stack breakdown revealed TWO distinct, opposite, computed findings (not guessed):
(A) **BB is TOO TIGHT calling jams at 5-6bb** (agreement 74%/78%, rising monotonically to 95% by
20bb) — folds K-/Q-high suited hands Nash CALLS. This is the trustworthy, actionable finding: a real
spot, at stacks the model DOES train on, with range-conditioned equity. Overlaps [FMT-1]'s
ultra-short myopia and the 5bb training-floor boundary. (B) SB agreement instead ERODES with depth
(88% at 5bb → 73% at 20bb), but this is mostly a YARDSTICK ARTIFACT not a leak: the deep
disagreements are the model COMMITTING weak suited hands (Q4s/J6s/K2s) that pure jam-or-fold Nash
folds — and it commits them via a sized RAISE, which is normal/correct real-poker open-raising at
20bb HU where a jam-or-fold game is unrealistic. The SB check is most trustworthy at short stacks
(87-88%), where jam-or-fold is the actual game. Also reconfirmed at scale: 851/851 SB commits use a
sized raise, never a literal jam (V29 anti-jam, [BET-1]).

**Finding (A) DIAGNOSED (2026-07-20, cheap no-retrain probe — done before deciding on a version)**:
computed pot-odds threshold t=(S-1)/(2S) and margin=eq_vs_jam−t for every BB call/fold cell, then
compared agreement at EQUAL margin across stack depth. Result is decisive: it is a STACK-SPECIFIC
calibration gap, NOT boundary noise. At an equal +0.03–0.07 price-edge the model calls 0% of
Nash-call hands at 5-6bb vs 57% at 8-20bb; mean P(fold) at that edge is 0.80 short vs 0.48 deep —
the model demands ~2x the price-edge before committing at 5-6bb. It is NOT gross: every wrongly
folded hand is only a modest +EV call (median margin +0.027, max +0.09; no clear +0.20 spot folded),
and both bands converge to 100% call once margin>+0.12. Root cause = a TRAINING-CURRICULUM FLOOR
effect (5-6bb is the thin bottom edge of the lowest `stack_depth_mix` band 5-14bb; the model
over-generalizes a deeper-stack "don't commit a big fraction on a thin edge" caution to a 5bb spot
where it's already priced in), NOT a feature/target bug. **Decision: do NOT spend a dedicated version
on it** — real but narrow/modest (marginal +EV calls, only the bottom two depths, a rare live 6-max
spot). Fold into a future short-stack/ICM pass ([FMT-1]) via extending `stack_depth_mix` below 5bb /
upweighting 5-8bb, then a retrain — a data-coverage lever, not a new mechanism. Further
external axes still open: PioSOLVER postflop spot-checks; a genuinely different (human/solver-like)
opponent for [OPP-1] stress. Still METHODOLOGY-flavored but now real external coverage of the whole
short-stack preflop push/fold space, both seats.

### [VAL-3] `free_check_low_fold` residual mass 🟡 PARTIAL
**Simple**: When there's no cost to seeing another card, the model's raw output occasionally still
shows some (fully masked, never-executed) desire to fold — which doesn't make sense, since folding
a free option is never correct.

**Technical**: Raw policy fold-mass when `call_amount=0` is masked to zero and renormalized by
`core/decision.py` before a decision is ever made, so this NEVER reaches a live action. But the
raw number itself is high in every version checked (V20_preflopEq_AI: 1.000, maximal). Root cause
not identified — a candidate is the equity-primary base head not fully internalizing that a free
continuation always dominates folding, but this hasn't been tested directly.

**Suggestion**: Not urgent (fully masked). Worth a dedicated look only if it turns out to
correlate with one of the OPEN items above (e.g. the same base-head calibration issue behind
[BET-1]).

### [VAL-4] New-feature live track record is thin 🟡 PARTIAL
**Simple**: The newest features (hand strength, equity edge, and the front/after "who's already in
the pot" logic) have only been tested in one real live session so far.

**Technical**: Verified via `model_verify` (both FAST sensitivity checks and SLOW field-winrate
checks) and one live Double-or-Nothing board. No long-run live statistics accumulated yet. The
turn-history recorder was only updated to persist these fields on 2026-07-17 (previously
live-only, discarded after each decision) — so accumulation starts now, not retroactively.

**Suggestion**: Keep monitoring live sessions now that the recorder captures these fields;
revisit this entry once enough live hands have accumulated to say something statistical.

### [VAL-5] Warm-started continuation degrades action diversity, independent of the variable being tuned 🔴 OPEN
**First identified**: V21_auxhead Phases 5-7 (2026-07-17), while mapping the aux-head weight
response curve.

**Simple**: Extending a model's training with a `--resume_path` continuation run (rather than
training fresh from scratch for the full hand count) seems to push it toward fold/all-in-only play,
regardless of what else changed in that continuation.

**Technical**: Four separate `+50k`-hand warm-started continuations off the same clean
`phase2_fresh_100k.pth` base (aux-head weights 0.05/0.20/0.10/0.35 — otherwise identical) ALL showed
`action_diversity` collapse to 2 distinct actions (`{fold:9,allin:12}`) and `stack_full_sweep`
argmax pinned to `allin` at every one of 9 stack points. The base checkpoint's OWN 100k-hand fresh
run (Phase 2) had the best bet-sizing diversity of the whole investigation (4 actions, a real
call→raise_pot→raise_66 progression). Since the collapse appeared identically across four different
aux-weight values, the continuation MECHANISM itself is implicated, not the variable under test —
plausibly the same total-hands/shove-trend pattern behind [BET-1], but not yet directly proven.
Re-training the fully chosen aux config FRESH to 100k (Phase 8) recovered most (not all) of the lost
diversity (3 actions vs Phase 2's 4), confirming continuation is at least A cause.

**Suggestion**: Until root-caused, prefer fresh full-length training runs over stacking
`--resume_path` continuations when evaluating anything sensitive to `action_diversity`/
`stack_full_sweep` (hyperparameter sweeps, ablations). If continuation must be used for cost
reasons, always re-validate the final candidate with `model_verify --full` rather than trusting an
earlier fresh-run's numbers to still apply. Worth a dedicated investigation (does hand-COUNT alone
reproduce it on a single fresh run taken past 100k, or is it specific to the optimizer/scheduler
reset `--resume_path` causes — see the unfixed gap noted in versions/v21_auxhead/SPECS.md) before
V22 relies on any warm-started extension.

---

## Resolved (kept for regression pattern-matching — do not re-flag as new)

### [RESOLVED-1] VPIP does not adapt to opponent style 🟢 RESOLVED
Originally V16 `[P4]` (VPIP-vs-style flatness — hero's preflop entry range didn't tighten/loosen
with opponent tightness, only postflop aggression did). Range-aware equity + the V20_preflopEq
Finding 2 front/after fix appear to have resolved this: `vpip_adapts_to_style` PASSES with a
growing margin — V20_preflopEq 75k (short +6.6pt, deep +7.1pt) → V20_preflopEq_AI 150k (short
+11.5pt, deep +9.6pt), both clearing the 5pt gate comfortably.

### [RESOLVED-2] model_verify FAST checks fed V20 the wrong context-feature scale 🟢 RESOLVED
Found and fixed 2026-07-17 while extending `model_verify` for V20_preflopEq: `scenarios.py`'s
`build_ctx` hardcoded the legacy `/400,/1000` scale for every version, but V20 (`contract_version`
4+) actually uses a clamped `/100,/250` scale. Every FAST check that varies stack/pot/call had been
silently testing V20 at a ~4x-wrong scale since it shipped. Fixed via a `contract_version`-aware
`_money_scale()` helper; SLOW checks (real simulator) were never affected. V20's own historical
FAST-check narrative in its SPECS.md predates this fix and should be treated as unreliable if
referenced.

### [RESOLVED-3] `vectorize_hand_samples` never received V20's own rescale 🟢 RESOLVED
Found and fixed 2026-07-17 while building V20_preflopEq: `train.py::vectorize_hand_samples` (the
function that builds the ACTUAL gradient-training tensors) kept a stale `/400,/1000` copy of the
context math while every inference path (rollout, live) used the new `/100,/250` scale — a real
train/serve mismatch baked into the deployed V20 model. Fixed by factoring the scale/clamp math
into shared helpers in `contract.py` that both paths import.

### [RESOLVED-4] Unknown HUD color silently dropped from live equity 🟢 RESOLVED
V20_preflopEq Finding 1 (2026-07-17): an opponent with no classified HUD color yet was excluded
entirely from the live equity calculation, as if not contesting the pot. Fixed by mapping unknown
→ 'Yellow' (the codebase's existing "no info" convention), in `PHPHelp.py`.

### [RESOLVED-5, historical] `hero_position` never set during training queries 🟢 RESOLVED
V19: every training-time model query (hero's own AND every opponent's) silently defaulted
`BoardState.hero_position` to Button — confirmed universal across V12-V18, training-only (live
serve was always correct). Fixed by threading each actor's real button-relative position through.

### [RESOLVED-6, accepted-as-policy] Cross-scale predecessor comparison gap 🟢 RESOLVED
Originally `[VAL-2]`, first identified V20 (`beats_frozen_predecessor` SKIP vs frozen `nit`/`tag`
when `contract_version`/context_dim changed), reconfirmed V20_preflopEq, worked around (not fixed)
in V20_preflopEq_AI. **2026-07-17: accepted as by-design, not a gap to close.** Decision: an older
model trained under a stale/incompatible contract is not a useful opponent or comparison baseline
for a newer version regardless of whether a cross-scale query mechanism existed — those older
models weren't strong enough in general for beating them to be meaningful signal, and building a
mechanism to query each frozen opponent through its own original contract would serve a comparison
we don't actually want while the model line is still moving quickly on base behavior. Revisit only
if the model line matures to a point where genuinely strong historical checkpoints exist and a
same-generation baseline becomes valuable again; until then, `beats_frozen_predecessor` SKIPping
across an incompatible contract change is expected, not a defect.

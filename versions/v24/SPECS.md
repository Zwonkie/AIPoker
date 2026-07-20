# V24 SPECS

Branches from `versions/v23` (`expert_main.pth`, 150k hands) -- same context/contract
(context_dim=44, contract_version=7, `pot_type` unchanged). V24 touches ONLY
`simulator.py`'s `_mc_target_evs_sized` (hero's own training-target computation) and
`opponent_bots.py` (a new per-personality trait). No network input features change.

## Motivation: V23's regression

V23 applied the [BET-1] price-sensitivity fix directly inside `opponent_bots.py`'s
`decide_postflop`/`decide_preflop` -- the SAME functions `_mc_target_evs_sized` calls to sample
opponent fold rates for computing hero's own per-size EV target. Result: `action_diversity`/
`deep_stack_ood_guard` REGRESSED (see `versions/v23/SPECS.md`). Root cause, confirmed by code
inspection: making bots fold more to oversized bets doesn't just describe more realistic live
play -- it ALSO mechanically inflates hero's own ALLIN training target, since `p_all_fold * pot`
is credited straight into that size's counterfactual EV. The fix's two effects (opponents demand
more to continue vs. hero gets more fold-equity credit for shoving) pointed in opposite
directions, and the wrong one won.

## Fix 1: decoupling

`_mc_target_evs_sized` no longer calls `bot.decide_preflop`/`decide_postflop` directly. New
`SixMaxSimulator._ev_target_fold_decision` is used instead -- independent of the live,
BET-1-fixed decide_* functions (which stay unchanged, still used by live self-play opponents via
`_opponent_decide`, so self-play itself stays realistic). The decoupled function reverts the VALUE
branch to the pre-BET-1 flat `need_for_value` for target-EV purposes specifically, keeping the
original (validated, pre-dates BET-1) P1b `continue_bar` price-sensitivity, which was never the
regression's cause.

## Fix 2: "show of strength" for raises (not all-in)

New per-personality trait `bot_bluff_perc` (added to `FuzzyPlayerArchetype`, `opponent_bots.py`):
how often THIS bot's own raise is "for show" (representing strength it doesn't actually have).
Used, when this bot is the one FACING a raise, as an inverse read of how skeptical it is of
others' raises -- a bot that bluffs a lot itself assumes others do too (low respect, harder to
fold via a raise); one that rarely bluffs trusts raises more (high respect, easier to fold via a
raise). Deliberately a NEW, general-purpose trait (not derived from `base_bluff_freq`, which only
governs a bot's own below-the-bar bluff-raise frequency) -- scoped to just this mechanism for now,
intended for reuse in other bluff-related calculations later (e.g. preflop bluff-raise
frequency) per this version's own discussion.

**Mechanism** (in `_ev_target_fold_decision`): for non-all-in raise sizes, with probability
`1.0 - bot.current_bluff_perc`, the opponent "respects" the raise and folds MORE than raw price
alone would justify. All-in gets NONE of this bonus -- priced honestly on raw pot odds. This
directly targets the no-middle-gear problem: fold-equity today scales monotonically with size
(bigger bet -> more folds), which makes all-in dominate by construction; a raise-only,
non-price-scaling fold-equity source breaks that monotonicity -- a raise can now extract MORE
folds per dollar risked than a shove, a genuine incentive distinct from "less bad scaling."

### A real implementation bug found and fixed during calibration

First attempt implemented the bonus as a boost to `need_for_value` (the value bar), mirroring
[BET-1]'s own `VALUE_PRICE_SENSITIVITY` mechanism. Direct EV-arithmetic calibration (isolating
the `is_allin` flag's effect at MATCHED pot_odds, not the full noisy `_mc_target_evs_sized`
pipeline) found this had **no effect at realistic raise-pot price levels** (pot_odds~0.5): the
value bar never gates the fold-vs-continue decision except when `continue_bar` is INDEPENDENTLY
already high (i.e. shove-level pot odds) -- both the value AND continue branches return
"continue," so boosting the value bar alone can't create a fold at normal raise sizes; it can only
matter when the price is already big enough that continue_bar bites too. Fixed by boosting
`continue_bar` directly instead -- the actual fold-gate at any price level.

**This was caught before committing to a retrain, not after** -- direct EV/decision-function
arithmetic checks (not a full noisy training run) surfaced the flaw in minutes rather than costing
another ~2h wasted cycle, mirroring the same discipline [BET-1]'s own calibration used.

### Calibration

Isolated `_ev_target_fold_decision` directly (not the full `_mc_target_evs_sized` pipeline, which
adds real-card-equity noise that obscures the mechanism) at MATCHED pot_odds=0.50 (a realistic
pot-sized-raise price) for `is_allin=True` vs `False`, across a range of opponent equities
bracketing each archetype's own `continue_bar`, for `RAISE_RESPECT_BOOST` in
{0.05, 0.08, 0.10}. `RAISE_RESPECT_BOOST=0.10` chosen: produces a real, non-degenerate
`P(fold|raise) - P(fold|allin)` gradient for all four archetypes (TAG up to +0.52 at its own
marginal-equity edge; LAG +0.33; NIT up to +0.92 at ITS edge -- the strongest response, fitting
since NIT is the tightest archetype; CALLING_STATION +0.18-0.31).

**CALLING_STATION required its own `base_bluff_perc` recalibration mid-process**: its naturally
low `continue_bar` (very sticky, by design) meant even a small, infrequent respect-bonus swung a
wide equity range to fold -- directly contradicting its real identity as the hardest personality
to fold via any signal. Raised `base_bluff_perc` from an initial 0.40 to 0.70 (30%
respect-application rate instead of 60%) to keep the bonus rare enough not to override that.

**Final `base_bluff_perc` values**: TAG=0.20, LAG=0.35, NIT=0.05, CALLING_STATION=0.70 --
deliberately NOT copied from `base_bluff_freq` (CALLING_STATION's own bluff-raise frequency is
LOW at 0.05, but that says nothing about how skeptical it is of OTHERS -- those are different
questions).

## Training setup

- `target_hands: 150000` -- matches V23 (another population-behavior-changing fix, same
  reasoning: more exposure before trusting the result).
- Everything else (aux-head config, actor-critic cutover, deep-stack curriculum, entry-sizing,
  `pot_type`, opponent pool) inherited unchanged from V23.
- Fresh from-scratch run (no `--resume_path`), per [VAL-5].

## Verification (pre-training)

- 40 real simulated hands via `SixMaxSimulator` + `vectorize_hand_samples` end to end -- no
  exceptions, confirmed the decoupled fold model + show-of-strength mechanism run cleanly inside
  real self-play, context tensor still 44-wide (no feature change).

## Results (2026-07-18, `expert_main.pth`, 150k hands)

Training completed cleanly (Val Loss 0.80, notably healthier than V23's late spike to 17.6). Final
dashboard: Hero +56.3 BB/100 (best of the lineage so far), Action Entropy ended at 0.075. Cumulative
`ACTION USAGE` looked balanced (r33 8.9% / r66 8.7% / rPot 9.9% / All-In 8.8%) -- **this cumulative
appearance turned out to be misleading, same as V23's own.**

**Frozen backup preserved**: `versions/v24/weights/frozen_v24_150k.pth` (byte-identical copy of
`expert_main.pth`), made immediately after training completed and before any further
tests/training, per explicit request.

**A real bug found and fixed during verification**: `_ev_target_fold_decision` assumed every
opponent bot was a `FuzzyPlayerArchetype` (accessing `.current_value_threshold` etc. directly),
crashing `model_verify`'s `beats_offformula_stress` check (which uses `TieredLookupBot`, a
different class without those attributes). Fixed with a `hasattr` fallback that calls the bot's
own `decide_preflop`/`decide_postflop` directly for non-`FuzzyPlayerArchetype` bots -- safe here
since only `FuzzyPlayerArchetype.decide_*` carries the BET-1 fix this function exists to decouple
from; a bot without these attributes was never part of that coupling problem.

### The 5-point checklist (pre-registered before results came in)

**1. Does `action_diversity` show a raise bucket winning argmax anywhere?** **NO.**
`{'fold': 9, 'allin': 12}` -- the exact same 2-action degenerate pattern as V23. No improvement.

**2. Does `deep_stack_ood_guard` PASS?** **NO.** Still FAILS, now at `eq=0.55, stack=40bb ->
ALLIN@0.46` -- slightly WORSE than V23's 0.40.

**3. Does the allin-to-next-best Q-value gap narrow (direct comparison at the same cell used for
V22 vs V23)?** **NO -- it WIDENED, at all three tested stack depths:**

| stack | V22 (healthy) | V23 (regressed) | V24 (this version) |
|---|---|---|---|
| 15bb | allin=0.31 (not even top; raise_66=0.46 was) | allin=2.08, next-best=0.85 (2.45x) | allin=1.65, next-best=0.60 (2.75x) |
| 25bb | allin=0.06 (not even top) | allin=2.42, next-best=0.98 (2.47x) | allin=2.45, next-best=0.78 (3.14x) |
| 40bb | allin=0.49 (not even top; raise_66=0.64 was) | allin=2.99, next-best=1.24 (2.41x) | allin=3.50, next-best=1.10 (3.18x) |

The mechanism, even after fixing the `need_for_value`-vs-`continue_bar` implementation bug and
calibrating via direct EV arithmetic, made the shove-preference MORE pronounced, not less.

**4. Any collateral regression vs V22/V23 in overall win-rate metrics?** Mostly NO -- these
actually held up well or improved: `bb100_vs_standard_fields` PASS across all 4 fields, all
comparable to or slightly ABOVE V23's own numbers (e.g. loose_deep +73.3 vs V23's +62.2, tight_deep
+51.4 vs +48.8). `beats_offformula_stress` PASS, comparable to V23. `vpip_adapts_to_style` PASS
but with slightly smaller deltas than V23 (+11.9/+9.0pts vs V23's +14.7/+14.8pts) -- not a failure,
just modestly weaker style-adaptation. **One real regression found**: `committed_sensitivity`
dropped from V23's PASS (0.109) to WARN (0.011) -- the entry-sizing feature that had been
improving with every version's additional training suddenly went flat. Plausible explanation (not
confirmed): with the policy even MORE concentrated on all-in than before, sizing-relevant features
matter less when the policy barely uses graduated sizes at all -- worth re-checking in any future
version, not yet root-caused. `pot_type_sensitivity` unchanged (still WARN, 0.004).

**5. Does `stack_full_sweep` show r33 disproportionately dominating r66/rpot (the flagged
uncalibrated-by-size gap)?** **NO -- a different, arguably more concerning pattern**: the argmax
path is `allin` at ALL 9 stack points (5-180bb). None of the three raise buckets won ANY point --
the flagged r33-vs-rpot asymmetry never got a chance to matter, because all-in's dominance grew
strong enough to crowd out every raise bucket entirely.

**Overall verdict: this is a clean, unambiguous NEGATIVE RESULT for the core no-middle-gear goal,
by every measure in the pre-registered checklist.** Interestingly, the model's aggregate win-rate
against real fields (`bb100_vs_standard_fields`, `beats_offformula_stress`) held up or slightly
improved -- suggesting the trained policy IS a genuinely profitable strategy against this training
population (over-shoving isn't being meaningfully punished in aggregate EV terms), just an even
more all-in-committed one at the specific marginal-equity spots the diagnostic checks probe. The
`bot_bluff_perc`/"show of strength" mechanism demonstrably creates real behavioral differentiation
in isolation (confirmed via the direct calibration probe on `_ev_target_fold_decision` itself,
before this retrain) -- but at 150k hands of real training, whatever fold-equity edge raises
gained wasn't enough to overcome all-in's other advantages (e.g. genuine equity realization,
avoiding future street decisions/variance the training population doesn't heavily punish).

**Verdict on deployment**: V24 is NOT recommended for live deployment. V22 remains the
best-validated candidate (16 PASS/4 WARN/0 FAIL/1 SKIP). `frozen_v24_150k.pth` is preserved for
live testing/inspection per explicit request, independent of this deploy recommendation.

**Next step (per the pre-agreed contingency, not yet built)**: a quick, deliberately EXTREME
diagnostic run -- `bot_bluff_perc` pushed to near-0 for all archetypes (maximizing respect
probability rather than aiming for a realistic calibration) and/or a much larger
`RAISE_RESPECT_BOOST`, `target_hands: 50000` for fast turnaround, and a simplified curriculum
(single fixed stack depth around 30-40bb matching `deep_stack_ood_guard`'s own failure zone,
bootstrap warmup skipped) so training converges quickly on just the one variable being tested. This
would answer "can this mechanism move the needle AT ALL" (mechanism validity) before spending more
time re-calibrating a value that might be inherently too weak at realistic, production-scale
settings -- pending explicit go-ahead, not yet started.

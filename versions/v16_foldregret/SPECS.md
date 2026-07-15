# V16_foldregret — isolated regret-baseline experiment

**Purpose (2026-07-15):** a single-variable comparison against `versions/v16` (the main line).
Same config, same opponent pool (`fish/tag/nit/past`, frozen-V15 in the `past` seat), same
stack-depth curriculum, trained FRESH (no warm-start) — identical to how V16 itself was trained.
The ONLY code change is `regret_match_policy` in `self_play/train.py`.

## What changed and why

Diagnosed from the live training dashboard's Equity Action Matrix: the `<20%` (Pure Air) and
`20-40%` (Draws) equity buckets consistently net-LOSE chips over the course of training, yet the
hero continues/raises with them roughly half the time (Fold only ~50-53% in those buckets, All-In
still ~15-19%). Traced to `regret_match_policy`'s baseline:

- **Before (every other version)**: `regret[a] = max(v[a] - mean(all action values), 0)`. A
  bluff-raise's fold-equity (a legitimately positive EV even with a weak hand, from
  `_mc_target_evs_sized`'s `p_all_fold` term) pulls the MEAN up. That inflated mean can then let
  an independently-negative action (e.g. calling with air) still show positive regret relative to
  it, because it's only being compared to "the average of everything including worse options,"
  not to whether it beats folding.
- **After (this version only)**: `regret[a] = max(v[a] - v[fold], 0)`, where `v[fold]` is always
  0 by construction. Every action now has to independently justify itself against the
  always-available zero-risk baseline. Genuine +EV semi-bluffs are unaffected (they beat fold on
  their own merits); actions that are worse than folding outright lose the free ride they got from
  being compared to a mean dragged up by other options. Degenerate-tie fallback also changed: if
  literally nothing beats folding, fold outright (weight 1.0) rather than mixing uniformly across
  options that are all at-or-below the zero-risk baseline.

Side effect (not the primary motivation, but real): this also makes the existing
`POLICY_TIGHTNESS_BB` discount (2.0bb, applied uniformly to all non-fold actions) hit at full
strength instead of being diluted to ~1/6th of its nominal value by mean-centering (a flat
subtraction from 5-of-6 actions moves the mean almost as much as each individual action, so the
old formula mostly just made folding relatively cheaper without differentially punishing the
worst offender).

**Considered and deliberately NOT done here**: a real actor-critic redesign (blending the
network's own learned Q-values into the actor's regret-matching target, instead of the fresh
per-decision simulator Monte-Carlo estimate `mc_evs`) — a bigger, riskier architectural change
that would need its own careful validation. This experiment is the cheap, surgical, low-risk
alternative; revisit the bigger redesign only if this doesn't move the Equity Action Matrix enough.

## Budget & validation plan

100,000 hands (not the main line's 200k) — a deliberately short run to check the fix's directional
effect before committing to a full budget. **Sanity check at 50k hands**: compare the Equity
Action Matrix's `<20%`/`20-40%` Fold% and Net Chips against V16's own numbers at a comparable
point, before letting it run to completion. Must also pass the usual gates (`overfit_sanity`,
`tools/model_verify`) before any deploy consideration — this is a comparison experiment, not
automatically a replacement for V16.

## Outcome (2026-07-15) -- trained to completion, NOT deployed

**Training**: 100,001 hands, ~1h8m, clean (zero NaN/crash across the run). Final hero cumulative
+17.4 BB/100 in training self-play. Equity Action Matrix confirms the fix engaged as designed and
held stable from the 46k-hand checkpoint through completion:
- `<20%` Pure Air: Fold 65.9%, All-In 16.0% (was ~50-56%/13-19% on V16's own run)
- `20-40%` Draws: Fold 84.3%, All-In 5.6% (was ~50-56%/13-19% on V16's own run) -- the biggest
  move: draws lost almost all of their free-riding CALL/ALL-IN mass once regret was measured
  against FOLD's value instead of the mean. Marginal/Strong/Nuts tiers stayed clearly profitable
  (+128.9k / +233.5k / +181.4k net chips respectively) -- not an over-tightening collapse, a
  targeted fix at the weak-equity end, exactly as intended.

**`tools/model_verify --full`: 9 PASS, 1 WARN, 2 FAIL** (`results/v16_foldregret__expert_main.pth.json`):
- `deep_stack_ood_guard` FAIL (eq=0.55 stack=15bb -> ALL-IN argmax @ 0.32) -- **pre-existing,
  not a regression**: V16 itself fails the identical check (eq=0.55 stack=40bb -> ALL-IN argmax
  @ 0.52), and this is the already-tracked `[P0-recheck]` carried-in defect from V15. Both
  versions ship with this gap; foldregret didn't introduce or worsen it in kind (arguably a
  smaller argmax mass, though it now surfaces at a shorter stack too).
- `vpip_adapts_to_style` FAIL: short delta +6.3pts (clears the 5pt bar), **deep delta only
  +2.0pts (fails it)**. V16 itself PASSES this cleanly (short +8.7pts / deep +8.4pts). **This is
  a genuine regression introduced by the fold-relative baseline change** -- the [P4] deep-stack
  opponent-style adaptation that V16's range-aware-equity substitution had fixed got measurably
  weaker under fold-relative regret-matching. Working theory (not yet verified): at deep stacks
  the range-aware-equity preflop EV edge over calling/raising is small relative to stack, so it
  sits closer to FOLD's zero baseline -- exactly the kind of small-but-real edge the fold-relative
  fix is designed to zero out along with the bad ones. The mean-baseline (V16's original formula)
  apparently preserved enough of that small style-conditioned edge to still shift the policy;
  the fold-relative baseline may be discarding it as collateral damage.

**Decision: NOT deployed.** Per standing policy (diagnose+retrain rather than ship a regression),
2 FAILs blocks promotion -- one is a shared pre-existing gap (doesn't block on its own), but the
`vpip_adapts_to_style` deep-stack regression is new and real. Kept as a validated experiment
result, not promoted over `versions/v16` (still active/default going forward pending its own
resolution of the P4/P0 gaps).

**Follow-up options (not started)**: (a) blend the two baselines -- e.g. regret vs
`max(fold_value, alpha * mean)` for some alpha, keeping the fold-anchoring for clearly-bad actions
while not fully discarding the mean's small-edge signal deep-stack relies on; (b) scale the
fold-relative baseline's bite by street/stack depth so it's strongest preflop-shallow (where it
worked) and gentler deep; (c) re-run with a wider `n_hands_style` sample to confirm the deep-stack
+2.0pt delta isn't partly sampling noise before concluding the mechanism theory above. Any retry
should re-run `deep_stack_ood_guard` too since that carried gap still needs its own fix
independent of this line.

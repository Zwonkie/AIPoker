# V24_extreme SPECS

Deliberately EXTREME, throwaway DIAGNOSTIC -- NOT a production candidate, no deploy consideration.
Same mechanism as `versions/v24` (decoupled EV-target fold model + `bot_bluff_perc` "show of
strength" bonus for non-all-in raises); only parameters and training recipe change.

## Why

V24's calibrated settings (`RAISE_RESPECT_BOOST=0.10`, realistic per-archetype `bot_bluff_perc`)
demonstrably created a real, non-degenerate fold-equity gradient favoring raises over all-in when
tested in ISOLATION -- but the full 150k-hand retrain showed no improvement on
`action_diversity`/`deep_stack_ood_guard`, and the allin-vs-next-best Q-value gap actually WIDENED
at every tested stack depth (see `versions/v24/SPECS.md`). This diagnostic answers: is the
mechanism fundamentally unable to overcome all-in's other advantages at this population, or was
the calibrated magnitude just too subtle to show up over a full training run?

Also worth flagging (2026-07-18 discussion, not addressed by this diagnostic): the deeper likely
root cause is that `_mc_target_evs_sized`'s `ev_if_called = true_equity * (pot + 2*raise_size -
to_call) - raise_size` treats a called raise as a TERMINAL, single-street outcome -- there's no
representation anywhere of the value a smaller raise preserves by keeping future streets' betting
alive (bet again on the turn/river if still ahead) vs. an all-in that either folds out immediately
or goes to a showdown-equivalent right now. Both this run and V23/V24 test the opponent-response
side of the problem; if this ALSO shows no movement, that's added evidence the real fix has to be
in the target-EV computation's structural myopia, not opponent behavior at all.

## What's different from V24

- `bot_bluff_perc` pushed to near-0 (0.02) for ALL FOUR archetypes (maximizing "respect"
  probability to ~98%), including CALLING_STATION (V24 had pushed IT up to 0.70 for archetype-
  realism reasons -- deliberately ignored here, since maximizing the lever's effect is the point).
- `RAISE_RESPECT_BOOST` 0.10 -> 0.40.
- `target_hands: 50000` (fast turnaround) instead of 150000.
- `stack_depth_mix` removed; `fixed_stack_bb: 35.0` instead (matches `deep_stack_ood_guard`'s own
  failure zone).
- `disable_bootstrap: true` (skip heuristic-anchoring warmup entirely).
- Everything else (aux-head config, `pot_type`, entry-sizing, opponent pool, contract/context)
  inherited unchanged from V24.

## Reading the result

- If THIS run ALSO shows no movement on `action_diversity`/the Q-value gap: strong evidence the
  opponent-response-shaping approach is a structural dead end for this problem at this scale --
  redirect fully to the target-EV computation itself (the `ev_if_called` myopia above).
- If it DOES move the needle at this extreme, non-production-safe setting: confirms the mechanism
  is viable in principle, and the search should shift to finding a calibrated value between V24's
  (too weak) and this run's (too strong) settings.

## Verification (pre-training)

Code/mechanism identical to V24 (already smoke-tested there); only parameters changed here.
Config-level sanity checked (`fixed_stack_bb`/`disable_bootstrap` read paths confirmed in
`train.py`).

## Results (2026-07-18, `expert_main.pth`, 50k hands)

Training completed cleanly (55m53s, 15 hands/sec -- slower than V24's 22/sec, likely the deeper
per-hand action variety this setting produces). Final dashboard looked genuinely different from
every prior version: cumulative `ACTION USAGE` Fold 26.2% / Call 15.4% / r33 16.0% / r66 16.6% /
rPot 16.8% / All-In 9.0% -- all three raise buckets now individually HIGHER than all-in. Hero's own
cumulative performance was NEGATIVE (-27.4 BB/100) during training, though the final trained
policy's real-field win-rate (see below) is positive -- the negative cumulative number reflects
a rocky, still-adapting training trajectory over only 50k hands, not the converged result.

### The core question this diagnostic answers: **partially YES, the lever can move the needle**

Direct Q-value comparison at the same cells used throughout this investigation (`eq=0.55`):

| stack | V24 (calibrated 0.10) | V24_extreme (0.40, all other changes) | gap ratio (V24 -> extreme) |
|---|---|---|---|
| 15bb | allin=1.65, next-best=0.61 | allin=2.79, next-best=1.59 | 2.75x -> 1.75x |
| 25bb | allin=2.45, next-best=0.78 | allin=2.69, next-best=1.53 | 3.14x -> 1.76x |
| 35bb | allin=3.28, next-best=1.03 | allin=2.47, next-best=1.47 | 3.18x -> **1.68x** |
| 40bb | allin=3.50, next-best=1.10 | allin=2.31, next-best=1.44 | 3.18x -> **1.60x** |

The gap roughly HALVED at every tested stack depth. All-in is still technically the argmax at
these specific cells, but `model_verify`'s full grid shows real movement for the first time in
this whole investigation:

- **`action_diversity`: `{'fold': 8, 'raise_pot': 2, 'allin': 11}` -- `raise_pot` won 2 argmax
  cells.** No raise bucket has EVER won a cell in this check across the entire V20-V24 lineage
  until now.
- **`stack_full_sweep`'s argmax path**: `['allin','allin','allin','allin','raise_pot','raise_pot',
  'fold','fold','fold']` -- a genuine transition through actions by stack depth (5-180bb), not one
  action dominating the whole sweep (every prior version from V22 on showed a single action -- call
  or allin -- winning ALL 9 points).
- **Several long-flat features became load-bearing for the first time**: `committed_sensitivity`
  0.011 (V24, WARN) -> 0.075 (PASS); `pot_type_sensitivity` 0.004 (WARN, every version since V23)
  -> 0.122 (first-ever PASS); `opponent_style_sweep` ~0.001-0.010 (WARN, every version since
  V20_preflopEq_AI) -> 0.151 (first-ever PASS).

### The cost: a new regression, and a likely-confounded result

**`model_verify --full`: 18 PASS / 1 WARN / 2 FAIL / 1 SKIP** (vs V24's 17/4/1/0):
- `deep_stack_ood_guard` still FAILS (moved to a 15bb cell, was 40bb in V22-V24 -- the persistent
  pattern of this check's exact failing cell moving between runs without ever clearing, continues).
- **NEW regression**: `vpip_adapts_to_style` FAILS (short delta only +1.9pts, needs >=5pt; deep
  +6.1pts, passes). Every prior version (V22/V23/V24) passed this with deltas of 9.0-15.6pts. The
  root cause is visible in the raw VPIP numbers: hero now enters 61-75% of hands almost
  regardless of opponent tightness (`bb100_vs_standard_fields`: loose_short VPIP 70%, tight_short
  VPIP 75%) -- an extremely loose, largely undiscriminating range. Real win-rate held up anyway
  (`bb100_vs_standard_fields` PASS across all 4 fields, +27.6 to +42.8 BB/100; `beats_offformula_
  stress` PASS) -- so this is still a genuinely profitable strategy in aggregate, just one that
  stopped adapting ENTRY decisions to who's at the table, likely because opponents now fold to
  raises so readily that a wide-and-aggressive default works almost everywhere.
- `beats_frozen_predecessor` SKIPped (no frozen checkpoint saved for this quick diagnostic,
  expected/not meaningful to fix here).

**This diagnostic changed FIVE things at once** (bluff_perc, RAISE_RESPECT_BOOST, fixed vs mixed
stack depth, bootstrap on/off, 50k vs 150k hands) -- it cannot cleanly attribute how much of the
improvement (or the VPIP regression) came from the bluff mechanism specifically vs. the simplified
curriculum/faster convergence in general. That ambiguity is intentional for a first pass ("can
ANYTHING move the needle") but means the result doesn't directly hand over a next calibrated value.

### Verdict

**The opponent-response-shaping approach is NOT a structural dead end** -- pushed hard enough, it
demonstrably produces real, multi-dimensional behavioral change (the first raise-bucket argmax win
and three newly-load-bearing features in this entire investigation). But the specific setting
tested here is not production-safe (a genuine new regression, VPIP no longer adapting to style),
and the result is confounded across five simultaneous changes. Two, not mutually exclusive, paths
forward:
1. Find a calibrated point between V24's 0.10 (too weak, no movement) and this run's 0.40 (moves
   the needle, but overshoots into a VPIP regression) -- and isolate which of the five
   simultaneous changes here is actually doing the work (a cleaner follow-up would vary ONE of them
   at a time against the fixed-35bb/no-bootstrap/50k baseline established here).
2. Also pursue the user's own, independently-motivated hypothesis (2026-07-18 discussion, not
   addressed by this diagnostic): `_mc_target_evs_sized`'s `ev_if_called` formula treats a called
   raise as a terminal, single-street outcome, with no representation of the value a raise
   preserves by keeping future streets' betting alive vs. an all-in that forecloses that option
   entirely. This is a plausible, structurally distinct, and likely complementary root cause --
   worth attacking directly rather than relying solely on opponent-response tuning to compensate
   for a target-EV computation that doesn't represent the advantage a raise is supposed to have.

Not recommended for deployment (this was never the goal -- diagnostic only). No frozen backup
needed (not a candidate for live testing).

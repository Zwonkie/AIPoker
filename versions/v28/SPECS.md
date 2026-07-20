# V28 SPECS

Branches from `versions/v27` (fresh weights, not resumed -- per [VAL-5]). SAME architecture,
contract, opponent pool, and inherited fixes as V27 (context_dim=44, contract_version=7, the
multi-street `_rollout_continuation_ev` fix, the two TreeOpponent real-data seats, the [VAL-3]/
[OPP-7] fixes -- see `versions/v25/SPECS.md`, `versions/v26/SPECS.md`, `versions/v27/SPECS.md` for
those, not re-litigated here). V27 -- not V26 -- was kept as the base per explicit user decision
despite showing a real regression cluster in its own results (see `versions/v27/SPECS.md`
"Results"): both of V27's fixes were verified correct in isolation before training, and the
principle applied was to judge the CODE CHANGE, not solely the aggregate training outcome.

V28 changes ONE thing: adds a closed-form risk/variance penalty to `_mc_target_evs_sized`'s
per-size EV target, targeting the [BET-1] shove-preference problem directly at its now-best-
understood root cause.

## What changed: risk-adjusted (variance-penalized) sized-action target

**Motivation**: a permanent `allin_vs_nextbest_qgap` `model_verify` check (added during this same
diagnostic effort, first run against V26/V27) showed the all-in-vs-next-best Q-gap gets WORSE the
deeper the stack (V27: 15bb=+0.35 -> 40bb=+0.61, worst-cell fraction of pot) -- backwards from
correct theory. Investigated a "range narrows on a bigger bet" hypothesis first; ruled it out for
training specifically (the simulator has full information -- oracle equity from real dealt cards --
so there's no hidden range to narrow; verified the math is internally consistent for a single fixed
opponent hand). The better-supported lead: `_mc_target_evs_sized`'s target is a raw point-estimate
EV with NO risk/variance penalty, and because a bigger bet's `raise_size` scales with stack while
its outcome variance scales too, the same marginal equity edge produces a linearly bigger raw EV
number at deeper stacks, with nothing counteracting all-in's much higher variance than a smaller
raise. Confirmed via code research: no variance-aware mechanism exists anywhere in this codebase
(`target_clip_bb` is a flat, symmetric magnitude clamp, not a variance-based penalty) -- this is a
genuinely new but well-motivated concept, not a rediscovery of an existing lever.

**Design**: for a given sized action, the outcome is a 3-point discrete mixture -- fold (prob
`p_all_fold`, net `pot`), call-and-win (prob `(1-p_all_fold)*true_equity`, net
`base_pot_if_called - raise_size`), call-and-lose (prob `(1-p_all_fold)*(1-true_equity)`, net
`-raise_size`). `Var[X] = E[X^2] - E[X]^2` is closed-form from these three ALREADY-COMPUTED
quantities -- no new sampling. Verified the decomposition's `E[X]` algebraically matches the
existing `p_all_fold*pot + (1-p_all_fold)*ev_if_called` line exactly (confirmed via a 20-trial
randomized unit test, not just asserted). `risk_adjusted_ev = raw_ev - RISK_AVERSION_COEFFICIENT *
sqrt(Var[X])`, applied UNIFORMLY to every sized action (raise_33/raise_66/raise_pot/allin alike) --
deliberately NOT an `is_allin`-special-cased patch, unlike [BET-1]'s earlier `RAISE_RESPECT_BOOST`
attempt. All-in's `raise_size` is far larger than a smaller raise's, so its variance (which scales
with `raise_size^2`) is naturally far larger too -- the same coefficient penalizes all-in the most
because its outcome genuinely IS riskier, matching the user's own stated preference for emergent
fixes over hand-tuned per-action patches.

**Implementation**: `SixMaxSimulator._outcome_variance` (new static method, `simulator.py`) computes
the closed-form variance; `_mc_target_evs_sized`'s raise-size loop subtracts the penalty before
`evs.append(...)` when `self.risk_aversion_coefficient > 0`. New config knob
`risk_aversion_coefficient` (default semantics: 0.0 = off, matching every prior version's implicit
behavior) threaded through the full chain: `config.yaml` -> `main()`'s `t_cfg.get(...)` ->
`run_training(...)` -> per-batch worker `args` tuple -> `simulate_worker(...)` ->
`sim.risk_aversion_coefficient` (assigned unconditionally, never `hasattr`-guarded -- the
`policy_temperature` bug precedent showed a guarded assignment on a never-pre-declared attribute
silently no-ops).

## Calibration (2026-07-19, before any training)

Standalone script (scratchpad, mirrors `versions/v23/self_play/calibrate_bet1.py` /
`versions/v25/self_play/calibrate_multistreet_ev.py`'s established pattern): called
`_mc_target_evs_sized` directly (stubbing `_calculate_equity` to a fixed target value, matching the
same controlled-probe convention this investigation's own earlier `RAISE_RESPECT_BOOST` probe
used), averaged over 40 trials per cell (a single call is too noisy to read -- both
`bot.start_new_hand()`'s trait fuzzing and the fold-decision's own internal sampling are
stochastic; an unaveraged first pass of this script produced misleading single-draw numbers, caught
before choosing a coefficient).

Swept coefficients 0.0-0.20 across the diagnosed problem grid (eq 0.35/0.45/0.55, stack 15/25/40bb)
and a value-shove sanity spot (eq=0.85, stack=15bb):

| coef | eq=0.55, 15bb | eq=0.55, 25bb | eq=0.55, 40bb (worst cell) | eq=0.85, 15bb (value shove) |
|------|------|------|------|------|
| 0.00 | +0.18 | +1.14 | **+2.68** | +2.80 |
| 0.05 | -0.20 | +0.25 | +0.99 | +2.66 |
| 0.08 | -0.46 | -0.23 | +0.07 | +2.57 |
| **0.10** | **-0.62** | **-0.68** | **-0.60** | **+2.51** |
| 0.20 | -1.10 | -2.89 | -4.51 | +2.23 |

**Chose 0.10**: fully flips the worst diagnosed cell from strongly all-in-favoring to clearly (not
just barely) all-in-disfavoring, while the legitimate value shove barely moves (+2.80 -> +2.51, a
~10% reduction) -- 0.20 achieves a similar worst-cell fix but starts pushing OTHER cells to large
negative numbers (-4.51 and beyond), risking suppression of genuinely fine shoves elsewhere that
weren't part of the diagnosed problem. `risk_aversion_coefficient: 0.10` set in `config.yaml`.

## Verification (pre-training)

- Randomized unit test (20 trials, random equity/pot/stack/raise_size combinations): confirmed
  `_outcome_variance`'s 3-point mixture's `E[X]` matches the existing `p_all_fold*pot +
  (1-p_all_fold)*ev_if_called` formula exactly (not just algebraically argued), and `Var >= 0`
  always.
- Calibration table above -- see full averaged sweep output for every tested cell, not just the
  headline row.
- `target_hands: 100000` (matches V27's own scale), fresh weights, no `--resume_path`.

## Results (2026-07-19, `expert_main.pth`, 100k hands complete)

**`model_verify --full`: 19 PASS, 4 WARN, 1 FAIL, 0 SKIP** (`tools/model_verify/results/v28__expert_main.json`,
saved to `.agents/skills/OFK/references/V28/model_verify_report.html` per the standing convention).

**The core targeted metric, `allin_vs_nextbest_qgap` [BET-1], improved meaningfully and the
pathological pattern is GONE**:

| | V27 (before) | V28 (after) |
|---|---|---|
| by stack 15/20/25/30/40bb | +0.35/+0.40/+0.46/+0.52/**+0.61** | +0.23/+0.24/+0.26/+0.28/**+0.27** |
| by archetype NIT/TAG/LAG/STATION | +0.18/+0.17/+0.15/+0.13 | +0.10/+0.09/+0.08/+0.06 |

V27's gap MONOTONICALLY WORSENED with stack depth (the exact backwards-from-theory pattern that
motivated this version) -- V28's is roughly FLAT across depths, and every single cell shrank by
~40-50%. This is exactly the shape of change the risk-adjustment was designed to produce.

**V27's regression cluster substantially resolved**:
- `action_diversity`: V27 `{'fold':9,'allin':12}` (2 actions) -> V28 `{'fold':9,'call':2,
  'raise_33':1,'allin':9}` (4 actions) -- call and raise_33 both win argmax cells again.
- `stack_full_sweep`'s argmax path: V27 was all-`allin` across the full 5-180bb sweep -> V28 is a
  coherent `call`->`raise_33` progression with ZERO all-in wins in this sweep.
- `position_sweep`: V27 WARN (spread 0.022, nearly flat) -> V28 PASS (spread 0.653 -- even better
  than V26's own 0.378). The V27-introduced regression is resolved, not just partially recovered.
- `deep_stack_ood_guard`: still FAILS (the standing target every version since V19 has failed to
  clear), but at meaningfully lower confidence (V27: ALL-IN argmax @ 0.33 -> V28: @ 0.24).
- VPIP came down from V27's ~40-48% toward ~29-47% (`vpip_adapts_to_style`: still PASS, deltas
  +7.1/+6.5pts, comparable to V27's +8.1/+5.8pts) -- not all the way back to V26's ~16-26%, but a
  real partial reversal of V27's doubling.

**Real, direct win**: `beats_frozen_predecessor` PASS, **+29.7 BB/100** over 4000 hands vs a field
including a frozen V27 snapshot (`versions/v28/weights/frozen_v27.pth`) -- smaller margin than V27's
own +40.2 vs V26, consistent with the risk penalty trading away some raw exploitative aggression for
a structurally sounder policy, not a like-for-like comparison of the same lever.

**Still weak, not fully resolved**: `beats_offformula_stress` deep-stack (+1.5 BB/100, barely PASS,
similar to V27's own +5.6) -- the risk-adjustment didn't meaningfully help this specific field.
`allin_exploits_opponent_foldiness` [OPP-8] remains WARN (spread 0.022, still low) -- this fix
targets a different mechanism (target-EV risk-awareness, not opponent-foldiness perception) and was
never expected to move OPP-8's own metric.

**Verdict**: a clean, validated win on the specific problem this version targeted -- the Q-gap
shrank, flattened across stack depth, and several of V27's collateral regressions resolved
alongside it, all without a fabricated all-in-specific patch (the penalty is uniform across every
sized action, all-in is simply the biggest natural beneficiary of the correction because its
variance genuinely is largest). `deep_stack_ood_guard` remains the one standing failure this whole
BET-1 lineage has never cleared -- worth its own dedicated investigation, separate from BET-1's
general shove-preference framing (per the same open note every version since V25 has carried).

## Status

Training complete and verified 2026-07-19. **DEPLOYED LIVE 2026-07-19** per explicit user request --
`core/decision.py`'s `active_model_name` is `'Herocules (v28)'`. V25/V26/V27 remain in the registry
as rollback options.

See `versions/v27/SPECS.md` ([VAL-3]/[OPP-7] + the regression cluster) | `versions/v26/SPECS.md`
(opponent pool) | `versions/v25/SPECS.md` (EV-fix mechanism) |
`.agents/skills/OFK/references/known-shortcomings-backlog.md` ([BET-1])

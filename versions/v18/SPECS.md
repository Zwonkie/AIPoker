# V18 — Roadmap / consolidated backlog

Planning doc only (2026-07-15) — no code yet. **V17 is now deployed live**
(`core/models/v17_engine.py`, `Herocules (v17 Actor-Critic)` in `core/decision.py`; V15/V14/V13 kept
as fallbacks). This carries forward everything from `versions/v17/SPECS.md` that wasn't addressed by
the actor-critic/fold-relative fix, per the plan agreed when V17 shipped ("push the other change
suggestions to V18").

## [P0] Deep-stack OOD trash-jam — top priority, now 4 versions running unaddressed

`deep_stack_ood_guard` FAILS on V15, V16, v16_foldregret, AND V17 — the exact same failure signature
(eq≈0.55, 15-40bb stack, single modest bet -> ALL-IN argmax) has now survived four consecutive
versions as a side effect of unrelated work each time. Needs a DEDICATED pass, not another incidental
fix: revisit the critic's `target_clip_bb=40` behavior at depth (V15 flagged Q-loss running high at
30-50bb as a known-not-urgent gap — may be the same root cause), add explicit deep-stack
decision-quality curriculum weight, or a dedicated diagnostic sweep at the exact incident conditions
as a first-class TRAINING signal, not just an eval-time check. This is the single most-carried,
least-addressed item in the whole V14→V17 line.

## [P5]/[P6] Input-contract gaps — promote over further target-formula tuning

Three versions running (P4, v16_foldregret, V17) have now squeezed real gains out of tuning the
actor's TARGET/regret-matching mechanism. The model still has no encoding of who raised, how many
opponents raised, or bet-size patterns in the action history (`act` tensor is hero-only; history
tokens are size-blind, `to_call` is /400-normalized so small raises are near-invisible). This is a
genuinely new information channel, not another reweighting of existing signal — likely a cleaner
lever than continuing to iterate on the target/loss side, which is showing diminishing returns
(each round trades one field/style for another rather than a clean win). Contract bump + retrain;
bundle P5 (bet-size perception) and P6 (opponent-action attribution) together since both touch the
same input-contract revision. Full detail in `versions/v16/SPECS.md` [P5]/[P6] (unchanged since).

## [P7] Opponent-pool NN personalities / "Yellow"-LAG gap — still backlog, lower priority

Checked empirically (V15): the live model's response across the untrained VPIP=0.26-0.35 gap is
smooth/monotonic, so not an urgent correctness bug. Mechanism to close it properly (dedicated NN
opponent personalities via the dormant `hero_personality` forcing path) is fully scoped in
`versions/v16/SPECS.md` [P7] if it becomes worth prioritizing later.

## MC equity_sims budget — ANALYZED (2026-07-16), recommend reverting to 2000

V17 bumped `equity_sims` 2000->5000 mid-experiment, on the assumption it runs on CUDA and would cost
"not much more." That assumption was wrong for the path that actually matters: `_calculate_equity`
(`versions/v17/self_play/simulator.py`), when called with `specific_opponents` (the oracle-equity
path `_mc_target_evs_sized` uses for EVERY decision's target, and the ONLY path exercised at that
call site — `_cuda_evaluator` is explicitly skipped whenever `specific_opponents is not None`), runs
`core/evaluator.py`'s `calculate_equity`: a pure-Python `for _ in range(num_simulations)` loop doing
`random.sample` + `treys` hand evaluation — entirely CPU-bound, no GPU involvement at all.

**Measured directly** (`core/evaluator.py`, 20-call average per sample size, representative flush-draw
spot vs a specific opponent hand):

| sims | ms/call | relative |
|---|---:|---:|
| 500 | 8.2 | 1.0x |
| 2000 | 32.3 | 4.0x |
| **5000** | **79.5** | **9.7x** |
| 10000 | 157.5 | 19.3x |

Scales linearly with sims, as expected for this loop shape. 2000->5000 is a measured **2.46x** cost
increase per call — and `_calculate_equity` is called MULTIPLE times per decision (hero oracle
equity + once per active opponent for `opp_base_eq`, so 3-5+ calls in a typical multi-way spot),
which compounds this and fully explains the observed training slowdown (~20-50 -> ~11-12 hands/sec).

**Noise reduction actually achieved** (60 repeated calls per sample size, same spot):

| sims | measured std dev | theoretical SE |
|---|---:|---:|
| 2000 | 0.87% equity-pts | 0.98% |
| 5000 | 0.55% equity-pts | 0.62% |

Real, but modest (~0.3 percentage-point absolute reduction) — matches the theoretical
`1/sqrt(n)` binomial scaling almost exactly, no surprise extra variance source.

**Verdict:** this modest per-hand noise reduction is very unlikely to have been the reason V17
worked. The validated mechanism (see `versions/v17/SPECS.md` "Round 2") is fold-relative
regret-matching applied to the CRITIC's Q-values — and the critic gets its own, far more powerful
denoising for free, by regression across THOUSANDS of hands over many epochs (the entire point of
routing the actor through it instead of a fresh per-hand sample). A <1-percentage-point reduction in
one hand's equity noise is small next to that. Recommend **reverting `equity_sims` to 2000 for V18**
(the cost is real and roughly linear, the benefit is marginal and likely subsumed by the critic's own
averaging) rather than spending a full comparison training run on a question this direct measurement
+ mechanism reasoning already answers with reasonable confidence. If V18 wants extra insurance at low
cost, 2500-3000 is a defensible middle ground (~1.25-1.5x cost, ~15-20% less noise than 2000) — but
plain revert to 2000 is the recommended default absent a specific reason to pay more.

## Also worth carrying into V18

- **Widen the frozen-opponent pool.** Every version in this line has used a SINGLE frozen predecessor
  in the `past` seat. A gauntlet (multiple frozen checkpoints, or frozen-V15 + frozen-V16 + frozen-V17)
  would test generalization instead of just "beats the immediate parent."
- **model_verify weighted composite score** (carried from V17 SPECS, still not built): a score
  weighted toward the actual live field mix (loose-heavy) so a tradeoff like foldregret's loose-deep
  collapse surfaces in the summary line, not just by manually cross-referencing the raw per-field table.

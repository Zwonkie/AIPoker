# V48 SPECS — true short-handed geometry: variable seat count 3–6 (DRAFT)

**Status**: REVISED 2026-07-22 (evening) after V47's gate review — V47 landed 20/7/0, head-to-head
PARITY with V44, DEPLOYED live by user decision; its findings are folded in below. IN BUILD.
**Base**: clone of `versions/v47` (confirmed — its realism changes and healed sensitivities are
keepers per the gate review), fresh weights, no
`--resume_path` ([VAL-5] stands). **Contract unchanged** (`context_dim=54`, `contract_version=9`):
the 5 opponent slots + `is_active` flags + `num_active` + per-seat stacks already REPRESENT a
3-handed table — what is missing is simulator truth, not tensor width. Every change below is
simulator/curriculum-side, so the V48 engine is again a declaration-only clone and deploy is
one registry line.
**Target**: 100k hands fresh, `model_verify --full` under the Phase-0/P48-0 harness, mirrored-deal
paired head-to-head vs the frozen predecessor.

## Why this scope

Live play is Double-or-Nothing: tables START 6-handed and SHRINK to 3–4 as players bust — the
bubble, where every live "too tight / too loose" complaint concentrates, is short-handed AND
short-stacked. The simulator has only ever dealt a 6-seated table, approximating smaller fields by
pre-folding seats (`num_to_fold`). That gets the opponent COUNT roughly right but not the
GEOMETRY: a real 4-handed table has a faster blind orbit (blinds hit ~50% more often), a
compressed position map, and no dead stacks parked behind folded seats. Hero has never once
trained on the geometry it plays the most important phase of every session in.

Deliberately NOT this version (each needs isolation): ICM [FMT-1] (a TARGET-definition change —
a later version, after this version establishes the chip-EV 3-max baseline it will be measured
against); [OPP-3] size-aware action history (a contract bump, its own version — more urgent
post-V47 since opponents now produce real sizes hero cannot remember); [OPP-6] exploiter,
[OPP-8], replay buffer.

---

## V47 findings this version must answer (added at revision, 2026-07-22 evening)

- **[W1] `opponent_style_sweep` went FLAT** (fold-spread 0.127→0.027; isolated-ablation
  table-scalar TV 0.750→0.019). Hypothesis: occupant-true fold models ([M4]) made
  training-time folding depend on what is actually seated, homogenizing the archetype signal —
  a realism↔exploitability tension. V48 mitigation: keep occupant-true folding (it is CORRECT)
  but re-widen the POOL's behavioral spread using population fitting (Change 1b below) so
  distinct archetypes are genuinely distinct in-data; gate: the sweep must not stay at 0.027.
- **[W2] ≤5bb trash-looseness is lineage-wide** (identical Nash disagreement cells V44↔V47:
  92s–94s@5bb overplayed, T2s–T4s@5bb called vs jams). The sub-5bb curriculum band alone did
  not fix it; the 3-max Nash axis (P48-0.1) plus the bubble-band curriculum (Change 2) are
  this version's levers; track the cells explicitly.
- **[W3] Hold the V47 wins**: position_sweep spread 0.948, committed/pot_type sensitivities
  PASS, vpip_adapts +6.2/+7.9 — all now part of gate 4's regression watch.
- **[W4] `short_stack_polarization`** slipped 0.19→0.22 (WARN) — watch under the new curriculum.

## Change 0 — Generalized chip-identity collapse [now UNCONDITIONAL]

The fable resolution log (Tier 6, T-M9) measured the min-raise floor swallowing every pot
fraction — `raise_33/66/pot` = the SAME min-raise chips — on **40.7% of decisions / 56% preflop**,
vs the 2.4% all-in aliasing V47's Change 2 collapses. Duplicated buckets also TRIPLE-COUNT one
physical action's regret in the actor-target normalization: a structural min-raise over-weight at
preflop entry (a plausible contributor to the historical VPIP-inflation failure class).

**Design**: generalize V47's machinery — group raise buckets by RESOLVED chip amount at the node;
one canonical per group (ALLIN for the shove group, lowest-index bucket otherwise); duplicates get
the canonical's exact EV, zero actor-target regret, and the serve-side sampler mask (the
`collapse_aliased_allin` engine flag generalizes to `collapse_aliased_buckets`). Same
train≡serve invariant pair, wider trigger.

**Sequencing rule OUTCOME (2026-07-22 evening)**: V47 landed with NO VPIP inflation and the
post-run measurement (`probe_minraise_aliasing.py`) found no pathology — V47 is actually
slightly TIGHTER at aliased cells (raise mass 0.242 vs V44's 0.274; intra-group Q spread ratio
0.01) despite ~73% curriculum-weighted preflop prevalence (geometry-driven: 33%-pot floors to
the min-raise at ANY depth). So no V47.1; the collapse ships HERE as a correctness item (the
triple-counted regret is real even without a measured symptom), with the probe re-run
before/after as its own attribution evidence.

## P48-0 — evaluation tooling (version-independent; BEFORE any training run)

- **P48-0.1 Three-handed Nash push/fold solver** (`tools/model_verify/nash/solve_nash_3max.py`):
  extend the [VAL-1] external axis to the geometry this version introduces. 3-max push/fold
  (BTN jam/fold → SB call → BB call/overcall) is still tractably solvable in-repo, chip-EV.
  New FAST checks `nash_3max_*` mirroring the HU pair, scored on literal ALLIN-vs-fold from day
  one (P0.3 discipline — never re-introduce the composite artifact).
- **P48-0.2 True-table-size FAST sweeps**: today's `multiway_shortstack_aggression` varies
  `num_active_opp` INSIDE 6-seat geometry. Add sweeps where the TABLE itself is 3/4/5/6-handed
  (inactive trailing slots, correct position encoding) so a geometry-blind model is visible
  before any SLOW run.
- **P48-0.3 SLOW checks gain a `table_size` axis**: `_run_field`/head-to-head accept a seat-count
  mix so the paired head-to-head and the standard fields can be run 6-handed (comparable to every
  prior version) AND on the DoN mix (the new claim under test). Baselines recorded for both.

## Change 1 — True N-handed dealing (simulator geometry)

**Design**:
- `SixMaxSimulator` deals a table of N ∈ {3,4,5,6} live seats per hand (curriculum-sampled, see
  Change 2): N seats receive cards/stacks/blinds; the remaining slots are ABSENT (not "folded") —
  `is_active=False`, stack 0, excluded from blind rotation and position arithmetic.
- Blind orbit and button rotation over the N occupied seats only; position features encode the
  compressed map (the live path already handles occupied-ring positions since V42's #12-CE fix —
  the simulator now matches it; verify the encodings agree EXACTLY, that pairing is a new
  train≡serve surface).
- The pre-fold mechanism is KEPT alongside (both occur in reality: a 6-seat table where 3 fold is
  a different node than a 3-seat table — different dead money, same live count). Curriculum
  covers the joint.
- Opponent pool/`STYLE_SLOT` stat bucketing unchanged — styles fill however many seats exist.

### Change 1b — Population fitting from the hand store (NEW at revision; answers [W1])

The hand-history corpus (4,015 hands / 183 sessions / ~100 recurring opponents with 40+ shared
hands, `live2/historydb`) replaces hand-authored guesses with the measured population
(FITTED 2026-07-22, hero excluded, `history/population_fit.json`):
- **Measured mixture**: TAG 0.19 / LAG 0.30 / Nit 0.20 / Calling Station 0.30 (99 players).
- **Measured signature**: preflop raises are BIG (pot-plus to jam; sub-pot fractions barely
  exist preflop — NB fit semantics are amount/pot-BEFORE, map to the sim's
  `_raise_size_for_fraction` pot-after-call semantics when copying tables); postflop bets are
  SMALL (0.33/0.50-pot dominate every cluster). Cluster spread is real: Station limps 21.8%
  with AF 0.63 while Nit jams 46.6% of its raises at VPIP 10.8.
- **Fit `RAISE_SIZE_DISTRIBUTIONS` per archetype** from the fitted histograms (the C1 harness
  re-scores the fitted tables); **fit limp rates and the archetype mixture** for pool weights.
- **Widen behavioral spread deliberately**: [W1]'s mitigation — verify the fitted bots
  reproduce the measured spread in C1 fold-curve/size terms before training.

**Verification before training**: geometry probe — over 2k hands per N: blind frequency per seat
≈ 1/N ± tolerance; position feature range matches occupied ring; zero "absent seat acts" events;
encoder parity spot-check vs the live `to_observation` path for a shrunk table (V42 #12-CE
convention), asserted tensor-level per the V41 #11 lesson (verify what the ENCODER sees).

## Change 2 — DoN-shaped joint curriculum (seat count × stack depth)

**Design**: replace the independent stack-depth mixture with a joint distribution shaped like a
DoN's life cycle: 6-handed × deeper early; 4–5-handed × mid; 3–4-handed × short (the bubble
band, where [VAL-1(A)]'s 2–8bb work from V47 concentrates). Weights MEASURED from the hand
store's real DoN life-cycles (seat-count-over-blind-level trajectories per session — resolved
at revision, was an open question); overlapping bands per [STACK-2] precedent. `stack_depth_mix`
stays honored when a config sets it flat (verify-mode compatibility).

**Verification before training**: distribution check on 5k simulated hands (joint densities as
configured); confirm the marginal stack-depth distribution does not regress V47's 2–5bb coverage.

## Change 3 — MOVED to `versions/v49_liveRebuild` (revision 2026-07-22 evening)

The entire live-side scope of the earlier draft (3a capture/OCR overhaul, 3c riders) moved to
the **v49_liveRebuild** slice (`versions/v49_liveRebuild/SPECS.md`) — the live layer is being
rebuilt around the LiveObservation boundary in a new `live2/` root rather than patched in
place, motivated by the flagged JJ-fold case study (phantom timer-seat + three-components-
three-prices; see resolution-log). 3b's ingestion is BUILT (`live2/historydb`): decoder, XML
backfill (4,015 hands), windowed stats engine, population fitting, live watcher — validated
against a real session same-day. V48 is a pure TRAINING version. (The superseded live-side
draft text lives in this file's git history / the earlier draft.)

Scope guard (still applies to anything live-touching): a change that would ALTER a live input
feature's distribution (the V42 lesson: that is a serve-side contract change) ships behind its
own before/after measurement and is individually revertible.

---

## Acceptance gates (run under the P48-0 harness)

1. **Mirrored-deal paired head-to-head vs frozen predecessor**, CI excluding 0 — run BOTH
   6-handed (comparability) and on the DoN seat mix (the point of the version).
2. **Hold V47's gate set** at 6-handed: `vpip_adapts_to_style` ≥ +5pts both depths,
   `multiway_shortstack_aggression` PASS, `deep_stack_ood_guard` PASS,
   `allin_vs_nextbest_qgap` non-positive worst cells, VPIP level vs predecessor
   (Change-0-sensitive — the aliasing probe re-run is the attribution evidence).
3. **The point of the version**: `nash_3max_*` agreement at or above the HU checks' level;
   true-table-size sweeps show geometry sensitivity (blind-pressure-aware, not flat across N);
   head-to-head on the DoN mix at least matches the 6-handed result (the shrunk-table phase must
   not be subsidized by full-ring play).
4. **Regression watch** (now anchored to V47's actual numbers, see [W3]): `committed` (0.030) /
   `pot_type` (0.167) sensitivities stay PASS, `action_diversity`, `position_sweep` spread near
   0.948 (the compressed map must not flatten it), HU Nash checks (3-max training must not
   degrade the 2-max axis), `opponent_style_sweep` must move OFF 0.027 ([W1] — the Change-1b
   claim under test), [W2] Nash cells tracked, `short_stack_polarization` ([W4]).
5. Standard: full report to OFK `references/V48/model_verify_report.html`; backlog statuses
   updated in place ([OPP-4], live seconds-tier items, [VAL-1] 3-max extension, T-M9 remainder).

**Rollback**: predecessor stays active until gates pass and the user calls it; V41 remains the
MILESTONE fallback. Deploy = declaration-clone engine + one registry line.

## Suggested build order (revised)

`live2/historydb` query layer (DONE) → Change 1b population fitting (DONE, see above) →
P48-0 (3-max solver + geometry checks) → Change 0 (+ aliasing probe re-run) → Change 1 + geometry
probes → Change 2 + distribution checks (weights measured from the hand store) → 10k smoke
(blind-frequency telemetry, hands/sec) → 100k fresh → gates → OFK backfill + review vs this
document. v49_liveRebuild proceeds in parallel during the 100k (webapp → service → assembler);
its shadow sessions double as V47's live validation.

## Open questions (user calls)

1. Seat-count floor: 3 (current draft) or include true heads-up 2? HU maximizes the Nash axis's
   value but is OOD for DoN until the final two; including it doubles the geometry test surface.
2. ~~Joint-curriculum weights source~~ RESOLVED at revision: measured from the hand store
   (4,015 hands of real DoN life-cycles — seat-count-over-blind-level trajectories per session).
3. [OPP-9] — now a v49-scope question, deferred there (v49 SPECS lists it as a non-goal for the
   rebuild slice; it remains backlog-tracked).

---

## Build log (2026-07-22)

- Slice cloned from `versions/v47` (weights: `frozen_v47.pth` = v47 expert_main; v47 training
  logs/checkpoints excluded). `live2/historydb` built + population fitted BEFORE the clone.
- **Change 0 DONE**: generalized chip-identity grouping in `_mc_target_evs_sized`
  (`collapse_by_chips_all` flag, canonical = ALLIN for shove group else lowest index, EV copy
  + aliased flags; V47 allin-only kept as fallback); serve mirror in core/decision.py behind
  NEW engine flag `collapse_aliased_buckets`; `core/models/v48_engine.py` created (declares
  buckets flag, NOT registered). 30-hand smoke: 60 aliased flags, 0 errors.
- **Change 1b DONE**: STREET-SPLIT fitted repertoires in opponent_bots.py
  (`RAISE_SIZE_DISTRIBUTIONS_PREFLOP` + postflop verbatim from the fit), street stash
  `_betting_street_idx` at the betting-loop call site (no signature change -- C2 subclass
  compat); pool weights = measured mixture (maniac .23 / fish .23 / tag .14 / nit .15,
  past .25). C1 re-run: no degenerate shapes.
- **Change 1 DONE**: `present` seats + `present_ring`, button/blinds via `_next_present`
  orbit, absent seats stack-0/inactive/never-act, pre-fold + live_players operate on present
  ring, ring-relative `actor_position` in `_build_query_board_state` (falls back to 6-ring
  when no ring supplied), `_sample_table_size` + `table_size_mix`. Geometry probe: positions
  exactly 0..N-1 at every N in {3,4,5,6}, ring math 5k-draw clean per N, 0 query errors.
- **Change 2 DONE**: `table_stack_joint_mix` (19 measured rows; joint precedence over
  stack_depth_mix via `_joint_depth_bb` stash consumed by `_get_starting_stack`), wired
  config→run_training→worker starmap→sim. 5k-draw check: seat marginal {3:.009 4:.229
  5:.250 6:.511} vs target {.010 .235 .247 .505}; depth 2-5bb marginal 0.06 (the measured
  real value; V47 reference 0.08 -- [W2] watch). MEASUREMENT NOTE: DoN ends at 3 players →
  3-handed only 1% of real hands, true HU 0.2% — settles open question 1 (floor stays 3).
- **verify_v47_inherited: 29/29** after updating the two C1-wiring probes to the FITTED
  expectations (station DOES jam in the measured population -- old "never jams" was
  hand-authored fiction).
- **P48-0.1 IN PROGRESS**: `tools/model_verify/nash/solve_nash_3max.py` written (smoothed
  stochastic FP; reuses HU pairwise matrix; lazy 3-way MC cache; anchors) -- solver running
  (3-way equity cache building, ~18MB @ 19:40).
- **P48-0.2 DONE**: `check_table_size_sweep` + `nash3_btn_jam`/`nash3_bb_call` registered in
  FAST_CHECKS (`tools/model_verify/nash/pushfold3_check.py`). Extended FAST battery on
  frozen_v47: 17 PASS / 7 WARN / 0 FAIL / 2 SKIP -- table_size_sweep correctly reads V47
  weights as geometry-blind (agg spread 0.015 across 3..6-handed); nash3 checks SKIP until
  the solver lands.
- **P48-0.3 DONE**: SLOW checks run BOTH table-size axes. `_run_field` + the head-to-head's
  `_run_leg` accept `table_size_mix`; `_don_seat_mix(rc)` derives the marginal ({3:.010,
  4:.236, 5:.248, 6:.507}) from the version's OWN config `table_stack_joint_mix`;
  `_supports_table_mix(rc)` capability-checks that the sim PRE-DECLARES the attribute
  (dead-attribute guard -- pre-V48 sims keep the single 6-handed axis, no silent no-op).
  `bb100_vs_standard_fields` records 8 baseline keys (`<field>` + `<field>@donmix`);
  `beats_frozen_predecessor` runs both paired head-to-heads, GATES on the 6-handed pair
  (cross-version comparability) and downgrades PASS→WARN if the DoN-mix pair decisively
  disagrees. Micro-run (40 hands) verified both axes execute and collect.
- **10k smoke PASSED**: 13.3 hands/sec, val loss 12.05→2.58 monotone, hero -13.3→+0.9
  BB/100, action entropy 1.39, equity-monotone action table (fold 88% air / jam 72% nuts).
- **100k fresh RUN** launched 19:27 (2026-07-22); orphaned at 43,583 hands by a harness
  restart, then PARKED overnight by user instruction.
- **P48-0.1 DONE (2026-07-23 ~01:00)**: 3-max solver completed all four stacks, `anchors OK`
  (4.26M cached 3-way triples; two crash-fix iterations along the way: atomic+non-fatal
  cache flush after a transient OSError-22, then flush interval scaled with cache size —
  fixed-interval flushing was ~40% of wall time by the 100MB mark). `nash_3max_solved.json`
  49KB. FIRST SCORED nash3 BASELINE on frozen V47 weights: `nash3_btn_jam` **74%** WARN,
  `nash3_bb_call` **73%** WARN (threshold 0.75) — e.g. BTN folds AJs–A5s@5bb that 3-max
  Nash jams. Extended FAST battery: **17 PASS / 9 WARN / 0 FAIL / 0 SKIP** (every check
  scores now). NOTE: equity3_cache.json (224MB) is gitignored — regenerate via the solver
  if ever lost; nash_3max_solved.json IS tracked.
- **100k RESUMED (2026-07-23 ~01:05)** from checkpoints/main_hands43583.pth with full
  optimizer/scheduler state (verified by startup notices).
- **100k COMPLETE (2026-07-23 ~04:00)**: 43,583 + 56,418 = 100k hands, final val loss 5.91,
  hero +76.1 BB/100 cumulative vs training pool, equity-monotone final action table (air
  fold 92.6% → nuts jam 78.1%), action entropy 0.708.
- **Aliasing probe DONE (Change 0 attribution)**: prevalence unchanged as expected
  (structural: 73.2% of curriculum-weighted preflop raise decisions sit at min-raise-aliased
  cells). Q-coherence CLEAN: intra-group Q spread 0.081 vs inter-action 3.512 (ratio 0.02;
  V47 control 0.01) — the collapse did NOT introduce the memorized-inconsistency signature.
  Behavior at aliased cells shifted looser: fold 0.642→0.382, call 0.116→0.307, raise mass
  0.242→0.310 (probe output labels the control 'v44_frozen' — cosmetic leftover, line 155
  actually loads frozen_v47.pth). Verdict on aliased-regret leak: no pathology in V48.
- **model_verify --full --update-baseline DONE (2026-07-23 ~07:30)**: **20 PASS / 9 WARN /
  1 FAIL / 0 SKIP** (report: OFK references/V48/model_verify_report.html; raw JSON:
  tools/model_verify/results/v48__expert_main.pth.json; 8-key bb100 baseline recorded).

## Gate verdict (2026-07-23)

- **[W1] opponent_style_sweep: 0.027 → 0.105 PASS** — the headline gate MET; V47's
  realism↔exploitability flattening is undone. `opponent_color_isolated_ablation` healthy
  (table-scalar TV 0.323 / per-seat 0.739). Caveat: `allin_exploits_opponent_foldiness`
  still flat (0.031, [OPP-8] stands).
- **table_size_sweep: 0.015 → 0.027 PARTIAL** — spread nearly doubled and is monotone
  (3h 0.739 → 6h 0.766) but still under the 0.03 WARN bar; geometry is no longer
  invisible, not yet strongly load-bearing.
- **nash3 vs 74/73 baseline: btn_jam 74% → 79% PASS; bb_call 73% → 73% WARN** — BTN
  first-in geometry learned (Change 1's compressed-ring cells), BB-facing-jam unchanged.
- **beats_frozen_predecessor: PARITY (tie-breaker final, 2026-07-23 ~09:10).** Background:
  two concurrent batteries on the same weights gave contradictory verdicts (+15.7/+23.0 vs
  −28.4/−53.9, ±42-44 CIs; even deterministic FAST checks drifted between them — concurrent
  heavy load corrupts measurements, ALWAYS run batteries solo). SOLO tie-breaker (8000
  paired hands/axis, sha-logged weights 4391a3f3/8ffb8308, CIs ~±30):
  **6-handed −4.1 ±29.8 (dead parity), DoN mix +13.7 ±31.2 (positive lean, not decisive)**.
  Both extreme concurrent measurements were artifacts; truth is parity with a mild edge in
  V48's actual live environment. Same verdict class V47 itself deployed on (+2.6 ±45 vs
  V44). No rollback trigger.
- **bb100 fields (new baselines)**: classic 6-handed +23.9..+132.7; DoN-mix axis notably
  stronger short-stacked (+40.8/+47.1 vs classic +27.6/+23.9) — the joint curriculum shows
  up exactly where it trained.
- **Change 0 semantics visible**: nash_pushfold composite-commits expressed as literal
  ALLIN instead of a sized raise: 897/971 → 202/971 (the collapse routes chip-identical
  raises to the canonical jam action). Aliasing probe clean (Q-coherence ratio 0.02).
- **NEW FAIL: deep_stack_ood_guard** (eq 0.55 @ 15bb → ALLIN argmax 0.34) — REGRESSION vs
  V47's 0-FAIL card (V29 was the first pass; V47 held it). Plausible mechanism: the
  measured joint curriculum concentrates mass at 4-6-handed short/mid depths, thinning
  deep-stack coverage vs V47's uniform 5-50bb. Track before deploy.
- **WATCH: position_sweep flattened** (V47 spread 0.948 → 0.023 WARN) — likely interacts
  with Change 1's ring-relative position encoding at sampled table sizes; needs a
  table-size-conditioned probe before concluding the feature went dead.
- **vpip_adapts_to_style HELD: PASS** (+10.2/+7.7pts) — the V44 [P4] fix survives the
  geometry package.
- **DEPLOYED LIVE 2026-07-23 (morning) by explicit user decision** — with the
  deep_stack_ood_guard FAIL and the UNRESOLVED head-to-head (tie-breaker still running)
  known at deploy time; both recorded in the registry provenance comment
  (`core/decision.py`). Handover parity smoke on the active V48 engine: **14/14 PASS**.
  ROLLBACK: `Herocules (v47)` (one line). pipeline-flow.md invariants updated with the
  generalized chip-collapse train≡serve pair (H3 ↔ LD3 `collapse_aliased_buckets`).

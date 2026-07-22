# V47 SPECS — opponent-behavior realism + target alignment

**Status**: IN BUILD 2026-07-22 — Phase 0 SHIPPED (repo-wide, see "Build log" at the bottom),
slice cloned from v44, Changes 1–5 IMPLEMENTED, pre-training probes 28/28
(`self_play/verify_v47.py`), C1/C2 calibration + smoke/100k training pending.
**Base**: clone of `versions/v44` (guardrails §6: copy the slice, FRESH weights, no `--resume_path`
— [VAL-5] stands). **Contract unchanged**: `context_dim=54`, `contract_version=9` — every change
below is simulator/training-side, so V44's live game-state work, bridge, and `live_features()`
carry over; the V47 engine is a declaration-only clone of `v44_engine.py` (post-v46, that is the
entire live hookup).
**Target**: 100k hands fresh, `model_verify --full` under the Phase-0-corrected harness,
duplicate-deal head-to-head vs frozen V44.

## Why this scope

V40/V41 fixed what the model was allowed to see ([BET-3] bundle: post-check nodes, dead blinds,
asymmetric stacks, NN-opponent inputs); V43/V44 fixed what it was told to want (prior cleanup,
effective-field equity_edge). The largest remaining gap is **what its opponents are allowed to
do** (one raise size, ever) and **whether the targets describe the opponents actually seated**
(heuristic fold-proxies for a 60%-NN/tree pool). Everything here attacks that, plus the two
cheap corrections (bucket aliasing, sub-5bb curriculum floor) sitting in the same neighborhood.

**Deliberately deferred**: variable seat count 3–6 (→ V48 — a state-distribution change; stacking
it on an opponent-behavior change would reproduce the V24_extreme/V27 confound pattern), full ICM
[FMT-1], replay buffer, [OPP-6] exploiter, [OPP-8] fold_pressure_color (HELD), padding rework (L3).

---

## Phase 0 — evaluation tooling (version-independent; DO BEFORE any training run)

The point is that V47's acceptance evidence must be decisive, not directional. All four land in
`tools/model_verify/` + `shared/manifest.py`, none touch `versions/*`:

- **P0.1 Confidence intervals + duplicate-deal head-to-head** [#5/M1, sim-M8]. Every BB/100
  verdict reports mean ± 95% CI; `beats_frozen_predecessor` becomes a **mirrored-deal pairing**
  (same shuffled decks played twice with hero/opponent models swapped, paired-difference test).
  Gate on the CI excluding 0, not on `bb100 > 0`.
- **P0.2 `gameplay_eval.py` serve-temp fix** — set `policy_temperature = 0.5` (one line; the trap
  checks.py already fixed).
- **P0.3 Nash check re-scoring** — score `nash_pushfold_vs_chart` on ALLIN-vs-fold (the question
  Nash actually answers) with the aggression-mass composite kept as a secondary column; re-baseline
  V41/V43/V44 so the V40-introduced "regression" is re-measured on the corrected metric BEFORE V47
  is judged against it.
- **P0.4 `contract_version` hard validation** [#12] — `shared/manifest.py` fails loud on a
  contract_version mismatch, not just width. V44 changing ctx[35]'s MEANING at width 54 is exactly
  the case width-checking cannot catch.

---

## Change 1 — [#6 / SIM-RAISE] Opponents get a real raise-size repertoire

**Motivation**: `simulator.py`'s opponent raise branch hardcodes `min(pot*0.75, stack)` for every
seat type; NN opponents' own bucket choices and TreeOpponent's 6-class size prediction are
discarded. Hero has never faced an open-jam, min-raise, overbet, or small probe in any training
run in this project's history. The [OPP-2] per-seat raise features and every fold-vs-raise
response are calibrated against a world with exactly one opponent bet size. Last open member of
the [BET-3] bundle; the reviewer's nomination for the next version.

**Design**:
- **NN opponents (lagged-self / frozen)**: EXECUTE the bucket they chose — `raise_33/66/pot/allin`
  map through the same `_raise_size_for_fraction` hero uses (min-raise floor, stack cap). No new
  behavior model; stop discarding the one they already have.
- **TreeOpponent**: map its existing size-class prediction to the same fraction set instead of
  collapsing to 'raise'.
- **Heuristic archetypes**: per-archetype size distribution replacing the 0.75 constant, sampled
  per raise event — e.g. NIT {2.5x-open/0.75pot-value heavy}, LAG {min-raise probes, overbets,
  occasional jam}, CALLING_STATION {rare, small}, TAG {balanced}. Constants live in
  `opponent_bots.py` next to the personality traits; calibrated (C1 below), not guessed.
- **Betting-engine compliance**: all sizes flow through the V41-fixed min-raise/reopen rules —
  no new rule surface.
- **Targets**: hero's counterfactual mechanism (`p_all_fold` per HERO size) is structurally
  unchanged; what changes is the realized state distribution and returns. No formula edits in
  `_mc_target_evs_sized` for this change.

**Verification before training**:
- C1: standalone probe — per-archetype size histograms + fold-vs-size response curves; confirm no
  degenerate archetype (e.g. a size that never occurs, or jam-spam).
- C2: instrumented 1000-hand run (the H1 methodology): assert hero actually FACES min-raises,
  overbets, and open-jams at material frequencies; assert NN-opponent executed sizes match their
  chosen buckets bucket-for-bucket.

**Risk watch**: opponent-population changes are this repo's historically highest-collateral class
(V24_extreme broke `vpip_adapts_to_style`; V27 doubled VPIP). Acceptance gates below carry the
specific regression set.

## Change 2 — [M9 / ALIAS] Chip-identical raise buckets stop being four different actions

**Motivation**: at short stacks `_raise_size_for_fraction` clamps raise_33/66/pot/allin to the
same all-in chip amount, but only `frac is None` gets `is_allin=True` — so the clamped buckets
escape the all-in treatment in the fold model, receive the show-of-strength bonus a jam doesn't,
and the model trains four "different" actions with different targets for ONE physical shove.
Directly implicated in "851/851 commits are sized raises, never a literal jam" ([VAL-1]) and
[STACK-3].

**Design**: at target-generation time, any bucket whose resolved chip size equals the all-in
amount (within epsilon) is treated as all-in semantics (`is_allin=True` for the fold model, no
show-of-strength exemption), and in the actor target, chip-identical buckets are collapsed onto
the canonical ALLIN bucket (duplicates masked to 0 so probability mass concentrates instead of
splitting four ways). Live serve mirror: `decision.py`'s sampler masks the duplicate buckets the
same way when their slider chips are identical (train≡serve; small, contained change to the
sized path — the invariant list gains a pair).

**Verification before training**: unit tests at stacks where clamping engages (all four buckets →
one candidate + ALLIN carries the mass) and where it doesn't (no behavior change); assert the
pre/post actor-target mass conservation.

## Change 3 — [M4 / FOLD-MODEL] Counterfactual fold probabilities come from the seated opponent

**Motivation**: `_ev_target_fold_decision` estimates every seat's fold-vs-continue using the
HEURISTIC archetype proxy (`agent.recording_bot`), while ~60% of pool weight is lagged-self NN +
TreeOpponents. Realized returns come from NN/tree behavior; counterfactual targets assume
heuristic-threshold folding — the taken/untaken targets for the same Q-head are drawn from
different opponent models.

**Design**: dispatch the fold-probability estimate by actual occupant —
- NN seats: one policy forward pass on the hypothetical post-raise state → P(FOLD) directly
  (replaces 10 Bernoulli heuristic rolls with an analytic number: less noise AND correct model).
- TreeOpponent seats: its own class-probability output.
- Heuristic seats: analytic threshold probability where derivable, else the existing rolls.
  (This also delivers the L4 noise fix — 0.1-granularity Bernoulli quantization — for most seats.)

**Verification before training**: A/B probe on fixed spots — old proxy vs occupant-true fold
probs per seat type; measured per-hand cost budget (NN query per size per NN seat; batch the
sizes into one forward pass; abort criterion if hands/sec drops below ~10, per the CUDA-backlog
threshold).

## Change 4 — [CURR / VAL-1(A)] Stack curriculum floor below 5bb

**Motivation**: [VAL-1] Finding A's diagnosis was decisive and quantified: BB demands ~2x the
price-edge before calling jams at 5–6bb — a training-floor artifact (5bb is the thin bottom edge
of the lowest band), not a feature/target bug. Cheap data-coverage half of [FMT-1], no ICM.

**Design**: extend `stack_depth_mix` with a 2–5bb band (~0.08–0.10 weight, overlapping style per
[STACK-2] precedent) and modestly upweight 5–8bb. No formula changes.

**Verification before training**: distribution check on 5k simulated hands (band densities as
configured); confirm `STACK_CEIL/scaling` untouched (no contract implication).

## Change 5 — [HYGIENE / M6+M7] Training-loop correctness (scoped; no paradigm changes)

- `CosineAnnealingLR(T_max=100)` stepped once per sim batch ≈ 50 steps/run — anneal never
  completes. Set T_max from the actual planned step count.
- `--resume_path` restores weights only — save/restore optimizer moments + scheduler position in
  the checkpoint (fixes the prime [VAL-5] suspect; V47 itself still trains fresh).
- Validation: replace `random_split` within the same 2k-hand batch (memorization metric) with
  held-out simulation seeds/opponent draws generated alongside each batch.
- Explicitly NOT this pass: replay buffer (real learning-dynamics change — own version if ever).

**Verification**: unit test that a save→resume round-trip is bit-identical in optimizer state;
val-loss curves sanity-checked against the old metric on a 10k smoke run.

---

## Acceptance gates (run under the Phase-0 harness)

1. **Duplicate-deal head-to-head vs frozen V44**: paired CI excluding 0 in V47's favor
   (frozen V44 seated as a real NNOpponent — the post-#4 mechanism).
2. **Hold the wins**: `vpip_adapts_to_style` ≥ +5pts both depths (first-ever pass in V44 — must
   not be the collateral this time), `multiway_shortstack_aggression` PASS ([BET-3] guard),
   `deep_stack_ood_guard` PASS + `allin_vs_nextbest_qgap` non-positive worst cells ([BET-1]
   guard — Change 1 alters all-in economics, watch it specifically).
3. **The point of the version**: corrected-Nash literal-jam agreement improves (Change 2's
   expected signature); [STACK-3] probe — actor-vs-critic fold divergence in the 7–14bb first-in
   band shrinks; opponent-raise-size exposure confirmed in training telemetry (C2 rerun on the
   real run).
4. **Regression watch**: `committed_sensitivity` / `pot_type_sensitivity` (V44's two new WARNs —
   flag if they worsen), `action_diversity`, `position_sweep`, VPIP level vs V44 (the V27-class
   failure mode).
5. Standard: full report saved to OFK `references/V47/model_verify_report.html` (standing rule).

**Rollback**: V44 stays active until the gates pass and the user calls it; V41 remains the
MILESTONE fallback. Deploy = declaration-clone engine + one registry line (post-v46 world).

## Suggested build order

Phase 0 (tooling) → C1/C2 calibrations + Change 2/3 unit probes → Changes 1–5 implemented in the
clone → 10k smoke (telemetry sanity: sizes faced, hands/sec, val metric) → 100k fresh →
`model_verify --full` + gates → review vs this document, update OFK backlog statuses
([STACK-3], [VAL-1](A), review #5/#6/#12, M4/M6/M7/M9, L4).

---

## Build log (2026-07-22)

**Phase 0 — SHIPPED (repo-wide, before any V47 training):**
- P0.1 `tools/model_verify/checks.py`: `_ci95_bb100` + `_play_hands` (per-hand profit series;
  optional per-hand global-RNG seeding — treys' `Deck()` shuffles from the global `random`
  module, so same seed base = same decks/stacks/seatings). Every SLOW BB/100 now reports ±95% CI;
  `bb100_vs_standard_fields`' regression call is CI-aware; `beats_frozen_predecessor` is a
  MIRRORED-DEAL PAIRED head-to-head (two legs, hero/frozen swapped, same seeded deals; gate =
  paired-diff CI excluding 0: PASS above / WARN straddling / FAIL below). Validated on a 150-hand
  smoke: paired diff +223.4 ±254.7 → WARN-inconclusive, exactly the honesty the old `bb100>0`
  gate lacked (its single-run CIs measured ±60–230 BB/100 at that n). Also fixed
  `beats_offformula_stress`'s hasattr-guarded temp assignment (the documented silent-skip trap).
- P0.2 `versions/v44/self_play/gameplay_eval.py` (inherited by this clone): evaluates at serve
  temp 0.5, assigned unconditionally.
- P0.3 `tools/model_verify/nash/pushfold_check.py`: SB check PRIMARY score = literal
  ALLIN-vs-FOLD; the old commit-vs-fold composite is a secondary column. RE-BASELINED (FAST run,
  results/ JSONs): V41 82% primary / 78% composite, V43 65%/65%, V44 71%/66% — and 100% of every
  model's composite-commits are sized raises, never a literal jam (the [M9] artifact, now cleanly
  measured; V47 gate 3 judges against V44's 71% on the SAME metric).
- P0.4 `shared/manifest.py::load_state_dict` + `shared/registry.py::load_model`: hard
  `contract_version` validation (width-equal ≠ semantics-equal, the V43/V44 ctx[35] case);
  `allow_contract_mismatch=True` is the explicit, printed opt-in for deliberate cross-contract
  frozen-OPPONENT seating (used by `check_beats_frozen_predecessor`).

**Changes 1–5 — IMPLEMENTED in this slice (defaults ON in `SixMaxSimulator.__init__`:
`opponent_raise_realism` / `allin_by_chips` / `occupant_fold_models`, so training and
model_verify's SLOW checks run the same world; calibration scripts flip them off for A/B):**
1. [#6] `_opponent_raise_fraction` + betting-loop execution via the shared
   `_raise_size_for_fraction`; NN `raise_k` buckets executed as chosen; TreeOpponent's 6-class
   prediction carried through as `raise_k` (`tree_opponent._RAISE_CLASS_TO_BUCKET`); heuristics
   sample `opponent_bots.RAISE_SIZE_DISTRIBUTIONS` (per-archetype, C1-validated). Also fixed the
   recording-bot VPIP/PFR/AGG funnel dropping every NN 'raise_k' string (pre-existing since V18).
2. [M9] `allin_by_chips` adopted (V43's T-M9 gate, now default ON) + aliased buckets' EVs COPIED
   from the canonical ALLIN bucket + `allin_aliased` flags recorded per decision →
   `vectorize_hand_samples` 13th column → `regret_match_policy`/`regret_match_policy_torch`
   zero aliased regrets (mass concentrates on ALLIN). SERVE MIRROR: `core/decision.py`'s sampler
   masks chip-identical duplicate buckets for engines declaring `collapse_aliased_allin=True`
   (declared by `core/models/v47_engine.py` — created, NOT registered; no pre-V47 engine declares
   it, so deployed models are untouched). NEW TRAIN≡SERVE INVARIANT PAIR.
3. [M4/L4] `occupant_fold_models`: `_mc_target_evs_sized` dispatches per-seat fold estimates by
   actual occupant — NN seats via `_nn_fold_probs_for_sizes` (ONE batched policy pass over
   hypothetical post-raise states built through the extracted `_build_query_board_state`;
   histories copied, never mutated), Tree seats via `TreeOpponent.fold_prob` (proba[0]),
   heuristic seats via the analytic `_heuristic_fold_prob` (exact closed-form expectation of the
   sampled decision — the L4 quantization-noise fix); non-Fuzzy bots keep the legacy 10-roll
   path. Failures fall back loudly (`_note_query_error`). Measured cost: 1.05x total vs pre-V47
   behavior single-process (analytic forms offset the NN forwards) — far above the 10 hands/sec
   abort criterion.
4. [VAL-1(A)] `config.yaml` stack_depth_mix: new [2,5,0.08] + [5,8,0.07] bands, others scaled
   down proportionally (sum 1.00).
5. [M6/M7/VAL-5] `train.py`: CosineAnnealingLR T_max = actual planned batch count; every
   resumable checkpoint saves `optimizer_state`+`scheduler_state` and `--resume_path` restores
   both (loud NOTICE when absent); validation = HELD-OUT stream (the last worker's records —
   own simulator instance/RNG/opponent draws — never enter the train split; single-worker
   ramp-up batch falls back to a sequential 90/10 split). NO replay buffer.

**Pre-training verification: `self_play/verify_v47.py` 28/28** (M9 EV-unification + actor-mass
collapse both paths + without-mask regression demo; M4 analytic==sampled-mean, tree/NN validity,
history non-mutation; C1 wiring; curriculum band densities; VAL-5 bit-identical round trip;
P0.4 raise/opt-in; serve-mirror declarations + clamp parity). C1/C2:
`self_play/calibrate_raise_sizes.py`.

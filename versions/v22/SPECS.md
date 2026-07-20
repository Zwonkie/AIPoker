# V22 SPECS

Branches from `versions/v21_auxhead` (the live foundation, `expert_main.pth`, Phase 8) with two
bundled structural additions, both scoped from V21's own deferred FOLLOWUP items (see
`versions/v21/SPECS.md` items 8/9). Not another training-recipe or aux-head experiment -- that
thread closed with V21_auxhead; its aux-head configuration (corrected `opp_bluff_prob` label,
sqrt-dampened reweighting, per-head weights bluff=0.05/strength=0.10/equity=0.05) is inherited
unchanged.

Both changes touch the same context vector, so they're bundled into ONE `contract_version` bump
(5->6, `context_dim` 37->43) and ONE retrain, rather than two separate versions.

## 1. Deeper stack curriculum ([STACK-2])

Training had never gone past 50bb effective stacks (`STACK_CEIL_BB=50`); a live-serve clamp
mitigated the resulting OOD extrapolation past that depth but never taught the model anything
about genuinely deep play. `versions/v22/core/contract.py`:

- `STACK_CEIL_BB` 50.0 -> 100.0, `POT_CEIL_BB` 100.0 -> 200.0, `CALL_CEIL_BB` 50.0 -> 100.0.
  `STACK_SCALE`/`POT_SCALE`/`CALL_SCALE` unchanged (100/250/100) -- same per-bb resolution as
  before. Stack/call's old ceiling/scale ratio was exactly 0.5 (50/100), so doubling the ceiling
  now uses the FULL previously-half-wasted `[0,1]` normalized range, no rescale needed. Pot's
  ratio was already sub-1.0 by design (100/250=0.4, finer per-bb resolution) -- its new ratio
  (200/250=0.8) uses more of the range without hitting 1.0; not a completeness requirement, just
  never needed the same fix stack/call did.
- `versions/v22/self_play/config.yaml`'s `stack_depth_mix` widened:
  `[5-14bb:0.40, 14-30bb:0.30, 30-60bb:0.20, 10-100bb:0.10]`. The last band deliberately OVERLAPS
  the others (uniform-sampled across 10-100bb, not a disjoint 60-100bb bucket) so there's no
  training-density cliff at the seam between bands -- every depth from 10bb to 100bb gets SOME
  exposure, not just the four nominal buckets.

## 2. Per-opponent/hero entry-sizing ([OPP-2]/[OPP-3]-adjacent, deliberately CHEAP scope)

Two opponents with identical remaining stack and identical VPIP/AGG color can have gotten there
via very different lines this hand (e.g. one limped in, one 3-bet) -- nothing in the existing
context could tell them apart, since `opp_stack` is a remaining-stack snapshot, not a this-hand
action signal. This is NOT the full [OPP-2]/[OPP-3] fix (a genuine per-street, per-seat
action-token sequence, which the backlog itself flags as "a real architecture change") -- it's the
cheap slice V21's SPECS.md item 8 scoped out: `simulator.py` already tracks a per-seat `committed[]`
array (hand-total chips in, used for side-pot math) but never surfaced it into the context.

**New APPENDED features** (every existing index 0-36 stays stable):
- `opp_committed_this_hand_bb` per opponent slot, ctx[37:42] (seat order matches the existing
  per-opponent block) -- chips that seat has already put into this hand's pot, scaled via the
  SAME `scaled_stack_bb` helper as `opp_stack` (same domain: `committed + remaining == starting
  stack`). 0.0 for an inactive/absent seat.
- `hero_committed_this_hand_bb`, ctx[42] (global, not per-seat) -- the symmetric hero-side value.

**What this does NOT give**: it's a single cumulative scalar, not an action sequence. It can't
distinguish HOW the money went in (one big raise vs. limp-then-call-a-raise) or WHEN (preflop vs.
river) -- only how much. A companion `pot_type` feature (limped/single-raised/3-bet+, derivable
from the same `committed[]` data) was considered and deliberately deferred to the OFK backlog
rather than bundled here (see `.agents/skills/OFK/references/known-shortcomings-backlog.md`).

**Wiring** (train-time; live-serve wiring deferred to deploy time, same discipline V21 used):
- `core/board_state.py`: `SeatState.committed` (new, default 0.0) and `BoardState.hero_committed`
  (new, default 0.0) -- additive/optional fields, inert for every earlier version's contract
  (mirrors `hand_strength`'s own pattern).
- `versions/v22/self_play/simulator.py`:
  - `HandRecordV4.add_decision`/`decision_points` gained `opponents_committed` (5-element list,
    mirrors `opponents_stacks`); hero's own value reuses the already-existing `committed_before`
    field.
  - The per-decision `table_state` dict now carries the full-table `committed` array (absolute
    seat 0-5, hero==0).
  - `_query_model_decide` reads it: `hero_committed` = the CURRENT querying actor's own committed
    amount (mirrors how `hero_stack`/`hand_cards` already mean "this actor's own", not literally
    seat 0); per-opponent `committed` values use the REAL per-seat amount (an improvement over the
    pre-existing `stack=hero_stack` placeholder used for opponent stacks in that same function --
    not touched, out of scope, but worth noting `committed` is more accurate than that placeholder
    from day one since `committed[]` is already a full, correctly-indexed table array).
- `versions/v22/self_play/train.py`'s `vectorize_hand_samples`: appends the matching 6 features
  from `dp['opponents_committed']`/`dp['committed_before']`, mirroring `contract.py` exactly (same
  scaling helper, same seat order) -- this is the SAME train/serve duplication point that caused
  V20's original rescale bug, kept in lockstep deliberately this time.

**Opponent-pool compatibility fix** (found while implementing, not part of the original scope):
V21_auxhead's `maniac`/`fish` pool slots loaded frozen V20_preflopEq NN checkpoints
(`frozen_50k.pth`/`frozen_25k.pth`, context_dim=37/contract_version=5). `_query_model_decide`
builds ONE shared context for every queried model regardless of which checkpoint it is -- V22's
own 43-dim context would shape-mismatch-crash a 37-dim-input frozen model. This is the exact
situation V20 itself hit with its own rescale (frozen pre-V20 checkpoints became incompatible
then, too) -- same precedented fix: `maniac`/`fish` reverted to plain heuristic archetypes for
this version (`versions/v22/self_play/config.yaml`). `past` (lagged self-play mirror of THIS run)
is unaffected -- always version-native by construction.

## Verification (pre-training)

Full plumbing smoke-tested before committing to a real training run:
- `PokerEVModelV4()` forward pass with a 43-dim context tensor -- correct output shapes.
- `shared.registry.get_manifest('v22')` auto-discovers the new manifest (context_dim=43,
  contract_version=6) with no registry changes needed.
- `tools/model_verify/run.py --version v22` against throwaway random-init weights -- runs to
  completion with no crashes (checks themselves are meaningless against untrained weights; this
  only validates the plumbing). Added two new FAST checks for this version's own new features:
  `committed_sensitivity` (paired isolate-one-slot ablation, mirrors `hand_strength_sensitivity`)
  and reused `opponent_color_isolated_ablation` (added earlier this session against V21_auxhead,
  applies unchanged here).
- 30 real simulated hands via `SixMaxSimulator` + `vectorize_hand_samples` end to end -- confirmed
  `opponents_committed`/`committed_before` carry real, sensible non-zero values (e.g. blind-post
  amounts appearing correctly scaled: a 5-chip small blind at bb_size=10 -> ctx value 0.005,
  matching `scaled_stack_bb(5, 10) = min(5/10, 100)/100 = 0.005` exactly), and the resulting
  context tensor is 43-wide at every timestep.

## Training setup

- `target_hands: 100000`, `checkpoint_dump_interval: 20000` -- matches V21/V21_auxhead's own
  cadence for a like-for-like comparison.
- Aux-head config, actor-critic cutover, bootstrap/exploration schedule, policy target source,
  range-aware equity, realization discount -- all inherited unchanged from V21_auxhead.
- Opponent pool: `past` (lagged self, 0.25) / `maniac` (heuristic, 0.20) / `fish` (heuristic,
  0.15) / `tag` (heuristic, 0.25) / `nit` (heuristic, 0.15) -- see compatibility fix above.
- Fresh from-scratch run (no `--resume_path`) -- matches this session's own finding
  ([VAL-5] in the OFK backlog) that warm-started continuations degrade action diversity
  independent of whatever else is being changed; V22 gets a clean, confound-free 100k-hand result.

## Results (2026-07-17, `expert_main.pth`, 100k hands)

Training completed cleanly, no NaN/crashes. Final dashboard: Hero +44.3 BB/100 vs the field, all
six actions represented in `ACTION USAGE`, healthy loss decomposition (Bluff/Str/Eq losses all
low and stable, matching V21_auxhead's own numbers).

**`model_verify --full`: 16 PASS / 4 WARN / 0 FAIL / 1 SKIP -- the FIRST zero-FAIL result in this
entire lineage.**

- **`deep_stack_ood_guard` PASSED for the first time ever** (FAILed in every prior version since
  V14/V15: V19, V20, V20_preflopEq, V20_preflopEq_AI, V21, V21_auxhead all failed this gate). The
  deeper stack curriculum ([STACK-2] fix: `STACK_CEIL_BB` 50->100 + `stack_depth_mix`'s real
  10-100bb tail, replacing a hard extrapolation clamp with actual training density) appears to
  have genuinely resolved the marginal-equity/15-40bb trash-jam this check has tracked since the
  live K9o incident. Backlog updated (see `known-shortcomings-backlog.md` [STACK-1]/[STACK-2]).
- **`committed_sensitivity` PASSED** (TV=0.077, comparable to `hand_strength_sensitivity`'s own
  0.120) -- the new `opp_committed_this_hand_bb`/`hero_committed_this_hand_bb` features are
  load-bearing, not dead weight, on the first training exposure.
- `bb100_vs_standard_fields`: strong across all 4 fields (loose_short +29.2, loose_deep +96.0,
  tight_short +27.4, tight_deep +70.1 BB/100). `vpip_adapts_to_style`/`beats_offformula_stress`
  both PASS with healthy deltas.
- `stack_full_sweep`'s argmax path is now `call` at every one of the 9 stack points (5-180bb) --
  a genuinely different shape than prior versions' allin-heavy paths, plausibly downstream of
  real training exposure across a much wider stack range.
- **New WARN**: `short_stack_polarization` (avg P(call)=0.35 in shove-or-fold spots) -- this is
  [BET-2] (tracked since V15/V16), not a new issue, but worth noting the deeper stack curriculum
  didn't help it and may have mildly worsened it (same "more training worsens this" pattern V20
  showed).
- Unchanged, pre-existing WARNs: `opponent_style_sweep` ([OPP-5], root-caused this session as a
  training-population artifact, untouched by V22's changes) and `position_sweep`.
- `action_diversity` still shows no raise_33/raise_66/raise_pot winning argmax anywhere in the
  grid ({'fold':9, 'call':3, 'allin':9}) -- [BET-1] ("no middle gear") is completely untouched by
  V22, exactly as expected since V22 didn't target it.
- `beats_frozen_predecessor` SKIPped -- no `frozen_v*.pth` saved yet in `versions/v22/weights/`.
  Not urgent (every other gate is strong), but worth adding a preserved V21_auxhead/V22 frozen
  checkpoint for a future direct predecessor comparison.

**Verdict**: V22 is a clean, validated improvement over V21_auxhead -- resolves a long-standing
open regression ([STACK-1]) as a side effect of the deep-stack curriculum work, and the new
entry-sizing feature is confirmed load-bearing. Deploy-live decision and live-serve wiring
(mirroring V21_auxhead's own bridge, since the contract changed -- a NEW bridge is needed this
time, contract_version 6 is not byte-identical to any live-wired version) not yet actioned.

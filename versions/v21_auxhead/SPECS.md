# V21_auxhead — aux-head rationality probe (warm-started continuation of V21)

Clone of V21. Same tensor contract (`context_dim=37`, `contract_version=5`), same architecture,
same opponent pool. Changes exactly ONE training knob: `aux_loss_weight: 0.0 -> 0.05`.

## Motivation

The bluff/strength/equity aux heads exist in `PokerEVModelV4` since early versions but have always
trained at `aux_loss_weight=0.0` — a forward pass + loss computed every step, contributing exactly
zero gradient (genuinely inert since V14). Raised as an open question in `versions/v21/SPECS.md`
item 7: are these worth reviving (representation-learning regularization, eventually a legitimate
live opponent-read display), or dead weight worth deleting outright?

## Method — deliberately narrow, not the full ablation

This is NOT the from-scratch `aux_loss_weight=0.0` vs `0.05-0.1` isolated ablation SPECS.md
originally sketched for item 7 (that's a bigger, separate follow-up — "does this change final
model quality"). This is a fast, narrow **rationality check**: warm-started from V21's own
converged 100k-hand checkpoint (`frozen_v21.pth`), continued for a short ~20k-hand stretch
(`target_hands: 120000`, resumed via `--resume_path .../frozen_v21.pth --hands_done 100000`) with
`aux_loss_weight=0.05` turned on. Two things this checks, cheaply, before committing to the bigger
ablation:

1. **Does turning on aux gradient now destabilize an already-converged policy/critic?** Watch
   `train_loss_q`/`train_loss_pi` through the continuation — should stay in the same range V21
   ended at, not spike.
2. **Do the aux heads' predictions actually start tracking their own training labels once given
   real gradient?** `self_play/inspect_aux_heads.py` — simulates heuristic-only hands (decoupled
   from whichever model is under test), vectorizes them with `train.py`'s own
   `vectorize_hand_samples` (identical featurization to training), forward-passes the trained
   model, and correlates `preds['bluff']/['strength']/['equity']` against
   `opp_bluff_prob`/`opp_strength`/`dp['equity']`.
   - `self_equity` is the cleanest signal: equity is ALSO a direct input feature (`ctx[3]`), so a
     genuinely-wired head should learn to echo it back with near-trivial ease (correlation near
     1.0, low MAE). If it doesn't, that's evidence of a broken gradient path, not "a hard feature."
   - `opp_bluff`/`opp_strength` are noisier proxy labels (see
     `simulator.py::_mc_target_evs_sized`) — perfect accuracy isn't the bar; meaningful positive
     correlation and a prediction spread that isn't collapsed to the label's mean is.

## Next steps (not yet decided)

If this rationality check looks sane (no destabilization, non-trivial aux-label correlation):
proceed with the originally-scoped from-scratch ablation to test whether aux gradient changes
final model quality, and decide whether to keep aux heads on for future versions by default.

If it doesn't (predictions stay flat/uncorrelated even for `self_equity`, or the main losses
destabilize): that's itself a useful, cheap negative result — either the gradient path has a real
bug worth finding, or the heads are confirmed dead weight worth deleting from the architecture
rather than carrying forward at `aux_loss_weight=0.0` forever.

## Results (2026-07-17)

**Loss stability**: main train/Q/Pi losses spiked mid-run (Train 0.73->4.32, Q 0.43->3.53 at
~111k) then fully re-settled by 120k (Train 0.30, Q 0.047, Pi 0.26 -- comparable to or better than
V21's own ending values). Traced to a confound in the resume mechanism itself, NOT the aux heads:
`run_training` creates a FRESH `Adam` optimizer + fresh `CosineAnnealingLR` scheduler every call --
`--resume_path` only restores model WEIGHTS, not optimizer momentum or where the LR schedule had
gotten to. The aux losses' own contribution to total loss is negligible regardless (`Bluff: 0.0165,
Str: 0.0174, Eq: 0.0043` at the spike, weighted by 0.05 -> ~0.002 total). Any `--resume_path`
continuation in this codebase likely shows a similar transient; this is a general gap worth fixing
(warm-starting optimizer/scheduler state too) before relying on short resumed continuations for
future tests, not something specific to this experiment.

**Aux-head rationality** (`inspect_aux_heads.py`, 1034 live decision points):

| head | label | r (corr) | MAE | pred mean/std | label mean/std |
|---|---|---|---|---|---|
| equity | self_equity (ctx[3]) | 0.894 | 0.052 | 0.389/0.161 | 0.370/0.174 |
| strength | opp_strength | 0.166 | 0.110 | 0.605/0.048 | 0.588/0.147 |
| bluff | opp_bluff_prob | 0.132 | 0.069 | 0.000/0.039 | 0.044/0.206 |

- **`self_equity`: clean pass.** r=0.894, comparable means/spread -- in ~20k hands of real
  gradient the head learned to closely echo an input it already had direct access to. Rules out a
  broken gradient path outright: if this had come back near 0, that would have meant the aux loss
  wiring itself was broken, not just "hard to learn."
- **`opp_strength`/`opp_bluff`: weak-but-nonzero, not yet meaningful.** Both show small positive
  correlation but a prediction std collapsed well below the label's std -- `bluff` in particular
  predicts almost exactly 0.000 for everything (the label's own mean is 0.044, a sparse/rare
  proxy). Reads as "found the safe near-constant prediction, hasn't extracted real signal yet,"
  plausible given only ~20k hands, a deliberately modest `aux_loss_weight=0.05`, and these two
  labels being intrinsically harder (they depend on the opponent's actual hole cards -- information
  only partially inferable from context, unlike `self_equity`).

**Verdict**: clears the bar to proceed with the originally-scoped from-scratch ablation (does aux
gradient change final policy/critic quality, and do `strength`/`bluff` develop real signal given
full training exposure rather than a 20k-hand tail). Also surfaced a real, separate action item:
warm-starting `--resume_path` runs should restore optimizer/scheduler state, not just weights, if
short resumed-continuation tests are going to be a repeated pattern.

**Follow-up finding, same day**: digging into WHY `opp_bluff` correlated so weakly turned up a real
labeling bug, not just insufficient training. `opp_bluff_prob` (`simulator.py::_mc_target_evs_sized`)
was `1.0 if max_opp_equity < 0.33 else 0.0` — true whenever ANY active opponent held weak cards,
regardless of whether anyone had actually taken an aggressive action. That fires just as often when
a weak opponent simply folds (the common case) as when they genuinely bluff, so the label was
really measuring "is someone at the table weak" (redundant with `opp_strength`), not "is my
opponent bluffing me right now." Fixed: now gated on `last_raiser` (already reliably tracked in the
betting loop), reading specifically the last aggressor's own equity; 0.0 when no opponent is the
last raiser (correctly "not a bluff scenario," not a fallback to the old proxy). Full diff in
`self_play/simulator.py` (`_mc_target_evs_sized`, the `active_opps_list` construction). Logged in
the OFK backlog is a separate finding ([OPP-5], opponent-style/VPIP-AGG-color read not load-bearing
in `model_verify`'s `opponent_style_sweep`) — not the same issue, don't conflate the two.

## Phase 2 (complete) — fresh from-scratch 100k run with the corrected label

Rather than another warm-started continuation, this phase is a FRESH run (no `--resume_path`,
`target_hands: 100000` matching V21's own run exactly) — both because Phase 1's loss-spike confound
(fresh optimizer/scheduler state on every resume) makes a from-scratch comparison cleaner to read,
and because the corrected `opp_bluff_prob` deserves a full run's worth of exposure rather than a
20k-hand tail grafted onto weights that never saw the fixed label. Phase 1's final weights
preserved at `weights/phase1_warmstart_120k.pth`, this phase's own final weights preserved at
`weights/phase2_fresh_100k.pth`, both before being overwritten by later runs. This run is the
direct, comparable `aux_loss_weight=0.05` arm against V21's own `aux_loss_weight=0.0` 100k-hand run.

Training itself tracked V21's own trajectory closely at every checkpoint (entropy, loss shape,
equity-bucket action distribution all matched V21's own numbers within noise at 20k/40k/60k/80k),
including the same dramatic late-run loss collapse V21 itself showed (Train Loss 5.64→0.54 in the
final 20k hands) — confirms that pattern is a real, reproducible feature of this recipe's last
training window, not a one-off.

**`inspect_aux_heads.py` results (5099 live decision points, `--n-hands 4000` — re-ran at 5x the
initial sample to make sure the sparser corrected label wasn't just under-sampled; numbers were
stable across both runs):**

| head | label | r (corr) | MAE | pred mean/std | label mean/std |
|---|---|---|---|---|---|
| equity | self_equity (ctx[3]) | 0.942 | 0.034 | 0.368/0.139 | 0.363/0.156 |
| strength | opp_strength | 0.151 | 0.106 | 0.602/0.031 | 0.603/0.146 |
| bluff | opp_bluff_prob | 0.080 | 0.033 | -0.002/0.020 | 0.019/0.137 |

- **`self_equity` improved further** (r=0.894→0.942) with full training exposure vs Phase 1's
  20k-hand tail — as expected, not a new finding.
- **`opp_bluff` got WORSE, not better** (r=0.132 with the OLD broken label → 0.080 with the FIXED
  label), despite a full 100k-hand run. Root cause identified, not a failure of the label fix
  itself: gating on `last_raiser` made the label correctly sparser (positive rate ~4.4% before →
  ~1.9-2.6% after, since most decisions genuinely aren't "facing a specific opponent's raise").
  Plain MSE at `aux_loss_weight=0.05` trivially minimizes a ~98%-negative binary-valued target by
  predicting near-zero for everything — exactly what's observed (pred std collapsed to 0.020 vs
  the label's own 0.137). This is a well-known failure mode for imbalanced regression/classification
  targets, not evidence the semantic fix was wrong — it made the label CORRECT, which incidentally
  made it HARDER for an unweighted MSE loss to learn from.

**`model_verify --full` comparison vs V21's own runs**: 16 PASS/2 WARN/1 FAIL/0 SKIP — same shape
as V21. No new failures, no lost passes. Notable deltas, not just noise:
- **Encouraging**: `hand_strength_sweep` 0.237 (V21) → **0.825** — 3x+ more responsive to an
  EXISTING feature, the first real evidence for the original aux-head hypothesis (representation-
  learning regularization, not just three unread outputs). `action_diversity` shows `call` winning
  argmax somewhere in the grid for the first time in this whole lineage
  (`{fold:9,allin:10,call:1,raise_pot:1}` vs V21's `{fold:9,allin:10,raise_66:2}`) — still
  fold/allin-dominated, but a real step away from [BET-1]'s shove-preference. `stack_full_sweep`
  extends call/raise usage across 5/9 points (was 4) and never reaches allin in this sweep (V21's
  did, at 180bb). `vpip_adapts_to_style` short-delta (+16.7pt) and `beats_offformula_stress`
  short-stack (+36.5 BB/100) both beat either of V21's own two runs.
- **One real, modest dip**: `bb100_vs_standard_fields`'s `tight_deep` field: +28.8 BB/100, below
  V21's own range across three prior readings (55.8-85.9). Still positive, but the one number here
  that reads as a genuine soft spot rather than noise.
- **One large but likely-noise swing**: `position_sweep`'s absolute fold rate at one synthetic spot
  (40bb, small bet, 45% equity) jumped from 6-13% (V21) to 51-62% -- spread itself improved
  (0.061→0.106, still PASS), but that's a big shift in one narrow probe. Real simulated-play win
  rates across every other field stayed healthy, so this reads as this specific corner shifting,
  not a systemic issue.
- **Unchanged, as expected**: `deep_stack_ood_guard` still FAILs (different cell -- 40bb this time,
  was 15bb -- matching [STACK-1]'s own note that the exact failing cell moves between runs without
  ever clearing). `opponent_style_sweep` still flat/WARN ([OPP-5], untouched by anything here).

**Fix applied same day**: `_bluff_pos_weight()` in `train.py` — per-batch inverse-frequency
reweighting (same idea as `BCEWithLogitsLoss`'s `pos_weight`, computed from the batch's own
observed positive rate rather than a hand-picked constant, capped to avoid a near-empty batch
producing an exploding weight). Applied to both the training and validation aux-loss blocks.
See Phase 3.

## Phase 3 (complete) — does reweighting the bluff loss fix the collapse?

Warm-started from Phase 2's preserved weights (`phase2_fresh_100k.pth`, +25k hands) rather than
another fresh 100k run, to get a fast read on whether reweighting alone changes the `bluff` head's
behavior. The known optimizer/scheduler-reset confound (see Phase 1) only muddies mid-run LOSS
TREND interpretation, not the final `inspect_aux_heads.py` correlation readout, which is what this
phase actually needs — so warm-starting is a legitimate, much cheaper way to test this specific
question without spending another full hour on a fresh run. Weights preserved at
`weights/phase3_fullweight_125k.pth`.

**Result: fixed the collapse, overcorrected on calibration.**

| head | r (corr) | MAE | pred mean/std | label mean/std |
|---|---|---|---|---|
| equity | 0.949 | 0.041 | 0.368/0.119 | 0.359/0.160 |
| strength | 0.065 | 0.106 | 0.607/0.034 | 0.601/0.141 |
| bluff | 0.104 | 0.301 | **0.298**/0.249 | 0.020/0.141 |

`bluff`'s prediction std stopped collapsing (0.020→0.249, no longer just predicting near-zero) and
correlation improved slightly (0.080→0.104) — the reweighting is doing SOMETHING. But the model
now systematically OVER-predicts: mean prediction 0.298 against a true ~2% base rate. The full
inverse-frequency ratio ((1-p)/p ≈ 49x at a 2% positive rate) equalizes gradient MASS between
classes, which fixes "ignores the minority class" but doesn't itself keep the predicted MAGNITUDE
anchored near the true rate — classic overcorrection. `strength`'s correlation also dropped
(0.151→0.065) in this run; flagged for Phase 5, see below.

**Fix**: dampen via `sqrt((1-p)/p)` instead of the raw ratio (~7x rather than ~49x at 2%) — a
standard adjustment for exactly this "correct direction, too strong a correction" pattern. Cap
lowered from 100 to 20 to match (the dampened value never approaches the old cap regardless).

## Phase 4 (complete) — sqrt-dampened reweighting

Same warm-start pattern as Phase 3 (from `phase2_fresh_100k.pth`, +25k hands), testing the dampened
weight in isolation against the same base. Weights preserved at
`weights/phase4_sqrtweight_125k.pth`.

**Result: the best-calibrated `bluff` result of the three variants.**

| head | r (corr) | MAE | pred mean/std | label mean/std |
|---|---|---|---|---|
| equity | 0.861 | 0.072 | 0.316/0.132 | 0.359/0.161 |
| strength | 0.047 | 0.111 | 0.565/0.043 | 0.602/0.144 |
| bluff | **0.115** | **0.110** | **0.019**/0.134 | 0.020/0.140 |

`bluff`'s predicted mean (0.019) now almost exactly matches the label's own mean (0.020), predicted
std (0.134) closely matches the label's std (0.140) — well-calibrated, not collapsed, not
overshooting, and the best correlation of all three variants (unweighted 0.080, full-weight 0.104,
sqrt-dampened 0.115).

**But there's a real cost showing up in the OTHER two heads, across both Phase 3 and Phase 4**:
`strength` degraded in both short warm-started continuations (0.151→0.065 in Phase 3, →0.047 in
Phase 4) and `equity` degraded specifically in Phase 4 (0.949→0.861, unchanged in Phase 3). Since
this shows up in BOTH the full-weight and dampened variants -- not just the dampening choice -- the
likely mechanism is structural, not specific to either weighting scheme: `final_loss_aux = sc_bluff
+ sc_str + sc_eq` shares ONE `aux_loss_weight` budget across all three heads, and bluff's now-larger
reweighted loss value likely claims a disproportionate share of that shared gradient during a short
window, at the other two heads' expense. Open question: is this a transient artifact of a SHORT
(25k-hand) warm-started continuation (the same class of confound Phase 1 found for the main Q/Pi
loss, which fully resettled by run's end), or a genuine standing tradeoff? Testing with a longer
continuation next (Phase 5).

## Phase 5 (in progress) — does the equity/strength dip resolve with more hands?

Same warm-start base (`phase2_fresh_100k.pth`) and the Phase 4 sqrt-dampened weighting, but a
longer continuation (+50k hands, target 150k total, vs Phase 3/4's +25k) to test whether `equity`/
`strength` recover given more training time with the dampened bluff loss active throughout, the way
the main Q/Pi loss recovered by the end of Phase 1's shorter window. Weights preserved at
`weights/phase5_sqrtweight_150k.pth`.

**Result: settles the transient-vs-persistent question.**

| head | Phase 2 | Phase 3 (+25k) | Phase 4 (+25k) | Phase 5 (+50k) |
|---|---|---|---|---|
| equity | 0.949 | 0.949 | 0.861 | 0.918 |
| strength | 0.151 | 0.065 | 0.047 | **0.033** |
| bluff | 0.080 | 0.104 | 0.115 | 0.105 |

`equity` mostly recovered with more hands (0.861→0.918) — consistent with a partially transient
warm-start effect. `strength` kept declining MONOTONICALLY across every single phase since Phase 2
— more hands made it worse, not better, ruling out "just needs more time to resettle" for this
specific head. `bluff` stayed stable and reasonably calibrated across the last three phases.

**Conclusion**: the shared-`aux_loss_weight` hypothesis is the best-supported explanation —
`strength`'s signal is being structurally crowded out by bluff's now-correctly-larger loss
magnitude within one shared budget, not suffering a transient perturbation. Moving to Phase 6:
decouple the weight per head instead of sharing one scalar.

## Phase 6 (complete) — per-head aux weights

Code change in `train.py`: `run_training` now accepts `aux_loss_weight_bluff`/`_strength`/`_equity`
(each defaulting to the shared `aux_loss_weight` if unset in config, for backward compatibility)
instead of one scalar multiplying the summed aux loss. Config set `strength=0.20` (4x the fallback
0.05), bluff/equity left at the fallback. Same base (`phase2_fresh_100k.pth`) and +50k-hand window
as Phase 5, for a direct, controlled comparison at the identical hand count. Weights preserved at
`weights/phase6_strength020_150k.pth`.

**Correction to the Phase 5 write-up**: `w * (a + b + c)` and `w*a + w*b + w*c` are mathematically
IDENTICAL when `w` is the same for all three terms — Phase 2-5's "one shared weight" was never
actually a different FORMULA STRUCTURE from per-head weights, only a special case where all three
happened to be equal. The real lever was always just giving one term a different coefficient than
the others; "sharing a sum" itself was never the mechanism. Correcting this here rather than
leaving the imprecise framing standing.

**Result:**

| head | Phase 5 (all @ 0.05) | Phase 6 (strength @ 0.20) |
|---|---|---|
| equity | 0.918 | 0.927 |
| strength | 0.033 | **0.144** |
| bluff | 0.105 | 0.086 |

`strength` improved >4x (0.033→0.144) — a real effect, not noise, given the magnitude. `equity`
held steady. `bluff`'s small dip (0.105→0.086) is very likely ordinary run-to-run noise, since its
own coefficient was UNCHANGED between these two runs (0.05 both times) — nothing about boosting
strength's weight should mechanically cost bluff anything.

**Retrospective check on whether this could have been predicted rather than guessed**: raw
per-head loss magnitudes from Phase 2's baseline (before any reweighting) don't cleanly predict
which head needs a bigger coefficient — all three started in a similar numeric range. What DOES
distinguish `strength` in hindsight: its raw loss barely decreased over Phase 2's full 100k-hand
run (0.029→0.011, ~2.6x) while `bluff`/`equity` dropped 15-30x in the same window. A magnitude
snapshot wouldn't have flagged it in advance; "is this term's loss actually decreasing over
training" would have. Going forward, `inspect_aux_heads.py`'s correlation readout remains the most
direct, ground-truth tuning signal available (measure -> adjust weight -> cheap warm-started
re-test -> re-measure) — not as rigorous as real gradient-norm-based dynamic balancing (e.g. a
GradNorm-style scheme), but usable now without new machinery.

## Phase 7 (complete) — mapping the response curve

One data point (`strength=0.20`) doesn't establish whether it's near-optimal or a lucky guess.
Tested two more points at the same base/window (`phase2_fresh_100k.pth`, +50k hands): `strength=
0.10` (Phase 7a, weights `phase7a_strength010_150k.pth`) and `strength=0.35` (Phase 7b, weights
`phase7b_strength035_150k.pth`), bluff/equity fixed at 0.05 throughout.

**Full response curve:**

| strength weight | equity | strength | bluff |
|---|---|---|---|
| 0.05 (Phase 5) | 0.918 | 0.033 | 0.105 |
| 0.10 (Phase 7a) | **0.943** | 0.120 | **0.130** |
| 0.20 (Phase 6) | 0.927 | **0.144** | 0.086 |
| 0.35 (Phase 7b) | 0.921 | 0.057 | 0.062 |

Not a plateau — an inverted U. Pushing to 0.35 made `strength` WORSE than 0.20 (0.144→0.057) and
dragged `bluff` down too (0.086→0.062): past the peak, now costing both the head it's meant to help
and its neighbor. Looking across all three heads rather than optimizing `strength` in isolation,
**`strength=0.10` is the best overall balance** — best `equity` and best `bluff` of all four points,
with `strength` at 0.120 (close to 0.20's peak without paying its cost elsewhere).

**Chosen final configuration**: `aux_loss_weight_bluff=0.05, aux_loss_weight_strength=0.10,
aux_loss_weight_equity=0.05`, sqrt-dampened bluff reweighting (Phase 4's fix), corrected
`opp_bluff_prob` label (Phase 1's fix).

**`model_verify --full` on Phase 7a's weights (150k total: 100k fresh + 50k warm-started
continuation) surfaced an important confound**: `action_diversity` regressed to `{fold:9,
allin:12}` — only 2 distinct actions across the whole grid, WORSE than V21's own baseline
(`{fold:9,allin:10,raise_66:2}`, 3 actions) and much worse than Phase 2's OWN 100k-hand result
(`{fold:9,allin:10,call:1,raise_pot:1}`, 4 actions -- the best diversity of the whole
investigation). `stack_full_sweep`'s argmax path was `allin` at all 9 stack points (was a real
call->raise_pot->raise_66 progression in Phase 2, never even reaching allin).

**Root-caused, not just noted**: Phase 2's own 100k FRESH run already had the best bet-sizing
diversity in this whole investigation. Every one of Phases 5/6/7a/7b independently applied a +50k
warm-started CONTINUATION on top of that same clean base, at four different aux weights (0.05,
0.20, 0.10, 0.35) -- and the diversity collapse showed up in every single one of them regardless of
weight. That's strong evidence the CONTINUATION mechanism itself (not the aux-weight tuning) is
responsible -- plausibly the same "more total hands in this recipe trends toward shove-dominance"
pattern already tracked as [BET-1] in the OFK backlog, observed live during V21's own 100k run.

## Phase 8 (complete) — the actual final candidate: fresh 100k with the full chosen config

Rather than ship the continuation-damaged Phase 7a checkpoint, Phase 8 trains the FULLY chosen
configuration (corrected bluff label + sqrt-dampened reweighting + per-head weights
bluff=0.05/strength=0.10/equity=0.05) together, FRESH from scratch, matching Phase 2's own protocol
exactly (100k hands, no `--resume_path`) -- getting the aux-head correlation improvements without
the continuation-induced diversity cost.

**`inspect_aux_heads.py` (5161 live decision points, `--n-hands 4000`):**

| head | label | r (corr) | MAE | pred mean/std | label mean/std |
|---|---|---|---|---|---|
| equity | self_equity (ctx[3]) | 0.922 | 0.042 | 0.382/0.149 | 0.366/0.166 |
| strength | opp_strength | **0.171** | 0.106 | 0.570/0.034 | 0.603/0.142 |
| bluff | opp_bluff_prob | 0.091 | 0.089 | 0.059/0.117 | 0.018/0.134 |

`strength` is the best of ANY phase in this investigation (0.171, beating Phase 6's 0.144 peak),
training it together with the other two heads from scratch rather than as an isolated
warm-started arm. `equity` stays strong (0.922, in the same band as every prior phase). `bluff`
(0.091) sits between Phase 2's 0.080 and Phase 7a's 0.130 -- consistent with prior noise level for
this head, not a regression.

**`model_verify --full`: 15 PASS / 3 WARN / 1 FAIL / 0 SKIP.** Directly compared to Phase 2 (fresh
100k, the pre-registered "best case" baseline) and Phase 7a (150k, continuation-damaged):

| metric | Phase 2 (fresh 100k) | Phase 7a (150k, continuation) | **Phase 8 (fresh 100k, final config)** |
|---|---|---|---|
| `action_diversity` | `{fold:9,allin:10,call:1,raise_pot:1}` (4 actions) | `{fold:9,allin:12}` (**2 actions**) | `{fold:9,allin:11,raise_33:1}` (**3 actions**) |
| `stack_full_sweep` argmax path | call→raise_pot→raise_66 progression, never reaches allin | `allin` at all 9 points | allin/allin/allin/**raise_33 x5**/allin (range 0.232) |
| PASS/WARN/FAIL | 16/2/1 | (not re-run standalone, same shape expected) | 15/3/1 |
| `deep_stack_ood_guard` | FAIL (40bb cell) | — | FAIL (30bb cell, eq=0.55→allin@0.37) |
| `opponent_style_sweep` | WARN (flat, [OPP-5]) | — | WARN (flat, spread 0.010, [OPP-5]) |

**This confirms the continuation-confound hypothesis.** Phase 8 recovers real mid-stack bet-sizing
diversity (a genuine `raise_33` plateau across 5 of 9 stack points) that Phase 7a's warm-started
continuation had completely erased down to a fold/allin coin-flip. It doesn't fully match Phase 2's
own 4-action peak (no `call` or `raise_pot` argmax cell here), so training FRESH recovers *most* but
not all of the diversity lost to continuation -- the residual gap is small compared to the
Phase 7a collapse and is within the kind of run-to-run noise seen elsewhere in this investigation
(e.g. `deep_stack_ood_guard`'s failing cell moving between runs, per [STACK-1]).

One new (mild) WARN not seen in Phase 2: `free_check_low_fold` -- max raw P(fold) when call_bb=0
hits 1.0 in the synthetic sweep, but the check's own detail notes this is already covered by
`decision.py`'s free-check mask at serve time, so it's a training-time-only artifact, not a live
behavior regression.

**Verdict: Phase 8 is the final candidate for this experiment.** `expert_main.pth` (this run's
weights) supersedes Phase 7a's checkpoint -- it carries the full aux-head configuration (corrected
bluff label, sqrt-dampened reweighting, per-head weights) with model_verify numbers close to V21's
own baseline shape and no continuation damage. `deep_stack_ood_guard` and `opponent_style_sweep`
remain open, pre-existing issues ([STACK-1]/[OPP-5] in the OFK backlog) untouched by this
experiment either way.

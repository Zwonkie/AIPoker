# V20_preflopEq — range-aware equity calibration + field-size-aware features

Planned subversion of V20 (clone target once building starts). **NOT YET BUILT — no folder
scaffold, no code, no training.** This document is the analysis/planning record from a live
investigation (2026-07-16/17, prompted by a live slider-execution bug report that led into a deep
dive on `compute_range_aware_equity`) — captured here per user request before any implementation.
Three findings, two of them real calibration bugs in the shared train/serve equity function, one a
proposed new set of engineered features. All verified with standalone scripts against the real
codebase (`versions/v17_gauntlet/self_play/simulator.py`) before writing this up — same
verify-the-mechanism-first discipline the project already uses (see V19's [P0], V20's own
[policy_tightness_bb] re-examination).

## How this started

Live dashboard work surfaced a report that RAISE_66/RAISE_POT signals sometimes executed as
all-in on screen. Traced the decision math (`core/decision.py::_v14_size_to_slider`) and confirmed
it was NOT the cause — verified against real recorded live history (`history/*/turns.jsonl`) that
the model's own computed slider fractions were sane in every non-forced-all-in case; the live bug
was in `action_executor.py`'s pixel calibration (fixed separately, not part of this doc). That
investigation led to a broader question — "verify the model's signals are actually pot-relative
during simulation" — which surfaced the two calibration issues below.

## Finding 1 — unknown-color opponents are silently dropped from the equity calc

`PHPHelp.py`'s live equity call builds `opp_colors` as:
```python
opp_colors = [o.get('vpip_color') for o in active_opps if o.get('vpip_color')]
```
Any opponent whose HUD color hasn't been classified yet (`vpip_color` is `None`) is filtered out
entirely — not treated as an average/unknown opponent, just erased, as if that seat weren't
contesting the pot at all. This is LIVE-ONLY (training always has ground-truth VPIP for every
seat, so this exact gap can't occur in simulation) but it's a real live-serving bug: an opponent
who has demonstrably called is invisible to the equity number.

**Quantified** (hero `Ah Jd`, one still-to-act `Blue` opponent, one front caller of unknown color):

| Treatment of the unknown caller | Equity |
|---|---|
| Dropped (current live behavior) | 0.68 (QQ example) / 0.36 (87s example) |
| Proxy → Blue/Green (tight) | ≈ same or slightly below dropped |
| Proxy → Yellow (LAG) | above dropped |
| Proxy → Red (loose) | further above dropped |
| "Guaranteed in + truly uniform/full-range" | highest of all (see Finding 2 for why "full range" is an optimistic, not neutral, choice) |

**Recommendation:** map `None` → `'Yellow'` before the equity call, not "full range." Two reasons:
(1) Yellow is already the codebase's existing "no information" convention elsewhere
(`opponents_profiles.get(..., {}).get('vpip', 0.3)` → `_vpip_to_color(0.3)` lands in Yellow's
`[0.26, 0.35)` band) — consistent, not a new default invented for this fix. (2) A truly uniform
"full range" opponent is *weaker on average than even the widest color band* (`Red`'s own band
tops out at the 85th percentile, `(0.40, 0.85)` — never touches the bottom ~15% of hands a fully
random deal would include), so "full range" is actually the most hero-favorable option of
everything tested, not a cautious middle ground — the wrong direction to be wrong in for a
real-money decision tool.

## Finding 2 — equity has no concept of "already acted" vs "still to act" (bigger issue)

### The simulator has full positional ground truth — it just isn't used for equity

Confirmed directly in `versions/v17_gauntlet/self_play/simulator.py`:
```python
button_seat = random.randint(0, 5)                    # line 799 — fresh every hand
...
if street_idx == 0:
    current_actor = (button_seat + 3) % 6              # preflop first-to-act = UTG
else:
    current_actor = (button_seat + 1) % 6              # postflop first-to-act = left of button
...
while not betting_ended:
    if not folded[current_actor] and stacks[current_actor] > 0:
        ...                                            # this seat's decision
    current_actor = (current_actor + 1) % 6             # line 1179 — proper rotation
```
This is a fully correct, ground-truth sequential action-order simulation — hero (always seat 0,
but at a randomized position relative to `button_seat` every hand) genuinely faces opponents who
have already acted this street (front) and opponents who haven't yet (after), exactly like real
poker.

But the code that builds hero's equity-facing opponent list ignores all of it:
```python
if self.range_aware_equity and current_actor == 0:      # line 1001 — only runs at hero's decision
    opp_colors = [
        _vpip_to_color(opponents_profiles.get(f"seat_{s}", {}).get('vpip', 0.3))
        for s in range(1, 6) if not folded[s]            # lines 1003-1006 — fixed seat-number order
    ]
```
Filtered only by `folded[s]`, iterated in fixed numeric order — `current_actor` and `button_seat`
are in scope in the very same loop and simply never consulted. `compute_range_aware_equity` itself
then applies the SAME VPIP-fold-roll to every color in that list regardless of position, preflop:
```python
if is_preflop and random.random() >= _COLOR_TO_VPIP.get(color, 0.30):
    continue   # this opponent folds preflop -> not in the pot
```
— meaning an opponent who has *already called* still gets modeled as possibly not really there.

### Quantified impact

Mixed-treatment experiment (front = guaranteed-in + color-range sample, after = normal VPIP-roll),
hero `8h 7h` / `Qs Qd`, front=`[Blue]`, after=`[Yellow, Blue]`:

| | Today (all 3 flat, VPIP-rolled) | Corrected (front guaranteed) |
|---|---|---|
| 87s | 0.36 (43.0% of samples counted) | **0.31** (100% counted) |
| QQ | 0.68 (43.0% of samples counted) | **0.61** (100% counted) |

Full-field case — hero `Ah Jd`, 3 opponents (`Green, Yellow, Red`) **all already called**, nobody
left to act:

| Treatment | Equity |
|---|---|
| Today (flat, all 3 VPIP-rolled) | 0.52 |
| Corrected (all 3 guaranteed-in) | **0.26** |
| *(reference)* vs 1/2/3 literal random hands | 0.63 / 0.45 / 0.35 |

0.26 sits *below* even the vs-3-random baseline (0.35) — expected, not a bug: color-based ranges
sample from the *top* p% of hands (even `Red` excludes the bottom 15%), so three real,
already-committed opponents holding better-than-random hands is genuinely harder than three
literal random hands. Today's 0.52 is the miscalibrated number — it's quietly assuming a
substantial fraction of those three probably aren't really there.

**Why this happens mechanically** (breakdown by which subset of opponents actually appears in each
MC sample, hero `8h7h`, colors=`['Blue']` vs `['Blue','Yellow']`, 20k sims):

```
colors=['Blue']:               nobody: 90.2%   Blue-only: 9.8% (counted; equity 0.34 in this subset)
colors=['Blue','Yellow']:      nobody: 62.9%   Yellow-only: 26.9% (eq 0.40)  Blue-only: 7.3% (eq 0.33)  both: 2.9% (eq 0.26)
```
Adding Yellow doesn't uniformly stack a second threat onto every sample — it mostly converts
previously-discarded "nobody's here" samples (excluded from the metric by design — this function
computes equity *conditional on being called*, deliberately not counting uncontested folds as wins)
into "Yellow showed up alone" samples, which are *easier* than the original Blue-only mix. This is
why naively adding an opponent can *raise* the computed equity — the intuition that "one more
opponent should mechanically lower equity" only holds once presence is no longer probabilistic.

### Fair-share framing (why this needs a paired feature, not just a bugfix)

Raw equity is meaningless without knowing N. Guaranteed-in equity vs `N` copies of a fixed color,
hero holding AA / AJo / 72o:

| N opponents | AA equity | AA edge vs fair share `1/(N+1)` | AJo equity | AJo edge | 72o equity | 72o edge |
|---|---|---|---|---|---|---|
| 1 | 0.86 | 1.72x | 0.59 | 1.18x | 0.29 | 0.58x |
| 2 | 0.74 | 2.22x | 0.38 | 1.14x | 0.19 | 0.57x |
| 3 | 0.64 | 2.56x | 0.27 | 1.08x | 0.15 | 0.60x |
| 4 | 0.56 | 2.80x | 0.20 | 1.00x | 0.12 | 0.60x |
| 5 | 0.50 | **3.00x** | 0.15 | **0.90x** | 0.10 | 0.60x |

AA's absolute equity craters (0.86→0.50) but its edge *relative to fair share* actually **grows**
with N (adding weak-average hands drags the field average down faster than it drags AA down). AJo
crosses *below* fair share by 5-way — a below-average multiway hand despite being a perfectly fine
heads-up one. 72o is uniformly bad at any N. This is the mechanism that makes "just fix equity" not
the whole story — the network needs the *relationship* between equity and N, not only a corrected
equity number.

## Proposed new engineered features

### `equity_edge` — equity's edge over the field-size fair share

Something like `equity * (num_active + 1)` (1.0 = exactly average for this field size, >1 better,
<1 worse) — NOT plain `fair_share` (`1/(N+1)`) alone. Plain fair-share is a pure, deterministic,
*unary* function of the already-present `num_active` feature (`ctx[5]`) — a network reconstructs a
1-input-to-1-output transform trivially, so it would add close to nothing on its own. The genuinely
hard part is the *interaction* between equity and N (a multiplicative relationship), which plain
feedforward layers are known to approximate inefficiently from raw inputs alone — precomputing the
cross-term directly is the same rationale recommendation systems use explicit cross-product
features for (factorization machines, wide-and-deep) rather than trusting depth alone to
rediscover a multiplication. Correlated inputs are not a multicollinearity problem for a
gradient-descent-trained network the way they'd be for closed-form linear regression — the
determining question is "does this reduce a real learning burden," not "are these correlated."

### `hand_strength` — pure card quality, independent of field

Decouples "how good is my hand on its own merits" from "how good is it given this specific field"
— maps onto the classic academic poker-AI hand-strength-vs-potential framing. Cheap to compute:

- **Preflop**: found a better concrete source than originally proposed — `preflop_equities.csv`
  (repo root, generated by `scripts/math/generate_equities.py`) already contains exactly this
  feature, precomputed: all 169 canonical starting hands (13 pairs + 78 suited + 78 offsuit),
  each with equity vs 1 random opponent from a dedicated 10,000-sim run. Sample:
  ```
  AA,0.8515   KK,0.8256   QQ,0.8038 ... AJo,0.6412 ... 52o,0.3419  32o,0.3268
  ```
  This is a strictly cleaner source than the in-simulator `_get_preflop_ranked()` cache: that
  cache scores 1326 *specific card combos* (not canonicalized to 169 hand classes) at only 80 MC
  sims each — noisy per-combo, and it only stores rank order, not an actual equity value, because
  its only current consumer is "top p% of this list == a p-wide range" slicing. `preflop_equities.csv`
  gives a real, low-variance (10k-sim) equity number directly, ready to use as-is with an O(1)
  lookup — no runtime MC call needed for the preflop case at all. One piece of new plumbing this
  requires: a canonicalize-to-hand-class helper (2 concrete hole cards -> one of the 169 strings,
  e.g. `AhKs` -> `"AKs"`) — searched the codebase, this doesn't exist yet (only the reverse
  direction, canonical-hand -> concrete-card sampling, exists in `_get_preflop_ranked`). Trivial
  (rank pair + suited/offsuit/pair flag) but not free.
- **Postflop**: hero's raw equity vs exactly 1 random opponent (no color/range modeling, no
  participation uncertainty) — cheaper than the range-aware MC call already being made. No
  precomputed table applies here (board is variable), so this stays a live MC call, same as
  originally proposed.

Honest caveat: hero's raw hole cards and the board are already fed to the model directly as
dedicated embedding tensors (`ContractV12.to_tensors` returns `hole, board, ctx, act` as four
separate tensors, not folded into the scalar `ctx` vector), so the network could in principle
derive pure hand strength itself, especially preflop (a fixed 169-canonical-hand lookup, a
well-scoped pattern-recognition task). Precomputing it anyway is consistent with the project's
existing philosophy — `equity`, VPIP/AGG aggregates, and pot odds are ALL precomputed rather than
left for the network to reconstruct from raw cards/history, for the same sample-efficiency reason,
on a model this size (not a giant foundation model with correspondingly large data).

## What "fixing Finding 2" actually requires (scope note, not yet designed in detail)

Unlike Finding 1 (live-only), this touches the SHARED `compute_range_aware_equity` /
`_calculate_range_aware_equity` used by both training self-play (line 1007) and live serving
(`PHPHelp.py`'s import). A real fix needs the function (or a new sibling function) to accept a
front/after split instead of one flat `opp_colors` list, and the training call site
(lines 1000-1007) to actually pass `current_actor`/`button_seat`-derived groupings instead of a
flat `range(1,6)` scan. This changes what `equity` means for every training example — same
category of change as V20's own contract-rescale (a foundation-level recalibration), and per the
project's established practice needs a fresh version/training run to recalibrate against, not a
patch to an in-flight run. `equity_edge` and `hand_strength` are additive context-vector entries
(`ContractV12`, `versions/v13/core/contract.py`) — a `context_dim`/`contract_version` bump, same
shape as V20's own `contract_version` 3→4 bump.

## Implementation (2026-07-16/17)

Built per explicit user request ("implement the specs in V20_preflopEq"), disregarding V20 root's
own carried-forward backlog (that stays a separate future pass). `versions/v20_preflopEq/` scaffolded
as a clone of `versions/v20/`.

**Extra finding made while implementing** (not in the original analysis): `versions/v20/self_play/
train.py::vectorize_hand_samples` — the function that builds the ACTUAL gradient-training context
tensors — never received V20's own `/100,/250` rescale. It kept a separate, hardcoded `/400,/1000`
copy of the same feature math, while every inference path (self-play rollout via `simulator.py::
_query_model_decide`, and live serving via `core/decision.py`) already used the new `/100,/250`
scale through `contract.py`. That means the currently-deployed V20 model is gradient-fit to one
scale but ACTED ON (both during its own training rollout and live) at a 4x-different scale for the
same real chip amount — a genuine train/rollout+live mismatch, confirmed by direct code trace
(`ctx_t` from `vectorize_hand_samples` is what `model(...)` backprops against — see train.py line
~984). Flagged to the user before proceeding (would otherwise have been silently inherited by this
clone); user chose to fix it as part of this build. Fixed by factoring the scale/clamp math into
one place (`contract.py`'s `scaled_stack_bb`/`scaled_pot_bb`/`scaled_call_bb`) that both
`ContractV12.to_tensors` and `vectorize_hand_samples` now import and share, so this exact class of
drift can't recur. (Whether the currently-deployed V20 live model itself should be retrained is a
separate decision, out of scope here — not touched.)

**What was built:**
- `core/contract.py`: `context_dim` 35→37. `equity_edge_feature(equity, num_active)` (pure,
  `equity * (num_active+1)`) and a `preflop_equities.csv`-backed `hand_strength` lookup
  (`canonical_hand_key` + `preflop_hand_strength`, O(1), 169 canonical hands / 10k sims each —
  the exact table the user pointed at mid-investigation, a strictly cleaner source than the
  originally-proposed in-simulator rank cache). Both appended as new ctx indices 35/36 — every
  existing index unchanged.
- `core/board_state.py` (shared): added `hand_strength: float = 0.5` to `BoardState` (additive,
  inert for every other version's contract).
- `self_play/simulator.py`: `compute_range_aware_equity`/`_calculate_range_aware_equity` gained a
  `front_colors` param (opponents already acted+committed this betting round — guaranteed in, no
  VPIP fold-roll; an all-in seat counts as front too). The training call site now tracks
  `acted_this_round` per seat (reset each street, reset again on any raise since that reopens
  action for everyone else) and splits hero's opponent list into front/after by that ground truth,
  instead of one flat `range(1,6)` scan. `_hand_strength()` added (CSV lookup preflop, a cheap
  200-sim vs-1-random MC call postflop), computed once per decision alongside `eq` and threaded
  through `table_state_dict` to `_query_model_decide` (covers hero's own rollout decisions AND
  every opponent NN query uniformly) and into the hero's recorded training decision point.
- `self_play/train.py`: `vectorize_hand_samples` fixed (see above) + the two new ctx entries.
- `core/manifest.py`: `contract_version` 4→5. `self_play/config.yaml`: `version: "v20_preflopEq"`,
  `target_hands: 75000` (first production run, per user request — checkpoints every 25k already
  give natural 25k/50k sanity-check points).
- **Finding 1 fix, live-only** (`PHPHelp.py`): `opp_colors` construction and
  `_classify_opponents_by_action_order`'s `colors_for` both changed from silently dropping a
  `None` HUD color to mapping it to `'Yellow'`. This is independent of which model is active, so
  it applies immediately regardless of v20_preflopEq's own training/deployment status.
- **Deferred, deliberately**: full live wiring for v20_preflopEq itself (a `V20PreflopEqModelEngine`,
  a `core/decision.py` registry entry + its own bridge, and switching `PHPHelp.py`'s
  `compute_range_aware_equity` call to the front/after signature for this specific model) — same
  pattern V19/V20 themselves followed (train → verify → wire into serving only once validated).
  `_classify_opponents_by_action_order` already computes a front/after split for DISPLAY today;
  reusing it to actually feed the equity call is the concrete next step once this version trains
  and passes `model_verify`.

**Verified standalone before training** (scratch scripts, not committed): `canonical_hand_key` +
`preflop_hand_strength` against known CSV rows (AA 0.8515, AKo 0.6579, 72o≈0.35); `ContractV12.
to_tensors` produces a 37-wide ctx with `equity_edge`/`hand_strength` landing correctly in the last
two slots; `compute_range_aware_equity` with `front_colors` reproduces the SPECS.md-quantified
direction (front-guaranteed corrected equity measurably BELOW the old flat-rolled number, ~0.32 vs
~0.38 for the 87s/[Blue,Yellow] case, matching the ~0.31/~0.36 figures above) and `front_colors=None`
still matches the old flat call; 80 real simulated hands all carry a valid `hand_strength` in
[0,1]. `overfit_sanity` (both v20 and v20_preflopEq, same script) — critic MAE is borderline-noisy
at the script's default 600 steps on a 64-hand batch (v20 0.96bb PASS, v20_preflopEq 1.06-1.32bb
across reruns, threshold 1.0bb) purely from batch-composition/GPU nondeterminism (the front/after
split changes how many `random()` calls happen per simulated hand, so the same seed no longer
produces an identical fixed batch); re-run at 2000 steps converges cleanly to 0.43bb, confirming
correct end-to-end gradient wiring through all 37 context features, not a plumbing bug.

## model_verify extended (2026-07-17)

Per user invitation ("if you think the verify model report is missing some analysis... feel free
to extend it"), extended `tools/model_verify/`:

**Bug found and fixed, not just an extension**: `tools/model_verify/scenarios.py::build_ctx` --
the synthetic-context builder EVERY FAST check uses -- had ALWAYS hardcoded the legacy
`/400,/1000` money-feature scale, with no awareness that V20 (contract_version=4, currently
DEPLOYED LIVE) uses a different `/100,/250`-clamped scale (see `versions/v20/core/contract.py`).
Confirmed by checking every version's `contract_version`: v13-v19 are all `<=3` (so the hardcoded
legacy scale happened to be correct for them by construction), only V20 (4) and this version (5)
use the new scale. Every FAST check that varies `stack_bb`/`pot_bb`/`call_bb` -- which is all of
them except `action_diversity`'s equity-only axis -- had therefore been feeding V20 a systematically
~4x-wrong-scale synthetic stack/pot/call for every run since V20 shipped (e.g.
`deep_stack_ood_guard`'s stack=15-40bb sweep landed at ctx[1]=0.0375-0.10, when V20's real
pipeline computes 0.15-0.40 for those same depths). SLOW checks (`vpip_adapts_to_style`,
`bb100_vs_standard_fields`, `beats_frozen_predecessor`, `beats_offformula_stress`) were NOT
affected -- they run the real simulator/contract.py, never `scenarios.py`.

Fixed by adding `_money_scale(contract_version)` to `scenarios.py` (legacy uncapped /400,/1000 for
`<=3`; clamped /100,/250 for `>=4`) and threading `contract_version=rc.manifest.contract_version`
through every `FAST_CHECKS` call site in `checks.py` (8 call sites). `build_ctx` also now accepts
`contract_version>=5`-gated `hand_strength`/`equity_edge` overrides (auto-derived defaults so
existing checks don't need to change), and `build_tensors`' context width is now dynamic instead
of a hardcoded 35.

**Re-ran FAST checks against the LIVE V20 model (`expert_main_200k.pth`) with the fix** to see
whether this actually mattered, not just theoretically: it does -- `equity_ablation_monotonic`
now reads P(fold) 1.00→0.00 (a sharp, fully decisive boundary) vs whatever softer curve the wrong
4x-compressed stack scale produced before; `air_folds_mostly` now reads 100% fold at ~12% equity
(PASS, was WARN-range material under the old scale); `deep_stack_ood_guard` still FAILs but at a
DIFFERENT specific cell (`eq=0.55, stack=15bb`) than the "1 failing cell... roughly FLAT across
stack depth" narrative recorded in `versions/v20/SPECS.md`/`versions/v19/SPECS.md`. This means:
V20's real decision behavior under correctly-scaled synthetic probes is more extreme/decisive than
the historical FAST-check record showed, and any FAST-check-specific narrative detail in V20's own
SPECS.md history should be treated as unreliable (re-run needed) going forward -- though the actual
deployment-gating checks (the SLOW ones) were never affected by this, so V20's deploy decision
itself doesn't need revisiting on this basis alone. Flagged to the user; no action taken on V20
itself beyond this note (out of scope for this pass).

**Two new FAST checks added** (`hand_strength_sensitivity`, `equity_edge_sensitivity`, both
SKIP when `context_dim<37`, WARN-only/diagnostic -- no established expected magnitude yet, this
is the first version carrying these features): hold equity/stack/pot/field FIXED and swing ONLY
the new feature (a synthetic ablation -- in real play it's never this decoupled, but isolates
whether the network reads that ctx slot at all) via total-variation distance between the resulting
policies. Run against the 25k restore checkpoint: `hand_strength_sensitivity` PASS (0.105 avg
shift), `equity_edge_sensitivity` PASS (0.353 avg shift) -- both new features are measurably
influencing the policy already at 25k hands, not inert padding.

## Status

**Training complete (75k hands, `expert_main.pth`), verified.** 25k checkpoint FAST model_verify:
7 PASS/2 WARN/1 FAIL. 50k: 8 PASS/1 WARN/1 FAIL, `equity_ablation_monotonic` already fully sharp
(P(fold) 1.00→0.00) matching live V20's shape. Clean run throughout -- no errors/warnings/
tracebacks in the training log (1400+ lines).

**Final full `model_verify` (FAST + SLOW, `expert_main.pth`, 75k) -- 11 PASS / 1 WARN / 1 FAIL / 1 SKIP:**
- `vpip_adapts_to_style` **PASS** -- short +6.6pt, deep +7.1pt (both clear the 5pt gate). This is
  the metric most directly downstream of Finding 2's front/after equity fix; a real positive
  signal the calibration change paid off in measured behavior, not just synthetic sensitivity.
- `bb100_vs_standard_fields` **PASS**, positive across all 4 fields (loose_short +16.8, loose_deep
  +36.2, tight_short +22.8, tight_deep +61.1 BB/100) -- no prior baseline existed (first run for
  this version); recorded as the new baseline in `tools/model_verify/baselines.json` for future
  regression tracking.
- `beats_offformula_stress` **PASS** (+31.3 BB/100 short, +66.0 deep vs the structurally-different
  `TieredLookupBot`) -- no overfit-to-training-formula signal.
- `hand_strength_sensitivity` **PASS** (0.107) / `equity_edge_sensitivity` **PASS** (0.641) -- both
  new features still measurably driving the policy at full training length (up from 0.105/0.353
  at 25k for hand_strength/equity_edge respectively -- equity_edge's influence grew substantially
  over the run).
- `deep_stack_ood_guard` **FAIL** (eq=0.55, stack=15bb) and `free_check_low_fold` **WARN** -- both
  the SAME long-standing soft spots every version in this line carries at full maturity (V19, V20
  included); neither targeted nor regressed by this pass.
- `beats_frozen_predecessor` **SKIP** -- copied V20's `expert_main_200k.pth` in as `frozen_v20.pth`
  (the established convention), but it fails to load: V20 is context_dim=35, this version is 37 --
  the exact "no per-model contract-selection mechanism yet" limitation V20's own SPECS.md already
  flagged as backlog when it made the same call about `nit`/`tag` frozen opponents. Not fixed here
  (out of scope); means this version has no direct head-to-head number against its immediate
  predecessor, only the field/style/generalization checks above.

## Deployed live (2026-07-17)

Per explicit user decision (given the `beats_frozen_predecessor` SKIP -- no direct V20 comparison
exists -- weighed against the strong `bb100_vs_standard_fields`/`vpip_adapts_to_style`/
`beats_offformula_stress` PASSes): wired in as the active live model.

- `core/models/v20_preflopEq_engine.py` (new) -- same pattern as `v20_engine.py`, loads
  `expert_main.pth`.
- `core/decision.py` -- registered as `'Herocules (v20_preflopEq)'`, set as `active_model_name`.
  Added `self.bridge_v20_preflopEq` (own `ContractV12` instance, 37-dim) since it can't share
  `bridge_v13` (35-dim, old scale) or `bridge_v20` (35-dim, new scale) -- gated by a new
  `is_v20_preflopEq_model` flag, resolved BEFORE `is_v20_model` (substring trap: `'v20'` is
  contained in `'v20_preflopeq'`, so the existing name-based fallback checks would otherwise
  misfire true for this model under the old ordering). Included in `is_sized_model`/
  `is_actor_policy` alongside V14/V15/V17/V17_gauntlet/V19/V20 (same 6-action head + slider
  sizing). V20 stays fully intact in the registry as the one-line rollback (`active_model_name =
  'Herocules (v20)'`).
- `PHPHelp.py` -- added a `'v20_preflopeq'` branch (checked BEFORE `'v20'`, same substring trap)
  importing this version's own `compute_range_aware_equity`. For this model specifically, the
  front/after split (Finding 2) now actually DRIVES the equity call -- `_classify_opponents_by_
  action_order`'s `colors_in_pot`/`colors_still_to_act` (previously display-only) are passed as
  `front_colors`/`opp_colors`, falling back to the flat legacy call if the dealer button wasn't
  detected that frame. Also added a live `hand_strength` computation (preflop O(1) lookup,
  postflop a 200-sim vs-1-random MC call via the same `self.evaluator` already used elsewhere),
  gated to only run when this model is active, set directly onto `board_state.hand_strength`
  after `to_board_state()` (that method doesn't take it as a constructor param -- the field is
  additive/optional, same pattern `equity` itself already uses via `equity_meta`).

Verified end-to-end with a synthetic `BoardState` through the real `PokerDecisionEngine`: routes
through the new 37-dim bridge, loads the checkpoint, produces a sane decision (AKs/66% equity/deep
stack -> ALLIN, reasonable). `PHPHelp.py`/`core/decision.py` both compile clean.

## Carried forward (unchanged, not addressed by this doc)

Everything already open on `versions/v20/SPECS.md`'s backlog — `policy_tightness_bb`
threshold-near-eq-0.45 lead for `deep_stack_ood_guard`, `short_stack_polarization` [P3]
regression at 200k, the Past-Self seat-loop rework, [P5]/[P6] (size-blind history tokens, no
opponent-action attribution), `model_verify` weighted composite score. This subversion is scoped
narrowly to the preflop/range-aware equity calibration items above; it does not touch or resolve
any of those.

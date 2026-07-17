# Known Model Shortcomings — Tracked Backlog

**Date Recorded**: 2026-07-17
**Related Files**: spans the whole model line — see each entry's own references.

## Purpose

A standing, living catalog of identified weaknesses in the Herocules model line, independent of
which version is currently active. This is NOT a per-version snapshot (those live in
`references/V*/specs.md`) — it's the cross-version comparison point.

**How to use this doc:**
- Before diagnosing a new live-play or training-time issue, check whether it matches an existing
  entry here first — a "new" observation is very often a known item resurfacing.
- When investigating or shipping a new version, re-check each OPEN/PARTIAL item's status against
  that version's actual behavior (model_verify results, live sessions) and update the entry's
  **Last confirmed** field + status. Don't just append a new entry for the same underlying issue.
- New shortcomings get a new entry in the relevant category, following the same
  Simple / Technical / Suggestion structure.
- Move an item to **Resolved** only when a version has actually verified the fix (not just
  attempted it) — keep resolved items listed (briefly) so a future regression is recognized as a
  regression, not treated as a fresh discovery.

## Status legend
- 🔴 OPEN — actively present, unaddressed
- 🟡 PARTIAL — improved, tried-and-failed, or resolved-with-a-tradeoff
- 🟢 RESOLVED — fixed and verified; kept for history/regression pattern-matching
- ⚪ METHODOLOGY — a gap in how we validate/tool, not a model behavior bug

---

## Betting behavior

### [BET-1] No middle gear — shove-preference 🔴 OPEN
**First identified**: V15 SPECS.md (`[P1]`, framed as a raise-fraction-floor artifact, believed
subsumed by V15's stack-range widening). **Reconfirmed**: V20_preflopEq / V20_preflopEq_AI
(2026-07-17) — NOT actually resolved by V15 as once believed.

**Simple**: We rarely bet a normal amount — decisions are mostly fold, call, or shove all-in;
medium-sized raises (33%/66%/pot) are almost never actually chosen even though the option exists.

**Technical**: `model_verify`'s `action_diversity` check shows RAISE_33/RAISE_66 essentially never
winning argmax across the whole equity×stack test grid, in every checkpoint of V20_preflopEq and
V20_preflopEq_AI (`{'fold':9,'allin':10-11,...}`, at most 1-5 raise-bucket cells out of 21). Root
cause traced directly in `opponent_bots.py`'s `FuzzyPlayerArchetype`: once a training opponent's
equity clears its `need_for_value` threshold, it calls/raises "regardless of price" (verbatim in
the code) — a min-raise and a shove extract identical zero-extra-fold-cost value from those hands,
so the critic learns ALLIN as the dominant action. Confirmed via the trained critic's own Q-values:
clean, monotonic ALLIN > RAISE_POT > RAISE_66 > RAISE_33 > CALL at every equity/stack combo tested,
roughly double the next-biggest size, no smooth EV gradient favoring a middle size.

**Suggestion**: Make the heuristic bots' value-branch price-sensitive (continuation probability
should decay with bet size even above the value threshold, not stay flat at "always continue").
**Already tried and failed**: diversifying the opponent POOL toward real NN opponents
(V20_preflopEq_AI, 2026-07-17) — a promising early read at 35k hands fully faded by 140k/150k.
Pool diversity alone is not sufficient; the fix needs to go directly at the opponent response
function.

### [BET-2] Short-stack polarization — residual flatting 🔴 OPEN
**First identified**: tracked since V15/V16 era as `[P3]`. **Last confirmed**: V20 (WORSENED with
more training, 0.12→0.25 avg P(call) between 120k→200k).

**Simple**: In clear shove-or-fold spots (short stack facing a raise sized near the stack), we
sometimes just call instead of shoving or folding — a real strong player would rarely flat here.

**Technical**: `short_stack_polarization` (WARN-gated, not a hard FAIL). Avg P(call) in a
shove-or-fold equity/stack grid should be near 0; it isn't, and it's trending worse with more
training in at least one version, not better. Root cause not yet isolated — a candidate is the
counterfactual EV target not sharply enough penalizing the dominated middle option at these
depths, but this hasn't been directly tested.

**Suggestion**: Dedicated investigation needed — hasn't been root-caused. Worth checking whether
this correlates with [BET-1]'s mechanism (deterministic opponent response smoothing away the
penalty for flatting) before assuming a separate cause.

---

## Stack depth

### [STACK-1] Deep-stack OOD trash-jam 🔴 OPEN
**First identified**: live incident, V14/V15 era (K9o jammed 20bb into a single limper).
**Last confirmed**: V20_preflopEq_AI 150k (`eq=0.55, stack=20bb -> ALLIN@0.37`) — present in
EVERY version checked so far (V19, V20, V20_preflopEq, V20_preflopEq_AI), at slightly different
specific cells each time.

**Simple**: At medium-deep stacks (15-40bb) facing a modest bet with a marginal hand (~45-55%
equity), we sometimes jam all-in when we should just call or fold.

**Technical**: `deep_stack_ood_guard` regression check. V20's own investigation found the failure
grid shape changes across versions (V19: flat across the whole 15-40bb range; V20: narrowed to
fewer cells) but never actually clears the gate. Likely shares [BET-1]'s mechanism (shoving reads
as "free" value against a population that doesn't punish overbets), though not directly proven —
worth testing together.

**Suggestion**: Same lever as [BET-1] — the opponent price-sensitivity fix is the most likely
thing to move this too, since it may be the same underlying cause in a different equity band.

### [STACK-2] Beyond-50bb extrapolation (clamped, not solved) 🟡 PARTIAL
**First identified**: V20 pre-deployment smoke test (a flopped set of aces showed fold%
climbing 10%→32% from 45bb→150bb, a real OOD extrapolation artifact).
**Status**: mitigated via a live-serve clamp (stack/pot/call-derived context features capped at
the training ceiling regardless of real table depth), not via actually extending training depth.

**Simple**: We've never actually trained beyond 50bb effective stacks. At truly deep tables
(80-150bb+), the live clamp keeps us from breaking, but we're not applying any deep-stack-specific
skill — we just can't perceive the extra depth at all.

**Technical**: `stack_depth_mix` curriculum (5-50bb bands) has capped training depth since V15.
`contract.py`'s stack/pot/call_amount-derived features are clamped to that ceiling before being
fed to the model at serve time, verified to hold fold-rate flat (~13%) from 20bb to 150bb+. This
is a safety patch against a real OOD failure mode, not learned deep-stack strategy (multi-street
implied odds, etc. past 50bb were never in any training example).

**Suggestion**: Extend `stack_depth_mix` to cover deeper bands in a future version if deep-stack
play quality (not just safety) becomes a priority.

---

## Opponent modeling

### [OPP-1] Overfitting-to-deterministic-training-formula risk 🟡 PARTIAL
**First identified**: discussion 2026-07-15, motivated `check_beats_offformula_stress`.
**Last confirmed**: V20_preflopEq_AI 150k — PASSES (+34.7/+65.0 BB/100 short/deep vs
`TieredLookupBot`), but [BET-1] shows the model HAS learned something population-specific
(the shove-preference), so this isn't fully clean.

**Simple**: We've mostly practiced against predictable, formula-driven opponents. Overall winrate
against a structurally different opponent still holds up, but at least one specific behavior
(shove-preference, see [BET-1]) is a direct product of that specific training population, not
general poker skill.

**Technical**: `FuzzyPlayerArchetype`'s fold/continue decision is a deterministic threshold given
equity+price (only the raise-vs-call split among continuing hands has randomness). The
off-formula stress test (`TieredLookupBot`, a price-insensitive-by-street lookup table) is itself
still a fairly mechanical opponent shape — it doesn't prove robustness against a genuinely
human-like or solver-like opponent.

**Suggestion**: [BET-1]'s fix (price-sensitive value branch) is the direct lever. Separately,
consider a genuinely different opponent archetype family for stress-testing beyond
`TieredLookupBot`.

### [OPP-2] No per-opponent action attribution 🔴 OPEN
**First identified**: V16 ROADMAP `[P6]`. **Status**: unchanged since — no architecture work done.

**Simple**: We don't track who specifically did what during a hand — we only see "someone raised,"
not which particular opponent (with their own known tendencies) did it.

**Technical**: Action-history tokens aren't per-seat — the sequence model sees a coarse
fold/call/raise token stream, not attributed to a specific opponent. HUD stats (VPIP/AGG color)
are static per-seat CONTEXT features, decoupled from the actual in-hand action sequence.

**Suggestion**: Would need a real architecture change (per-seat action tokens in the sequence
input), not a quick fix. Flagged as backlog across multiple versions, never built.

### [OPP-3] Size-blind action history 🔴 OPEN
**First identified**: V16 ROADMAP `[P5]` (generalized from an original call_amount-only framing).
**Status**: the CURRENT bet's size is fed via context features (and was rescaled in V20), but the
HISTORY of past bet sizes within a hand still isn't.

**Simple**: We don't fully track how big previous bets were earlier in the same hand — the action
history is somewhat size-blind, so unusual historical sizing from an opponent may not register.

**Technical**: `act_ints`/history tokens use a coarse vocabulary (fold→7, call→3, any-raise→6
regardless of size) — see `ContractV12.to_tensors`'s action sequence construction. The width was
never widened to carry size info alongside action type.

**Suggestion**: Same category as [OPP-2] — a sequence-encoding change, not addressed by any
contract iteration so far (V13→V20_preflopEq_AI all share this gap).

### [OPP-4] Live front/after equity — reopened-action blindness 🔴 OPEN (live-only)
**First identified**: 2026-07-17, while wiring V20_preflopEq's Finding 2 fix into live serving.

**Simple**: If an opponent calls, then someone else raises, we might still treat that first caller
as "definitely staying in" even though they haven't actually faced the new raise yet and could
still fold to it.

**Technical**: `PHPHelp.py`'s `_classify_opponents_by_action_order` is a pure TABLE-POSITION
heuristic (button-relative rotation order) — it has no real per-seat action-state to check whether
a subsequent raise reopened action for an earlier caller. Training's own simulator tracks this
correctly (`acted_this_round`, explicitly reset on every raise) because it has ground-truth access
to the full betting sequence; the live path only has vision-derived table state, not true
per-seat action history. This was a known, documented limitation of the classifier when it was
DISPLAY-only; it's now load-bearing (feeds the actual live equity computation for V20_preflopEq/
V20_preflopEq_AI), so the gap matters more than it used to.

**Suggestion**: Would need live per-seat action-state tracking (not just position) mirroring
training's exact mechanism. Scoped conceptually, not started — the live pipeline currently has no
per-seat "have they acted since the last raise" tracking at all.

### [OPP-5] Opponent-style/VPIP-AGG-color read may not be load-bearing 🔴 OPEN
**First identified**: `model_verify`'s `opponent_style_sweep` check, added 2026-07-15 (first
version to carry it). **Confirmed**: V20_preflopEq_AI (spread 0.000, completely flat) and V21
(spread 0.001, then re-confirmed at 0.004 after widening the sweep from 3 to 5 equity points to
cover the actual fold/continue transition zone — ruling out "the two saturated endpoints just
happened to hide a real mid-curve difference").

**Simple**: Facing the identical bet at the identical hand strength, we play essentially the same
way against a tight, disciplined opponent (a "nit") as against a loose, aggressive one (a
"maniac") — the model doesn't seem to actually use its read on who it's up against.

**Technical**: `check_opponent_style_sweep` holds equity/stack/pot/call fixed and only varies the
fed-in opponent VPIP/AGG archetype (Blue=nit through Red=maniac). Every version tested shows
P(fold) essentially flat across all four archetypes (spread ≤0.004), despite the model receiving
per-opponent VPIP-color/AGG-color context features since early versions. This is a DIFFERENT gap
from [OPP-2]/[OPP-3] (those are about the in-hand ACTION sequence not being per-opponent-attributed
or size-aware) — this is about whether the per-seat HUD-style CONTEXT features (present since early
versions, no architecture change needed to use them) are actually load-bearing at all, and the
answer looks like no.

**Suggestion**: Root cause not yet isolated. Worth checking whether this is a genuine dead input
(plausible: the deterministic heuristic-bot training population doesn't reward differentiating by
style enough to matter for the loss — a similar population-level flatness to [BET-1]'s root cause)
versus an actual wiring/normalization issue in how `VPIP_MAP`/`AGG_MAP` values reach the model. A
cheap first check: an isolated sensitivity ablation on just the VPIP/AGG color inputs (mirroring
`equity_edge_sensitivity`'s approach) before assuming it's purely a training-population artifact.

---

## Format fit

### [FMT-1] No ICM awareness 🔴 OPEN
**First identified**: 2026-07-17, live session on a Double-or-Nothing board.

**Simple**: In tournament formats (like Double or Nothing), we make decisions purely on chip
count, not on how much actually busting out costs you — so we may fold spots a human would call
wider in a real bubble/survival situation.

**Technical**: Trained purely on cash-game-style BB/100 profit; no ICM (tournament equity) model
anywhere in the pipeline (simulator payouts, EV targets, or evaluation). Confirmed live: folds
that are clean chip-EV-correct even at 2-3bb effective stack, where ICM-aware strategy might
differ.

**Suggestion**: Would need genuinely different training targets (ICM-weighted payouts instead of
raw chip profit) — a substantial, separate project, not a config tweak. Only relevant if
tournament/DoN formats are a priority over cash-style play.

---

## Validation & tooling (methodology gaps, not model behavior bugs)

### [VAL-1] No external GTO/solver ground truth ⚪ METHODOLOGY
**Simple**: We've never checked our play against real game-theory-optimal solutions — all
validation is "do we beat our own training opponents," not "are we close to unexploitable play."

**Technical**: The entire `model_verify` suite (FAST + SLOW checks) is self-referential — it tests
against the project's own simulator, its own heuristic archetypes, and its own frozen
predecessors. No external solver (e.g. PioSOLVER-derived ranges) has ever been used to validate a
specific spot.

**Suggestion**: Spot-check a handful of well-known solved situations (e.g. published heads-up
shove/fold charts) against actual model output, as an independent axis alongside the existing
suite.

### [VAL-3] `free_check_low_fold` residual mass 🟡 PARTIAL
**Simple**: When there's no cost to seeing another card, the model's raw output occasionally still
shows some (fully masked, never-executed) desire to fold — which doesn't make sense, since folding
a free option is never correct.

**Technical**: Raw policy fold-mass when `call_amount=0` is masked to zero and renormalized by
`core/decision.py` before a decision is ever made, so this NEVER reaches a live action. But the
raw number itself is high in every version checked (V20_preflopEq_AI: 1.000, maximal). Root cause
not identified — a candidate is the equity-primary base head not fully internalizing that a free
continuation always dominates folding, but this hasn't been tested directly.

**Suggestion**: Not urgent (fully masked). Worth a dedicated look only if it turns out to
correlate with one of the OPEN items above (e.g. the same base-head calibration issue behind
[BET-1]).

### [VAL-4] New-feature live track record is thin 🟡 PARTIAL
**Simple**: The newest features (hand strength, equity edge, and the front/after "who's already in
the pot" logic) have only been tested in one real live session so far.

**Technical**: Verified via `model_verify` (both FAST sensitivity checks and SLOW field-winrate
checks) and one live Double-or-Nothing board. No long-run live statistics accumulated yet. The
turn-history recorder was only updated to persist these fields on 2026-07-17 (previously
live-only, discarded after each decision) — so accumulation starts now, not retroactively.

**Suggestion**: Keep monitoring live sessions now that the recorder captures these fields;
revisit this entry once enough live hands have accumulated to say something statistical.

---

## Resolved (kept for regression pattern-matching — do not re-flag as new)

### [RESOLVED-1] VPIP does not adapt to opponent style 🟢 RESOLVED
Originally V16 `[P4]` (VPIP-vs-style flatness — hero's preflop entry range didn't tighten/loosen
with opponent tightness, only postflop aggression did). Range-aware equity + the V20_preflopEq
Finding 2 front/after fix appear to have resolved this: `vpip_adapts_to_style` PASSES with a
growing margin — V20_preflopEq 75k (short +6.6pt, deep +7.1pt) → V20_preflopEq_AI 150k (short
+11.5pt, deep +9.6pt), both clearing the 5pt gate comfortably.

### [RESOLVED-2] model_verify FAST checks fed V20 the wrong context-feature scale 🟢 RESOLVED
Found and fixed 2026-07-17 while extending `model_verify` for V20_preflopEq: `scenarios.py`'s
`build_ctx` hardcoded the legacy `/400,/1000` scale for every version, but V20 (`contract_version`
4+) actually uses a clamped `/100,/250` scale. Every FAST check that varies stack/pot/call had been
silently testing V20 at a ~4x-wrong scale since it shipped. Fixed via a `contract_version`-aware
`_money_scale()` helper; SLOW checks (real simulator) were never affected. V20's own historical
FAST-check narrative in its SPECS.md predates this fix and should be treated as unreliable if
referenced.

### [RESOLVED-3] `vectorize_hand_samples` never received V20's own rescale 🟢 RESOLVED
Found and fixed 2026-07-17 while building V20_preflopEq: `train.py::vectorize_hand_samples` (the
function that builds the ACTUAL gradient-training tensors) kept a stale `/400,/1000` copy of the
context math while every inference path (rollout, live) used the new `/100,/250` scale — a real
train/serve mismatch baked into the deployed V20 model. Fixed by factoring the scale/clamp math
into shared helpers in `contract.py` that both paths import.

### [RESOLVED-4] Unknown HUD color silently dropped from live equity 🟢 RESOLVED
V20_preflopEq Finding 1 (2026-07-17): an opponent with no classified HUD color yet was excluded
entirely from the live equity calculation, as if not contesting the pot. Fixed by mapping unknown
→ 'Yellow' (the codebase's existing "no info" convention), in `PHPHelp.py`.

### [RESOLVED-5, historical] `hero_position` never set during training queries 🟢 RESOLVED
V19: every training-time model query (hero's own AND every opponent's) silently defaulted
`BoardState.hero_position` to Button — confirmed universal across V12-V18, training-only (live
serve was always correct). Fixed by threading each actor's real button-relative position through.

### [RESOLVED-6, accepted-as-policy] Cross-scale predecessor comparison gap 🟢 RESOLVED
Originally `[VAL-2]`, first identified V20 (`beats_frozen_predecessor` SKIP vs frozen `nit`/`tag`
when `contract_version`/context_dim changed), reconfirmed V20_preflopEq, worked around (not fixed)
in V20_preflopEq_AI. **2026-07-17: accepted as by-design, not a gap to close.** Decision: an older
model trained under a stale/incompatible contract is not a useful opponent or comparison baseline
for a newer version regardless of whether a cross-scale query mechanism existed — those older
models weren't strong enough in general for beating them to be meaningful signal, and building a
mechanism to query each frozen opponent through its own original contract would serve a comparison
we don't actually want while the model line is still moving quickly on base behavior. Revisit only
if the model line matures to a point where genuinely strong historical checkpoints exist and a
same-generation baseline becomes valuable again; until then, `beats_frozen_predecessor` SKIPping
across an incompatible contract change is expected, not a defect.

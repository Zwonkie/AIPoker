# V19 — P0 deep-stack fix + hero_position fix + Past-Self investigation

Clones V18 (pure plumbing refactor, never itself trained to completion). This version is a real
training-content experiment: three items pulled forward from V18's carried backlog, user-directed
via `/goal` on 2026-07-16 ("do P0 and point 3) hero position bug ... and 4) Past-self mystery.
After this do the regular smoke-test and full 200k hand training").

## [P0] Deep-stack OOD trash-jam fix

**Failure signature** (`tools/model_verify/checks.py::check_deep_stack_ood_guard`): marginal
equity (0.35-0.55), 15-40bb stack, facing one modest bet, preflop -> ALL-IN argmax. FAILED on 5
straight versions (V15, V16, v16_foldregret, V17, v17_gauntlet) despite each changing the training
algorithm (regret-matching, fold-relative regret, actor-critic) -- none of them touched the actual
defect, which is in target CONSTRUCTION, not the learning rule.

**Root cause:** `versions/v19/self_play/opponent_bots.py`'s `decide_preflop` accepted a `pot_odds`
argument but never used it for the fold/continue bar -- a fixed per-style `call_bar` regardless of
bet size. Contrast `decide_postflop`, explicitly fixed for exactly this in V14 [P1b]:
`continue_bar = pot_odds + style_shift` (rises with bet size). That fix was made once, postflop
only, and never mirrored to preflop -- every version since inherited the unfixed bot unchanged.

`_mc_target_evs_sized` (the per-size counterfactual EV target builder) samples
`bot.decide_preflop(oeq, size_pot_odds)` to estimate the opponent's fold probability for EVERY
raise size including all-in. Since `decide_preflop` ignored `size_pot_odds`, a min-raise and an
all-in shove got the IDENTICAL simulated fold rate -- systematically inflating the all-in EV target
in the equity band just above breakeven (exactly where the check fails worst: v17_gauntlet's own
worst cell was `eq=0.55, stack=40bb -> ALL-IN argmax @ 0.39`).

**Fix:** mirror the postflop `continue_bar = pot_odds + style_shift` pattern into `decide_preflop`'s
`facing_bet` branch (`pot_odds > 0`); the unopened-pot/RFI branch keeps the original flat
VPIP-proxy bar unchanged.

**Standalone spot-check (before committing to training):** for a marginal opponent holding (87s,
oeq~0.44 -- above TAG's old flat call_bar 0.42, below the new shove-facing bar ~0.51):

```
size        size_pot_odds   OLD fold rate   NEW fold rate
0.33xPOT            0.182            0.00            0.00
0.66xPOT            0.182            0.00            0.00
1.00xPOT            0.231            0.00            0.00
ALLIN                0.479            0.00            1.00
```

Confirms the mechanism: OLD code folded 0% to every size including a full shove (the bug,
reproduced exactly); NEW code still continues vs small raises but now folds to the shove. Whether
this shifts the AGGREGATE trained policy (averaged over the full population of dealt opponent
hands, not one contrived example) is what the post-training `deep_stack_ood_guard` gate settles.

## [hero_position] fix

**Root cause:** `BoardState.hero_position` (`core/board_state.py`) defaults to `0` (Button). Every
training-time query -- both the Hero's own decision via `_hero_decide` AND every opponent NN query
via `_opponent_decide`/`NNOpponent` -- funnels through the same `_query_model_decide`, which
constructed `BoardState(...)` WITHOUT ever setting `hero_position`. So every single training query,
for every seat, was fed position=0 (Button) regardless of where that seat actually sat relative to
the dealer button. This is a real, previously-uncaught input-contract gap distinct from the
suspected-but-ruled-out explanations for the Past-Self VPIP mystery (see below) -- confirmed via
research to be universal across every version line (V12 through v18), training-only (the LIVE serve
path, `core/table_state.py::to_board_state`, has always set this correctly from OCR'd dealer
position -- this bug never affected live play, only every training run to date).

`hero_position` is a real, consumed feature: `ContractV12.to_tensors` uses it for `ctx[0]` (the
querying actor's own normalized position) AND derives each opponent's relative position from it
(`ctx[11]`, `ctx[16]`, `ctx[21]`, `ctx[26]`, `ctx[31]` -- one per opponent seat). A stuck-at-0
`hero_position` therefore corrupted 6 tensor slots on every query, not one.

**Fix:** `simulate_hand`'s per-actor `table_state` dict (rebuilt fresh every actor's turn) now
carries `actor_seat` (the current actor's own seat -- 0 for Hero, `current_actor` for opponents)
and `button_seat`. `_query_model_decide` computes `actor_position = (actor_seat - button_seat) % 6`
and passes it as `BoardState(..., hero_position=actor_position)` -- generalizing the existing
`hero_position = (0 - button_seat) % 6` formula (previously only used for a telemetry field, never
reaching the tensor) to any querying seat, matching the live-serve path's own formula exactly.

**Verified end-to-end**, both at the arithmetic level and by constructing real `BoardState`s at
several `hero_position` values and confirming `ContractV12.to_tensors` output actually varies:

```
hero_position=0: ctx[0]=0.000  ctx[11]=0.200
hero_position=1: ctx[0]=0.200  ctx[11]=0.400
hero_position=3: ctx[0]=0.600  ctx[11]=0.800
hero_position=5: ctx[0]=1.000  ctx[11]=0.000
```

**Implication:** every prior version's model has NEVER seen a real position signal during
training -- this is a bigger deal than V18's SPECS.md framed it ("opponent NN queries only";
it's actually universal, hero included). V19 is the first version where position is a genuine,
correctly-labeled training feature for anyone.

## [Past-Self mystery] investigation

Investigated via close code reading (not fixed this version -- see recommendation below). Two
previously-undocumented, real, confirmed asymmetries found; neither fully closes the mystery on
its own.

**Ruled out again (independently re-confirmed, not just re-cited):** checkpoint-refresh staleness
(`past_checkpoint.pth` is written every 5k hands and reloaded FRESH from disk every single
`Pool.starmap` batch -- `train.py`'s `simulate_worker` builds a brand-new `SixMaxSimulator` and
calls `_load_worker_model` every invocation, no long-lived stale in-memory copy); stat-bucket
population differences (`seat_histories` resets fresh every worker/batch, the dashboard EMA is a
rolling window not lifetime-cumulative, curriculum seat-culling applies uniformly by seat).

**Finding 1 -- Hero has a permanent 15% heuristic-anchor floor that `past` never has.**
`_hero_decide`'s `model_share = 1.0 if disable_exploration else 0.80` (`simulator.py:646`) is
**decoupled from `bootstrap_alpha`** -- even once `bootstrap_alpha` decays to its hard, exact 0.0
floor at hand 30k (`train.py:877-885`, confirmed linear-to-zero, no asymptote), Hero's decision
mix stays `5% random + 80% model + 15% heuristic-chart-anchor` for the ENTIRE remaining 170k
hands, because `disable_exploration` is `false` for this training recipe. `_opponent_decide`
(what `past` goes through) has no equivalent floor: its only heuristic gate is
`force_heuristic = roll < bootstrap_alpha` (preflop only), which is genuinely 0 once alpha decays,
and postflop is hardcoded `force_heuristic=False` always. So Hero's OWN measured stats are a
permanent blend with a tighter heuristic (`tag_heuristic`, VPIP 0.22) that `past` never gets
diluted by. Real and provable, but doesn't cleanly explain the magnitude (a 15%-weighted pull
toward a 22%-VPIP anchor shouldn't by itself drag `past` all the way down near that anchor's own
baseline while Hero stays at 40%) or direction on its own.

**Finding 2 -- opponent NN queries get a wrong/incomplete "who's at the table" view (likely the
bigger factor).** `opponents_profiles` (`simulator.py:856-894`) is built ONCE per hand, keyed
`seat_1..seat_5`, describing the archetype styles seated at 1-5 **from Hero's perspective** --
Hero's own seat-0 profile is never in this dict at all. `_query_model_decide`'s 5-opponent-slot
loop (lines ~449-478) uses this SAME static dict regardless of who's actually querying. For Hero's
own query this is correct (seats 1-5 really are Hero's opponents). But when an OPPONENT queries
(including `past`), it gets the IDENTICAL dict -- meaning: (a) one of its 5 "opponent" slots is
describing **itself** (its own seat's profile, nonsensical as a self-view), and (b) **Hero never
appears anywhere in the view at all** -- the one opponent `past` is actually, literally playing
against on every hand is invisible to it. This is a real, confirmed input-contract bug affecting
every opponent-seat query uniformly (not `past`-specific), but it plausibly hits `past` hardest
specifically BECAUSE `past` is the one seat whose "correct" comparison point (Hero) is the one
that's missing.

**Not fixed this version.** Correctly reconstructing a per-actor opponent view (excluding self,
including Hero, for whichever seat is querying) is a real architectural change to
`_query_model_decide`'s seat-loop, not a narrow one-line fix like [P0]/[hero_position] -- it
touches every opponent query's input shape and has no live-serve equivalent to validate parity
against (live serve only ever builds Hero's own perspective). Given the size of that change and
that it needs its own dedicated verification, it's flagged as a concrete, well-scoped next step for
a future version rather than rushed into this training pass. The mystery is investigated, not
closed: two real contributing asymmetries identified, magnitude/direction not fully pinned down.
Recommended follow-up: direct A/B query logging (same state, Hero's model vs `past`'s loaded
checkpoint, log raw pre-blend action distributions) to isolate how much each finding actually
contributes before investing in the seat-loop rework.

## Training + validation results (2026-07-16)

**200k-hand production run**: fresh weights (not warm-started — target semantics genuinely
changed), 3h0m1s, 200,003 hands, zero NaN/crash/traceback throughout. `overfit_sanity` noisy at
the known unseeded-RNG boundary (0.73/1.03/1.86bb across 3 runs, third clean PASS on all metrics)
-- not a regression, matches the documented pattern every version in this line shows. 500-hand
smoke test clean before launch.

**`model_verify --full`: 10 PASS, 1 WARN, 1 FAIL.**

PASSES worth noting:
- `vpip_adapts_to_style`: short tight=28.4%/loose=38.1% (delta +9.7pt); deep tight=27.4%/loose=34.2%
  (delta +6.8pt). Real, measurable style adaptation.
- `bb100_vs_standard_fields`: positive across all 4 fields (loose_short +45.7, loose_deep +63.4,
  tight_short +25.2, tight_deep +46.3).
- `beats_frozen_predecessor`: +56.8 BB/100 over 4000 hands vs a field including frozen
  v17_gauntlet.
- `beats_offformula_stress`: short +25.3 BB/100 (VPIP 41%), deep +83.7 BB/100 (VPIP 39%).
- `free_check_low_fold`: WARN (pre-existing, covered by decision.py's live free-check mask, not a
  new issue).

**`deep_stack_ood_guard`: still FAILS -- [P0] did NOT resolve it.** Worst cell moved (v17_gauntlet:
eq=0.55/stack=40bb/0.39; v19: eq=0.48/stack=15bb/0.36) but the full failure grid tells the real
story: **13 of 25 cells (52%) argmax to ALL-IN**, spanning every equity >=0.43 across the ENTIRE
15-40bb sweep, with the all-in probability essentially FLAT across stack depth (15bb: 0.34-0.36,
40bb: 0.33-0.34 -- barely moves). This is the key diagnostic finding: [P0]'s hypothesis assumed the
failure should SCALE with stack size (a bigger shove compounds the EV-target inflation more at
deeper stacks), but the actual failure doesn't scale with stack at all -- it just switches on at a
roughly fixed equity threshold and stays constant. That threshold (PASS at eq=0.40, FAIL starting
eq=0.43) lines up suspiciously with `policy_tightness_bb`'s own config comment: *"realization
discount on the actor target... below eq 0.45"*. This points at a DIFFERENT root cause than [P0]
targeted -- something in the actor's policy-target discount mechanism creating a threshold
discontinuity near eq 0.45, not the preflop opponent fold-model [P0] fixed. [P0] itself is still a
real, verified, worthwhile fix (confirmed via the standalone spot-check above) -- it just isn't
THE cause of this particular check's failure mode. Not investigated further this pass.

**Deployment decision (user, 2026-07-16):** deploy this checkpoint live as-is rather than block on
`deep_stack_ood_guard` -- every other gate passes strongly enough that the tradeoff favors shipping
the [hero_position] fix and the confirmed wins now, with the OOD guard's new, more precise
diagnostic (threshold near eq 0.45, not stack-scaling) carried forward as backlog for a dedicated
future pass rather than another blind retrain cycle. `core/models/v19_engine.py` created (mirrors
v17_gauntlet_engine.py's pattern exactly -- same bridge, same `_v14_size_to_slider` sizing);
`core/decision.py` registry updated with `'Herocules (v19)'` as `active_model_name`, `is_v19_model`
added to the shared `is_sized_model` union; `PHPHelp.py` dropdown default and range-aware-equity
import branch both updated to v19. Smoke-tested live end-to-end (air hand folds clean, set-of-aces
value-raises appropriately).

## Carried-forward backlog (updated)

- **`policy_tightness_bb` threshold effect near eq 0.45** (NEW, most specific lead for
  `deep_stack_ood_guard` yet found) -- investigate whether the "realization discount below eq 0.45"
  mechanism creates a discontinuity that disproportionately favors ALL-IN right at/above that
  threshold, independent of stack depth. Best next step for finally closing this 6-version-old gap.
- **Past-Self mystery seat-loop rework** (per the investigation above) -- reconstruct a per-actor
  opponent view in `_query_model_decide` (exclude self, include Hero) instead of reusing Hero's
  static `opponents_profiles` dict for every opponent query.
- [P5]/[P6] input-contract gaps (size-blind history tokens, no opponent-action attribution) --
  unchanged, still promoted over further target-formula tuning.
- `model_verify` weighted composite score -- still not built.

## Opponent-architecture refactor (2026-07-16, carried from V18, unchanged this version)

**Purpose:** user-requested, after building `v17_gauntlet` overnight — "would it make sense to
restructure the simulation/training loop so it's easier to slot in a NN or Heuristic bots, so like
they have their own class?" This is a pure PLUMBING refactor: IDENTICAL training recipe/config to
`v17_gauntlet` (same actor-critic/fold-relative mechanism, same curriculum, same INTENDED
opponent-pool composition), verified behaviorally sound via a 20k-hand sanity run before any
further training investment. Not a new training experiment.

### The problem this fixes

Every prior version wired opponents via five separate `self.<style>_model` attributes on
`SixMaxSimulator`, a hardcoded `style -> model` `elif` chain in the seat-assignment loop, and
per-style ACTION-FORCING logic duplicated across `_opponent_decide`'s preflop/postflop branches.
Adding `v17_gauntlet`'s `tag` seat (a model-loading option that slot never had before) took six
separate touches: a new `simulate_worker` param, a new `self.tag_model` attribute, a new `elif`
branch, a new config key, a new POSITIONAL slot in the worker-args tuple, and a new forcing-bypass
condition. That shape is exactly what let a stray leftover line (`opp_model = self.tag_model`
immediately followed by an accidental `opp_model = None`) silently nullify the `tag` seat's model
load for `v17_gauntlet`'s entire 200k-hand run -- see `versions/v17_gauntlet/SPECS.md`
"CORRECTION". No structural reason existed that couldn't happen, and nothing would have caught it.

### The fix: `self_play/opponents.py`

A uniform `Opponent` interface:
- `HeuristicOpponent(style, bot, forced=True)` -- wraps a scripted archetype bot.
- `NNOpponent(style, model, query_fn, error_fn, recording_bot, forced=False)` -- wraps a loaded
  checkpoint, decided via an injected query function (the simulator's own `_query_model_decide`,
  passed in rather than imported, so this module has zero dependency on `SixMaxSimulator`).
- Both share `.decide_preflop(...)` / `.decide_postflop(...)` / `.apply_forcing_preflop/postflop(...)`.
  `apply_forcing_*` is a no-op unless `forced=True` -- the v17_gauntlet forcing-bypass fix
  (don't archetype-force a genuine trained network) is now an explicit per-opponent flag, not an
  implicit `if model is None` check duplicated at two call sites.
- `build_opponent_pool(pool_config, heuristic_bots, query_fn, error_fn, load_model_fn)` -- the
  factory. Given a declarative list of `{style, weight, model?, forced?}` dicts, returns
  `{style: Opponent}`. A style whose model path is absent or fails to load falls back to
  `HeuristicOpponent(forced=True)` automatically -- structurally, there is no branch where a
  requested model can silently resolve to "loaded but never queried."

`simulator.py` changes: `SixMaxSimulator.__init__` drops the five `*_model` params/attributes for
one `self.opponent_pool = {}` (populated post-construction, same pattern `hero_model` already
used). `_opponent_decide` now only owns genuinely SIMULATOR-level concerns (5% exploration mix,
the preflop-only bootstrap heuristic-anchor gate, reading `self.seat_histories` to feed
`apply_forcing_*`) and delegates the actual decision to `opponent['agent']`. The seat-assignment
loop in `simulate_hand` replaced the 5-branch `elif` chain with one dict lookup:
`self.opponent_pool.get(style)`.

`train.py` changes: `simulate_worker` drops `maniac_model_path`/`nit_model_path`/
`sticky_model_path`/`past_model_path`/`tag_model_path` (5 params) for one `opp_pool_config` (a
picklable list of dicts, safe for `multiprocessing.Pool.starmap`). `run_training` drops
`opp_pool`/`opp_weights`/`disable_past_self`/`freeze_past_self`/`frozen_past_filename`/
`nit_model_filename`/`tag_model_filename` (7 params) for the same single `opp_pool_config`. The
`past` seat's dynamic per-batch resolution (a true lagged mirror needs its snapshot path
re-checked every 5k hands, unlike static frozen files) is now driven by a `lagged_self: true`
marker on that entry, resolved fresh into `resolved_pool_config` each batch -- same underlying
mechanism as before (`past_checkpoint.pth`, saved every 5k hands), just declared once instead of
threaded through `freeze_past_self`/`frozen_past_filename` as separate params.

`config.yaml`'s `opponents.pool` is now a list of `{style, weight, model?, forced?, lagged_self?}`
dicts instead of a bare list of style-name strings plus a growing set of bespoke
`*_model_filename` config keys. A legacy list-of-strings `pool` (pre-V18 format) is still accepted
(upgraded to bare heuristic entries) for backward compatibility, though no version in this line
currently uses that path.

### Dashboard now shows WHAT is actually loaded per seat, NOT the archetype slot name

User request (after seeing the dashboard still labeling seats "Nit"/"TAG Bot" even when a real
frozen checkpoint is loaded there): "ensure the interface supports bot names/versions, which
carries over to the output telemetry of the loaded bots." First pass kept the archetype-slot name
as a prefix ("Nit: V15"); user then asked directly why the archetype/slot association is kept at
all when it's often not even accurate for HEURISTICS either (the 'maniac' slot's real bot is `LAG`,
'fish' is `CALLING_STATION` i.e. "Calling Station" -- neither literally named after their slot).
**Final scheme: `"{BotName} ({Heuristic|NN})"`, the archetype-slot name dropped from the display
entirely** (it's now purely an internal stat-bucket/forcing-rule bookkeeping key, `Opponent.style`,
never shown). Every `Opponent` carries `.display_name` + `.kind` ("Heuristic"|"NN") +
a `.label` property (`f"{display_name} ({kind})"`); `opponents.describe_pool_entry(entry)` returns
the `(name, kind)` tuple both the built `Opponent` and the dashboard's `seat_labels`
(`run_training`, built dynamically from live `opp_pool_config`, passed into `print_dashboard`)
derive from -- one source, so a label can never drift from reality (exactly the kind of silent
mismatch that hid v17_gauntlet's broken `tag` seat for an entire run). Heuristic display names come
from `_HEURISTIC_ARCHETYPE_NAMES` (matching each bot's own `.name` in `opponent_bots.py` --
TAG/LAG/Nit/"Calling Station"), not the style key. Verified via a fresh smoke test:

```
Opp 1: LAG (Heuristic)
Opp 2: V15 (NN)
Opp 3: Calling Station (Heuristic)
Opp 4: Lagged-Self (NN)
Opp 5: V16 (NN)
```

Also fixed the seat-name column padding (was a fixed 16 chars, already too narrow for
"Opp 4 (Past Self)" alone at 18 chars even before this change) to size dynamically off the actual
label widths, so columns stay aligned regardless of which labels are active. The outer box-border
width is still a fixed literal (shared with other dashboard sections) and can overflow slightly
with a long label -- purely cosmetic, pre-existing behavior, not fixed here.

### Verification (before any further training investment)

- **Unit tests** (`HeuristicOpponent`/`NNOpponent`/`build_opponent_pool`, standalone, no training
  loop): forcing rules produce identical output to the original scalar logic; `force_heuristic`
  correctly bypasses the model query; a model-query exception falls back to `recording_bot` and
  calls the error hook; **a missing/unloadable model path correctly falls back to
  `HeuristicOpponent(forced=True)` instead of silently producing a broken `NNOpponent`** -- the
  exact failure mode that broke v17_gauntlet's `tag` seat is now covered by an explicit test.
- **`overfit_sanity`**: noisy on the synthetic critic check across repeated runs (same known
  unseeded-RNG variance every version in this line shows -- 2 of 2 runs checked here passed after
  a re-run), real targets learnable. This check doesn't exercise the opponent-wiring code at all
  (pure synthetic hand data), so it mainly confirms the refactor didn't collaterally break the
  model/training-loop plumbing it shares a file with.
- **300-hand smoke test**: dashboard header confirms `nit -> FROZEN frozen_v15.pth (forced=False)`
  and `tag -> FROZEN frozen_v16.pth (forced=False)` -- **both genuinely load this time**, unlike
  v17_gauntlet. No warnings, no errors, dashboard renders cleanly (including the "FACING A BET
  ONLY" telemetry fix, carried over unchanged from v17_gauntlet).
- **20k-hand sanity run**: see results below once complete.

### Expected difference from v17_gauntlet (not a refactor bug)

Because `tag` now genuinely loads frozen V16 (v17_gauntlet's version never did), this run's
opponent field is REAL and slightly different from what v17_gauntlet actually trained against --
frozen V16 is really in the mix this time, not the TAG heuristic standing in for it. Any
behavioral difference from v17_gauntlet's own early trajectory should be attributed to that
fixed bug engaging correctly, not to something the refactor broke.

## 20k-hand sanity run results (2026-07-16) — PASS

20,001 hands, 10m30s, clean exit, zero NaN/crash/traceback. Weights saved successfully. Still in
the bootstrap-decay window at this budget (alpha=0.50 at completion, Phase 2), so this is a
plumbing-soundness check, not a converged-behavior comparison -- exactly the bar this run was for.

- Hero: +18.0 BB/100, VPIP 43.2%, AGG 45.8% -- healthy range, no runaway/collapse.
- `nit` (frozen V15): VPIP 25.6%, AGG 80.0%. `tag` (frozen V16): VPIP 35.3%, AGG 73.8%. `past`
  (lagged self): VPIP 22.4%, AGG 52.2%. All three genuinely queried this time (confirmed via the
  new per-seat labels, see above) -- these numbers reflect real network decisions, not a repeat of
  v17_gauntlet's silently-heuristic `tag` seat.
- Equity Action Matrix: air/draws fold rates already trending sensible (90.8%/91.8%) this early
  with bootstrap still active; Marginal/Strong/Nuts tiers all net-positive. No red flags.
- Dashboard rendering confirmed clean throughout, including the new dynamic seat labels and the
  "FACING A BET ONLY" telemetry (carried over unchanged from v17_gauntlet).

**Verdict: the refactor is behaviorally sound.** Not directly comparable to v17_gauntlet's own
200k-hand converged numbers (different budget, still bootstrap-anchored, AND the `tag` seat is
genuinely different now that V16 actually plays) -- that comparison isn't this run's job. The
plumbing works; a full 200k production run is the natural next step whenever that's wanted, using
this exact config (which is what v17_gauntlet's config always should have produced).

## Carried backlog (unchanged from v17_gauntlet's planning, not addressed by this pass)

This refactor deliberately touches ONLY opponent plumbing -- none of the items below are in scope
here; they remain open for a future training pass once the plumbing is validated.

### [P0] Deep-stack OOD trash-jam — still top priority, 5 versions running unaddressed

`deep_stack_ood_guard` FAILS on V15, V16, v16_foldregret, V17, AND v17_gauntlet — the same failure
signature (eq≈0.55, 15-40bb stack, single modest bet -> ALL-IN argmax) has now survived five
consecutive versions as a side effect of unrelated work each time.

### [P5]/[P6] Input-contract gaps — still promoted over further target-formula tuning

The model still has no encoding of who raised, how many opponents raised, or bet-size patterns in
the action history. See `versions/v16/SPECS.md` [P5]/[P6] for full detail (unchanged).

### [NEW] `hero_position` never set for opponent NN queries — real bug, unclear impact direction

`_query_model_decide` never sets `hero_position` for ANY opponent NN query (defaults to 0 =
Button, the loosest position, regardless of actual seat). Confirmed real; direction doesn't
explain the Hero-vs-PastSelf VPIP gap finding (see below) since Button is the widest-range
position, not the tightest. Not yet fixed -- worth doing as part of a future contract-adjacent
pass, threading the querying seat's own position (`(0 - button_seat) % 6`, same formula hero's own
uses) into the `BoardState` construction instead of leaving it defaulted.

### [OPEN QUESTION] Why does a lagged self-play mirror play tighter than the live hero?

v17_gauntlet's `past` seat (true lagged mirror) showed VPIP stable at ~24-25% for the entire
130k-200k hand stretch, while Hero itself was flat at ~40-41% over that same stretch (100,000+
hands after Hero's own policy had converged) -- if `past` were simply "Hero from ≤5,000 hands
ago," it should have converged onto Hero's own plateau once Hero stopped moving. It never did.
Ruled out: sampling/temperature asymmetry, silent load failures. The `hero_position` bug above
doesn't explain the direction. Unresolved; worth a dedicated look before trusting a lagged/frozen
seat's realized stats as a proxy for "what does this checkpoint actually do."

### model_verify weighted composite score

Still not built: a score weighted toward the actual live field mix (loose-heavy) so a tradeoff
like foldregret's loose-deep collapse surfaces in the summary line automatically.

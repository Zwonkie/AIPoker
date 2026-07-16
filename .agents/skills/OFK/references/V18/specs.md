# V18 — Opponent-architecture refactor (+ consolidated backlog)

## Opponent-architecture refactor (2026-07-16)

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

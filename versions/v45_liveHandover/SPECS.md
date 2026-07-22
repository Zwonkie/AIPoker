# V45_liveHandover — the live↔model handover boundary (live layer only, no retrain)

**Date**: 2026-07-22
**Serves**: `Herocules (v44)` — unchanged weights, unchanged contract, unchanged simulator.
Like V42_liveFixes, every change lives in the shared live layer; nothing under `versions/v44/`
is read differently, so there is no training slice and no retrain (same §6 reasoning as
`versions/v42_liveFixes/SPECS.md`).

## Goal (user-directed)

Make hooking a new model to the live dashboard trivial and safe: the dashboard owns a RAW,
rarely-changing table snapshot; the model side "hooks up to it" through its own contract and
preprocesses it to match how that version trained. The dashboard never computes model-specific
values again, and models never touch live tracking state.

## What was built

**1. `core/live_observation.py` — `LiveObservation` (+ `SeatObservation`), the frozen handover
object.** Produced by `TableState.to_observation()`. Raw facts only, in raw table chips: cards,
pot, stacks, per-seat committed/raise-attribution/all-in/HUD reads, occupied-ring positions +
contract slots, blind membership, dealer, the price hero faces (`call_amount` +
`call_amount_known`), button availability, hero's own action tokens. Boundary rules (documented
in the module): raw facts only; sentinels are `None`, never plausible values (an unread HUD badge
is `None`, not a color — the adapter picks the training default); append-only evolution with a
`schema` int; frozen + JSON-serializable.

**2. `core/live_adapter.py` — the model-side half.**
- `classify_front_after(obs)` — the front/after split, moved out of `PHPHelp` into a pure
  function over the observation (it is model-side interpretation, not raw state).
  `PHPHelp._classify_opponents_by_action_order` is now a thin delegate to it, so the copies
  cannot drift; the V42 `verify_front_colors.py` cases (7/7) now run through the delegate.
- `observation_to_board_state(obs, equity, hand_strength, effective_field)` — pure, field-for-
  field identical to `to_board_state()` + the caller-side threading it replaces.
- `BaseLiveAdapter.decide(obs) -> LiveDecision` — the exact pre-refactor call-site pipeline,
  relocated: resolve the version's own feature implementations (`live_feature_providers`, i.e.
  what the engine's `live_features()` declares), range-aware equity with front/after split and
  vs-random fallback, `hand_strength` only if the contract reads it, `effective_field` only if
  the contract exposes it (V44+), then BoardState assembly → the untouched
  `make_decision` tensor/policy machinery. Returns a `LiveDecision` carrying the action plus all
  HUD diagnostics (equity, sim_msg, equity_meta, hand_strength, the exact BoardState, the
  observation) so the dashboard renders instead of recomputing.
- An engine may declare `make_live_adapter(decision_engine)` for a custom adapter; none of the
  registered engines need it.

**3. `PokerDecisionEngine.decide(obs, **kwargs)`** — the one entry point the dashboard calls.
Adapters are cached per model name.

**4. `PHPHelp.py` call site** — the inline equity/hand_strength/effective_field/BoardState block
(~190 lines of model-side logic in a GUI file) replaced by: build `LiveObservation` after the
price block → `decision_engine.decide(obs, ...)` → render `LiveDecision`. The dashboard now
contains **zero** per-version logic on the decision path.

**5. Recorder** — `turns.jsonl` records gain an additive `"observation"` key (Layer 0, the frozen
raw snapshot; `format` stays 2). Any recorded turn can be replayed offline through ANY version's
adapter via `LiveObservation.from_json_dict` — verified lossless round-trip.

**6. `TableState`** — gained `to_observation()` and an informational `_dealer_seen` flag
(`dealer_detected` in the observation): "hero really has the button" vs "we never found the
button". No existing consumer changes behavior on it.

## What deliberately did NOT move

`make_decision`'s policy transforms (temperature ramp, fold-when-free mask, button masking,
sampling, slider sizing) stay in `core/decision.py`, shared and unchanged — they are
train≡serve-verified invariants and this package moves code, not semantics. Same for the
`is_vN` ladders that still back the LEGACY engines: additive change only. The remaining H4-class
step (declaring serve-transform constants in the version manifest so model_verify reads the same
values — the live-M3 temperature-ramp gap) is future work, now easy to slot into the adapter.

## How a new version hooks up now

1. Engine declares `make_bridge()`, `live_features()`, `is_sized`, `display_tag` (all existing
   V43/V44 conventions — nothing new).
2. Register it in `PokerDecisionEngine.models`.
That's it: the dashboard, the equity path, the recorder, and the diagnostics all follow the
declarations. Optionally: `make_live_adapter()` for a version that needs a different pipeline.

## Verification

`versions/v45_liveHandover/verify_handover.py` — **14/14**:
- BoardState parity, old inline recipe vs `observation_to_board_state`, field-for-field
  (incl. every SeatState) across full-table / short-handed slot-remap / postflop fixtures;
- LiveObservation JSON round-trip lossless, classifier agrees before/after;
- end-to-end decide() parity on V44's real weights: two fresh engines, identically stubbed
  feature providers, same seed → identical action, size, reason string, and full model-output
  dict, across a two-street sequence; hand-history/action buffers identical after it.

Regressions re-run: `versions/v42_liveFixes/verify_front_colors.py` **7/7** (now exercising the
delegate → shared classifier), `verify_fold_monotonic.py` **15/15**, `verify_v42.py` **all pass**
(sentinel/masking/slot-remap/`_parse_button_money`, plus PHPHelp import compile).

NOT verified against a real table this session — same caveat class as V42's own note; the parity
suite proves the new path computes exactly what the old path computed, so first live session risk
is confined to the call-site plumbing, and the old behavior is one `git revert` away.

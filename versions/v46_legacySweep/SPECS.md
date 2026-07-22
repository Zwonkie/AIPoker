# V46_legacySweep — pre-v30 legacy retirement (live layer + repo hygiene, no retrain)

**Date**: 2026-07-22
**Serves**: `Herocules (v44)` — unchanged weights, unchanged contract, unchanged simulator.
**Policy decision (user, 2026-07-22)**: all versions before v30 are LEGACY. Registered live
models are now **v44 (active), v43, v41, v40** — V29 was deregistered (it was the last
ladder-dependent entry). Every legacy version's slice, weights and engine file remain on disk
(deprecate-not-delete, guardrails Rule 6; V13 and V41 stay tagged MILESTONEs), but re-registering
one now requires giving its engine the standard declarations.

## What was removed (core/decision.py)

The finding-#16/H4 endgame — dispatch is now 100% "the engine declares, the shared layer asks":

- the **13-flag `is_vN` substring ladder** and the per-version tensor-bridge chain
  (`bridge_v9/v11/v13/v20/v20_preflopEq/v25/v29/v41` + their contract imports) — engine-declared
  `make_bridge()` is the only mechanism; a model without one is refused loudly at decision time;
- **`_LEGACY_LIVE_FEATURES`** (the name-ladder backing engines without `live_features()`) and
  `context_scales()`'s per-spec fallback column — the contract's own module constants are the
  single source;
- the **13-deep `display_tag` nested ternary** (fallback is now the registry name itself, which
  cannot mislabel);
- the **v9 river guardrail** (keyed to a model not registered for a long time), the
  **math-engine override**, and the **preflop-chart/bluff-engine layer flags** — all dead for
  every sized model, and only sized models exist; the four `use_*` kwargs are gone from
  `make_decision`/`decide()`/`BaseLiveAdapter.decide`, and PHPHelp's four orphaned toggle vars
  with them;
- the **legacy 3-way actor path** (`FOLD/CALL/RAISE`, `CHECK`, dynamic 0.75-pot sizing) and the
  critic-only argmax path — `make_decision` refuses (loud FOLD reason) any engine that does not
  declare `is_sized=True`, *before* touching the history buffers so they stay aligned;
- duplicate `is_v11_model` computation, unused `ModelEngine`/`PokerAction` imports.

`V40ModelEngine`/`V41ModelEngine` gained the `is_sized`/`display_tag`/`has_aux` declarations they
were missing (V43/V44 already had them) — the prerequisite for deleting the ladder.

## Also removed

- **PHPHelp.py**: module-level `SEAT_ORDER_CLOCKWISE` (lives in `core/live_observation.py` since
  V45), the dead `last_valid_hero_stack` fallback (review live-M5: fed a local nothing read), and
  the unreachable CHECK-safeguard block (sized models never emit `CHECK`; CALL is masked when the
  button is absent, V42 A3).
- **core/models/base.py**: the 15-argument `predict_action` ABC replaced by a documentation base
  naming what a registrable engine actually provides (predict_ev + loaded + the declaration set).
  Nothing is abstractly enforced — decision.py fail-louds on exactly the declarations it needs.
- **scripts/math/run_model_diagnostics.py** updated (it passed `use_math_engine=False`).

## Repo hygiene

- `git rm --cached` (files stay on disk): all tracked `__pycache__/*.pyc`, `active_training.log`,
  `model_verify_v43.log`, `tools/training_monitor/telemetry*.json`, and graphify's generated
  outputs (`graph.json`, `GRAPH_REPORT.md`, `manifest.json`, `cache/*`).
  `graphify-out/.graphify_labels.json` stays tracked (curated, not generated).
- `.gitignore` gained: `active_training*.log`, `model_verify_*.log`,
  `tools/training_monitor/telemetry*.json`, `history/` (live turn recordings are session data).
- `tools/self_play/` (the pre-`versions/` v8–v11 training stack) moved to
  `attic/tools_self_play/` — only self-referential imports, nothing current depends on it, its
  durable lessons live in OFK's `legacy-versions-history.md`. See `attic/README.md`.

## What deliberately stayed

- Legacy version slices (`versions/v13..v29`), engine files (`core/models/v13..v29_engine.py`)
  and weights: on disk, unregistered. The `is_vN` attrs on old engines are harmless relics.
- `make_decision`'s serve transforms (temp ramp, masks, sampling, slider sizing): unchanged —
  train≡serve-verified invariants. Declaring their constants per version (live-M3) remains the
  known follow-up, unchanged by this sweep.

## Verification

`versions/v46_legacySweep/verify_legacy_sweep.py` — **31/31**: source-level proof the ladders are
gone; registry is exactly {v44,v43,v41,v40} with every engine loaded + fully self-declaring +
own-bridge + engine-resolved live_features + contract-sourced scales; every registered model
produces an executable, correctly-tagged action through `decide(obs)`; unknown-name and
undeclared-engine paths refuse loudly.

Full regression battery re-run green after the sweep: `verify_handover.py` **14/14** (old-path vs
new-path parity still byte-identical — the sweep changed no behavior on the surviving path),
`verify_v42.py` **all pass**, `verify_front_colors.py` **7/7**, `verify_fold_monotonic.py`
**15/15**.

Net effect: ~430 lines of live-layer legacy deleted; "add a model" = engine declarations + one
registry line; "misroute a model silently" now has no remaining mechanism in the live layer.

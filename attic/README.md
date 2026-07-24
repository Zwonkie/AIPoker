# attic/

Superseded code kept for the record, out of the live tree.

- `tools_self_play/` -- the pre-`versions/` self-play training stack (v8-v11 era: six_max_simulator,
  train_selfplay, opponent_bots, telemetry). Moved here in v46_legacySweep (2026-07-22): only
  self-referential imports, no current code depends on it, and its durable lessons live in
  `.agents/skills/OFK/references/legacy-versions-history.md`. Internal `tools.self_play.*` imports
  are intentionally left as-is -- these scripts are not runnable anyway (their v8-v11 weights are
  gone); this is an archive, not a package.
- `phphelp_stack/` -- the pre-`live2` live stack (PHPHelp.py Tkinter dashboard + its private
  helpers `action_executor.py`/`state_machine.py` + the legacy heuristic/engine/base model
  classes + `decision_rules.json`). Moved here 2026-07-24: superseded by `live2/`
  (pilot/webapp/assembler), verified dead outside frozen version slices. Note:
  `versions/v42_liveFixes/verify_v42.py` and `verify_front_colors.py` import `PHPHelp` and are
  no longer runnable from disk without adding `attic/phphelp_stack` to `sys.path` -- they are
  historical one-shot verifiers whose findings are recorded in that slice's `SPECS.md`.

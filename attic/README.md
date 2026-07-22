# attic/

Superseded code kept for the record, out of the live tree.

- `tools_self_play/` -- the pre-`versions/` self-play training stack (v8-v11 era: six_max_simulator,
  train_selfplay, opponent_bots, telemetry). Moved here in v46_legacySweep (2026-07-22): only
  self-referential imports, no current code depends on it, and its durable lessons live in
  `.agents/skills/OFK/references/legacy-versions-history.md`. Internal `tools.self_play.*` imports
  are intentionally left as-is -- these scripts are not runnable anyway (their v8-v11 weights are
  gone); this is an archive, not a package.

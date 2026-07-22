"""live2.webapp -- view-only web dashboard (v49_liveRebuild).

Renders what the live layer emits; contains NO decision logic and mutates NO state
(SPECS principle: "The webapp is view-only"). Until the assembler exists it feeds off
the legacy recorder's history/<board_id>/turns.jsonl (format 2) plus the historydb
hand store -- the same files, so cutover later is a source swap, not a rewrite.

Run:  .venv/Scripts/python.exe -m live2.webapp   (http://127.0.0.1:8765)
"""

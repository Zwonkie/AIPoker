"""live2 pilot: the headless live runtime that replaces the PHPHelp dashboard.

One process owns the whole turn cycle; its eyes and hands are local modules:

  PrintWindow capture (live2/pilot/capture.py, works unfocused)
    -> legacy vision + TableState stabilization (core.vision / core.table_state --
       OCR quality improvements are a later, separate stage)
    -> ASSEMBLER corrections (live2.assembler.assemble) -- for the first time the
       corrected observation is what the MODEL decides from, not just a shadow record
    -> V45 handover boundary: decision_engine.decide(LiveObservation) (core.decision)
    -> turn record appended to history/<board_id>/turns.jsonl (format 2, same schema
       as the legacy recorder + an 'assembler' layer) -- the live2 webapp tails it
    -> action execution on the human motor model (live2/pilot/mouse.py: Fitts's-law
       durations, min-jerk arcs, overshoot+correction), gated behind --auto; default
       mode only recommends.

History note: capture.py + mouse.py lived in live2/phpserver/ ("PHPserver") until
2026-07-23; the standalone WS wrapper around them was retired to attic/live2_phpserver/
when the pilot made in-process calls the transport.

Run:  .venv/Scripts/python.exe -m live2.pilot            # recommend-only
      .venv/Scripts/python.exe -m live2.pilot --auto     # executes clicks
      .venv/Scripts/python.exe -m live2.pilot --probe    # one capture+vision pass, saves PNG
      .venv/Scripts/python.exe -m live2.pilot --list     # candidate table windows

Do NOT run `live2.assembler --watch` alongside the pilot: the pilot embeds the
assembler and writes shadow_turns.jsonl itself; the watcher would fight it.
"""

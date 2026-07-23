"""live2.assembler -- the single source of live state truth (v49_liveRebuild).

Fuses three feeds into a corrected `LiveObservation` where EVERY changed field carries
provenance, and contradictions between feeds are SURFACED, never silently resolved by
whichever code path ran last (the SPECS principle; the flagged JJ-fold of 2026-07-22 --
phantom seat + three-way price disagreement -- is the type specimen of the bug class
this component exists to kill):

  vision      -- what the screen said (today: the legacy recorder's turns.jsonl
                 observations; later: PHPserver frames + own CV)
  carry-over  -- EXACT facts the previous completed hand proved (live2/historydb):
                 seat ROSTER (tournament play: fixed field, nobody new sits in),
                 final stacks, blinds. One hand behind by construction.
  opponent DB -- windowed per-player stats resolved by name at sit-down.

v1 scope: fusion + provenance + shadow parity. The assembler REPLAYS any recorded
session (shadow.py) so its corrections are testable offline against ground truth;
live operation tails the same file the legacy recorder writes. It does not yet drive
its own vision.

Run:  .venv/Scripts/python.exe -m live2.assembler --replay <board_id>
      .venv/Scripts/python.exe -m live2.assembler --replay-all
"""

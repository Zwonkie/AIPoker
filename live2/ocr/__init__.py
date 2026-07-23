"""live2 OCR stage: replace general OCR on money fields with binarized digit-template
matching (owner decision 2026-07-23). Rationale: templates on a 10-glyph vocabulary in a
fixed font ABSTAIN on bad frames (low match score) instead of returning a confident wrong
number -- the exact failure mode that poisoned hero-stack reads through the monotonic
ratchet (970 read as 380, locked for the hand).

Workflow:
  harvest_digits.py  -- build labeled, binarized glyph templates from recorded frames
                        (flagged screenshots + pilot last_turn.png, labels from the
                        matching turn records + self-cleaning consensus pass) and emit a
                        review sheet. Templates are NOT wired into live reading until the
                        owner has reviewed them.
  templates/         -- canonical digit_<d>.png (binary, native scale) + samples/<d>/*.png

Run:  .venv/Scripts/python.exe -m live2.ocr.harvest_digits
"""

"""PHPserver -- the native bet365 interaction agent (v49_liveRebuild; named by the owner
2026-07-22, sibling of the PHPHelper dashboard).

Owns PIXELS and CLICKS only; zero poker logic, zero model imports (SPECS principle).

Capture is HEADLESS: PrintWindow PW_RENDERFULLCONTENT renders the board with the window
unfocused or fully occluded -- the client never needs to be in front to be observed.
Interaction follows the client's own behavior: bet365 raises its window when it becomes
hero's turn, so the click path VERIFIES foreground rather than forcing it (fallback focus
only if the client didn't raise itself). Mouse movement follows a human motor model --
Fitts's-law durations, curved min-jerk trajectories, micro-jitter, variable event timing,
overshoot + micro-correction (see interact.py).

Exposes a localhost-only WS/JSON API (see server.py). The assembler is the only intended
client; PHPserver never interprets what it captures.

Run:  .venv/Scripts/python.exe -m live2.phpserver   (ws://127.0.0.1:8766)
"""

"""live2.service -- the native bet365 interaction agent (v49_liveRebuild).

Owns PIXELS and CLICKS only; zero poker logic, zero model imports (SPECS principle).
Capture is PrintWindow PW_RENDERFULLCONTENT (works with the window occluded -- the mss
full-monitor grab it replaces required visibility). Interaction reuses the humanized
mouse primitives' approach but lives here so the legacy path can be deleted whole.

Exposes a localhost-only WS/JSON API (see server.py). The assembler is the only
intended client; the service never interprets what it captures.

Run:  .venv/Scripts/python.exe -m live2.service   (ws://127.0.0.1:8766)
"""

# live2 — the rebuilt live stack (v49_liveRebuild)

**Date Recorded**: 2026-07-22

**Related Files**:
- `versions/v49_liveRebuild/SPECS.md` (canonical spec + build log — read alongside this)
- `live2/webapp/` (app.py, sources.py, static/) · `live2/service/` (server.py, capture.py, interact.py) · `live2/historydb/` (decode_bet365.py, parse_xml.py, ingest_watch.py, backfill_xml.py, stats.py, population.py)
- `core/live_observation.py` (the frozen boundary), `core/live_adapter.py`
- `history/handhistory/<sessioncode>/hands.jsonl` (the hand store), `history/<board_id>/turns.jsonl` + `flags.jsonl` (legacy recorder)
- `tools/handhistory/*` = 2-line forwarding shims into `live2.historydb` (kept so old call sites don't break)

**Context**:
The legacy live layer (PHPHelp.py → TableState) is inference-heavy: it reconstructs facts vision can't read reliably. One flagged hand (JJ folded at 69.5bb, 2026-07-22) exposed three independent members of that bug class at once — a phantom seat from the countdown timer OCR'd as a player name, three components disagreeing silently about the price, and the general "vision is the only witness" lineage. Post-hand ground truth (the bet365 blob/XML history, see `bet365-handhistory-formats.md`) makes most of that inference deletable. live2 is the replacement, built decoupled under a NEW top-level root so the old path stays untouched until shadow parity and is deleted later in a legacy-sweep pass.

Components and hard boundaries:
- **`live2/service`** — native bet365 interaction agent. Owns PIXELS and CLICKS only, zero poker logic, zero model imports. WS/JSON API on `ws://127.0.0.1:8766`: `ping / list_windows / bind / capture(roi) / click(x,y) / move_slider(track,frac)`. Capture = ctypes PrintWindow `PW_RENDERFULLCONTENT` (works OCCLUDED — verified pixel-perfect on the real client 2026-07-22; the legacy mss monitor-grab needed the window visible). Coordinates are client-area relative everywhere. Focus is touched only for clicks. v1 scoping decision: the service returns pixels; ALL interpretation (CV/OCR) belongs to the assembler.
- **`live2/assembler`** — NOT YET BUILT. The single source of state truth: fuses vision (shrunk surface) ⊕ carry-over from the previous completed hand (exact stacks/roster/button — the phantom-seat killer) ⊕ opponent DB, emits `LiveObservation` with per-field provenance, owns the turns.jsonl recorder and the F12 flow. Contradictions between feeds are SURFACED, never silently resolved by whichever code path ran last.
- **`live2/historydb`** — hand-history ingestion + stats (blob decoder, XML parser, 2s-poll ingest watcher, backfill, windowed stats engine, population fitting). Source of truth = append-only `hands.jsonl` per session; SQLite is a DERIVED index only (rebuildable, see Guidelines).
- **`live2/webapp`** — view-only dashboard, `http://127.0.0.1:8765` (FastAPI + uvicorn + vanilla JS, no build chain). Live mirror + decision panel (WS push), opponent profiles (lifetime vs last-N), hand browser/replayer, F12 flag queue. NO decision logic, NO state mutation. Until the assembler exists it tails the LEGACY recorder's turns.jsonl — cutover is a source swap.
- **The model side is untouched**: `LiveObservation` (V45) stays the frozen handover; downstream adapter/engine/`make_decision` don't know live2 exists.

**Guidelines**:
- Run: `.venv/Scripts/python.exe -m live2.webapp` (dashboard), `-m live2.service` (interaction agent). Both bind localhost only.
- Migration gates before cutover (SPECS): shadow parity ≥3 real sessions with every per-turn disagreement adjudicated by post-hand ground truth (new path must win/tie ≥95% and introduce zero new false facts); the post-hand OCR self-validation diff running live; assembler tick ≤ old path latency; final cutover = user call with the old path kept runnable.
- The service's click/slider primitives have NEVER been fired at the real client yet (deliberate — first interaction test belongs to the shadow phase). Do not "quick-test" a click outside that context; the user may have a live registration.
- Keep the boundaries clean when extending: pixels/clicks → service; interpretation/state → assembler; rendering → webapp. If a change needs poker knowledge inside service or webapp, it belongs in the assembler.
- hands.jsonl stays the source of truth. Any SQLite/index schema change = delete the index file and rebuild from jsonl; never migrate in place, never write facts only to SQLite.
- Feeding EXACT opponent stats into the model's VPIP/AGG slots is NOT part of live2 — that changes the model's input distribution and needs its own A/B in a trained version (V42 rule). The webapp may DISPLAY exact stats while the model still consumes color bands.

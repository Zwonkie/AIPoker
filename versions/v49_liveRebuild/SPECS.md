# V49_liveRebuild — live-side restructure around the LiveObservation boundary (DRAFT)

**Status**: DRAFT 2026-07-22. Live-side slice only — NO model, contract, or simulator changes;
parallel-safe with V48 (training version). Follows the `v42_liveFixes`/`v45_liveHandover`
live-slice pattern. New code lives under a NEW top-level root `live2/` (user decision
2026-07-22) so the legacy path stays untouched until parity and is deleted later via a
`v46_legacySweep`-style pass.

## Why now — three measured failures on one flagged turn (2026-07-22, turn_10)

The current live layer is inference-heavy: it reconstructs facts vision can't read reliably.
One F12-flagged hand (JJ folded preflop at 69.5bb) exposed three independent members of that
bug class in a single decision:

1. **Phantom seat**: the seat countdown timer OCR'd as a player name ("Tid: 18") registered a
   4th opponent at a 3-opponent table → range-aware equity for JJ fell 0.45 → 0.38 → the net
   folded 0.92 (slot-ablation proved the knife edge: at eq 0.45 it folds 0.003). Hotfixed
   (timer-pattern names can't first-register a seat, `core/table_state.py`), but the CLASS
   remains: vision was the only witness to seat occupancy.
2. **Three components, three prices**: only the FOLD button was OCR'd; the true price was a
   raise to 5bb; the observation carried 1bb; and the tensor path would have encoded an
   unknown price as 0bb ("free") — firing the net's known free-check-fold pathology while
   the FOLD mask (which correctly treats unknown ≠ free, V42 #13) stands down. Hotfixed
   (`core/live_adapter.py`: unknown price floors at 1bb), but no single component owns "the
   price" — they disagree silently.
3. **The lineage**: the V29 table_state cluster, V42 rounds 1–2, [OPP-4], the
   raise-attribution inversion, pot under-reads/latching — all live in inference code that
   post-hand ground truth (`tools/handhistory/`, discovered 2026-07-22) now makes deletable.

## Principles

- **LiveObservation stays the frozen boundary** (V45). Downstream — engine-declared adapters,
  `core/decision.py`, engines — is UNTOUCHED. A new version still = declarations + registry.
- **The assembler is the single source of state truth.** Every fact carries provenance
  (vision / carry-over / history-db) and confidence; feed contradictions are SURFACED (logged
  + shown), never silently resolved by whichever code path ran last.
- **The webapp is view-only.** No decision logic, no state mutation. It renders what the
  assembler and decision engine emit.
- **The native service owns pixels and clicks only.** Zero poker logic.
- **Prefer exact history over OCR** for anything a completed hand already proved (stacks,
  roster, button, blinds); vision covers only what is genuinely live.

## Components (`live2/`)

### `live2/service` — native bet365 interaction agent
Capture (PrintWindow `PW_RENDERFULLCONTENT` → Windows.Graphics.Capture fallback, per V48 3a —
works occluded, no `SetFocus()` except for clicks), OCR invocation on request, click/slider
primitives. Exposes a localhost-only WS/JSON API: `capture(roi)`, `read(rois)`,
`click(action)`, `move_slider(frac)`. Single process, no model imports.

### `live2/assembler` — the state assembler (replaces TableState inference)
Fuses three feeds into `LiveObservation`:
- **Vision (shrunk)**: hero cards, board, current bets/pot, whose-turn, buttons/slider —
  roughly a third of today's OCR surface.
- **Carry-over (exact)**: previous completed hand's final stacks, blind level, button seat,
  and the SEAT ROSTER (account names) — the phantom-seat killer: a name vision reads that is
  not in the roster and not a plausible new sit-in is quarantined until confirmed by the next
  hand record. Stacks from carry-over are truth; vision drift-checks them.
- **Opponent DB**: per-seat windowed stats (`tools/handhistory/stats.py` — moves to
  `live2/historydb`) resolved at sit-down by name → account.
Also owns the `turns.jsonl` recorder (same format, adds per-field provenance) and the F12
flag flow. Decision path: assembler → `LiveObservation` → engine adapter → `make_decision`
(unchanged).

### `live2/historydb` — hand-history ingestion + stats (move of `tools/handhistory/`)
`ingest_watch` runs as an assembler-managed thread during play (files are EPHEMERAL — a
client relaunch wipes TempData and the prior session XML). Adds the **OCR self-validation
diff**: after every completed hand, diff the recorded observations against the decoded blob;
disagreements auto-append (with ground truth) to the vision regression corpus. This is the
arbitration mechanism for the migration gate below.

### `live2/webapp` — web dashboard (view-only)
FastAPI + WebSocket push, localhost bind, no frontend build chain (vanilla JS/htmx).
Views: live board mirror with per-field provenance/confidence; decision panel (policy, Q,
reason string, think history); per-seat opponent profiles (lifetime vs last-100, from the
DB); session results; hand replayer over `hands.jsonl`; flag-review queue (screenshot +
three-layer diagnostic, the `flagged/` flow rendered properly).

## Migration gates

1. **Shadow parity**: assembler runs in shadow alongside PHPHelp for ≥3 real sessions,
   per-turn `LiveObservation` diff; every disagreement adjudicated by post-hand ground truth.
   Gate: new path wins or ties ≥95% of adjudicated disagreements and introduces zero new
   false facts.
2. **Self-validation live**: the post-hand OCR diff runs automatically and appends to the
   corpus during the shadow sessions.
3. **Latency**: assembler tick time ≤ the old path's (measured over a session).
4. **Cutover**: user call. Old path stays runnable (registry-style fallback) until a later
   legacy sweep deletes it.

## Non-goals

- No model/contract changes. Feeding EXACT stats into the VPIP/AGG slots stays behind its own
  A/B in a future trained version (V42 rule) — the webapp may DISPLAY exact stats while the
  model still consumes color bands.
- No multi-table support (single table, like today).
- The bet365 stats-panel seed read stays a V48-3b optional item, not part of this slice.

## Build log

- 2026-07-22: **webapp skeleton BUILT** (`live2/webapp/`): FastAPI + uvicorn (installed to
  .venv), localhost:8765, vanilla JS (no build chain). Working today against existing data:
  live board mirror + decision panel (policy/Q bars, equity, reason) fed by WS push tailing
  the LEGACY recorder's `turns.jsonl` (assembler later replaces the tail with direct push);
  opponent profiles lifetime-vs-last-100 from `live2/historydb/stats.py`; hand browser +
  replayer over the 4,015-hand store; F12 flag-review queue (joins per-board `flags.jsonl`
  with turn records; artifact folders listed, screenshot rendering TODO). Verified against a
  live session (Double_Or_Nothing_1171690247) while the user played. View-only holds: no
  endpoint mutates state. Run: `.venv/Scripts/python.exe -m live2.webapp`.
  TODO next: session-results view, screenshot/artifact serving for flags, provenance badges
  (blocked on assembler records).
- 2026-07-22: **native service skeleton BUILT** (`live2/service/`): WS/JSON API on
  ws://127.0.0.1:8766 (`ping/list_windows/bind/capture/click/move_slider`), ctypes-only
  PrintWindow `PW_RENDERFULLCONTENT` capture (fail-loud, no pywin32), humanized
  click/slider primitives (client-relative coords, focus only on click, pyautogui
  FAILSAFE on). VERIFIED: captured the real bet365 client window pixel-perfect over the
  API (Direct2D content renders; the legacy mss monitor-grab requirement is gone).
  SCOPING DECISION: v1 returns PIXELS only -- no OCR in the service; `read(rois)`
  collapses into `capture(roi)` and ALL interpretation (CV/OCR) moves to the assembler.
  Click/slider deliberately NOT yet fired at the real client (user had a live
  registration); first interaction test belongs to the shadow-parity phase.
  Run: `.venv/Scripts/python.exe -m live2.service`.
- 2026-07-22: **SQLite derived index ADDED** (`live2/historydb/sqlindex.py`, user decision):
  `history/handhistory/index.sqlite` (hands/players/actions tables + full record JSON) as a
  DISPOSABLE query index -- hands.jsonl stays the source of truth; schema change = delete +
  `rebuild()` (auto on SCHEMA_VERSION mismatch), never in-place migration. Wired: ingest_watch
  updates it per hand (best-effort -- an index failure never blocks ground-truth capture),
  backfill_xml rebuilds after bulk adds, webapp browse endpoints (sessions/hands/hand detail,
  new per-player filter) query it with jsonl-scan fallback. Rebuilt at 4,139 hands / 188
  sessions / 57k actions after harvesting tonight's blob hands. OFK folder
  `references/live2/` created (architecture + hand-history formats docs, indexed in SKILL.md).

## Open questions (user)

- Web stack sign-off: FastAPI + vanilla JS/htmx (no node toolchain) — OK?
- service↔assembler transport: WebSocket (proposed) vs named pipe.
- Shadow-session count before cutover (proposed ≥3).

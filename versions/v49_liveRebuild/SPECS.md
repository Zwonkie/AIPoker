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

### `live2/phpserver` — "PHPserver", the native bet365 interaction agent
(Named by the owner 2026-07-22, sibling of the PHPHelper dashboard; folder renamed from
`live2/service`.) Capture (PrintWindow `PW_RENDERFULLCONTENT` → Windows.Graphics.Capture
fallback, per V48 3a — HEADLESS, works unfocused/occluded), click/slider primitives.
Focus model (owner decision): bet365 raises its own window when it becomes hero's turn, so
clicks VERIFY foreground rather than forcing it (fallback-raise once, then fail loud).
Mouse movement follows a human motor model (owner spec): Fitts's-law durations
(T = a + b·log2(2D/W), no static intervals), curved Bezier trajectories with min-jerk
accel/decel, decaying micro-jitter, variable ~125Hz event timing, frequent
overshoot/undershoot + micro-correction, 60–150ms normally-distributed click holds.
Exposes a localhost-only WS/JSON API: `capture(roi)`, `move_to`, `click`,
`move_slider(track, frac)`. Single process, no model imports.

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

## Build log — assembler (2026-07-23)

- **assembler v1 BUILT** (`live2/assembler/`: feeds.py / assemble.py / shadow.py): fuses the
  recorded vision observation ⊕ carry-over (hand store, joined by tournament id = board_id
  suffix) ⊕ opponent DB into a corrected LiveObservation with per-field provenance
  ('carry-over' | 'derived' | 'quarantine' | 'sticky-identity') + surfaced contradictions.
  v1 rules: aligned ROSTER verification with seat-identity STICKINESS (established seats
  get garbage/timer names REPAIRED, never dropped; timer names can never first-register);
  unread stacks filled from prev-hand finals minus committed; unknown prices DERIVED from
  street bet level (never encoded free, `call_amount_known` stays False); blind level
  cross-check. Replay-first: `python -m live2.assembler --replay <board_id>` runs any
  recorded session and writes `shadow_turns.jsonl`; the same `process_turn` will tail the
  live file in shadow mode. Replay hardening found+fixed three v1 defects: future-blind
  carry-over alignment (blob `start_utc_ms` vs XML `start_local` normalized to epoch);
  roster authority gating (a sparse store must not quarantine early-busted-unknown real
  players); vision-trust mode with unverified identities when no hand has completed yet.
- **Validated on the flagged JJ session** (96 turns, 44 ground-truth hands): catches every
  timer-name corruption (chronic: ~170 raw instances), repairs `leh369`→`Jeh369` OCR noise,
  isolates real blind-level transitions, surfaces a genuine price contradiction (turn 73:
  vision 465 vs derived 150), fills an unread stack from carry-over. 8-session breadth
  replay: 0 false quarantines after hardening.
- **DIAGNOSIS REVISION (JJ fold, turn 10)**: ground-truth adjudication shows `Tid: 18` was
  seat_3's REAL occupant Paul6969 (dealt in, limped, later bet flop) and the observation's
  4-active-opponent count was CORRECT at hero's decision — the phantom-seat count-inflation
  story does NOT hold for this turn. The equity 0.38 was computed on correct inputs, which
  points the too-tight JJ fold at the known MULTIWAY EQUITY DEPRESSION thread
  (v42_liveFixes round 2) instead. The two live hotfixes remain valid defenses (timer text
  chronically pollutes names), but the root cause moves.

- **live shadow mode BUILT** (`--watch`, 2026-07-23): tails the newest board's turns.jsonl,
  processes each turn as it lands (partial-write safe), appends shadow_turns.jsonl, prints
  corrections/contradictions live. Verified by drip-feed test: 20/20 turns, corrections
  byte-identical to batch replay, JJ repair fires in live mode. This is the process that
  runs alongside PHPHelp for the gate-1 shadow sessions.
- **OCR self-validation BUILT** (`selfcheck.py`, gate 2, 2026-07-23): post-hand diff of
  recorded observations vs the completed hand's ground truth (hero cards, board prefix,
  blind level, seat names vs dealt-in set, hero stack vs blob per-action `stack_before`
  with minor-timing/MAJOR severity). Turn→hand grouping is time-window BOUNDED (first dry
  run showed a sparse store otherwise maps unrelated turns onto its one hand). First
  corpus run over 12 recorded sessions: **110 entries in history/vision_regression.jsonl**
  — dominated by hero_stack MAJORs (stale reads, e.g. 1125 vs truth 2180) and seat_name
  timer corruptions; cards/board/blinds essentially clean on dense sessions.

- **shadow-session wiring COMPLETE (2026-07-23 morning)**: `--watch` now runs the blob
  ingester as an internal daemon thread (SPECS design: carry-over/roster stay fresh
  MID-session; ingest failure degrades, never kills the watch). Webapp gained
  `/api/shadow` + an "Assembler shadow" panel in the Live tab (per-turn corrections with
  provenance badges — quarantine/sticky/carry-over/derived — contradictions highlighted,
  clean turns confirmed). Board-ranking rule UNIFIED between watcher and webapp
  (turns.jsonl mtime, not dir mtime — appends don't touch the dir and sibling shadow
  files perturb it). BOTH PROCESSES ARMED and agreeing on the followed board; gate-1
  shadow data collection starts with the next real session.

- **SHADOW SESSION #1 COMPLETE (2026-07-23 morning, 2 DoN tables, 60 turns)**: first real
  gate-1 data. Assembler behavior on live play: sticky-identity repaired every timer/garbage
  name over real occupants ('4'→TavGameDev, '4\n4'→Aleks888bum, fuzzy Diuk123→Djuk123),
  ZERO false quarantines, 5 correct big-blind level-change contradictions, 3 carry-over
  stack fills. Selfcheck vs ground truth: cards/board/blinds/seat-names ALL CLEAN both
  boards; **hero_stack is the weak field** — 26 diffs, mostly ≤2bb posting-timing offsets
  but 3 MAJOR misreads (970 read as 380; 720 read as 15; 900 read as 735). NEW assembler
  rule 2b from this: `hero_committed > pot_size` is an impossible state (verified the pot
  display includes current-street bets) produced by pot misreads, stack misreads absorbed
  by the legacy committed tracker, or the tracker carrying corrupt state across hand
  boundaries (committed frozen at 1040 on a fresh preflop, board 1171684621 t60-63).
  Flags as a composite contradiction with evidence attached, no guessed repair —
  in-stream attribution is provably unreliable (stack+committed reconstructs the
  tracker's own start by construction). Dry-run over 9 recent boards: fires on ~6% of
  turns, catches every selfcheck MAJOR. Also fixed: watcher restart duplicated shadow
  files (append→truncate on follow, shadow_turns.jsonl is a 1:1 mirror again).
  Legacy-pipeline implication: ~6% of live model inputs carry an impossible
  pot/committed state — strongest quantified argument yet for the assembler handover.
  Session outcome: hero bubbled both DoNs (4 left, top 3 paid), 0 F12 flags.
  **Gate-1 tally: 1 of ≥3 sessions.**

- **SHADOW SESSION #2 COMPLETE (2026-07-23, 1 DoN table, 12 turns)**: assembler again
  clean — repaired heavy OCR garbage over real occupants ('ree ee ee…'→Rarefire368,
  'cen aa nd…'→MrGray86, fuzzy MrGray8s6/nueenbright18/8kMyMonevsk), 0 false
  quarantines. Selfcheck: cards/board/blinds clean; hero_stack 5 minor-timing diffs
  (no MAJOR); 1 seat_name corpus entry (raw garbage, assembler had repaired it).
  FOUND + FIXED a carry-over freshness defect: `latest()` treated the newest stored
  hand as completed only after a fixed 90s settle margin, but this turbo's hands run
  ~40s — carry-over lagged one EXTRA hand for up to a minute, surfacing as a big_blind
  contradiction on EVERY turn (8/12). Fix: end = start + duration_s + 5s (blob records
  carry exact durations; 90s stays the no-duration fallback). Replay-verified: 8→6
  contradictions, all six now genuine first-hand-after-level-change lag (vision kept),
  and roster/stack carry-over runs one hand fresher mid-session. NOTE: final turn
  (t12, A3s flop all-in 115 into 990) was HUMAN-EXECUTED (user forced the action) —
  model intent agreed (0.8% FOLD, chose the all-in slider itself; ~10:1 price), so no
  adjudication conflict, but the turn is excluded from model-behavior reads. Hero
  busted pre-money again (blind level 150 at 12 minutes, 6th/5th).
  **Gate-1 tally: 2 of ≥3 sessions.**

- **GATE DECISION (owner, 2026-07-23)**: shadow gate CLOSED at 2 sessions ("it is working
  correctly; the issues are OCR related which we will improve at a later stage") -- jump
  to full live2 pipeline connection.

- **PILOT BUILT (`live2/pilot/`, 2026-07-23)**: the headless runtime replacing PHPHelp.
  One process: PrintWindow capture (unfocused-safe) → legacy vision + TableState (OCR
  reuse; improvements deferred by owner decision) → **assembler in the DECISION PATH**
  (the model now consumes the corrected observation -- first time; corrected obs is
  stored as the record's `observation`, raw vision preserved under `observation_raw`
  when they differ) → V45 boundary `decide(LiveObservation)` → format-2 turns.jsonl
  (+`recorder: live2-pilot` + `assembler` layer) + self-owned shadow_turns.jsonl →
  webapp tails it unchanged. Clicks = phpserver motor model via `live2/pilot/actions.py`
  (legacy ActionExecutor geometry re-expressed client-relative: main buttons off the
  fold anchor (+90/+290/+460, +45), POT shortcuts (+65..455, -65), slider center-anchored
  track (1153,970)→(1508,970) @1536x1090), gated behind `--auto`; default recommend-only.
  Legacy behaviours kept: partial-board wait, <2-hero-cards guard, unknown-price
  semantics (FOLD never masked), post-action debounce. NEW: decide-once fingerprint
  (legacy recommend-mode re-decided the same turn every ~1.5s -- that's where the
  duplicated stack-frozen turns in old sessions came from). XML baseline seeding + blob
  ingest thread run inside the pilot. VALIDATED offline: 12/12 recorded turns from
  session #2 through the full new seam (assembler → from_json_dict → decide) reproduce
  the legacy action exactly (max policy drift ≤0.17, MC-equity noise only, actions
  identical); records serialize. NOT yet validated: live PrintWindow frame vs vision
  ROIs (`--probe` exists for this), first real click. Run: `python -m live2.pilot
  [--auto|--probe|--list]`. Do NOT run `live2.assembler --watch` alongside the pilot.

## Migration gates

1. **Shadow parity**: assembler runs in shadow alongside PHPHelp for ≥3 real sessions,
   per-turn `LiveObservation` diff; every disagreement adjudicated by post-hand ground truth.
   Gate: new path wins or ties ≥95% of adjudicated disagreements and introduces zero new
   false facts. **CLOSED EARLY at 2 sessions by owner decision (2026-07-23).**
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
- 2026-07-22 (later): **service renamed PHPserver** (`live2/phpserver/`, owner naming) and
  **interact.py rebuilt on a human motor model** (owner spec): Fitts's-law movement
  durations (validated: 30px→~330ms vs 1500px→~930ms, log growth; 8px target ~880ms vs
  64px ~550ms at same distance), min-jerk velocity profile (measured bell: 230→1300→33
  px/s), Bezier arc + decaying tremor, variable 4–12ms event intervals, 58% measured
  overshoot+micro-correction rate on long moves, 60–150ms gaussian click holds (mean
  96ms). Focus model changed to match the client: bet365 raises itself on hero's turn →
  `_ensure_foreground` verifies + fallback-raises once + raises FocusError rather than
  clicking blind. New `move_to` API method. All planning functions PURE (validated
  statistically without touching the cursor); real-client interaction still untested by
  design. Run: `.venv/Scripts/python.exe -m live2.phpserver`.
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

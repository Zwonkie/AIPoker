# bet365 hand-history ground truth — wire formats, locations, hand store

**Date Recorded**: 2026-07-22

**Related Files**:
- `live2/historydb/decode_bet365.py` (blob decoder — full wire-format notes in its docstring), `parse_xml.py`, `ingest_watch.py`, `backfill_xml.py`, `stats.py`, `population.py`
- Store: `history/handhistory/<sessioncode>/hands.jsonl` + ledger `history/opponent_db.json`
- Snapshots: `history/opponent_stats.json`, `history/population_fit.json`

**Context**:
The bet365 client writes TWO machine-readable records of every completed hand — exact ground truth that supersedes OCR for anything a finished hand already proved (stacks, roster, button, blinds, winners, showdowns):

1. **Protobuf blob**, one file per hand, filename = hand_id. Written when the complete preflop→river cycle ends (so live consumers are always ONE HAND BEHIND — correct for aggregates/carry-over, never for the current hand). Location (EPHEMERAL — wiped on client relaunch): `%LOCALAPPDATA%\Poker at Bet365.DK\data\Zwonkie\History\TempData\<epoch>\Data\Tournaments\<sessioncode>\`. Richest source: per-action stack_before, ms timestamps (→ think-time), account_ids (stable identity), avatars (client portraits — NOT player notes; same avatar recurs across accounts), showdown reveals. Key wire facts: 4-byte LE length prefix; small ints zigzag; money = zigzag/2/100 = display chips; header timestamps zigzag ms-epoch but per-event timestamps PLAIN ms-epoch. Full field map in the decoder docstring.
2. **iPoker session XML**, one file per session (sessioncode), each hand a `<game gamecode=hand_id>`. PERSISTENT across relaunches at `...\History\Data\Tournaments\` (183 sessions backfilled → 4,014 hands). Less rich (no think-times, no account_ids, no stack_before) but complete. Action enum: 0=fold 1=SB 2=BB 3=call 4=check 5=bet 6=allin 15=ante 23=raise; round 0=blinds 1=preflop 2=flop 3=turn 4=river. Danish decimal commas in amounts.

Both funnel into ONE schema in `hands.jsonl` (`source: blob|xml`; blob wins dedup via the processed-hand ledger). The client's own HUD stats panel is lifetime-aggregated across all tables and holds only ~10–20 hands more than the parseable files — negligible; our own DB with rolling last-N windows is strictly better (stats track how a player plays NOW).

**Guidelines**:
- Blob hands must be harvested DURING play (`ingest_watch`, 2s poll) — TempData does not survive a client relaunch. XML backfill is the safety net but loses the blob-only fields.
- Identity = account_id where known (blob), with a name→account map learned from blob hands merging XML history into the same profile. Names alone collide across time; prefer account_id.
- Stats are QUERIES over the store (`stats.py build(window, min_hands)`), not stored aggregates — lifetime and rolling last-50/100 come from the same facts. The DB file is a processed-hand ledger only.
- Population fitting (`population.py`) MUST exclude hero (`exclude=('Zwonkie',)`) — hero's ~4k hands otherwise dominate every histogram (this bug happened; the fit was garbage until excluded).
- Fit semantics trap: measured raise sizes are amount/pot-BEFORE; the simulator's fractions are pot-AFTER-call. Convert before comparing (see `versions/v48/SPECS.md` Change 1b — measured tables).
- The blob money rule (raw/2/100) was cross-validated against the XML export of the same hand; if a future client update changes amounts, re-derive against XML before trusting either.

"""Derived SQLite index over the hand store (user decision 2026-07-22).

`history/handhistory/<sessioncode>/hands.jsonl` stays the SOURCE OF TRUTH -- append-only,
schema-free, replayable, diffable against the client's own records. This module maintains
`history/handhistory/index.sqlite` as a DISPOSABLE query index for the webapp and windowed
stats: any schema change or suspected corruption is handled by deleting the file and
`rebuild()`ing from the jsonl (seconds at current corpus size). Never write a fact only to
SQLite, never migrate the schema in place.

Tables: hands (one row per hand, full record kept as JSON in `raw` for detail views),
players (per seat), actions (per action, ordered). Indexed by session, player name and
account_id.

Run:  .venv/Scripts/python.exe -m live2.historydb.sqlindex --rebuild
      .venv/Scripts/python.exe -m live2.historydb.sqlindex --status
"""
import argparse
import json
import os
import sqlite3

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INDEX_PATH = os.path.join(REPO, 'history', 'handhistory', 'index.sqlite')

SCHEMA_VERSION = 1   # bump on ANY table change; connect() auto-rebuilds on mismatch

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS hands (
    hand_id      INTEGER PRIMARY KEY,
    sessioncode  TEXT,
    source       TEXT,
    start_local  TEXT,
    tournament_name TEXT,
    sb REAL, bb REAL,
    num_players  INTEGER,
    last_street  TEXT,
    winner_seat  INTEGER,
    winner_name  TEXT,
    pot_won      REAL,
    board        TEXT,
    raw          TEXT
);
CREATE TABLE IF NOT EXISTS players (
    hand_id INTEGER, seat INTEGER, name TEXT, account_id TEXT,
    start_stack REAL, final_stack REAL, dealer INTEGER, hole_cards TEXT,
    PRIMARY KEY (hand_id, seat)
);
CREATE TABLE IF NOT EXISTS actions (
    hand_id INTEGER, ord INTEGER, street TEXT, seat INTEGER, name TEXT,
    action TEXT, amount REAL, stack_before REAL, aggressive INTEGER, think_time_s REAL,
    PRIMARY KEY (hand_id, ord)
);
CREATE INDEX IF NOT EXISTS ix_hands_session ON hands (sessioncode);
CREATE INDEX IF NOT EXISTS ix_players_name ON players (name);
CREATE INDEX IF NOT EXISTS ix_players_acct ON players (account_id);
CREATE INDEX IF NOT EXISTS ix_actions_hand ON actions (hand_id);
"""


def connect(path=INDEX_PATH, auto_rebuild=True):
    """Open (creating if needed) the index. On SCHEMA_VERSION mismatch the file is deleted
    and rebuilt from the jsonl store -- the index is derived, so this is always safe."""
    fresh = not os.path.exists(path)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if row is None:
        conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        conn.commit()
        if not fresh and auto_rebuild:
            _fill_from_store(conn)
    elif int(row[0]) != SCHEMA_VERSION:
        conn.close()
        os.remove(path)
        conn = sqlite3.connect(path)
        conn.executescript(_SCHEMA)
        conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        conn.commit()
        if auto_rebuild:
            _fill_from_store(conn)
    conn.row_factory = sqlite3.Row
    return conn


def add_hands(hands, conn=None):
    """INSERT OR REPLACE decoded hand dicts (blob or xml schema). Safe to re-add: blob and
    xml versions of the same hand share hand_id, last writer wins -- callers should add the
    richer blob AFTER the xml when both exist (ingest order already does)."""
    own = conn is None
    if own:
        conn = connect()
    try:
        for h in hands:
            res = h.get('result') or {}
            blinds = h.get('blinds') or {}
            conn.execute(
                "INSERT OR REPLACE INTO hands VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (h['hand_id'], str(h.get('game_id')), h.get('source'), h.get('start_local'),
                 h.get('tournament_name'), blinds.get('sb'), blinds.get('bb'),
                 h.get('num_players'), h.get('last_street'), res.get('winner_seat'),
                 res.get('winner_name'), res.get('pot_won'),
                 json.dumps(h.get('board') or []), json.dumps(h, ensure_ascii=False)))
            conn.execute("DELETE FROM players WHERE hand_id=?", (h['hand_id'],))
            conn.execute("DELETE FROM actions WHERE hand_id=?", (h['hand_id'],))
            for p in h.get('players', []):
                conn.execute(
                    "INSERT OR REPLACE INTO players VALUES (?,?,?,?,?,?,?,?)",
                    (h['hand_id'], p['seat'], p['name'],
                     str(p['account_id']) if p.get('account_id') else None,
                     p.get('start_stack'), p.get('final_stack'),
                     int(bool(p.get('dealer'))), json.dumps(p.get('hole_cards') or [])))
            for i, a in enumerate(h.get('actions', [])):
                conn.execute(
                    "INSERT OR REPLACE INTO actions VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (h['hand_id'], i, a.get('street'), a.get('seat'), a.get('name'),
                     a.get('action'), a.get('amount'), a.get('stack_before'),
                     int(bool(a.get('aggressive'))), a.get('think_time_s')))
        conn.commit()
    finally:
        if own:
            conn.close()


def _fill_from_store(conn):
    from live2.historydb.stats import load_hands
    hands = load_hands()
    add_hands(hands, conn=conn)
    return len(hands)


def rebuild():
    """Delete + rebuild the whole index from the jsonl store. -> hand count."""
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
    conn = connect(auto_rebuild=False)
    try:
        return _fill_from_store(conn)
    finally:
        conn.close()


def status():
    if not os.path.exists(INDEX_PATH):
        return {'exists': False}
    conn = connect()
    try:
        out = {'exists': True, 'path': INDEX_PATH,
               'schema_version': SCHEMA_VERSION,
               'hands': conn.execute("SELECT COUNT(*) FROM hands").fetchone()[0],
               'sessions': conn.execute("SELECT COUNT(DISTINCT sessioncode) FROM hands").fetchone()[0],
               'players': conn.execute("SELECT COUNT(DISTINCT name) FROM players").fetchone()[0],
               'actions': conn.execute("SELECT COUNT(*) FROM actions").fetchone()[0]}
        return out
    finally:
        conn.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--status', action='store_true')
    args = ap.parse_args()
    if args.rebuild:
        n = rebuild()
        print(f"rebuilt {INDEX_PATH}: {n} hands")
    print(json.dumps(status(), indent=1))

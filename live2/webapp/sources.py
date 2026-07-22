"""Read-only data sources for the dashboard. Everything here OPENS files and returns
plain dicts; nothing writes. Three feeds:

  live      -- history/<board_id>/turns.jsonl (legacy recorder format 2; the assembler
               will publish the same shape directly once it exists)
  handstore -- history/handhistory/<sessioncode>/hands.jsonl (ground truth, one hand behind)
  stats     -- live2.historydb.stats windowed profiles

Provenance note: format-2 turn records predate per-field provenance, so the live view
labels them source='legacy-recorder'. Assembler records will carry real provenance.
"""
import glob
import json
import os

from live2.historydb import stats as hstats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HISTORY = os.path.join(REPO, 'history')
HANDSTORE = os.path.join(HISTORY, 'handhistory')


# ------------------------------------------------------------------ live (turns.jsonl)

def _board_dirs():
    out = []
    for d in glob.glob(os.path.join(HISTORY, '*')):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'turns.jsonl')):
            out.append(d)
    return sorted(out, key=os.path.getmtime, reverse=True)


def latest_board():
    """(board_id, turns_path) of the most recently written session, or (None, None)."""
    dirs = _board_dirs()
    if not dirs:
        return None, None
    return os.path.basename(dirs[0]), os.path.join(dirs[0], 'turns.jsonl')


def read_turns(turns_path, offset=0):
    """New complete JSON lines after byte `offset` -> (records, new_offset).
    Tolerates a partially-written last line by not advancing past it."""
    records = []
    try:
        size = os.path.getsize(turns_path)
        if size <= offset:
            return records, offset
        with open(turns_path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            chunk = f.read()
        consumed = 0
        for line in chunk.splitlines(keepends=True):
            if not line.endswith('\n'):
                break                      # partial write in flight -- retry next poll
            consumed += len(line.encode('utf-8'))
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records, offset + consumed
    except OSError:
        return records, offset


def live_snapshot():
    """Latest turn record of the latest session, wrapped with feed metadata."""
    board_id, path = latest_board()
    if not path:
        return {'board_id': None, 'turn': None, 'feed': 'none'}
    records, _ = read_turns(path)
    return {
        'board_id': board_id,
        'feed': 'legacy-recorder',
        'mtime': os.path.getmtime(path),
        'turn': records[-1] if records else None,
        'turn_count': len(records),
    }


def flagged_turns(limit=50):
    """F12 flags, newest session first. The live flag flow writes a per-board flags.jsonl
    ({turn, ts, dir, action}) plus an artifact folder; join each entry with its full turn
    record from turns.jsonl so the review queue can show the whole decision."""
    out = []
    for d in _board_dirs():
        flags_path = os.path.join(d, 'flags.jsonl')
        if not os.path.exists(flags_path):
            continue
        records, _ = read_turns(os.path.join(d, 'turns.jsonl'))
        by_turn = {r.get('turn'): r for r in records}
        with open(flags_path, encoding='utf-8') as f:
            for line in f:
                try:
                    fl = json.loads(line)
                except json.JSONDecodeError:
                    continue
                art_dir = fl.get('dir') or ''
                artifacts = sorted(os.listdir(art_dir)) if os.path.isdir(art_dir) else []
                out.append({'board_id': os.path.basename(d), 'flag': fl,
                            'artifacts': artifacts, 'record': by_turn.get(fl.get('turn'))})
        if len(out) >= limit:
            break
    return out[:limit]


# ------------------------------------------------------------------ hand store
# Browse queries go through the derived SQLite index (live2/historydb/sqlindex.py) when it
# exists -- the jsonl scans remain as the fallback so a deleted/missing index degrades to
# slower, never to broken. hands.jsonl stays the source of truth either way.

def _index_conn():
    from live2.historydb import sqlindex
    if not os.path.exists(sqlindex.INDEX_PATH):
        return None
    return sqlindex.connect()


def list_sessions():
    conn = _index_conn()
    if conn:
        try:
            rows = conn.execute(
                "SELECT sessioncode, COUNT(*) AS hands, MAX(hand_id) AS latest "
                "FROM hands GROUP BY sessioncode ORDER BY latest DESC").fetchall()
            return [{'sessioncode': r['sessioncode'], 'hands': r['hands']} for r in rows]
        finally:
            conn.close()
    out = []
    for d in sorted(glob.glob(os.path.join(HANDSTORE, '*')), reverse=True):
        p = os.path.join(d, 'hands.jsonl')
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                n = sum(1 for line in f if line.strip())
            out.append({'sessioncode': os.path.basename(d), 'hands': n,
                        'mtime': os.path.getmtime(p)})
    return out


def list_hands(sessioncode=None, limit=100, player=None):
    """Newest-first hand summaries for the browser list. `player` filter is index-only."""
    conn = _index_conn()
    if conn:
        try:
            q = ("SELECT h.hand_id, h.sessioncode, h.source, h.winner_name, h.winner_seat, "
                 "h.pot_won, h.last_street FROM hands h")
            where, params = [], []
            if player:
                q += " JOIN players p ON p.hand_id = h.hand_id"
                where.append("p.name = ?")
                params.append(player)
            if sessioncode:
                where.append("h.sessioncode = ?")
                params.append(str(sessioncode))
            if where:
                q += " WHERE " + " AND ".join(where)
            q += " ORDER BY h.hand_id DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(q, params).fetchall()
            out = []
            for r in rows:
                names = [x['name'] for x in conn.execute(
                    "SELECT name FROM players WHERE hand_id=? ORDER BY seat", (r['hand_id'],))]
                out.append({'hand_id': r['hand_id'], 'sessioncode': r['sessioncode'],
                            'source': r['source'], 'players': names,
                            'winner': r['winner_name'] or r['winner_seat'],
                            'pot': r['pot_won'], 'last_street': r['last_street']})
            return out
        finally:
            conn.close()
    if sessioncode:
        paths = [os.path.join(HANDSTORE, str(sessioncode), 'hands.jsonl')]
    else:
        paths = [os.path.join(d, 'hands.jsonl') for d in
                 sorted(glob.glob(os.path.join(HANDSTORE, '*')), reverse=True)]
    hands = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding='utf-8') as f:
            for line in f:
                try:
                    hands.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if len(hands) >= limit * 3:        # enough to sort/trim without loading all 4k
            break
    hands.sort(key=lambda h: h['hand_id'], reverse=True)
    out = []
    for h in hands[:limit]:
        res = h.get('result') or {}
        out.append({
            'hand_id': h['hand_id'],
            'sessioncode': h.get('sessioncode') or h.get('game_id'),
            'source': h.get('source'),
            'players': [p['name'] for p in h['players']],
            'winner': res.get('winner_name') or res.get('winner_seat'),
            'pot': res.get('pot_won'),
            'last_street': h.get('last_street'),
        })
    return out


def get_hand(sessioncode, hand_id):
    conn = _index_conn()
    if conn:
        try:
            row = conn.execute("SELECT raw FROM hands WHERE hand_id=?",
                               (int(hand_id),)).fetchone()
            if row:
                return json.loads(row['raw'])
        except (ValueError, TypeError):
            pass
        finally:
            conn.close()
    p = os.path.join(HANDSTORE, str(sessioncode), 'hands.jsonl')
    if not os.path.exists(p):
        return None
    with open(p, encoding='utf-8') as f:
        for line in f:
            try:
                h = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(h.get('hand_id')) == str(hand_id):
                return h
    return None


# ------------------------------------------------------------------ stats

def opponent_profiles(window=100, min_hands=10):
    profiles, total = hstats.build(window=window, min_hands=min_hands)
    rows = sorted(profiles.items(), key=lambda kv: -kv[1]['lifetime']['hands'])
    return {'total_hands': total, 'window': window,
            'players': [{'key': k, **v} for k, v in rows]}

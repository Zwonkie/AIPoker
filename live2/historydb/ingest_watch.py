"""Live hand-history ingester for the bet365 client (V48 3b work item, built 2026-07-22).

Watches the client's History tree for new per-hand protobuf blobs (written the moment a hand
completes -- see decode_bet365.py for the wire format), decodes each, and maintains:

  history/handhistory/<sessioncode>/hands.jsonl   -- one decoded hand (JSON) per line
  history/opponent_db.json                        -- processed-hand ledger only (restarts and
                                                     the XML backfill never double-count).
                                                     Stats are computed on demand from the
                                                     store by tools/handhistory/stats.py
                                                     (lifetime + rolling-window).

Prints a one-line summary per ingested hand. Polling (default 2s) -- no deps.

Run:  .venv/Scripts/python.exe tools/handhistory/ingest_watch.py
      [--root "C:/Users/zwonk/AppData/Local/Poker at Bet365.DK/data/Zwonkie/History"]
      [--once]   # ingest the backlog and exit (no watch loop)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from live2.historydb.decode_bet365 import decode  # noqa: E402

DEFAULT_ROOT = r"C:\Users\zwonk\AppData\Local\Poker at Bet365.DK\data\Zwonkie\History"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(REPO, 'history', 'handhistory')
DB_PATH = os.path.join(REPO, 'history', 'opponent_db.json')

AGGRESSIVE = {'bet', 'raise'}
VOLUNTARY = {'call', 'bet', 'raise'}


def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, encoding='utf-8') as f:
            return json.load(f)
    return {'players': {}, 'processed_hands': []}


def save_db(db):
    tmp = DB_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=1, ensure_ascii=False)
    os.replace(tmp, DB_PATH)


def _unused_update_stats(db, hand):  # superseded by stats.py (kept for reference only)
    """Accumulate per-player counters from one decoded hand."""
    by_seat = {p['seat']: p for p in hand['players']}
    per_seat = {p['seat']: {'vpip': False, 'pfr': False, 'bets': 0, 'calls': 0,
                            'think': [], 'saw_showdown': False} for p in hand['players']}
    last_street_actors = set()
    for a in hand['actions']:
        s = per_seat.get(a['seat'])
        if s is None:
            continue
        if a['street'] == 'preflop':
            if a['action'] in VOLUNTARY:
                s['vpip'] = True
            if a['action'] == 'raise':
                s['pfr'] = True
        if a['action'] in AGGRESSIVE:
            s['bets'] += 1
        elif a['action'] == 'call':
            s['calls'] += 1
        if a['action'] not in ('post_sb', 'post_bb'):
            s['think'].append(a['think_time_s'])
        if a['street'] == hand.get('last_street'):
            last_street_actors.add(a['seat'])

    river_folds = {a['seat'] for a in hand['actions']
                   if a['street'] == hand.get('last_street') and a['action'] == 'fold'}
    for seat in last_street_actors - river_folds:
        if hand.get('last_street') == 'river':
            per_seat[seat]['saw_showdown'] = True

    for seat, s in per_seat.items():
        p = by_seat[seat]
        key = str(p['account_id'] or p['name'])
        rec = db['players'].setdefault(key, {
            'name': p['name'], 'hands': 0, 'vpip': 0, 'pfr': 0, 'bets': 0, 'calls': 0,
            'showdowns': 0, 'think_total_s': 0.0, 'think_actions': 0, 'avatars': [],
        })
        rec['name'] = p['name']
        rec['hands'] += 1
        rec['vpip'] += int(s['vpip'])
        rec['pfr'] += int(s['pfr'])
        rec['bets'] += s['bets']
        rec['calls'] += s['calls']
        rec['showdowns'] += int(s['saw_showdown'])
        rec['think_total_s'] += sum(s['think'])
        rec['think_actions'] += len(s['think'])
        # avatars are client-assigned portraits (recur across accounts) -- kept only as a
        # weak cross-check, never identity; account_id is identity.
        if p.get('avatar') and p['avatar'] not in rec['avatars']:
            rec['avatars'].append(p['avatar'])


def find_blobs(root):
    """-> {hand_id_str: full_path} for every numeric-named file under Tournaments/Tables."""
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        if 'Tournaments' not in dirpath and 'Tables' not in dirpath:
            continue
        for fn in files:
            if fn.isdigit():
                out[fn] = os.path.join(dirpath, fn)
    return out


def ingest(path, db):
    hand = decode(path)
    hand['source'] = 'blob'
    sess = str(hand['game_id'])
    os.makedirs(os.path.join(OUT_DIR, sess), exist_ok=True)
    with open(os.path.join(OUT_DIR, sess, 'hands.jsonl'), 'a', encoding='utf-8') as f:
        f.write(json.dumps(hand, ensure_ascii=False) + '\n')
    try:   # derived index only -- a SQLite hiccup must never block ground-truth capture
        from live2.historydb import sqlindex
        sqlindex.add_hands([hand])
    except Exception as e:
        print(f"  ! sqlindex update failed ({e}) -- jsonl written, rebuild index later", flush=True)
    r = hand.get('result') or {}
    hero = next((p for p in hand['players'] if p['name'] == 'Zwonkie'), None)
    hero_cards = ' '.join(c or '?' for c in (hero or {}).get('hole_cards', [])) or '--'
    print(f"[{time.strftime('%H:%M:%S')}] hand {hand['hand_id']} | blinds "
          f"{hand['blinds']['sb']}/{hand['blinds']['bb']} | hero {hero_cards} | "
          f"board {' '.join(hand['board']) or '--'} | pot {r.get('pot_won', 0)} -> "
          f"{r.get('winner_name', '?')} | {len(hand['actions'])} actions", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--poll', type=float, default=2.0)
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    db = load_db()
    done = set(db['processed_hands'])
    print(f"watching {args.root} (already ingested: {len(done)} hands)", flush=True)

    while True:
        blobs = find_blobs(args.root)
        new = sorted((hid, p) for hid, p in blobs.items() if hid not in done)
        for hid, path in new:
            try:
                ingest(path, db)
            except Exception as e:  # mid-write race: retry next poll
                print(f"  ! {hid}: {e} (will retry)", flush=True)
                continue
            done.add(hid)
            db['processed_hands'].append(hid)
        if new:
            save_db(db)
        if args.once:
            break
        time.sleep(args.poll)


if __name__ == '__main__':
    main()

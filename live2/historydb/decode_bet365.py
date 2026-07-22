"""Decode a bet365 protobuf hand-history blob (file named <hand_id>, e.g. 12389144736) into
semantic JSON.

Wire facts established from hand 12389144736 (2026-07-22), verified against the hand's own
internal consistency (stack arithmetic, blind posts, pot progression, winner stack):

- File = 4-byte little-endian payload length + protobuf message.
- Small ints (ids, seats, streets, action codes, per-player counters) are ZIGZAG-encoded
  (sint): stored = 2*value for value >= 0. Hand id field 4 zigzag-decodes to the filename;
  tournament id field 2 zigzag-decodes into the same 1171xxx range as the live recorder's
  Double_Or_Nothing_<id> folders.
- Header timestamps (fields 7/31) are zigzag ms-epoch; per-event timestamps (event field 13)
  are PLAIN ms-epoch (they equal header/2's magnitude directly).
- Money amounts appear as bare varints (event field 4) and as {1: amount, 2: 4} submessages.
  RESOLVED (2026-07-22, cross-checked against the client's official XML export of the same
  session, 8675336738.xml game 12389144736): amounts are zigzag-encoded with two implied
  decimals -- display_chips = raw / 2 / 100 (pb 2000 == blinds "10", pb 140000 == stack
  "700"). `--raw-money` disables the conversion for wire-level debugging.
- Event: 1 street(1..4) | 2 seat | 3 action code | 4 chips committed by this action
  (incremental) | 5 actor stack before acting | 6 actor's per-hand action counter | 7 flag
  (14 on blind posts, 6 on some folds -- auto-action?) | 8 cards shown ("X"=hidden) |
  9 aggressive flag | 13 ms timestamp.
- Player: 1 name | 3 final stack | 5 seat | 9 hole cards | 11 flag | 15/22 numeric hole cards
  {1: suit, 2: rank-2} (suit 1=h 2=s 3=c, 7/14=hidden) | 19 {1: avatar} (client avatar/portrait
  name, e.g. "fish" -- user-confirmed 2026-07-22 NOT a player note; same avatar recurs on
  different accounts) | 24 account id.
- Top-level: 6 player count | 11 board strings ("C2"=2c) | 13 pot resolution (1 winner seat,
  2 total collected, 3 main-pot amount, 4 uncalled returned, 6 winner's resulting stack) |
  14 last street | 17 per-street pot | 21/22 SB/BB | 24 numeric board cards.

Run:  .venv/Scripts/python.exe tools/handhistory/decode_bet365.py <file> [--raw-money] [--out x.json]
"""
import argparse
import json
import os
import struct
import sys

STREETS = {1: 'preflop', 2: 'flop', 3: 'turn', 4: 'river'}
ACTIONS = {0: 'fold', 1: 'post_sb', 2: 'post_bb', 3: 'call', 4: 'check', 5: 'bet', 23: 'raise'}
SUITS = {1: 'h', 2: 's', 3: 'c', 4: 'd', 0: 'd'}


def read_varint(b, i):
    out, shift = 0, 0
    while True:
        v = b[i]
        i += 1
        out |= (v & 0x7F) << shift
        if not v & 0x80:
            return out, i
        shift += 7


def zz(n):
    """zigzag sint decode."""
    return (n >> 1) ^ -(n & 1)


def parse_msg(b):
    """-> dict field -> list of values (int | bytes | nested dict attempt deferred)."""
    fields = {}
    i = 0
    while i < len(b):
        key, i = read_varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = read_varint(b, i)
        elif wt == 1:
            v = struct.unpack('<d', b[i:i + 8])[0]
            i += 8
        elif wt == 2:
            ln, i = read_varint(b, i)
            v = b[i:i + ln]
            i += ln
        elif wt == 5:
            v = struct.unpack('<f', b[i:i + 4])[0]
            i += 4
        else:
            raise ValueError(f"wiretype {wt}")
        fields.setdefault(f, []).append(v)
    return fields


def first(fields, f, default=None):
    v = fields.get(f)
    return v[0] if v else default


def money(v, raw=False):
    """v is a {1: amount, 2: 4} submessage (bytes) or a bare int. Wire value is zigzag with
    two implied decimals: display chips = raw / 200 (validated vs the XML export)."""
    if isinstance(v, bytes):
        v = first(parse_msg(v), 1, 0)
    if raw:
        return v
    chips = zz(v) / 100.0
    return int(chips) if chips == int(chips) else chips


def card_str(s):
    """'C2' -> '2c', 'X' -> None."""
    s = s.decode() if isinstance(s, bytes) else s
    if s == 'X' or len(s) < 2:
        return None
    return s[1:] + s[0].lower()


def decode(path, raw_money=False):
    raw = open(path, 'rb').read()
    plen = struct.unpack('<I', raw[:4])[0]
    top = parse_msg(raw[4:4 + plen])

    start_ms = zz(first(top, 7, 0))
    end_ms = zz(first(top, 31, 0))

    seat_names = {}
    players = []
    for pb in top.get(10, []):
        p = parse_msg(pb)
        seat = zz(first(p, 5, 0))
        name = first(p, 1, b'?').decode('utf-8', 'replace')
        seat_names[seat] = name
        avatar = None
        if p.get(19):
            avatar = first(parse_msg(p[19][0]), 1, b'').decode('utf-8', 'replace') or None
        players.append({
            'seat': seat,
            'name': name,
            'hole_cards': [card_str(c) for c in p.get(9, [])],
            'final_stack': money(first(p, 3, 0), raw_money),
            'avatar': avatar,
            'account_id': first(p, 24),
        })

    actions = []
    prev_ms = start_ms
    for eb in top.get(9, []):
        e = parse_msg(eb)
        code = zz(first(e, 3, 0))
        ts = first(e, 13, 0)
        seat = zz(first(e, 2, 0))
        actions.append({
            'street': STREETS.get(zz(first(e, 1, 0)), f"street_{zz(first(e, 1, 0))}"),
            'seat': seat,
            'name': seat_names.get(seat, f"seat{seat}"),
            'action': ACTIONS.get(code, f"action_{code}"),
            'amount': money(first(e, 4, 0), raw_money),
            'stack_before': money(first(e, 5, 0), raw_money),
            'aggressive': bool(first(e, 9, 0)),
            'think_time_s': round(max(0, ts - prev_ms) / 1000.0, 2),
            'cards_shown': [c for c in (card_str(c) for c in e.get(8, [])) if c],
        })
        prev_ms = ts

    pot_by_street = {}
    for sb_ in top.get(17, []):
        s = parse_msg(sb_)
        st = zz(first(s, 1, 0))
        inner = parse_msg(first(s, 2, b''))
        pot_by_street[STREETS.get(st, 'end' if st == 4 else f"street_{st}")] = \
            money(first(parse_msg(first(inner, 2, b'')), 1, 0), raw_money)

    result = {}
    if top.get(13):
        r = parse_msg(top[13][0])
        wseat = zz(first(r, 1, 0))
        result = {
            'winner_seat': wseat,
            'winner_name': seat_names.get(wseat),
            'total_collected': money(first(r, 2, 0), raw_money),
            'pot_won': money(first(parse_msg(first(r, 3, b'')), 2, 0), raw_money),
            'uncalled_returned': money(first(r, 4, 0), raw_money),
            'winner_final_stack': money(first(r, 6, 0), raw_money),
        }

    return {
        'hand_id': zz(first(top, 4, 0)),
        'tournament_id': zz(first(top, 2, 0)),
        'game_id': zz(first(top, 3, 0)),
        'start_utc_ms': start_ms,
        'duration_s': round((end_ms - start_ms) / 1000.0, 1),
        'num_players': zz(first(top, 6, 0)),
        'blinds': {'sb': money(first(top, 21, 0), raw_money),
                   'bb': money(first(top, 22, 0), raw_money)},
        'money_scale': 'raw' if raw_money else 'display_chips',
        'players': sorted(players, key=lambda p: p['seat']),
        'board': [card_str(c) for c in top.get(11, [])],
        'last_street': STREETS.get(zz(first(top, 14, 0))),
        'actions': actions,
        'pot_by_street': pot_by_street,
        'result': result,
    }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--raw-money', action='store_true',
                    help="emit wire-level money values (no /200 conversion)")
    ap.add_argument('--out', help="write JSON here instead of stdout")
    args = ap.parse_args()
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    doc = decode(args.path, raw_money=args.raw_money)
    text = json.dumps(doc, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)

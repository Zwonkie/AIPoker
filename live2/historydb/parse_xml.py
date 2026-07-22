"""iPoker session-XML parser -> the same hand-JSON schema as decode_bet365.py.

The client persists every tournament session as XML under
History\\Data\\Tournaments\\<sessioncode>.xml (183 sessions back to 2025-04 at discovery).
XML hands lack the blob-only extras (per-action think-time, stack_before, account ids,
avatars) -- fields are set None/omitted; `source: "xml"` marks the thinner provenance
(blob-ingested hands carry `source: "blob"`).

Action `type` enum (iPoker): 0=fold 1=post_sb 2=post_bb 3=call 4=check 5=bet 6=allin
15=ante 23=raise. Round no: 0=blinds 1=preflop 2=flop 3=turn 4=river.

Run:  .venv/Scripts/python.exe tools/handhistory/parse_xml.py <session.xml> [--out x.json]
      (library use: parse_session(path) -> list of hand dicts)
"""
import json
import os
import sys
import xml.etree.ElementTree as ET

ACTIONS = {0: 'fold', 1: 'post_sb', 2: 'post_bb', 3: 'call', 4: 'check', 5: 'bet',
           6: 'allin', 7: 'muck', 15: 'ante', 23: 'raise'}
ROUND_STREET = {0: 'blinds', 1: 'preflop', 2: 'flop', 3: 'turn', 4: 'river'}
AGGRESSIVE = {'bet', 'raise', 'allin'}


def _num(s):
    """'1,335' (thousands) / '0,50' (decimal) / '10' -> number."""
    if s is None or s == '':
        return 0
    s = s.strip()
    if ',' in s:
        head, tail = s.rsplit(',', 1)
        if len(tail) == 3:  # thousands separator
            s = s.replace(',', '')
        else:
            s = head.replace(',', '') + '.' + tail
    v = float(s)
    return int(v) if v == int(v) else v


def _card(tok):
    """'C2'->'2c', 'D10'->'10d', 'X'->None."""
    tok = tok.strip()
    if not tok or tok.upper() == 'X':
        return None
    return tok[1:] + tok[0].lower()


def parse_session(path):
    root = ET.parse(path).getroot()
    sessioncode = int(root.get('sessioncode'))
    gen = root.find('general')
    tournamentcode = (gen.findtext('tournamentcode') or '0').strip()
    tournamentname = (gen.findtext('tournamentname') or '').strip()

    hands = []
    for game in root.findall('game'):
        g_gen = game.find('general')
        players = []
        seat_names = {}
        wins = {}
        for pl in g_gen.find('players').findall('player'):
            seat = int(pl.get('seat'))
            name = pl.get('name')
            seat_names[name] = seat
            wins[seat] = _num(pl.get('win'))
            players.append({
                'seat': seat,
                'name': name,
                'hole_cards': [None, None],
                'start_stack': _num(pl.get('chips')),
                'final_stack': None,          # not in XML (start chips + win only)
                'dealer': pl.get('dealer') == '1',
                'avatar': None,
                'account_id': None,
            })
        by_seat = {p['seat']: p for p in players}

        actions = []
        board = []
        last_street = None
        for rnd in game.findall('round'):
            street = ROUND_STREET.get(int(rnd.get('no')), f"round_{rnd.get('no')}")
            for cards in rnd.findall('cards'):
                ctype = (cards.get('type') or '').lower()
                toks = [(_card(t)) for t in (cards.text or '').split()]
                if ctype == 'pocket':
                    p = by_seat.get(seat_names.get(cards.get('player')))
                    if p is not None and any(toks):
                        p['hole_cards'] = toks
                elif ctype in ('flop', 'turn', 'river'):
                    board.extend(c for c in toks if c)
            for act in rnd.findall('action'):
                code = int(act.get('type'))
                name = act.get('player')
                a = ACTIONS.get(code, f"action_{code}")
                actions.append({
                    'street': 'preflop' if street == 'blinds' else street,
                    'seat': seat_names.get(name, -1),
                    'name': name,
                    'action': a,
                    'amount': _num(act.get('sum')),
                    'stack_before': None,
                    'aggressive': a in AGGRESSIVE,
                    'think_time_s': None,
                    'cards_shown': [],
                })
                if street != 'blinds':
                    last_street = street

        winners = [(s, w) for s, w in wins.items() if w > 0]
        winners.sort(key=lambda sw: -sw[1])
        result = {}
        if winners:
            wseat = winners[0][0]
            result = {
                'winner_seat': wseat,
                'winner_name': by_seat[wseat]['name'],
                'pot_won': winners[0][1],
                'all_winners': [{'seat': s, 'name': by_seat[s]['name'], 'won': w}
                                for s, w in winners],
            }

        hands.append({
            'hand_id': int(game.get('gamecode')),
            'tournament_id': int(tournamentcode) if tournamentcode.isdigit() else tournamentcode,
            'tournament_name': tournamentname,
            'game_id': sessioncode,
            'start_local': (g_gen.findtext('startdate') or '').strip(),
            'num_players': len(players),
            'blinds': {'sb': _num(g_gen.findtext('smallblind')),
                       'bb': _num(g_gen.findtext('bigblind'))},
            'source': 'xml',
            'players': sorted(players, key=lambda p: p['seat']),
            'board': board,
            'last_street': last_street,
            'actions': actions,
            'result': result,
        })
    return hands


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--out')
    args = ap.parse_args()
    doc = parse_session(args.path)
    text = json.dumps(doc, indent=2, ensure_ascii=False)
    if args.out:
        open(args.out, 'w', encoding='utf-8').write(text)
        print(f"wrote {args.out} ({len(doc)} hands)")
    else:
        print(text)

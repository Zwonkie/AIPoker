"""Windowed opponent-stats query engine over the hand store (the authoritative stats path --
the DB written by ingest_watch.py holds only the processed-hand ledger).

Loads every history/handhistory/*/hands.jsonl record, resolves identity (account_id when
known -- blob hands -- else name; a name->account map learned from blob hands merges XML
history into the account profile), orders each player's hands by hand_id (monotonic over
time), and computes stats LIFETIME and over a rolling last-N window so profiles track how a
player plays NOW, not their 2025 self.

Per player: hands, VPIP%, PFR%, limp%, 3bet%, AF (bets+raises)/calls, WTSD%, WSD%,
avg think-time (blob hands only), avg preflop raise size (bb).

Run:  .venv/Scripts/python.exe tools/handhistory/stats.py [--window 100] [--min-hands 20]
      [--player NAME] [--out history/opponent_stats.json]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.join(REPO, 'history', 'handhistory')

VOLUNTARY = {'call', 'bet', 'raise', 'allin'}
AGGRESSIVE = {'bet', 'raise', 'allin'}


def load_hands():
    hands = []
    for path in glob.glob(os.path.join(OUT_DIR, '*', 'hands.jsonl')):
        with open(path, encoding='utf-8') as f:
            for line in f:
                try:
                    hands.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    hands.sort(key=lambda h: h['hand_id'])
    return hands


def identity_maps(hands):
    """name -> account_id learned wherever a blob hand knew both."""
    name_to_acct = {}
    for h in hands:
        for p in h['players']:
            if p.get('account_id'):
                name_to_acct[p['name']] = p['account_id']
    return name_to_acct


def per_player_hand_facts(hands, name_to_acct):
    """-> {player_key: [fact dict per hand, ordered by hand_id]}"""
    out = defaultdict(list)
    for h in hands:
        seats = {p['seat']: p for p in h['players']}
        facts = {s: {'vpip': False, 'pfr': False, 'limp': False, 'threebet': False,
                     'bets': 0, 'calls': 0, 'think': [], 'pf_raise_bb': [],
                     'showdown': False, 'won_sd': False} for s in seats}
        bb = (h.get('blinds') or {}).get('bb') or 1
        pf_raises = 0
        for a in h['actions']:
            f = facts.get(a['seat'])
            if f is None:
                continue
            act = a['action']
            if a['street'] == 'preflop':
                if act in VOLUNTARY:
                    f['vpip'] = True
                if act == 'call' and pf_raises == 0:
                    f['limp'] = True
                if act in ('raise', 'allin', 'bet'):
                    f['pfr'] = True
                    if pf_raises >= 1:
                        f['threebet'] = True
                    pf_raises += 1
                    if a['amount']:
                        f['pf_raise_bb'].append(a['amount'] / bb)
            if act in AGGRESSIVE:
                f['bets'] += 1
            elif act == 'call':
                f['calls'] += 1
            if a.get('think_time_s') is not None and act not in ('post_sb', 'post_bb', 'ante'):
                f['think'].append(a['think_time_s'])

        if h.get('last_street') == 'river':
            river_folds = {a['seat'] for a in h['actions']
                           if a['street'] == 'river' and a['action'] == 'fold'}
            river_actors = {a['seat'] for a in h['actions'] if a['street'] == 'river'}
            wseat = (h.get('result') or {}).get('winner_seat')
            for s in river_actors - river_folds:
                facts[s]['showdown'] = True
                facts[s]['won_sd'] = (s == wseat)

        for s, f in facts.items():
            p = seats[s]
            key = str(p.get('account_id') or name_to_acct.get(p['name']) or p['name'])
            f['name'] = p['name']
            f['hand_id'] = h['hand_id']
            out[key].append(f)
    return out


def summarize(fact_list):
    n = len(fact_list)
    if n == 0:
        return None
    s = {'hands': n}
    s['vpip'] = 100.0 * sum(f['vpip'] for f in fact_list) / n
    s['pfr'] = 100.0 * sum(f['pfr'] for f in fact_list) / n
    s['limp'] = 100.0 * sum(f['limp'] for f in fact_list) / n
    s['threebet'] = 100.0 * sum(f['threebet'] for f in fact_list) / n
    bets = sum(f['bets'] for f in fact_list)
    calls = sum(f['calls'] for f in fact_list)
    s['af'] = bets / calls if calls else None
    s['wtsd'] = 100.0 * sum(f['showdown'] for f in fact_list) / n
    sd = sum(f['showdown'] for f in fact_list)
    s['wsd'] = 100.0 * sum(f['won_sd'] for f in fact_list) / sd if sd else None
    think = [t for f in fact_list for t in f['think']]
    s['think_avg_s'] = sum(think) / len(think) if think else None
    sizes = [x for f in fact_list for x in f['pf_raise_bb']]
    s['pf_raise_avg_bb'] = sum(sizes) / len(sizes) if sizes else None
    return s


def build(window=None, min_hands=1):
    hands = load_hands()
    name_to_acct = identity_maps(hands)
    per = per_player_hand_facts(hands, name_to_acct)
    out = {}
    for key, facts in per.items():
        if len(facts) < min_hands:
            continue
        rec = {'name': facts[-1]['name'], 'lifetime': summarize(facts)}
        if window:
            rec[f'last_{window}'] = summarize(facts[-window:])
        out[key] = rec
    return out, len(hands)


def fmt(s):
    if s is None:
        return "--"
    af = f"{s['af']:.1f}" if s['af'] is not None else "inf"
    th = f"{s['think_avg_s']:.1f}" if s['think_avg_s'] is not None else "  --"
    return (f"{s['hands']:>5}  {s['vpip']:>5.0f} {s['pfr']:>4.0f} {s['limp']:>5.0f} "
            f"{s['threebet']:>5.1f} {af:>5} {s['wtsd']:>5.0f} "
            f"{(s['wsd'] if s['wsd'] is not None else 0):>4.0f} {th:>6}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=100)
    ap.add_argument('--min-hands', type=int, default=20)
    ap.add_argument('--player')
    ap.add_argument('--out')
    args = ap.parse_args()

    stats, total = build(window=args.window, min_hands=args.min_hands)
    rows = sorted(stats.items(), key=lambda kv: -kv[1]['lifetime']['hands'])
    if args.player:
        rows = [(k, v) for k, v in rows if args.player.lower() in v['name'].lower()]
    print(f"{total} hands in store | {len(stats)} players with >= {args.min_hands} hands")
    print(f"{'player':<18} {'scope':<9} {'hands':>5}  {'VPIP':>5} {'PFR':>4} {'limp%':>5} "
          f"{'3bet%':>5} {'AF':>5} {'WTSD':>5} {'WSD':>4} {'think':>6}")
    for key, rec in rows[:40]:
        print(f"{rec['name']:<18} {'lifetime':<9} {fmt(rec['lifetime'])}")
        wkey = f'last_{args.window}'
        if rec.get(wkey) and rec[wkey]['hands'] < rec['lifetime']['hands']:
            print(f"{'':<18} {wkey:<9} {fmt(rec[wkey])}")
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=1, ensure_ascii=False)
        print(f"wrote {args.out}")

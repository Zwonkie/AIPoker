"""[V48 Change 1b] Population fitting from the hand store — replaces hand-authored opponent
archetypes with the measured player pool.

Assigns every recurring player (>= min-hands shared hands) to one of the simulator's four
archetypes by (VPIP, AF, limp%), then fits per-archetype:
  - preflop/postflop raise-size histograms as pot fractions, mapped onto the simulator's
    bucket grid {0.33, 0.50, 0.66, 0.75, 1.00, 1.50, jam} (pot state reconstructed per hand
    by accumulating action amounts; jam = amount >= ~95% of the actor's reconstructed stack),
  - limp rate, and
  - the archetype MIXTURE (player-weighted — each opponent seat is a draw from the pool).

Archetype rules (aligned with the sim's TAG/LAG(maniac)/Nit/Calling Station semantics):
  Nit             VPIP < 18
  TAG             18 <= VPIP < 30 and AF >= 1.5
  Calling Station VPIP >= 30 and AF < 1.2, or (VPIP >= 25 and limp% >= 12)
  LAG             everything else aggressive (VPIP >= 18, AF >= 1.2 outside TAG bounds)

Output: printed tables + optional --out JSON. The V48 clone copies the fitted tables into its
opponent_bots.py with provenance (this module never edits version slices).

Run:  .venv/Scripts/python.exe live2/historydb/population.py [--min-hands 40] [--out x.json]
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from live2.historydb.stats import load_hands, identity_maps, per_player_hand_facts, summarize  # noqa: E402

BUCKETS = [0.33, 0.50, 0.66, 0.75, 1.00, 1.50]   # pot fractions; jam is its own class
AGGRESSIVE = {'bet', 'raise', 'allin'}


def archetype_of(s):
    """s = summarize() dict -> archetype name."""
    vpip, af, limp = s['vpip'], (s['af'] if s['af'] is not None else 99.0), s['limp']
    if vpip < 18:
        return 'Nit'
    if vpip >= 30 and af < 1.2 or (vpip >= 25 and limp >= 12):
        return 'Calling Station'
    if vpip < 30 and af >= 1.5:
        return 'TAG'
    return 'LAG'


def nearest_bucket(frac):
    return min(BUCKETS, key=lambda b: abs(b - frac))


def size_events_by_player(hands):
    """-> {name: [(street_kind, bucket_or_'jam')]} with pot/stack reconstructed per hand."""
    out = defaultdict(list)
    for h in hands:
        stacks = {}
        for p in h['players']:
            st = p.get('start_stack')
            if st is None:
                st = p.get('final_stack')
            stacks[p['seat']] = float(st or 0.0)
        pot = 0.0
        for a in h['actions']:
            amt = float(a['amount'] or 0.0)
            seat = a['seat']
            if a['action'] in AGGRESSIVE and pot > 0:
                stack_before = a.get('stack_before')
                if stack_before is None:
                    stack_before = stacks.get(seat, 0.0)
                is_jam = stack_before > 0 and amt >= 0.95 * stack_before
                kind = 'preflop' if a['street'] == 'preflop' else 'postflop'
                out[a['name']].append((kind, 'jam' if is_jam else nearest_bucket(amt / pot)))
            pot += amt
            if seat in stacks:
                stacks[seat] = max(0.0, stacks[seat] - amt)
    return out


def fit(min_hands=40, exclude=('Zwonkie',)):
    """`exclude`: hero must NOT be in the OPPONENT population — with 4k+ hands hero would
    dominate a cluster's size histograms with our own model's behavior."""
    hands = load_hands()
    name_to_acct = identity_maps(hands)
    per = per_player_hand_facts(hands, name_to_acct)
    sizes = size_events_by_player(hands)

    players = {}
    for key, facts in per.items():
        if len(facts) < min_hands:
            continue
        name = facts[-1]['name']
        if name in exclude:
            continue
        s = summarize(facts)
        players[key] = {'name': name, 'stats': s, 'archetype': archetype_of(s)}

    mixture = Counter(p['archetype'] for p in players.values())
    n_players = sum(mixture.values())

    dists = {}
    for arch in ('TAG', 'LAG', 'Nit', 'Calling Station'):
        members = [p for p in players.values() if p['archetype'] == arch]
        pre, post = Counter(), Counter()
        limps, vpips, afs = [], [], []
        for p in members:
            for kind, b in sizes.get(p['name'], []):
                (pre if kind == 'preflop' else post)[b] += 1
            limps.append(p['stats']['limp'])
            vpips.append(p['stats']['vpip'])
            afs.append(p['stats']['af'] if p['stats']['af'] is not None else 5.0)
        def norm(c):
            t = sum(c.values())
            return {str(k): round(v / t, 3) for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))} if t else {}
        dists[arch] = {
            'players': len(members),
            'mixture_weight': round(len(members) / n_players, 3) if n_players else 0.0,
            'avg_vpip': round(sum(vpips) / len(vpips), 1) if vpips else None,
            'avg_af': round(sum(afs) / len(afs), 2) if afs else None,
            'avg_limp_pct': round(sum(limps) / len(limps), 1) if limps else None,
            'preflop_raise_fracs': norm(pre),
            'postflop_raise_fracs': norm(post),
            'n_size_events': sum(pre.values()) + sum(post.values()),
        }
    return {'min_hands': min_hands, 'players_fitted': n_players, 'archetypes': dists,
            'sample_players': {a: [p['name'] for p in players.values() if p['archetype'] == a][:5]
                               for a in dists}}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-hands', type=int, default=40)
    ap.add_argument('--out')
    args = ap.parse_args()
    r = fit(min_hands=args.min_hands)
    print(f"fitted {r['players_fitted']} players (>= {args.min_hands} hands)")
    for arch, d in r['archetypes'].items():
        print(f"\n{arch}: {d['players']} players (weight {d['mixture_weight']}), "
              f"VPIP {d['avg_vpip']}, AF {d['avg_af']}, limp {d['avg_limp_pct']}% "
              f"[{d['n_size_events']} size events]")
        print('  preflop :', d['preflop_raise_fracs'])
        print('  postflop:', d['postflop_raise_fracs'])
        print('  e.g.', ', '.join(r['sample_players'][arch]))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(r, f, indent=1)
        print(f"\nwrote {args.out}")

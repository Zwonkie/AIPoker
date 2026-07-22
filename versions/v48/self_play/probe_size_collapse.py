"""[V43] How often are the four "different" raise sizes the SAME NUMBER OF CHIPS?

`probe_commit_ev_decomposition.py` found that at the Nash check's node (pot 1.5bb, to_call 0.5bb)
raise_33, raise_66 and raise_pot are ALL 1.5bb -- not because of stack-capping (review T-M9) but
because the legal min-raise floor `to_call + min_increment` exceeds every pot fraction when the pot
is small. Three action heads, one physical bet, one identical target.

That matters twice over:
  - the model spends 3 of its 6 heads on one action at those nodes, and
  - anything that scores "aggression mass" by summing the aggressive heads (e.g.
    nash_pushfold_vs_chart's `agg_mass > p_fold`) counts that one action three times against
    FOLD's one.

This measures how often it actually happens in REAL training hands, by instrumenting
`_raise_size_for_fraction` during live self-play. No model needed -- heuristic hero.

Run:  .venv/Scripts/python.exe -m versions.v48.self_play.probe_size_collapse --hands 300
"""
import argparse
import os
import sys
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from versions.v48.self_play.simulator import SixMaxSimulator     # noqa: E402
from versions.v48.self_play.opponents import build_opponent_pool  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hands', type=int, default=300)
    ap.add_argument('--seed', type=int, default=3)
    args = ap.parse_args()

    import random
    random.seed(args.seed)

    sim = SixMaxSimulator(bb_size=10.0, equity_sims=100)
    try:
        sim.opponent_pool = build_opponent_pool(
            [{'style': s} for s in ('tag', 'lag', 'nit', 'fish', 'maniac')], sim, {})
    except Exception:
        pass   # heuristic fallback is fine for this measurement

    stats = Counter()
    by_street = Counter()
    orig = sim._mc_target_evs_sized

    def wrapped(hero_cards, pot, to_call, hero_stack, street_idx, active_opponents,
                board_str, raise_fracs, **kw):
        sizes = [sim._raise_size_for_fraction(f, pot, to_call, hero_stack,
                                              min_increment=kw.get('min_increment'))
                 for f in raise_fracs]
        distinct = len({round(s, 6) for s in sizes})
        stats[distinct] += 1
        # how many of the three SIZED raises (excluding all-in) are identical?
        sized_distinct = len({round(s, 6) for s in sizes[:-1]})
        by_street[(street_idx, sized_distinct)] += 1
        stats['total'] += 1
        if sized_distinct == 1:
            stats['sized_all_identical'] += 1
        if round(sizes[0], 6) == round(sizes[-1], 6):
            stats['smallest_is_allin'] += 1
        return orig(hero_cards, pot, to_call, hero_stack, street_idx, active_opponents,
                    board_str, raise_fracs, **kw)

    sim._mc_target_evs_sized = wrapped

    for i in range(args.hands):
        try:
            sim.simulate_hand()
        except Exception as e:
            print(f"hand {i} failed: {e!r}")
            break

    total = stats['total'] or 1
    print(f"\nhero decisions with a sized-EV target computed: {total}\n")
    print("DISTINCT CHIP AMOUNTS AMONG THE 4 ACTIONS {raise_33, raise_66, raise_pot, ALLIN}")
    for k in sorted(k for k in stats if isinstance(k, int)):
        print(f"  {k} distinct: {stats[k]:5}  ({stats[k]/total:5.1%})")
    print()
    print(f"  all THREE sized raises chip-identical : {stats['sized_all_identical']:5} "
          f"({stats['sized_all_identical']/total:5.1%})")
    print(f"  raise_33 is already an ALL-IN         : {stats['smallest_is_allin']:5} "
          f"({stats['smallest_is_allin']/total:5.1%})   <- review T-M9")
    print()
    print("BY STREET (0=preflop) -- distinct chip amounts among the three SIZED raises")
    streets = sorted({s for s, _d in by_street})
    for s in streets:
        row = {d: by_street[(s, d)] for _s, d in by_street if _s == s}
        tot = sum(row.values()) or 1
        label = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}.get(s, str(s))
        print(f"  {label:<8} " + "  ".join(f"{d}:{row.get(d,0):4} ({row.get(d,0)/tot:4.0%})"
                                           for d in sorted(row)))


if __name__ == '__main__':
    main()

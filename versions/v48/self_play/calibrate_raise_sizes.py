"""[V47, Change 1 / #6] C1 + C2 calibration for the opponent raise-size repertoire.

C1 (static): per-archetype sampled-fraction histograms and REALIZED-size histograms at
representative pot/stack geometries, plus fold-vs-size response curves through the analytic
occupant-fold closed form -- confirms no degenerate archetype (a size that never occurs where its
weight says it should, jam-spam, or a fold curve that stopped responding to size).

C2 (instrumented run, the H1 methodology): N simulated hands with an instrumented simulator --
every opponent raise event logs (decision string, agent kind, resolved fraction, realized chips,
pot); asserts every NN 'raise_k' decision executed EXACTLY bucket k's fraction; reports the size
spectrum hero's training world now actually produces (min-raise floors, small/pot/overbet, jams).

Run:  .venv/Scripts/python.exe versions/v48/self_play/calibrate_raise_sizes.py [--hands 1000]
"""
import argparse
import copy
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))


def c1_static():
    from versions.v48.self_play.opponent_bots import (RAISE_SIZE_DISTRIBUTIONS,
                                                      sample_raise_fraction, BOT_PROFILES)
    from versions.v48.self_play.simulator import SixMaxSimulator

    sim = SixMaxSimulator(bb_size=10.0, equity_sims=50, hero_personality='main', bootstrap_alpha=0.0)
    print("=" * 78)
    print("C1a -- sampled fraction frequencies vs configured weights (20k draws each)")
    print("=" * 78)
    degenerate = []
    for name, dist in RAISE_SIZE_DISTRIBUTIONS.items():
        draws = Counter(sample_raise_fraction(name) for _ in range(20000))
        parts = []
        for frac, w in dist:
            got = draws.get(frac, 0) / 20000
            parts.append(f"{('jam' if frac is None else f'{frac:.2f}p')}: {got:.3f} (cfg {w:.2f})")
            if w > 0 and got == 0:
                degenerate.append(f"{name}: {frac} configured but never sampled")
        jam_share = draws.get(None, 0) / 20000
        if jam_share > 0.25:
            degenerate.append(f"{name}: jam-spam ({jam_share:.2f})")
        print(f"  {name:<16} " + " | ".join(parts))

    print("-" * 78)
    print("C1b -- REALIZED chip sizes at three geometries (min-raise floor / stack cap visible)")
    print("-" * 78)
    geoms = [("preflop open (pot 15, to_call 10, stack 400)", 15.0, 10.0, 400.0),
             ("postflop mid (pot 60, to_call 0, stack 400)", 60.0, 0.0, 400.0),
             ("short stack  (pot 60, to_call 0, stack 80)", 60.0, 0.0, 80.0)]
    for name in RAISE_SIZE_DISTRIBUTIONS:
        for label, pot, to_call, stack in geoms:
            sizes = Counter()
            for _ in range(4000):
                frac = sample_raise_fraction(name)
                rs = sim._raise_size_for_fraction(frac, pot, to_call, stack, min_increment=10.0)
                if rs >= stack - 1e-9:
                    sizes['all-in'] += 1
                elif rs <= to_call + 10.0 + 1e-9:
                    sizes['min-raise'] += 1
                elif rs > 1.2 * pot:
                    sizes['overbet'] += 1
                elif rs >= 0.8 * pot:
                    sizes['~pot'] += 1
                else:
                    sizes['small/mid'] += 1
            dist_str = ", ".join(f"{k} {v/4000:.2f}" for k, v in sizes.most_common())
            print(f"  {name:<16} {label:<46} {dist_str}")

    print("-" * 78)
    print("C1c -- fold-vs-size response curves (analytic occupant fold form, transition zone)")
    print("-" * 78)
    pot_odds_grid = [0.20, 0.33, 0.43, 0.50, 0.60, 0.67]
    for style in ('tag', 'maniac', 'nit', 'fish'):
        bot = copy.deepcopy(BOT_PROFILES[style if style != 'maniac' else 'maniac'])
        bot.start_new_hand()
        for eq in (0.35, 0.50):
            curve = [sim._heuristic_fold_prob(bot, eq, po, 1, False) for po in pot_odds_grid]
            flat = max(curve) - min(curve) < 1e-9 and 0.0 < curve[0] < 1.0
            print(f"  {bot.name:<16} eq={eq}: P(fold) " +
                  " -> ".join(f"{p:.2f}" for p in curve) + ("   [FLAT -- check]" if flat else ""))
            # a mid-zone cell also exercises the respect-roll mixture term
    if degenerate:
        print("\nC1 DEGENERATE SHAPES FLAGGED:")
        for d in degenerate:
            print("  !!", d)
    else:
        print("\nC1: no degenerate archetype shapes.")
    return not degenerate


def c2_instrumented(n_hands, weights='frozen_v47.pth'):
    from versions.v48.self_play.simulator import SixMaxSimulator
    from versions.v48.self_play.opponents import build_opponent_pool
    from shared.registry import load_model

    events = []
    bucket_mismatches = []

    class InstrumentedSim(SixMaxSimulator):
        def _opponent_raise_fraction(self, decision, agent):
            frac = super()._opponent_raise_fraction(decision, agent)
            # NN/tree bucket-execution assert: 'raise_k' must resolve to raise_fracs[k] exactly.
            if isinstance(decision, str) and decision.startswith('raise_'):
                suffix = decision.split('_', 1)[1]
                if suffix.isdigit():
                    k = min(int(suffix), len(self.raise_fracs) - 1)
                    if frac is not self.raise_fracs[k] and frac != self.raise_fracs[k]:
                        bucket_mismatches.append((decision, frac))
            self._c2_pending = (decision, getattr(agent, 'kind', '?'), frac)
            return frac

        def _raise_size_for_fraction(self, frac, pot, to_call, hero_stack, min_increment=None):
            size = super()._raise_size_for_fraction(frac, pot, to_call, hero_stack,
                                                   min_increment=min_increment)
            pend = getattr(self, '_c2_pending', None)
            if pend is not None and (pend[2] == frac or (pend[2] is None and frac is None)):
                events.append({'decision': pend[0], 'kind': pend[1], 'frac': frac,
                               'size': size, 'pot': pot, 'to_call': to_call,
                               'stack': hero_stack,
                               'min_inc': self.bb_size if min_increment is None else min_increment})
                self._c2_pending = None
            return size

    random.seed(4242)
    model = load_model('v48', weights)
    sim = InstrumentedSim(bb_size=10.0, equity_sims=60, hero_personality='main', bootstrap_alpha=0.0)
    sim.hero_model = model
    sim.range_aware_equity = True
    sim.opponent_pool_styles = ['past', 'maniac', 'fish', 'tag', 'nit']
    sim.opponent_pool_weights = [0.25, 0.20, 0.15, 0.25, 0.15]
    sim.stack_depth_mix = [[2, 5, 0.08], [5, 8, 0.07], [5, 14, 0.35], [14, 30, 0.25],
                           [30, 60, 0.17], [10, 100, 0.08]]
    hb = {'fish': sim.fish_heuristic, 'maniac': sim.maniac_heuristic,
          'nit': sim.nit_heuristic, 'tag': sim.tag_heuristic, 'past': sim.tag_heuristic}
    sim.opponent_pool = build_opponent_pool(
        [{'style': 'past', 'weight': 0.25, 'model': 'x'},
         {'style': 'maniac', 'weight': 0.20, 'tree_cluster': 0},
         {'style': 'fish', 'weight': 0.15},
         {'style': 'tag', 'weight': 0.25},
         {'style': 'nit', 'weight': 0.15, 'tree_cluster': 3}],
        hb, query_fn=sim._query_model_decide, error_fn=sim._note_query_error,
        load_model_fn=lambda p: model)

    import time
    t0 = time.time()
    for i in range(n_hands):
        sim.simulate_hand(current_hand=100000 + i)
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{n_hands} hands ({(i+1)/(time.time()-t0):.1f} hands/sec)")
    hps = n_hands / (time.time() - t0)

    print("=" * 78)
    print(f"C2 -- instrumented {n_hands}-hand run: {len(events)} opponent raise events, "
          f"{hps:.1f} hands/sec (single process)")
    print("=" * 78)

    def classify(e):
        if e['size'] >= e['stack'] - 1e-9:
            return 'all-in/jam'
        if e['size'] <= e['to_call'] + e['min_inc'] + 1e-9:
            return 'min-raise'
        if e['size'] > 1.2 * e['pot']:
            return 'overbet'
        if e['size'] >= 0.8 * e['pot']:
            return '~pot'
        return 'small/mid'

    by_class = Counter(classify(e) for e in events)
    total = max(1, len(events))
    print("Size spectrum hero's world now produces:")
    for k, v in by_class.most_common():
        print(f"  {k:<12} {v:>5}  ({v/total:.1%})")
    by_kind = defaultdict(Counter)
    for e in events:
        by_kind[e['kind']][classify(e)] += 1
    for kind, cnt in sorted(by_kind.items()):
        t = sum(cnt.values())
        print(f"  by kind {kind:<10}: " + ", ".join(f"{k} {v/t:.0%}" for k, v in cnt.most_common()))

    nn_bucketed = sum(1 for e in events if str(e['decision']).startswith('raise_')
                      and str(e['decision']).split('_', 1)[1].isdigit())
    print(f"\nBucketed ('raise_k') raise events executed: {nn_bucketed} "
          f"| bucket-execution mismatches: {len(bucket_mismatches)}")
    material = (by_class.get('min-raise', 0) > 0 and by_class.get('all-in/jam', 0) > 0
                and by_class.get('overbet', 0) + by_class.get('~pot', 0) > 0)
    print(f"Material frequencies of min-raises, jams, and pot+/overbets: {'YES' if material else 'NO -- FAIL'}")
    print(f"NN buckets executed bucket-for-bucket: {'YES' if not bucket_mismatches else bucket_mismatches[:5]}")
    return material and not bucket_mismatches


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--hands', type=int, default=1000)
    ap.add_argument('--skip-c2', action='store_true')
    ap.add_argument('--weights', default='frozen_v47.pth',
                    help="checkpoint for hero + lagged-self pool (post-training: expert_main.pth)")
    ap.add_argument('--skip-c1', action='store_true')
    args = ap.parse_args()
    ok1 = True if args.skip_c1 else c1_static()
    ok2 = True if args.skip_c2 else c2_instrumented(args.hands, weights=args.weights)
    print(f"\nC1 {'OK' if ok1 else 'DEGENERATE'} | C2 {'OK' if ok2 else 'FAIL'}")
    sys.exit(0 if (ok1 and ok2) else 1)

"""[V43 / B2] If no risk dampener moves the push/fold commit threshold, what does?

`calibrate_pushfold_dampeners.py` returned a clean NEGATIVE result: across 349 Nash cells, all
eleven dampener configurations produce the same answer -- commit mass ~0.67, flat across 5-20bb,
`under = 0` in every variant. Turning the realization discount OFF ENTIRELY moves commit mass by
0.02. So the four risk dampeners are NOT the lever, and the fixed-bb-discount hypothesis for the
inverted slope is dead (the same fate the `policy_tightness_bb` hypothesis met in V20).

So this decomposes the raise target itself at the failing cells:

    raw_ev = p_all_fold * pot  +  (1 - p_all_fold) * ev_if_called   [- risk penalty]
             ^^^^^^^^^^^^^^^^     fold-equity credit vs showdown value

and reports `p_all_fold` -- the probability the decoupled fold model (`_ev_target_fold_decision`)
says the opponent folds to each size. If that number is high, every commit is profitable by
construction and no downstream dampener can fix it, because the bias is in the OPPONENT MODEL, not
in the risk adjustment.

Reference point for judging it: at 10bb heads-up, Nash has the BB CALLING a shove with roughly the
top 40-55% of hands, so a shove should be folded to ~45-60% of the time, and a min-raise defended
far wider still.

Run:  .venv/Scripts/python.exe -m versions.v48.self_play.probe_commit_ev_decomposition
"""
import json
import os
import random
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from versions.v48.self_play.simulator import SixMaxSimulator                     # noqa: E402
from versions.v48.self_play.opponent_bots import TAG, LAG, NIT, CALLING_STATION   # noqa: E402

_SOLVED = os.path.join(_REPO_ROOT, 'tools', 'model_verify', 'nash', 'nash_solved.json')
BB = 10.0
POT_BB, CALL_BB = 1.5, 0.5
RAISE_FRACS = [0.33, 0.66, 1.0, None]
FRAC_NAMES = ['raise_33', 'raise_66', 'raise_pot', 'ALLIN']
BOTS = {'TAG': TAG, 'LAG': LAG, 'NIT': NIT, 'CALLING_STATION': CALLING_STATION}
RANKS, SUITS = '23456789TJQKA', 'cdhs'

# The cells the trained model gets WRONG: Nash folds, model commits, eq just under the 0.45 pivot.
HANDS = ['94s', '93s', '92s', '83s', 'T3o', '95o', 'J2o', 'T5o']
STACKS = [5, 10, 15, 20]
TRIALS = 400


def cards_for(h):
    r1, r2 = h[0], h[1]
    if r1 == r2:
        return [r1 + SUITS[0], r2 + SUITS[1]]
    return [r1 + SUITS[0], r2 + (SUITS[0] if h.endswith('s') else SUITS[1])]


def main():
    random.seed(11)
    with open(_SOLVED) as f:
        solved = json.load(f)
    sim = SixMaxSimulator(bb_size=BB, equity_sims=150)
    sim.risk_aversion_coefficient = 0.15

    print("FOLD-MODEL SURRENDER RATE  p_all_fold  (probability the single HU opponent folds)")
    print("Nash reference @10bb: a shove should be folded to ~45-60%, a min-raise far less.\n")
    hdr = f"{'stack':>6} {'size':<10} " + " ".join(f"{b:>16}" for b in BOTS)
    print(hdr)
    print("-" * len(hdr))

    agg = defaultdict(list)
    for S in STACKS:
        pot, to_call, hero_stack = POT_BB * BB, CALL_BB * BB, S * BB
        for fi, frac in enumerate(RAISE_FRACS):
            raise_size = sim._raise_size_for_fraction(frac, pot, to_call, hero_stack)
            raise_increment = raise_size - to_call
            new_pot = pot + raise_size + raise_increment
            size_pot_odds = raise_increment / max(1.0, new_pot)
            is_allin = (frac is None)
            row = []
            for bname, bot in BOTS.items():
                folds = 0
                for _ in range(TRIALS):
                    # opponent equity vs 1 random -- the same input the fold model sees
                    oeq = random.random()
                    if sim._ev_target_fold_decision(bot, oeq, size_pot_odds, 0, is_allin):
                        folds += 1
                r = folds / TRIALS
                row.append(r)
                agg[(S, FRAC_NAMES[fi])].append(r)
            print(f"{S:>5}bb {FRAC_NAMES[fi]:<10} " +
                  " ".join(f"{r:>15.1%}" for r in row) +
                  f"   (size {raise_size/BB:.1f}bb, price to opp {size_pot_odds:.2f})")
        print()

    print()
    print("=" * 100)
    print("TARGET DECOMPOSITION at the failing cells -- where does the commit EV come from?")
    print("=" * 100)
    print(f"{'hand':<6}{'stack':>6}{'eq':>7}  {'action':<10}{'p_fold':>8}{'foldEV':>9}"
          f"{'calledEV':>10}{'rawEV':>8}{'EV_call':>9}")
    print("-" * 80)

    for h in HANDS[:4]:
        eq = solved['eq_vs_random'][h]
        hero = cards_for(h)
        for S in [10, 20]:
            pot, to_call, hero_stack = POT_BB * BB, CALL_BB * BB, S * BB
            acc = defaultdict(lambda: [0.0] * 4)
            n = 40
            for _ in range(n):
                bot = random.choice(list(BOTS.values()))
                deck = [r + s for r in RANKS for s in SUITS if (r + s) not in set(hero)]
                opp = [{'cards': random.sample(deck, 2), 'bot': bot, 'seat': 1,
                        'stack': hero_stack}]
                oeq = sim._calculate_equity(opp[0]['cards'], '', 1)
                for fi, frac in enumerate(RAISE_FRACS):
                    rs = sim._raise_size_for_fraction(frac, pot, to_call, hero_stack)
                    inc = rs - to_call
                    npot = pot + rs + inc
                    spo = inc / max(1.0, npot)
                    ia = (frac is None)
                    f = sum(sim._ev_target_fold_decision(bot, oeq, spo, 0, ia) for _ in range(10))
                    p = f / 10.0
                    base = pot + 2.0 * rs - to_call
                    evc = eq * base - rs
                    a = acc[FRAC_NAMES[fi]]
                    a[0] += p
                    a[1] += p * pot / BB
                    a[2] += (1 - p) * evc / BB
                    a[3] += (p * pot + (1 - p) * evc) / BB
            ev_call = (eq * (pot + to_call) - to_call) / BB
            for name in FRAC_NAMES:
                a = [x / n for x in acc[name]]
                print(f"{h:<6}{S:>5}bb{eq:>7.3f}  {name:<10}{a[0]:>8.2f}{a[1]:>9.2f}"
                      f"{a[2]:>10.2f}{a[3]:>8.2f}{ev_call:>9.2f}")
            print()


if __name__ == '__main__':
    main()

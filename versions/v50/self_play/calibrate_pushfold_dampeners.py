"""[V43 / B2] Which risk-dampener actually controls the heads-up push/fold commit threshold?

WHY THIS EXISTS. `nash_pushfold_vs_chart` sits at 78% (V40/V41) vs 83% (V29), and
`probe_nash_regression.py` showed the aggregate hides two separable defects:

  1. LEVEL   -- V40/V41 commit ~+0.35 more actor mass at marginal cells than V29 did.
  2. SLOPE   -- commit propensity RISES with stack depth (94s @ eq 0.432: 0.57 @5bb -> 0.68 @20bb
                in V41, 0.20 -> 0.31 in V29). Push/fold requires the opposite. Present in BOTH
                versions, so it is not a V40 regression -- it is [STACK-1], unresolved since V19.

Every failing cell sits at eq in [0.404, 0.457], straddling `POLICY_TIGHTNESS_PIVOT = 0.45`. The
realization discount is a FIXED bb amount (`POLICY_TIGHTNESS_BB * (pivot-eq)/pivot`) subtracted
from every non-fold action's value, while those values scale with pot and stack -- so its relative
bite shrinks as stacks deepen, which is exactly the inverted slope. That is a HYPOTHESIS, and
`policy_tightness_bb` is one of the LOCKED V12 validated fixes (see OFK
versioned-architecture-guardrails.md §0), so it does not get changed on a hypothesis. This script
measures it first.

WHAT IT MEASURES. For each (hand, stack) cell of the solved Nash grid it rebuilds the ACTOR TARGET
the way training does -- `_mc_target_evs_sized` (model-free counterfactual EVs, oracle equity,
decoupled fold model) -> clip -> realization discount -> fold-relative regret matching -> ALLIN
veto -> normalise -- under several dampener configurations, and scores each against Nash.

No trained network is involved anywhere. This is the ground-truth training SIGNAL, which is what
the four dampeners actually act on; the critic is fit to it and the actor is fit to the critic.

HONEST LIMITATION. Changing a dampener also changes what the critic learns, which feeds back into
the post-cutover actor target through Q. This measures the first-order effect on the target only.
It can tell you which lever moves the threshold and in which direction; it cannot predict the
trained model's final policy. That still needs the retrain -- this exists so the retrain tests ONE
justified change instead of five guesses.

Run:  .venv/Scripts/python.exe -m versions.v50.self_play.calibrate_pushfold_dampeners
      (--hands 24 --samples 6 for a fast pass; --full for the whole grid)
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from versions.v50.self_play.simulator import SixMaxSimulator                    # noqa: E402
from versions.v50.self_play.opponent_bots import TAG, LAG, NIT, CALLING_STATION  # noqa: E402

_SOLVED = os.path.join(_REPO_ROOT, 'tools', 'model_verify', 'nash', 'nash_solved.json')
_CLEAR_HI, _CLEAR_LO = 0.95, 0.05

BB = 10.0                      # chip scale; every reported number is in bb
POT_BB, CALL_BB = 1.5, 0.5     # HU SB-first geometry, identical to the nash check
PIVOT = 0.45                   # POLICY_TIGHTNESS_PIVOT
RAISE_FRACS = [0.33, 0.66, 1.0, None]

# The opponent pool the fold model is sampled against, matching training's own archetype spread.
BOTS = [TAG, LAG, NIT, CALLING_STATION]

RANKS = '23456789TJQKA'
SUITS = 'cdhs'


# --------------------------------------------------------------------------- variants
# Each variant is a first-order change to ONE dampener (plus two deliberate combinations at the
# end). `discount` selects how the realization discount scales:
#   'fixed_bb'  -- today's behaviour: a flat bb amount, independent of what the action risks.
#   'per_risk'  -- proportional to the chips THAT action actually commits. This is the form the
#                  discount's own rationale implies ("the all-in-equity counterfactual overvalues
#                  speculative entries" -- the overvaluation is proportional to money put in, not a
#                  constant), and it is the only variant that can move the SLOPE, since it is the
#                  only one whose bite grows with stack depth.
#   'off'       -- no discount at all, to bound how much of the behaviour it owns.
# `tight` is in bb for 'fixed_bb'; for 'per_risk' it is a fraction, calibrated so that an action
# committing 2bb (a typical preflop entry) at eq=0 receives the SAME penalty it does today.
VARIANTS = [
    ('base (V41)',          dict(risk=0.15, discount='fixed_bb', tight=2.0, clip=40.0, veto=0.15, allin_chips=False)),
    ('T-M9 allin-by-chips', dict(risk=0.15, discount='fixed_bb', tight=2.0, clip=40.0, veto=0.15, allin_chips=True)),
    ('risk 0.10',           dict(risk=0.10, discount='fixed_bb', tight=2.0, clip=40.0, veto=0.15, allin_chips=False)),
    ('risk 0.20',           dict(risk=0.20, discount='fixed_bb', tight=2.0, clip=40.0, veto=0.15, allin_chips=False)),
    ('risk 0.25',           dict(risk=0.25, discount='fixed_bb', tight=2.0, clip=40.0, veto=0.15, allin_chips=False)),
    ('clip 100',            dict(risk=0.15, discount='fixed_bb', tight=2.0, clip=100.0, veto=0.15, allin_chips=False)),
    ('veto off',            dict(risk=0.15, discount='fixed_bb', tight=2.0, clip=40.0, veto=0.0, allin_chips=False)),
    ('discount off',        dict(risk=0.15, discount='off',      tight=0.0, clip=40.0, veto=0.15, allin_chips=False)),
    ('discount per-risk',   dict(risk=0.15, discount='per_risk', tight=1.0, clip=40.0, veto=0.15, allin_chips=False)),
    ('per-risk + T-M9',     dict(risk=0.15, discount='per_risk', tight=1.0, clip=40.0, veto=0.15, allin_chips=True)),
    ('per-risk + T-M9 + clip100',
                            dict(risk=0.15, discount='per_risk', tight=1.0, clip=100.0, veto=0.15, allin_chips=True)),
]


def hand_to_cards(h):
    """'94s' / 'AKo' / 'TT' -> two concrete cards matching that canonical class."""
    r1, r2 = h[0], h[1]
    if r1 == r2:
        return [r1 + SUITS[0], r2 + SUITS[1]]
    suited = h.endswith('s')
    return [r1 + SUITS[0], r2 + (SUITS[0] if suited else SUITS[1])]


def deal_opponent(dead):
    deck = [r + s for r in RANKS for s in SUITS if (r + s) not in dead]
    return random.sample(deck, 2)


def realization_penalties(mode, tight, equity, committed_bb):
    """Per-action penalty vector for the non-fold actions, in bb. `committed_bb[i]` is what action
    i risks. 'fixed_bb' ignores it entirely -- which is the whole point of the comparison."""
    below = max(0.0, PIVOT - equity) / PIVOT
    if mode == 'off' or tight <= 0.0:
        return [0.0] * len(committed_bb)
    if mode == 'fixed_bb':
        return [tight * below] * len(committed_bb)
    if mode == 'per_risk':
        # tight=1.0 reproduces today's 2bb penalty for an action committing 2bb at eq=0.
        return [tight * below * c for c in committed_bb]
    raise ValueError(mode)


def target_distribution(evs_bb, committed_bb, equity, cfg):
    """The actor target training would build from these counterfactual EVs, reproducing
    train.py's chain exactly: clip -> fold=0 -> realization discount on non-fold ->
    fold-relative regret matching -> ALLIN veto -> normalise (fold-outright fallback)."""
    clip = cfg['clip']
    v = [max(-clip, min(clip, e)) for e in evs_bb]
    v[0] = 0.0
    pens = realization_penalties(cfg['discount'], cfg['tight'], equity, committed_bb)
    for i in range(1, len(v)):
        v[i] -= pens[i - 1]
    regrets = [max(0.0, x - v[0]) for x in v]
    if cfg['veto'] > 0.0 and len(v) > 2:
        best_non_allin = max(v[1:-1])
        if (best_non_allin - v[-1]) > cfg['veto']:
            regrets[-1] = 0.0
    total = sum(regrets)
    if total <= 1e-9:
        out = [0.0] * len(v)
        out[0] = 1.0
        return out
    return [r / total for r in regrets]


def cell_target(sim, hero_cards, stack_bb, equity, cfg, samples):
    """Average target distribution over `samples` opponent hands at one (hand, stack) cell."""
    pot, to_call, hero_stack = POT_BB * BB, CALL_BB * BB, stack_bb * BB
    sim.allin_by_chips = cfg['allin_chips']
    sim.risk_aversion_coefficient = cfg['risk']

    committed_bb = [to_call / BB]
    for frac in RAISE_FRACS:
        committed_bb.append(sim._raise_size_for_fraction(frac, pot, to_call, hero_stack) / BB)

    acc = None
    for _ in range(samples):
        bot = random.choice(BOTS)
        # 'stack' is required by [V41]'s _rollout_continuation_ev (asymmetric stacks). HU push/fold
        # is an effective-stack spot, so the opponent exactly covers hero -- the neutral choice.
        opp = [{'cards': deal_opponent(set(hero_cards)), 'bot': bot, 'seat': 1,
                'stack': hero_stack}]
        evs, _mx, _bl = sim._mc_target_evs_sized(
            hero_cards, pot, to_call, hero_stack, 0, opp, '', RAISE_FRACS,
            range_aware_eq=equity, last_raiser=-1)
        dist = target_distribution([e / BB for e in evs], committed_bb, equity, cfg)
        acc = dist if acc is None else [a + b for a, b in zip(acc, dist)]
    return [a / samples for a in acc]


def run_variant(job):
    """One variant over every cell. Module-level so Windows' spawn-based Pool can pickle it; each
    worker re-seeds identically, so all variants see the SAME opponent draws (paired comparison --
    the differences below are the dampener, not the deal)."""
    name, cfg, cells, samples, equity_sims, seed = job
    random.seed(seed)
    sim = SixMaxSimulator(bb_size=BB, equity_sims=equity_sims)
    agree = over = under = 0
    by_stack = defaultdict(list)
    for h, S, nash, eq in cells:
        dist = cell_target(sim, hand_to_cards(h), S, eq, cfg, samples)
        p_fold, agg = dist[0], sum(dist[2:])
        lean = 'shove' if agg > p_fold else 'fold'
        if lean == nash:
            agree += 1
        elif nash == 'fold':
            over += 1
        else:
            under += 1
        by_stack[S].append(agg)
    n = max(1, len(cells))
    means = {S: sum(v) / len(v) for S, v in by_stack.items()}
    return name, dict(agree=agree / n, over=over, under=under, means=means)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hands', type=int, default=28,
                    help="hands sampled from the decision-relevant equity band (0 = all)")
    ap.add_argument('--samples', type=int, default=6, help="opponent hands averaged per cell")
    ap.add_argument('--equity-sims', type=int, default=120)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--full', action='store_true', help="all 169 hands, all stacks")
    ap.add_argument('--workers', type=int, default=11, help="parallel variants (1 = serial)")
    args = ap.parse_args()

    random.seed(args.seed)
    with open(_SOLVED) as f:
        solved = json.load(f)

    # Unambiguous Nash cells only, same filter the check uses.
    cells = []
    for S in solved['stacks']:
        st = solved['per_stack'][str(S)]
        for h in solved['hands']:
            fq = st['sb_jam_freq'][h]
            if _CLEAR_LO < fq < _CLEAR_HI:
                continue
            cells.append((h, S, 'shove' if fq >= _CLEAR_HI else 'fold',
                          solved['eq_vs_random'][h]))

    if not args.full and args.hands:
        # Concentrate on the band every disagreement lives in -- that is where the threshold is.
        band = sorted({h for h, S, n, eq in cells if 0.38 <= eq <= 0.50})
        random.shuffle(band)
        keep = set(band[:args.hands])
        cells = [c for c in cells if c[0] in keep]

    stacks = sorted({S for _h, S, _n, _e in cells})
    print(f"cells={len(cells)}  hands={len({c[0] for c in cells})}  stacks={stacks}  "
          f"samples/cell={args.samples}")
    print()

    jobs = [(name, cfg, cells, args.samples, args.equity_sims, args.seed)
            for name, cfg in VARIANTS]
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(min(args.workers, len(jobs))) as pool:
            done = pool.map(run_variant, jobs)
    else:
        done = [run_variant(j) for j in jobs]

    header = (f"{'variant':<28}{'agree':>8}{'over':>7}{'under':>7}   "
              f"{'commit mass by stack':<46}{'slope':>8}")
    print(header)
    print("-" * len(header))

    results = {}
    lo, hi = min(stacks), max(stacks)
    for name, r in done:
        means = r['means']
        r['slope'] = means[hi] - means[lo]   # NEGATIVE is correct: commit less as stacks deepen
        trend = " ".join(f"{S}:{means[S]:.2f}" for S in stacks)
        print(f"{name:<28}{r['agree']:>7.1%}{r['over']:>7}{r['under']:>7}   "
              f"{trend:<46}{r['slope']:>+8.3f}")
        results[name] = r

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pushfold_calibration.json')
    with open(out, 'w') as f:
        json.dump({'cells': len(cells), 'samples': args.samples, 'seed': args.seed,
                   'results': results}, f, indent=2)
    print(f"\nwrote {out}")
    print()
    print("READING THIS TABLE")
    print("  agree  -- fraction of unambiguous Nash cells the TARGET leans the same way as Nash.")
    print("  over   -- Nash folds, target commits (V41's dominant error: 318 of 336).")
    print("  under  -- Nash shoves, target folds (V29's error mode).")
    print("  slope  -- mean commit mass at the deepest stack minus the shallowest. Push/fold wants")
    print("            this NEGATIVE (commit wider when short). V41's trained policy is POSITIVE.")
    print("            Only a dampener whose bite scales with the chips at risk can move it.")


if __name__ == '__main__':
    main()

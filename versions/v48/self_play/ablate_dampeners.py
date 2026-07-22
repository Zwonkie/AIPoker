"""[V43] Are the four risk dampeners still EARNING their place, or are they band-aids over root
causes that V40/V41 already fixed?

Each was introduced against a specific MEASURED pathology in the training target:

  risk_aversion_coefficient (V28, 0.10 -> V29 0.15)
        [BET-1] `allin_vs_nextbest_qgap`: ALLIN's value beat every other action by a gap that GREW
        with stack depth -- the pathological signature. Penalty = coeff * sqrt(Var[X]).
  realization discount (V12, POLICY_TIGHTNESS_BB=2.0 below eq 0.45)
        "the all-in-equity counterfactual overvalues speculative entries" -- hero entering too wide
        on hands whose equity it will never realise.
  critic-consistency veto (V29, margin 0.15bb)
        [STACK-3]: ALLIN keeping actor mass at states where the critic itself preferred something
        else.
  TARGET_CLIP_BB (V12, 40bb)
        critic stability -- damp the fat right tail of realised returns.

Since those were calibrated, V40 fixed the betting round ending on any check (so postflop nodes
exist at all) and CALL's exemption from both the variance penalty and the continuation credit, and
V41 fixed dead blinds, NN opponents playing a degraded self, symmetric stacks, the min-raise rule
and the [OPP-7] encoding. Several of those are plausible ROOT CAUSES of the very pathologies the
dampeners suppress. If a pathology is gone at the source, its dampener is now pure distortion.

WHAT THIS MEASURES, per dampener configuration, over the eq x stack grid the relevant checks use:

  allin_gap_bb   ALLIN's target EV minus the best non-fold alternative. The [BET-1] pathology is
                 this being POSITIVE and GROWING with stack depth. If it is already negative /
                 flat with the penalty OFF, the penalty is no longer doing the job it was hired for.
  p_allin        ALLIN's share of the actor target.
  commit         1 - fold mass (entry rate) -- what the realization discount exists to restrain.
  clip_bind      fraction of |target EV| that TARGET_CLIP_BB actually truncates. A clip that never
                 binds is inert; one that binds often at depth is silently reshaping the target.

HONEST LIMITS -- read before drawing conclusions:
  * This is TARGET space. The dampeners also shape what the critic learns, which feeds the actor
    post-cutover. A target that looks healthy undampened can still train badly.
  * TARGET_CLIP in particular guards a TRAINING dynamic (fat-tailed realised returns driving critic
    divergence), not a static target property. `clip_bind` tells you whether it binds, not whether
    removing it is safe.
  * The honest test of "do we still need it" is a short ablation RUN. This exists to pick WHICH one
    or two arms are worth that cost, rather than running five.

Run:  .venv/Scripts/python.exe -m versions.v48.self_play.ablate_dampeners
"""
import argparse
import os
import random
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from versions.v48.self_play.simulator import SixMaxSimulator                      # noqa: E402
from versions.v48.self_play.opponent_bots import TAG, LAG, NIT, CALLING_STATION    # noqa: E402

BB = 10.0
PIVOT = 0.45
RAISE_FRACS = [0.33, 0.66, 1.0, None]
BOTS = [TAG, LAG, NIT, CALLING_STATION]
RANKS, SUITS = '23456789TJQKA', 'cdhs'

# The grid `allin_vs_nextbest_qgap` / `deep_stack_ood_guard` actually probe: equity band x depth,
# heads-up and multiway, at a realistic single-raised-pot geometry.
EQ_GRID = [0.35, 0.45, 0.55, 0.65, 0.75, 0.90]
STACK_GRID = [5, 10, 20, 40, 60, 100]
N_OPPS = [1, 3]
POT_BB, CALL_BB = 3.0, 1.0

CONFIGS = [
    ('V41 (all four on)',   dict(risk=0.15, tight=2.0, veto=0.15, clip=40.0)),
    ('no variance penalty', dict(risk=0.00, tight=2.0, veto=0.15, clip=40.0)),
    ('no realization disc', dict(risk=0.15, tight=0.0, veto=0.15, clip=40.0)),
    ('no ALLIN veto',       dict(risk=0.15, tight=2.0, veto=0.00, clip=40.0)),
    ('clip 100 (=ceiling)', dict(risk=0.15, tight=2.0, veto=0.15, clip=100.0)),
    ('NONE (all four off)', dict(risk=0.00, tight=0.0, veto=0.00, clip=100.0)),
    # [V43 AS SHIPPED] variance penalty KEPT (its pathology is not gone at the source); realization
    # discount and ALLIN veto REMOVED; clip raised to the contract's stack ceiling.
    ('V43 (shipped)',       dict(risk=0.15, tight=0.0, veto=0.00, clip=100.0)),
    # [V43] RE-CALIBRATION of the one dampener kept, for the new clip ceiling. `risk_aversion_
    # coefficient=0.15` was calibrated (V28/V29) while targets were clipped at 40bb -- the clip was
    # itself suppressing ALLIN's deep-stack edge, so the coefficient only ever had to cover what
    # the clip left. At clip=100 the same 0.15 leaves the ALLIN-vs-next-best gap growing +6.17 bb
    # across 5->100bb where V41 held +2.38. These arms find the coefficient that restores V41's
    # damping in the new scale -- this is what "update the bb-normalised values accordingly" means
    # in practice: the clip and the penalty are not independent knobs.
    ('V43 + risk 0.20',     dict(risk=0.20, tight=0.0, veto=0.00, clip=100.0)),
    ('V43 + risk 0.25',     dict(risk=0.25, tight=0.0, veto=0.00, clip=100.0)),
    ('V43 + risk 0.30',     dict(risk=0.30, tight=0.0, veto=0.00, clip=100.0)),
    ('V43 + risk 0.35',     dict(risk=0.35, tight=0.0, veto=0.00, clip=100.0)),
]


def deal(dead, n=2):
    deck = [r + s for r in RANKS for s in SUITS if (r + s) not in dead]
    return random.sample(deck, n)


def build_target(evs_bb, equity, cfg):
    clip = cfg['clip']
    bound = sum(1 for e in evs_bb if abs(e) > clip)
    v = [max(-clip, min(clip, e)) for e in evs_bb]
    v[0] = 0.0
    if cfg['tight'] > 0.0:
        pen = cfg['tight'] * max(0.0, PIVOT - equity) / PIVOT
        for i in range(1, len(v)):
            v[i] -= pen
    regrets = [max(0.0, x - v[0]) for x in v]
    if cfg['veto'] > 0.0 and len(v) > 2:
        if (max(v[1:-1]) - v[-1]) > cfg['veto']:
            regrets[-1] = 0.0
    total = sum(regrets)
    if total <= 1e-9:
        dist = [0.0] * len(v)
        dist[0] = 1.0
    else:
        dist = [r / total for r in regrets]
    # ALLIN's edge over the best OTHER non-fold action, in the clipped/undiscounted value space.
    gap = v[-1] - max(v[1:-1])
    return dist, gap, bound


def run_config(job):
    name, cfg, samples, equity_sims, seed = job
    random.seed(seed)
    sim = SixMaxSimulator(bb_size=BB, equity_sims=equity_sims)
    sim.allin_by_chips = False
    sim.risk_aversion_coefficient = cfg['risk']

    gap_by_stack = defaultdict(list)
    commit_by_stack = defaultdict(list)
    pallin_by_stack = defaultdict(list)
    clip_hits = total_evs = 0
    commit_low_eq = []

    for S in STACK_GRID:
        pot, to_call, hero_stack = POT_BB * BB, CALL_BB * BB, S * BB
        for eq in EQ_GRID:
            for nopp in N_OPPS:
                for _ in range(samples):
                    hero = deal(set())
                    dead = set(hero)
                    opps = []
                    for k in range(nopp):
                        c = deal(dead)
                        dead |= set(c)
                        opps.append({'cards': c, 'bot': random.choice(BOTS), 'seat': k + 1,
                                     'stack': hero_stack})
                    evs, _m, _b = sim._mc_target_evs_sized(
                        hero, pot, to_call, hero_stack, 0, opps, '', RAISE_FRACS,
                        range_aware_eq=eq, last_raiser=-1)
                    evs_bb = [e / BB for e in evs]
                    dist, gap, bound = build_target(evs_bb, eq, cfg)
                    gap_by_stack[S].append(gap)
                    commit_by_stack[S].append(1.0 - dist[0])
                    pallin_by_stack[S].append(dist[-1])
                    clip_hits += bound
                    total_evs += len(evs_bb)
                    if eq <= 0.35:
                        commit_low_eq.append(1.0 - dist[0])

    mean = lambda xs: sum(xs) / max(1, len(xs))
    return name, dict(
        gap={S: mean(v) for S, v in gap_by_stack.items()},
        commit={S: mean(v) for S, v in commit_by_stack.items()},
        pallin={S: mean(v) for S, v in pallin_by_stack.items()},
        clip_bind=clip_hits / max(1, total_evs),
        commit_low_eq=mean(commit_low_eq),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=12)
    ap.add_argument('--equity-sims', type=int, default=120)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    jobs = [(n, c, args.samples, args.equity_sims, args.seed) for n, c in CONFIGS]
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(min(args.workers, len(jobs))) as pool:
            done = pool.map(run_config, jobs)
    else:
        done = [run_config(j) for j in jobs]

    print(f"grid: eq={EQ_GRID} x stack={STACK_GRID} x opps={N_OPPS}, "
          f"{args.samples} samples/cell, pot={POT_BB}bb to_call={CALL_BB}bb\n")

    print("ALLIN target EV minus best non-fold alternative, in bb  [BET-1 pathology = POSITIVE and")
    print("GROWING with stack depth. That is the pattern the variance penalty was hired to kill.]")
    hdr = f"{'config':<22}" + "".join(f"{str(S)+'bb':>9}" for S in STACK_GRID) + f"{'trend':>9}"
    print(hdr); print("-" * len(hdr))
    for name, r in done:
        g = r['gap']
        trend = g[STACK_GRID[-1]] - g[STACK_GRID[0]]
        print(f"{name:<22}" + "".join(f"{g[S]:>9.2f}" for S in STACK_GRID) + f"{trend:>+9.2f}")

    print()
    print("ENTRY RATE (1 - fold mass in the actor target)  [what the realization discount restrains]")
    print(hdr); print("-" * len(hdr))
    for name, r in done:
        c = r['commit']
        trend = c[STACK_GRID[-1]] - c[STACK_GRID[0]]
        print(f"{name:<22}" + "".join(f"{c[S]:>9.2f}" for S in STACK_GRID) + f"{trend:>+9.2f}")

    print()
    print("ALLIN share of the actor target  [what the veto + penalty jointly suppress]")
    print(hdr); print("-" * len(hdr))
    for name, r in done:
        p = r['pallin']
        trend = p[STACK_GRID[-1]] - p[STACK_GRID[0]]
        print(f"{name:<22}" + "".join(f"{p[S]:>9.2f}" for S in STACK_GRID) + f"{trend:>+9.2f}")

    print()
    print(f"{'config':<22}{'clip binds':>12}{'entry rate @ eq<=0.35':>24}")
    print("-" * 58)
    for name, r in done:
        print(f"{name:<22}{r['clip_bind']:>11.1%}{r['commit_low_eq']:>24.2f}")

    print()
    print("HOW TO READ IT")
    print("  * If 'no variance penalty' shows a FLAT or NEGATIVE allin-gap trend, the [BET-1]")
    print("    pathology is gone at the source and the penalty is now pure distortion.")
    print("  * If 'no realization disc' barely moves the entry rate at eq<=0.35, the discount is")
    print("    no longer restraining anything.")
    print("  * If 'clip binds' is ~0%, TARGET_CLIP_BB is inert and its 40-vs-100bb aliasing")
    print("    (review T-M5) costs nothing either way.")
    print("  * Target space only -- see the module docstring's limits before removing anything.")


if __name__ == '__main__':
    main()

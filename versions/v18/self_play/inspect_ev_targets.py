"""
EV-target instrument for V12-D.

Actor-sharpening did NOT reduce VPIP, which pinned the loose-collapse on the EV TARGETS
themselves: against a tight folding field the counterfactual scores RAISING weak hands as
+EV (fold equity), while realized play loses. This script quantifies that gap and tests
its mechanism.

  Part 1  per-equity-bucket mean target EV [fold/call/raise] (bb) + how often RAISE is the
          target argmax. Hypothesis: raise-EV > 0 (often argmax) even for trash.
  Part 2  weak-hand (equity<0.40) decision points where the hero RAISED: mean TARGET
          ev_raise vs mean REALIZED go-forward return (bb). The gap is the bug.
  Part 3  MECHANISM. For controlled weak spots, decompose ev_raise into p_all_fold and
          ev_raise_if_called, and compare hero equity UNCONDITIONAL vs CONDITIONAL on the
          opponent NOT folding. Unconditional >> conditional  ==  the overestimate.

Run:  .venv/Scripts/python.exe -m versions.v18.self_play.inspect_ev_targets
"""
import os
import sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from treys import Card, Deck
from versions.v18.self_play.simulator import SixMaxSimulator

EQ_BUCKETS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]


def make_sim(equity_sims=200):
    sim = SixMaxSimulator(bb_size=10.0, equity_sims=equity_sims)
    sim.opponent_pool_styles = ['nit', 'tag']
    sim.opponent_pool_weights = [0.5, 0.5]
    sim.live_players = 3
    sim.fixed_stack_bb = 100.0
    sim.disable_exploration = True
    return sim


def collect(n_hands=600, equity_sims=200):
    """Run the verify environment and pull every hero decision point's target EVs + realized
    go-forward return. Hero plays heuristically (model=None); the target EVs are computed for
    ALL three actions independent of what the hero chose, so the policy doesn't bias them."""
    sim = make_sim(equity_sims)
    rows = []
    for h in range(n_hands):
        rec = sim.simulate_hand(current_hand=h)
        if not rec or not rec.decision_points:
            continue
        bb = 10.0
        for dp in rec.decision_points:
            tevs = dp.get('target_evs') or [0.0, 0.0, 0.0]
            realized = (rec.final_hero_profit + dp['committed_before']) / bb
            rows.append({
                'equity': dp['equity'],
                'street': dp['street'],
                'action': dp['action'],           # 0 fold, 1 call, 2..K-1 raise sizes
                'ev': [e / bb for e in tevs],       # bb (length K)
                'realized': realized,               # bb, go-forward for the taken action
            })
    return rows


def part1(rows):
    print("\n" + "=" * 78)
    print("  PART 1: mean TARGET EV by equity bucket (bb)   [what the critic/actor is taught]")
    print("=" * 78)
    print(f"  {'equity':<12} {'n':>5} | {'EV_fold':>8} {'EV_call':>8} {'EV_bestRaise':>12} | {'raise=argmax':>12}")
    for lo, hi in EQ_BUCKETS:
        b = [r for r in rows if lo <= r['equity'] < hi]
        if not b:
            continue
        ef = sum(r['ev'][0] for r in b) / len(b)
        ec = sum(r['ev'][1] for r in b) / len(b)
        er = sum(max(r['ev'][2:]) for r in b) / len(b)                      # best raise size
        raise_arg = sum(1 for r in b if max(r['ev'][2:]) >= max(r['ev'][0], r['ev'][1])) / len(b)
        print(f"  {f'{lo:.1f}-{hi:.1f}':<12} {len(b):>5} | {ef:>8.2f} {ec:>8.2f} {er:>12.2f} | {raise_arg*100:>10.1f} %")


def part2(rows):
    print("\n" + "=" * 78)
    print("  PART 2: weak hands (equity<0.40) the hero RAISED -- TARGET vs REALIZED (bb)")
    print("=" * 78)
    weak_raises = [r for r in rows if r['equity'] < 0.40 and r['action'] >= 2]
    if not weak_raises:
        print("  (no weak-hand raises sampled)")
        return
    mean_target = sum(r['ev'][r['action']] for r in weak_raises) / len(weak_raises)
    mean_real = sum(r['realized'] for r in weak_raises) / len(weak_raises)
    print(f"  n weak-hand raises       : {len(weak_raises)}")
    print(f"  mean TARGET ev_raise     : {mean_target:>+7.2f} bb   (what the model is told raising is worth)")
    print(f"  mean REALIZED return     : {mean_real:>+7.2f} bb   (what actually happened)")
    print(f"  overestimate (target-real): {mean_target - mean_real:>+7.2f} bb")


def part3(equity_sims=400, n_spots=6, opp_samples=300):
    """MECHANISM: for random weak hero hands vs a Nit, decompose ev_raise and compare hero
    equity unconditional vs conditional on the Nit NOT folding."""
    print("\n" + "=" * 78)
    print("  PART 3: mechanism -- unconditional vs conditional equity when RAISING vs a Nit")
    print("=" * 78)
    sim = make_sim(equity_sims)
    nit = sim.nit_heuristic
    bb = 10.0
    pot = 3.0 * bb      # ~limped/blind pot
    to_call = 0.0
    print(f"  {'hero':<7} {'uncond_eq':>9} {'p_fold':>7} {'cond_eq':>8} | {'evR(uncond)':>11} {'evR(cond)':>10}")
    tried = 0
    shown = 0
    while shown < n_spots and tried < n_spots * 40:
        tried += 1
        deck = Deck()
        hero = [Card.int_to_str(c) for c in deck.draw(2)]
        uncond_eq = sim._calculate_equity(hero, [], 1)
        if uncond_eq >= 0.40:      # want weak hero hands
            continue
        # p_fold and conditional equity via sampled opponent hands
        nit.start_new_hand()
        raise_size = max(pot * 0.75, to_call + bb)
        new_pot = pot + raise_size + (raise_size - to_call)
        pot_odds = (raise_size - to_call) / max(1.0, new_pot)
        used = set(hero)
        fold_n = 0
        cont_eqs = []
        samp = 0
        while samp < opp_samples:
            d2 = Deck()
            opp = [Card.int_to_str(c) for c in d2.draw(2)]
            if opp[0] in used or opp[1] in used or opp[0] == opp[1]:
                continue
            samp += 1
            opp_eq = sim._calculate_equity(opp, [], 1)
            decision = nit.decide_preflop(opp_eq, pot_odds)
            if decision == 'fold':
                fold_n += 1
            else:
                # hero equity vs THIS continuing hand
                cont_eqs.append(sim._calculate_equity(hero, [], 1, specific_opponents=[opp]))
        p_fold = fold_n / opp_samples
        cond_eq = sum(cont_eqs) / len(cont_eqs) if cont_eqs else 0.0
        evR_uncond = uncond_eq * (pot + 2.0 * raise_size - to_call) - raise_size
        evR_cond = cond_eq * (pot + 2.0 * raise_size - to_call) - raise_size
        print(f"  {''.join(hero):<7} {uncond_eq:>9.3f} {p_fold:>7.2f} {cond_eq:>8.3f} | "
              f"{evR_uncond/bb:>+10.2f} {evR_cond/bb:>+9.2f}")
        shown += 1
    print("\n  ev_raise = p_fold*pot + (1-p_fold)*ev_raise_if_called.")
    print("  The sim uses evR(uncond); the TRUE value uses evR(cond). If cond_eq << uncond_eq,")
    print("  the target over-credits raising weak hands with showdown value it won't realize.")


def main():
    print("Collecting hero decision points from the verify environment...")
    rows = collect()
    print(f"Collected {len(rows)} decision points.")
    nonzero = sum(1 for r in rows if any(abs(x) > 1e-9 for x in r['ev']))
    print(f"  ({nonzero} have non-zero target EVs)")
    part1(rows)
    part2(rows)
    part3()


if __name__ == '__main__':
    main()

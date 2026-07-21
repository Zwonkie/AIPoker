"""[V25] calibration probe: BEFORE committing to a full retrain, sanity-check
`_rollout_continuation_ev` in isolation against representative hand scenarios. Same rigor as
`calibrate_bet1.py` -- direct calls into the mechanism under test, no simulator/training loop.

What this checks:
  1. River / all-in: the correction must be EXACTLY zero (no next street to roll out either way).
     If this isn't zero, something is wrong with the gating, not the mechanism.
  2. Shallow stack: as hero_stack_remaining -> 0, the correction should shrink toward zero (a
     smaller raise and an all-in should converge in value once there's no chip room left for
     another street of betting -- this is the intuition the whole fix rests on).
  3. Deep-stack flush draw on the flop: hero is a middling-equity DRAW (most of its value is in
     future streets, not right now) with a full stack behind. This is the single most direct test
     of the hypothesis -- if the correction doesn't show a meaningful POSITIVE delta here, the
     mechanism isn't doing what it's supposed to.
  4. Deep-stack strong made hand on the turn: hero is already well ahead, one street left. Should
     show a smaller but still positive delta (a "keep value-betting" bonus) since hero can extract
     more when already the favorite.
  5. Magnitude sanity: the correction should be a modest ADDITION relative to the base pot, not a
     multiple of it (if this fires, HERO_CBET_POT_FRACTION/trials need retuning before training).

Run: .venv/Scripts/python.exe -m versions.v43.self_play.calibrate_multistreet_ev
"""
import random

from versions.v43.self_play.simulator import SixMaxSimulator, _poker_evaluator
from versions.v43.self_play.opponent_bots import TAG, LAG, NIT, CALLING_STATION

PERSONALITIES = {'TAG': TAG, 'LAG': LAG, 'NIT': NIT, 'CALLING_STATION': CALLING_STATION}
N_REPEATS = 40   # outer repeats of the whole (already-averaged-over-4-trials) call, for a stable mean/spread
TRUE_EQ_SIMS = 3000   # high-precision oracle equity for the scenario's ACTUAL cards -- this MUST
                       # match what the real caller would compute (_calculate_equity's oracle_equity),
                       # not an eyeballed round number, or the delta just measures "guessed wrong",
                       # not the mechanism.


def compute_true_equity(hero_cards, board_str, opp_hands_list):
    if not opp_hands_list:
        return 1.0
    eq, _ = _poker_evaluator.calculate_equity(
        board_str, hero_cards, num_opponents=len(opp_hands_list),
        num_simulations=TRUE_EQ_SIMS, specific_opponents=opp_hands_list)
    return eq


def run_case(sim, label, hero_cards, board_str, opp_hands_list, base_pot, hero_stack_remaining,
             raise_size, street_idx, personality='TAG'):
    true_equity = compute_true_equity(hero_cards, board_str, opp_hands_list)
    bot = PERSONALITIES[personality]
    active_opponents = [{'bot': bot, 'stack': base_pot * 2.0, 'cards': h, 'seat': i + 1}
                         for i, h in enumerate(opp_hands_list)]
    deltas = []
    for _ in range(N_REPEATS):
        bot.start_new_hand()
        d = sim._rollout_continuation_ev(hero_cards, board_str, active_opponents,
                                          opp_hands_list, base_pot, hero_stack_remaining,
                                          raise_size, true_equity, street_idx)
        deltas.append(d)
    mean_d = sum(deltas) / len(deltas)
    spread = (max(deltas) - min(deltas))
    print(f"  {label:<52} true_eq={true_equity:.3f}  base_pot={base_pot:7.1f}  "
          f"mean_delta={mean_d:+8.2f}  (spread {spread:6.2f})  "
          f"as %% of base_pot={100.0*mean_d/max(1.0, base_pot):+6.2f}%%")
    return mean_d


def main():
    sim = SixMaxSimulator()

    print("=" * 100)
    print("1. RIVER / ALL-IN CONTROL -- must be exactly 0.0 (no next street either way)")
    print("=" * 100)
    d_river = sim._rollout_continuation_ev(
        ['As', 'Ks'], ['Qs', 'Js', '2c', '7d', '3h'], [{'bot': TAG, 'stack': 500.0, 'cards': ['2d', '3d'], 'seat': 1}],
        [['2d', '3d']], base_pot=200.0, hero_stack_remaining=300.0, raise_size=100.0,
        true_equity=0.85, street_idx=3)
    print(f"  river (street_idx=3): delta = {d_river}")
    assert d_river == 0.0, "river must be a no-op"
    print("  (all-in is never routed through this function at all -- gated by `is_allin` at the call site)")

    print()
    print("=" * 100)
    print("2. SHALLOW STACK -- correction should shrink toward 0 as hero_stack_remaining -> 0")
    print("=" * 100)
    hero_cards = ['As', 'Ks']
    board_flop = ['Qs', '7d', '2c']
    opp_cards = [['9h', '9c']]
    for stack_left in (400.0, 100.0, 20.0, 2.0, 0.0):
        run_case(sim, f"stack_remaining={stack_left}", hero_cards, board_flop, opp_cards,
                  base_pot=150.0, hero_stack_remaining=stack_left, raise_size=100.0,
                  street_idx=1)

    print()
    print("=" * 100)
    print("3. DEEP-STACK FLOP FLUSH DRAW -- the core hypothesis test")
    print("=" * 100)
    print("   Hero: As Ks on a two-tone flop (nut flush draw + overcards, ~35-40% true equity vs a")
    print("   made pair) with a full stack behind. If implied odds are real, mean_delta should be")
    print("   MEANINGFULLY POSITIVE here -- most trials miss and check for ~0, but the trials that")
    print("   hit the flush let hero bet big while still ahead, and that should show up in the mean.")
    hero_draw = ['As', 'Ks']
    board_draw = ['Qs', '7s', '2c']       # two spades down, hero holds As Ks -> nut flush draw
    opp_made_hand = [['Qh', 'Qd']]         # opponent has top set/overpair-ish made hand
    for personality in ('TAG', 'LAG', 'NIT', 'CALLING_STATION'):
        run_case(sim, f"flush draw vs {personality}, deep (400bb-equiv stack)", hero_draw, board_draw,
                  opp_made_hand, base_pot=150.0, hero_stack_remaining=400.0, raise_size=100.0,
                  street_idx=1, personality=personality)

    print()
    print("=" * 100)
    print("4. DEEP-STACK TURN, HERO ALREADY WELL AHEAD -- smaller, still-positive 'keep betting' delta")
    print("=" * 100)
    hero_strong = ['As', 'Ac']
    board_turn = ['Ad', '7s', '2c', '9h']   # hero flopped/turned a big hand (trip aces)
    opp_decent = [['Kh', 'Kd']]
    for personality in ('TAG', 'LAG', 'NIT', 'CALLING_STATION'):
        run_case(sim, f"trips vs {personality}, deep, turn", hero_strong, board_turn, opp_decent,
                  base_pot=300.0, hero_stack_remaining=400.0, raise_size=150.0,
                  street_idx=2, personality=personality)

    print()
    print("=" * 100)
    print("5. PREFLOP -> FLOP (3 cards dealt at once) -- sanity check the wider deal doesn't blow up")
    print("=" * 100)
    hero_pp = ['Ts', 'Tc']
    opp_pp = [['Ah', 'Kd']]
    run_case(sim, "TT vs AKo preflop shove-sized raise (not all-in), deep", hero_pp, [], opp_pp,
              base_pot=60.0, hero_stack_remaining=380.0, raise_size=40.0,
              street_idx=0)

    print()
    print("Done. Check: (1) exactly 0.0, (2) monotonically shrinking toward 0, (3) clearly positive")
    print("and bigger than (4)'s made-hand deltas, (5) no crash and a sane (not huge) magnitude.")


if __name__ == '__main__':
    main()

"""Regression test for the front/after equity split -- the bug that folded QQ at
history/Turbo_1171580052/flagged/turn_2_20260721_201440.

`front_colors` tells compute_range_aware_equity "these opponents are GUARANTEED to showdown, do
not roll them for a fold". The live classifier used to award that status on seat position alone,
so in an unopened pot every seat sitting before hero -- all of whom had folded -- was treated as a
locked-in showdown opponent. QQ read 0.38 instead of 0.64 and the bot clicked FOLD.

Run: python versions/v42_liveFixes/verify_front_colors.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.table_state import TableState
import PHPHelp

SEATS = ['hero', 'seat_1', 'seat_2', 'seat_3', 'seat_4', 'seat_5']


class _Stub:
    """Minimal stand-in for PHPHelpApp -- the classifier only touches `self.table_state`."""

    def __init__(self, table_state):
        self.table_state = table_state

    classify = PHPHelp.PHPHelpApp._classify_opponents_by_action_order


def build(dealer_idx, start_stacks, current_stacks, board=(), big_blind=20.0, raised=()):
    ts = TableState()
    ts.reset(big_blind=big_blind)
    ts.dealer_idx = dealer_idx
    ts.community_cards = list(board)
    ts.hero_stack = current_stacks['Hero']
    ts.opponents = {
        k: {'stack': current_stacks[k], 'is_active': True, 'vpip_color': 'Yellow'}
        for k in current_stacks if k != 'Hero'
    }
    ts.hand_start_stacks = dict(start_stacks)
    ts.raised_this_hand = {k: True for k in raised}
    state = {'community_cards': list(board), 'dealer_idx': dealer_idx}
    by_seat = {k: v for k, v in ts.opponents.items()}
    return _Stub(ts).classify(state, by_seat)


def check(label, got, expected_front_n, expected_after_n):
    front, after = got
    nf = len(front or [])
    na = len(after or [])
    ok = nf == expected_front_n and na == expected_after_n
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}\n"
          f"         front={nf} (expected {expected_front_n})  "
          f"after={na} (expected {expected_after_n})")
    return ok


def main():
    results = []
    full = {k: 1500.0 for k in ['Hero'] + SEATS[1:]}

    # --- The flagged hand -------------------------------------------------------------------
    # Button on seat_2, so seat_3 = SB (posts 10), seat_4 = BB (posts 20). Hero acts UTG+2 with
    # seat_5 and seat_1 in front of it, both folded without putting a chip in. Pot is 30 = the
    # blinds and nothing else, exactly as the screenshot's "Pulje: 30" reads.
    current = dict(full)
    current['seat_3'] = 1490.0   # SB posted
    current['seat_4'] = 1480.0   # BB posted
    print("The flagged hand -- unopened pot, folded seats sitting before hero:")
    results.append(check(
        "nobody is committed voluntarily -> front must be empty",
        build(dealer_idx=2, start_stacks=full, current_stacks=current),
        expected_front_n=0, expected_after_n=5))

    # --- A blind that has only posted is still foldable --------------------------------------
    # Same table, hero on the button so BOTH blinds sit 'after' it postflop but 'before' it
    # preflop under the old positional read. A posted blind is involuntary money.
    current = dict(full)
    current['seat_1'] = 1490.0
    current['seat_2'] = 1480.0
    print("\nPosted blinds only:")
    results.append(check(
        "a posted blind is not 'guaranteed in' -> front stays empty",
        build(dealer_idx=0, start_stacks=full, current_stacks=current),
        expected_front_n=0, expected_after_n=5))

    # --- A real raiser MUST be counted ---------------------------------------------------------
    # seat_5 opens to 60 from a 1500 stack. It is genuinely in the pot and must keep its
    # no-fold-roll status, or the fix would have swung the equity error the other way.
    current = dict(full)
    current['seat_3'] = 1490.0
    current['seat_4'] = 1480.0
    current['seat_5'] = 1440.0
    print("\nAn opponent has actually raised:")
    results.append(check(
        "the raiser is committed -> front counts exactly it",
        build(dealer_idx=2, start_stacks=full, current_stacks=current, raised=('seat_5',)),
        expected_front_n=1, expected_after_n=4))

    # --- A raiser BEHIND hero still counts --------------------------------------------------------
    # dealer_idx=2 makes seat_4 the big blind, which acts LAST preflop -- so it sits positionally
    # 'after' hero and the old code could never have counted it, even 3-betting to 100. Committed
    # chips see it; seat order cannot.
    current = dict(full)
    current['seat_3'] = 1490.0
    current['seat_4'] = 1400.0   # BB 3-bet to 100
    print("\nThe big blind 3-bet from behind hero:")
    results.append(check(
        "committed > blind -> counted despite acting after hero",
        build(dealer_idx=2, start_stacks=full, current_stacks=current, raised=('seat_4',)),
        expected_front_n=1, expected_after_n=4))

    # --- Postflop keeps the positional read ------------------------------------------------------
    # `committed` spans earlier streets postflop, so it cannot answer "has this seat acted on THIS
    # street" -- position does that, and it is reliable postflop. Dealer on seat_3 puts seat_4 and
    # seat_5 in front of hero.
    current = {k: 1400.0 for k in full}
    print("\nPostflop, everyone contested the pot:")
    results.append(check(
        "positional read preserved when chips really are in",
        build(dealer_idx=3, start_stacks=full, current_stacks=current, board=('Ah', '7d', '2c')),
        expected_front_n=2, expected_after_n=3))

    # --- Postflop, a folded seat still marked active ------------------------------------------
    # The failure mode from the flagged hand, one street later: vision never noticed seat_4 fold,
    # so it is still 'active' and sits before hero. Zero chips in the hand is what disqualifies it.
    current = {k: 1400.0 for k in full}
    current['seat_4'] = 1500.0   # never put a chip in
    print("\nPostflop with a phantom active seat:")
    results.append(check(
        "a seat with no chips in is never 'guaranteed to showdown'",
        build(dealer_idx=3, start_stacks=full, current_stacks=current, board=('Ah', '7d', '2c')),
        expected_front_n=1, expected_after_n=4))

    # --- No button, no order --------------------------------------------------------------------
    print("\nDealer button not detected:")
    front, after = build(dealer_idx=-1, start_stacks=full, current_stacks=dict(full))
    ok = front is None and after is None
    print(f"  [{'PASS' if ok else 'FAIL'}] falls back to the flat legacy equity call "
          f"(got {front!r}, {after!r})")
    results.append(ok)

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

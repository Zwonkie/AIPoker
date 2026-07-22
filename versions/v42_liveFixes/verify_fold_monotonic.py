"""Regression test: a folded seat must stay folded for the rest of the hand.

`is_active` sets the field size the model reasons about, and feeds BOTH the equity computation and
`equity_edge` (= equity x (num_active + 1)). Before this fix, `TableState.update()` assigned
`is_active = raw_active` unconditionally on every frame, so ONE bright read (deal animation, timer
overlay, a chip graphic over the name plate) put a folded seat back in the hand.

Measured cost of a single phantom seat, AKs preflop, training-computed equity:
    4 opponents -> CALL  (fold 0.12)
    5 opponents -> FOLD  (fold 0.91)

Run: python versions/v42_liveFixes/verify_fold_monotonic.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.table_state import TableState

SEATS = ['seat_1', 'seat_2', 'seat_3', 'seat_4', 'seat_5']


def frame(active_seats, stacks=None):
    stacks = stacks or {}
    return {'opponents': {
        s: {'name': f'p{s}', 'stack': stacks.get(s, 1500 if s in active_seats else 0),
            'is_active': s in active_seats,
            'state': 'Active' if s in active_seats else 'Folded',
            'vpip_color': 'Yellow' if s in active_seats else None,
            'agg_color': 'Green' if s in active_seats else None}
        for s in SEATS}}


def active_count(ts):
    return sum(1 for o in ts.opponents.values() if o.get('is_active'))


def check(label, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got {got}, want {want}")
    return ok


def main():
    results = []

    print("A folded seat survives a single bright frame:")
    ts = TableState()
    ts.reset(big_blind=20)
    ts.update(frame(SEATS))                       # everyone dealt in
    results.append(check("all 5 dealt in", active_count(ts), 5))
    ts.update(frame(['seat_2', 'seat_3', 'seat_4', 'seat_5']))   # seat_1 folds
    results.append(check("seat_1 folds", active_count(ts), 4))
    ts.update(frame(SEATS))                       # flicker: seat_1 reads bright again
    results.append(check("flicker must NOT resurrect seat_1", active_count(ts), 4))
    results.append(check("seat_1 state stays Folded",
                         ts.opponents['seat_1'].get('state'), 'Folded'))

    print("\nFolds accumulate as the hand narrows:")
    ts = TableState()
    ts.reset(big_blind=20)
    ts.update(frame(SEATS))
    for i, remaining in enumerate([['seat_2', 'seat_3', 'seat_4', 'seat_5'],
                                   ['seat_3', 'seat_4', 'seat_5'],
                                   ['seat_4', 'seat_5'],
                                   ['seat_5']], start=1):
        ts.update(frame(remaining))
        results.append(check(f"after {i} fold(s)", active_count(ts), 5 - i))
    ts.update(frame(SEATS))
    results.append(check("a fully bright frame cannot undo any of them", active_count(ts), 1))

    print("\nA seat already folded on the FIRST frame of the hand:")
    ts = TableState()
    ts.reset(big_blind=20)
    ts.update(frame(['seat_2', 'seat_3', 'seat_4', 'seat_5']))
    results.append(check("never counted as in", active_count(ts), 4))
    ts.update(frame(SEATS))
    results.append(check("and cannot join later", active_count(ts), 4))

    print("\nThe latch is per-hand, not permanent:")
    ts = TableState()
    ts.reset(big_blind=20)
    ts.update(frame(SEATS))
    ts.update(frame(['seat_2', 'seat_3']))
    results.append(check("3 folded this hand", active_count(ts), 2))
    ts.reset(big_blind=20)
    ts.update(frame(SEATS))
    results.append(check("next hand deals everyone back in", active_count(ts), 5))

    print("\nAll-in players stay in the hand (stack 0 is not a fold):")
    ts = TableState()
    ts.reset(big_blind=20)
    ts.update(frame(SEATS))
    f = frame(SEATS)
    f['opponents']['seat_2'].update({'stack': 0, 'state': 'All-In'})
    ts.update(f)
    results.append(check("all-in seat still active", active_count(ts), 5))
    results.append(check("all-in state preserved",
                         ts.opponents['seat_2'].get('state'), 'All-In'))

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

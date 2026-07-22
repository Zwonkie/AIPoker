"""[V45_liveHandover] Parity verification: the observation path must be BYTE-IDENTICAL to the
pre-refactor inline path. This refactor moves code behind a boundary; it must not move behavior.

Three layers, mirroring where drift could hide:

  1. BoardState parity  -- TableState.to_board_state(...) (+ the caller-side hand_strength /
     effective_field threading it replaces) vs observation_to_board_state(to_observation(...)),
     field-for-field including every SeatState, across fixtures that exercise short-handed slot
     remapping, unknown HUD colors, folded seats, all-in labels, and blind exemptions.
  2. Classifier parity  -- classify_front_after(obs) is additionally covered by the EXISTING
     versions/v42_liveFixes/verify_front_colors.py, whose 7 cases now run through the PHPHelp
     delegate into the shared implementation. Run that script alongside this one.
  3. End-to-end decide() parity -- two fresh PokerDecisionEngines, live-feature providers stubbed
     identically (fixed equity / hand_strength, no RNG), same seed: the OLD call-site recipe
     (manual BoardState -> make_decision) and the NEW one (to_observation -> decide) must return
     the same action, reason, bet size, and model-output dict, on the ACTIVE model's real weights,
     across a two-street sequence (proves the hand-history buffers stay aligned too).

Run: python versions/v45_liveHandover/verify_handover.py
"""
import os
import random
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.table_state import TableState
from core.live_adapter import observation_to_board_state, classify_front_after

RESULTS = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"\n         {detail}" if detail and not ok else ""))
    RESULTS.append(ok)
    return ok


# ---------------------------------------------------------------- fixtures
def make_ts(dealer_idx=2, big_blind=20.0, board=(), short_handed=False):
    """A deliberately awkward table: posted blinds, a raiser, a folded seat, an all-in seat,
    an unread HUD badge, and (optionally) empty seats to exercise the short-handed slot remap."""
    ts = TableState()
    ts.reset(big_blind=big_blind)
    ts.dealer_idx = dealer_idx
    ts._dealer_seen = True
    ts.community_cards = list(board)
    ts.hero_cards = ['Qs', 'Qd']
    ts.pot_size = 130.0
    ts.hero_stack = 1480.0

    ts.opponents = {
        'seat_1': {'name': 'A', 'stack': 1440.0, 'is_active': True, 'state': 'Active',
                   'vpip_color': 'Red', 'agg_color': 'Yellow'},
        'seat_2': {'name': 'B', 'stack': 1500.0, 'is_active': False, 'state': 'Folded',
                   'vpip_color': 'Blue', 'agg_color': 'Blue'},
        'seat_3': {'name': 'C', 'stack': 1490.0, 'is_active': True, 'state': 'Active',
                   'vpip_color': None, 'agg_color': None},          # unread HUD badge
        'seat_4': {'name': 'D', 'stack': 0.0, 'is_active': True, 'state': 'All-In',
                   'vpip_color': 'Green', 'agg_color': 'Red'},
    }
    if not short_handed:
        ts.opponents['seat_5'] = {'name': 'E', 'stack': 1500.0, 'is_active': True,
                                  'state': 'Active', 'vpip_color': 'Yellow', 'agg_color': 'Green'}

    ts.hand_start_stacks = {'Hero': 1500.0, 'seat_1': 1500.0, 'seat_2': 1500.0,
                            'seat_3': 1500.0, 'seat_4': 1500.0, 'seat_5': 1500.0}
    ts.raised_this_hand = {'seat_1': True}
    ts.raised_this_street = {'seat_1': True}
    ts.raise_count = 2
    ts.active_buttons = ['FOLD', 'CALL', 'RAISE']
    ts.action_history = ['c', 'r']
    ts._recompute_positions()
    return ts


def boardstate_old_path(ts, call_amount, equity, hand_strength, effective_field, big_blind):
    """The exact pre-refactor recipe: to_board_state + the two caller-side field assignments."""
    bs = ts.to_board_state(call_amount=call_amount, equity=equity, big_blind=big_blind)
    bs.hand_strength = hand_strength
    bs.effective_field = effective_field
    return bs


# ---------------------------------------------------------------- 1. BoardState parity
def test_boardstate_parity():
    print("1. BoardState parity (old inline recipe vs observation_to_board_state):")
    scenarios = [
        ("full table, preflop, raiser+blinds+all-in+unread HUD",
         make_ts(dealer_idx=2), 60.0),
        ("short-handed (seat_5 empty) -- slot remap engaged",
         make_ts(dealer_idx=3, short_handed=True), 40.0),
        ("postflop board, button on hero",
         make_ts(dealer_idx=0, board=('Ah', '7d', '2c')), 0.0),
    ]
    for label, ts, ca in scenarios:
        old = boardstate_old_path(ts, ca, equity=0.42, hand_strength=0.61,
                                  effective_field=2.5, big_blind=20.0)
        obs = ts.to_observation(call_amount=ca, call_amount_known=True,
                                check_call_available=True, bet_raise_available=True,
                                big_blind=20.0)
        new = observation_to_board_state(obs, equity=0.42, hand_strength=0.61,
                                         effective_field=2.5)
        same = asdict(old) == asdict(new)
        check(label, same,
              detail="" if same else f"old={asdict(old)}\n         new={asdict(new)}")


# ---------------------------------------------------------------- 2. Observation round-trip
def test_observation_roundtrip():
    print("\n2. LiveObservation JSON round-trip (turns.jsonl replayability):")
    from core.live_observation import LiveObservation
    import json
    ts = make_ts(dealer_idx=2)
    obs = ts.to_observation(call_amount=60.0, call_amount_known=False,
                            check_call_available=True, bet_raise_available=False,
                            big_blind=20.0, ts_epoch=1234.5)
    wire = json.loads(json.dumps(obs.to_json_dict()))
    back = LiveObservation.from_json_dict(wire)
    check("to_json_dict -> JSON -> from_json_dict is lossless", back == obs,
          detail=f"back={back}\n         obs={obs}")
    front_a, after_a = classify_front_after(obs)
    front_b, after_b = classify_front_after(back)
    check("classifier agrees on original vs round-tripped observation",
          (front_a, after_a) == (front_b, after_b))


# ---------------------------------------------------------------- 3. End-to-end decide() parity
def _stub_providers(with_eff=False):
    def stub_equity(hero, board, colors, sims=250, front_colors=None):
        return 0.62
    def stub_hand_strength(c1, c2):
        return 0.71
    eff = (lambda front, after: 2.4) if with_eff else None
    return {'equity_fn': stub_equity, 'hand_strength_fn': stub_hand_strength,
            'effective_field_fn': eff, 'use_front_colors': True,
            'source': 'stub', 'error': None}


def test_decide_parity():
    print("\n3. End-to-end decide() parity on the ACTIVE model's real weights:")
    from core.decision import PokerDecisionEngine

    class _StubEvaluator:
        """The vs-random fallback must never fire in this test -- fail loudly if it does."""
        def calculate_equity(self, *a, **k):
            raise AssertionError("vs-random fallback fired despite a stubbed range-aware equity_fn")

    engine_old = PokerDecisionEngine()
    engine_new = PokerDecisionEngine()
    providers = _stub_providers(with_eff=False)
    engine_old.live_feature_providers = lambda name=None: dict(providers)
    engine_new.live_feature_providers = lambda name=None: dict(providers)

    # Two consecutive decision points of the same hand: preflop facing a raise, then the flop --
    # exercises the hand-history/action buffers through both entry points.
    streets = [
        ("preflop", make_ts(dealer_idx=2), 60.0, True),
        ("flop", make_ts(dealer_idx=2, board=('Ah', '7d', '2c')), 0.0, True),
    ]

    for label, ts, ca, known in streets:
        obs = ts.to_observation(call_amount=ca, call_amount_known=known,
                                check_call_available=True, bet_raise_available=True,
                                big_blind=20.0)

        # OLD call-site recipe, replicated exactly (equity/hand_strength as the stub returns them;
        # hand_strength is postflop a real MC call in the old site ONLY via the evaluator -- the
        # adapter keeps that same branch, so parity here uses the preflop lookup on both sides
        # preflop and the SAME evaluator stub identity postflop, where the old site and the
        # adapter share code shape; effective_field absent -> 0.0 on both sides).
        hand_strength = 0.71 if len(ts.community_cards) == 0 else 0.5
        if len(ts.community_cards) > 0:
            # postflop old site would call evaluator MC; give both sides the same fixed value by
            # stubbing through equity_meta instead: the adapter's evaluator raises if touched, so
            # drop hand_strength_fn for the postflop step on BOTH engines.
            providers_pf = dict(providers, hand_strength_fn=None)
            engine_old.live_feature_providers = lambda name=None: dict(providers_pf)
            engine_new.live_feature_providers = lambda name=None: dict(providers_pf)
            hand_strength = 0.5

        bs_old = boardstate_old_path(ts, ca, equity=0.62, hand_strength=hand_strength,
                                     effective_field=0.0, big_blind=20.0)
        random.seed(4242)
        res_old = engine_old.make_decision(
            bs_old, bet_raise_available=True, check_call_available=True,
            call_amount_known=known)

        random.seed(4242)
        dec_new = engine_new.decide(obs, evaluator=_StubEvaluator(), fallback_sims=100,
                                    fallback_num_opponents=4)

        same_action = res_old[0] == dec_new.action and abs(res_old[2] - dec_new.bet_size) < 1e-9
        same_reason = res_old[1] == dec_new.reason
        # repr-compare: the model-output dict can hold numpy scalars/arrays where `==` is
        # elementwise; identical computations produce identical reprs.
        same_ev = (repr(res_old[3]) == repr(dec_new.ev_dict)) if len(res_old) > 3 \
            else dec_new.ev_dict is None
        check(f"{label}: action/size identical ({res_old[0]} vs {dec_new.action})", same_action)
        check(f"{label}: reason string identical", same_reason,
              detail=f"old={res_old[1]!r}\n         new={dec_new.reason!r}")
        check(f"{label}: full model-output dict identical", same_ev)
        check(f"{label}: adapter equity/meta wired ({dec_new.equity})",
              dec_new.equity == 0.62 and dec_new.equity_meta.get('method') == 'range-aware')

    # Buffer alignment after the sequence: both engines saw the same two states.
    check("hand-history buffers identical across both entry points",
          len(engine_old.hand_history_buffer) == len(engine_new.hand_history_buffer)
          and engine_old.hero_action_buffer == engine_new.hero_action_buffer)


def main():
    test_boardstate_parity()
    test_observation_roundtrip()
    test_decide_parity()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""V42_liveFixes verification: positions, seat slots, call-amount sentinel, CALL masking."""
import sys, os
sys.path.insert(0, r'c:\REPO\Antigravity\AIPoker')
from core.table_state import TableState
from core.decision import PokerDecisionEngine
from versions.v41.core.contract import ContractV12

FAIL = []
def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got={got} want={want}")
    if not ok:
        FAIL.append(name)


def build(seats, dealer_idx, hero_stack=1000):
    ts = TableState()
    ts.reset(big_blind=20.0)
    ts.hero_stack = hero_stack
    ts.hero_cards = ['Ah', 'Kd']
    ts.opponents = {f"seat_{i}": {'name': f'p{i}', 'stack': 1000, 'is_active': True,
                                  'state': 'Active', 'vpip_color': 'Red', 'agg_color': 'Red'}
                    for i in seats}
    ts.dealer_idx = dealer_idx
    ts._recompute_positions()
    return ts


print("=== 1. FULL 6-HANDED: must be byte-identical to the legacy formula ===")
for d in range(6):
    ts = build([1, 2, 3, 4, 5], d)
    check(f"6-handed dealer={d} hero_position", ts.hero_position, (0 - d) % 6)
    bs = ts.to_board_state(call_amount=0.0, equity=0.5, big_blind=20.0)
    # slot assignment must be the identity mapping seat_i -> seat_i
    check(f"6-handed dealer={d} slots", sorted(bs.seats), [f"seat_{i}" for i in range(1, 6)])

print()
print("=== 2. SHORT-HANDED 4 seats (hero, seat_1, seat_3, seat_4), button on seat_3 ===")
# ring = [Hero, seat_1, seat_3, seat_4]; dealer at ring index 2
# hero_position = (0-2) % 4 = 2  -> hero is the BIG BLIND (2 after the button). Legacy gave
# (0-3) % 6 = 3 == UTG.
ts = build([1, 3, 4], 3)
check("hero_position (occupied ring)", ts.hero_position, 2)
print(f"      legacy formula would have said {(0 - 3) % 6}  <-- the bug")
check("seat positions", ts.seat_positions, {'Hero': 2, 'seat_1': 3, 'seat_3': 0, 'seat_4': 1})

bs = ts.to_board_state(call_amount=0.0, equity=0.5, big_blind=20.0)
hp = ts.hero_position
decoded = {}
for j in range(5):
    key = f"seat_{j+1}"
    if key in bs.seats:
        decoded[bs.seats[key].name] = (j + 1 + hp) % 6   # ContractV12's own formula
check("contract-decoded opponent positions", decoded, {'p1': 3, 'p3': 0, 'p4': 1})
# p1->pos3 lands in slot (3-2)%6=1, p3->pos0 in slot (0-2)%6=4, p4->pos1 in slot (1-2)%6=5
check("occupied contract slots", sorted(bs.seats), ['seat_1', 'seat_4', 'seat_5'])

# per-seat features must travel with the opponent, not the physical key
ts.opponents['seat_4']['stack'] = 111
ts.raised_this_hand['seat_4'] = True
bs = ts.to_board_state(call_amount=0.0, equity=0.5, big_blind=20.0)
moved = [s for s in bs.seats.values() if s.name == 'p4'][0]
check("features follow the opponent (stack)", moved.stack, 111)
check("features follow the opponent (raised_this_hand)", moved.raised_this_hand, True)

print()
print("=== 3. HUD default: unknown colour must be Yellow/Green, not Blue ===")
ts = build([1], 0)
ts.opponents['seat_1']['vpip_color'] = None
ts.opponents['seat_1']['agg_color'] = None
bs = ts.to_board_state(call_amount=0.0, equity=0.5, big_blind=20.0)
s = list(bs.seats.values())[0]
check("unknown vpip_color", s.hud.vpip_color, 'Yellow')
check("unknown agg_color", s.hud.agg_color, 'Green')

print()
print("=== 4. Contract still encodes 54 features from a short-handed table ===")
c = ContractV12(max_seq_len=20)
ts = build([1, 3, 4], 3)
bs = ts.to_board_state(call_amount=40.0, equity=0.5, big_blind=20.0)
hole, board, ctx, act = c.to_tensors([bs], [])
check("ctx width", ctx.shape[-1], 54)
check("hero_position feature ctx[0]", round(float(ctx[0, -1, 0]), 4), round(2 / 5.0, 4))

print()
print("=== 5. call_amount sentinel + CALL masking (decision engine) ===")
eng = PokerDecisionEngine()
ts = build([1, 2, 3, 4, 5], 0)


def decide(call_amount, known=True, cc_avail=True, br_avail=True, n=400):
    acts = []
    for _ in range(n):
        eng.hand_history_buffer = []
        eng.hero_action_buffer = []
        eng._last_hole_cards = None
        b = ts.to_board_state(call_amount=call_amount, equity=0.35, big_blind=20.0)
        b.hand_strength = 0.4
        a, r, sz, ev = eng.make_decision(b, bet_raise_available=br_avail,
                                         check_call_available=cc_avail,
                                         call_amount_known=known)
        acts.append(a)
    return acts


def fold_sampling_mass(call_amount, known):
    """The [Fable #13] invariant is about MASKING, so read it off the POST-MASK sampling
    distribution the decision actually drew from (`ev['sampled_probs']`) -- a hard 0 iff FOLD was
    masked. The raw policy `ev['FOLD']` can't distinguish the two (masking is applied to the
    sampler, not the reported policy), and sampling-APPEARANCE over N draws was a proxy that broke
    on V44: at this spot raw P(FOLD)=0.038, which temp-0.5 sharpening drops to ~0.001, so FOLD is
    legal-but-rare and often absent from 400 draws -- a calibration artifact, not a masking bug."""
    eng.hand_history_buffer = []
    eng.hero_action_buffer = []
    eng._last_hole_cards = None
    b = ts.to_board_state(call_amount=call_amount, equity=0.35, big_blind=20.0)
    b.hand_strength = 0.4
    _a, _r, _sz, ev = eng.make_decision(b, bet_raise_available=True, check_call_available=True,
                                        call_amount_known=known)
    sp = ev.get('sampled_probs') or {}
    return float(sp.get('FOLD', 0.0))

fm_known = fold_sampling_mass(0.0, known=True)
fm_unknown = fold_sampling_mass(0.0, known=False)
print(f"      sampled P(FOLD): known free check={fm_known:.4f}  unknown price={fm_unknown:.4f}")
check("known free check -> FOLD masked (sampled P==0)", fm_known == 0.0, True)
check("unknown price -> FOLD available (sampled P>0)", fm_unknown > 0.0, True)
a = decide(200.0, known=True, cc_avail=False)
check("no call button -> CALL never chosen", 'CALL' in a, False)
a = decide(200.0, known=True, cc_avail=False, br_avail=False)
check("no call+no raise button -> FOLD only", sorted(set(a)), ['FOLD'])

print()
print("=== 6. _parse_button_money (decimal-stake units, mirrors vision digit-strip) ===")
try:
    from PHPHelp import PHPHelpApp
    p = PHPHelpApp._parse_button_money
    check("'KALD 0.20' on a cents table", p(None, "KALD 0.20"), 20.0)
    check("'CALL 1.50'", p(None, "CALL 1.50"), 150.0)
    check("'KALD 40' (integer chips)", p(None, "KALD 40"), 40.0)
    check("'KALD' (no digits) -> None", p(None, "KALD"), None)
    check("'CALL 0,20' (comma decimal)", p(None, "CALL 0,20"), 20.0)
except Exception as e:
    print(f"SKIP  could not import PHPHelp headlessly: {e!r}")

print()
print("=" * 60)
print("FAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)

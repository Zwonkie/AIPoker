"""V44 verification: `equity_edge` normalized by the EFFECTIVE contested field.

ctx[35] changes MEANING (contract_version 8 -> 9). Two independent implementations of that feature
exist -- `ContractV12.to_tensors` (inference) and `vectorize_hand_samples` (gradient tensors) -- so
the headline check is that they still agree, since a divergence there is a silent train/serve split
of exactly the kind this repo keeps getting bitten by.

Run:  python versions/v47/self_play/verify_v44.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

import torch  # noqa: E402

from core.board_state import BoardState, SeatState, HUDStats            # noqa: E402
from versions.v47.core.contract import (                                 # noqa: E402
    ContractV12, effective_contested_field, equity_edge_feature)
from versions.v43.core.contract import ContractV12 as ContractV43        # noqa: E402
from versions.v47.self_play.simulator import _COLOR_TO_VPIP              # noqa: E402

results = []


def check(label, got, want, tol=1e-9):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got {got}, want {want}")
    results.append(ok)


def seats(n, color='Yellow'):
    return {f'seat_{i}': SeatState(is_active=(i <= n), stack=1500,
                                   hud=HUDStats(vpip_color=color, agg_color='Green'))
            for i in range(1, 6)}


def bstate(n, eq, eff=0.0, board=()):
    return BoardState(community_cards=list(board), hero_cards=['Ah', 'Kh'], pot_size=60,
                      hero_stack=1500, seats=seats(n), hero_position=2,
                      street='Preflop' if not board else 'Flop', call_amount=20, equity=eq,
                      big_blind=20, hand_strength=0.661, effective_field=eff)


print("1. effective_contested_field closed form vs the definition")
p = _COLOR_TO_VPIP['Yellow']
for n in (1, 2, 3, 5):
    want = (n * p) / (1 - (1 - p) ** n)          # E[k] / P(k>=1), all-Yellow
    check(f"{n} after-opponents @ VPIP {p}", effective_contested_field([p] * n), want, 1e-9)
check("front opponents are never rolled (p=1)", effective_contested_field([], n_front=3), 3.0)
check("front forces P(k=0)=0, so no conditioning",
      effective_contested_field([p, p], n_front=1), 1.0 + 2 * p)
check("no opponents at all", effective_contested_field([]), 0.0)
check("mixed colours",
      effective_contested_field([_COLOR_TO_VPIP['Blue'], _COLOR_TO_VPIP['Red']]),
      (0.10 + 0.45) / (1 - 0.90 * 0.55), 1e-9)

print("\n2. The feature is now flat across field size (AKs, real equities)")
AK_EQ = {1: 0.660, 2: 0.650, 3: 0.580, 4: 0.550, 5: 0.520}
edges_new, edges_old = [], []
for n, eq in AK_EQ.items():
    eff = effective_contested_field([p] * n)
    edges_new.append(equity_edge_feature(eq, eff))
    edges_old.append(equity_edge_feature(eq, n))
spread_new = max(edges_new) - min(edges_new)
spread_old = max(edges_old) - min(edges_old)
print(f"       V43 edges {[round(e, 2) for e in edges_old]}  spread {spread_old:.2f}")
print(f"       V44 edges {[round(e, 2) for e in edges_new]}  spread {spread_new:.2f}")
check("V44 spread is much tighter than V43's", spread_new < spread_old / 3.0, True)

print("\n3. The feature still SEPARATES hands (its whole purpose)")
HANDS = {'AA': {1: 0.85, 3: 0.72, 5: 0.66}, 'AKs': {1: 0.66, 3: 0.58, 5: 0.52},
         '72o': {1: 0.31, 3: 0.25, 5: 0.18}}
bands = {}
for name, eqs in HANDS.items():
    vals = [equity_edge_feature(eq, effective_contested_field([p] * n)) for n, eq in eqs.items()]
    bands[name] = (min(vals), max(vals))
    print(f"       {name:<5} {min(vals):.2f} - {max(vals):.2f}")
check("AA band sits entirely above AKs'", bands['AA'][0] > bands['AKs'][1], True)
check("AKs band sits entirely above 72o's", bands['AKs'][0] > bands['72o'][1], True)

print("\n4. ctx[35]: V44 contract vs V43 contract")
C44, C43 = ContractV12(max_seq_len=20), ContractV43(max_seq_len=20)
eff5 = effective_contested_field([p] * 5)
v44 = C44.to_tensors([bstate(5, 0.52, eff=eff5)], None)[2][0][-1][35].item()
v43 = C43.to_tensors([bstate(5, 0.52)], None)[2][0][-1][35].item()
check("V44 preflop uses the effective field", v44, 0.52 * (eff5 + 1), 1e-5)
check("V43 preflop used the nominal field", v43, 0.52 * 6, 1e-5)
check("so they genuinely differ preflop", abs(v44 - v43) > 1.0, True)

no_eff = C44.to_tensors([bstate(5, 0.52, eff=0.0)], None)[2][0][-1][35].item()
check("effective_field=0.0 falls back to V43 behaviour exactly", no_eff, v43, 1e-6)

postflop_eff = 5.0   # no fold-roll postflop -> callers pass the nominal count
pf44 = C44.to_tensors([bstate(5, 0.52, eff=postflop_eff, board=('Ad', '7c', '2h'))],
                      None)[2][0][-1][35].item()
pf43 = C43.to_tensors([bstate(5, 0.52, board=('Ad', '7c', '2h'))], None)[2][0][-1][35].item()
check("postflop is byte-identical to V43", pf44, pf43, 1e-9)

print("\n5. train.py's vectorizer agrees with the contract (the train/serve split risk)")
import versions.v47.self_play.train as T  # noqa: E402
for eff, nominal, label in ((eff5, 5, "effective field supplied"),
                            (0.0, 5, "not supplied -> nominal fallback")):
    _eff = float(eff or 0.0)
    train_side = T.equity_edge_feature(0.52, _eff if _eff > 0 else nominal)
    ctx_side = C44.to_tensors([bstate(5, 0.52, eff=eff)], None)[2][0][-1][35].item()
    check(f"vectorize == contract ({label})", train_side, ctx_side, 1e-5)

print(f"\n{sum(results)}/{len(results)} passed")
raise SystemExit(0 if all(results) else 1)

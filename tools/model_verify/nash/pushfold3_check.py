"""[V48, P48-0.1] VAL-1 runtime checks against the in-repo 3-MAX push/fold Nash solution
(nash_3max_solved.json, produced offline by solve_nash_3max.py). Same discipline as the HU
pair: literal ALLIN-vs-FOLD scoring from day one (P0.3 -- no composite artifact), WARN-only
(3-max push/fold is a subgame, not the whole game), pure lookup + one forward pass per cell.

  check_nash3_btn_jam  -- BTN first-in at a TRUE 3-handed table (compressed position map,
                          num_active_opp=2): jam-vs-fold lean vs the solved BTN range.
                          Ctx equity = hand vs TWO random hands (lazy MC cache).
  check_nash3_bb_call  -- BB facing a BTN jam after SB folds: call-vs-fold lean vs the solved
                          C_bb1 range, equity RANGE-CONDITIONED on BTN's Nash jam range via
                          the cached pairwise matrix (no MC at check time).
"""
import json
import os
import random

from tools.model_verify.scenarios import build_ctx, run_policy

_HERE = os.path.dirname(__file__)
_SOLVED3_PATH = os.path.join(_HERE, 'nash_3max_solved.json')
_MATRIX_PATH = os.path.join(_HERE, 'equity_matrix.json')
_EQ2RAND_PATH = os.path.join(_HERE, 'nash3_eq_vs_2random.json')

_CLEAR_HI = 0.95
_CLEAR_LO = 0.05
_PASS_THRESHOLD = 0.75   # 3-max axis is new -- calibrated after the first scored runs
_BTN_POSITION = 0        # compressed 3-handed map: button acts first preflop


def _load(path):
    with open(path) as f:
        return json.load(f)


def _eq_vs_2random():
    """169-class equity vs two random hands, MC once then cached."""
    if os.path.exists(_EQ2RAND_PATH):
        return _load(_EQ2RAND_PATH)
    from treys import Deck, Evaluator
    from tools.model_verify.nash.solve_nash_pushfold import all_hands, _rep_cards
    ev = Evaluator()
    rng = random.Random(4849)
    out = {}
    for h in all_hands():
        cards = _rep_cards(h, set())
        wins = 0.0
        n = 400
        for _ in range(n):
            deck = [c for c in Deck().cards if c not in cards]
            rng.shuffle(deck)
            o1, o2, board = deck[0:2], deck[2:4], deck[4:9]
            scores = [ev.evaluate(board, list(cards)), ev.evaluate(board, o1), ev.evaluate(board, o2)]
            best = min(scores)
            if scores[0] == best:
                wins += 1.0 / scores.count(best)
        out[h] = round(wins / n, 4)
    with open(_EQ2RAND_PATH, 'w') as f:
        json.dump(out, f)
    return out


def _score(rc, cells):
    """cells: iterable of (hand, stack, nash_lean, ctx). -> CheckResult via literal lean."""
    from tools.model_verify.checks import CheckResult, _find
    ai = _find(rc.action_keys, 'allin')
    fi = _find(rc.action_keys, 'fold')
    agree = total = 0
    gross = []
    for h, S, nash, ctx in cells:
        policy, _q = run_policy(rc.model, ctx, rc.action_keys, device=rc.device)
        p_fold = policy[rc.action_keys[fi]]
        p_allin = policy[rc.action_keys[ai]]
        lean = 'shove' if p_allin > p_fold else 'fold'
        total += 1
        agree += int(lean == nash)
        if lean != nash and len(gross) < 6:
            gross.append(f"{h}@{S}bb(Nash {nash},model {lean}:allin={p_allin:.2f}/F={p_fold:.2f})")
    pct = agree / total if total else 0.0
    detail = (f"literal jam-vs-fold agreement over {total} unambiguous 3-max Nash cells: "
              f"{agree} ({pct:.0%})" + (f" | e.g. {'; '.join(gross)}" if gross else ""))
    status = 'PASS' if pct >= _PASS_THRESHOLD else 'WARN'
    return CheckResult(status, detail, {'agree': agree, 'total': total, 'pct': round(pct, 4)})


def check_nash3_btn_jam(rc):
    from tools.model_verify.checks import CheckResult, _find
    if _find(rc.action_keys, 'allin') is None or _find(rc.action_keys, 'fold') is None:
        return CheckResult('SKIP', "action space lacks ALLIN/FOLD")
    if not os.path.exists(_SOLVED3_PATH):
        return CheckResult('SKIP', "nash_3max_solved.json missing -- run tools.model_verify.nash.solve_nash_3max")
    solved = _load(_SOLVED3_PATH)
    eqr = _eq_vs_2random()
    cv = rc.manifest.contract_version
    cells = []
    for S in solved['stacks']:
        jam = solved['solutions'][str(S)]['jam_btn']
        for h, f in jam.items():
            if _CLEAR_LO < f < _CLEAR_HI:
                continue
            nash = 'shove' if f >= _CLEAR_HI else 'fold'
            ctx = build_ctx(equity=eqr[h], stack_bb=S, pot_bb=1.5, call_bb=1.0,
                            num_active_opp=2, position=_BTN_POSITION, street=0,
                            contract_version=cv)
            cells.append((h, S, nash, ctx))
    return _score(rc, cells)


def check_nash3_bb_call(rc):
    from tools.model_verify.checks import CheckResult, _find
    if _find(rc.action_keys, 'allin') is None or _find(rc.action_keys, 'fold') is None:
        return CheckResult('SKIP', "action space lacks ALLIN/FOLD")
    if not (os.path.exists(_SOLVED3_PATH) and os.path.exists(_MATRIX_PATH)):
        return CheckResult('SKIP', "nash_3max_solved.json / equity_matrix.json missing")
    solved = _load(_SOLVED3_PATH)
    mat = _load(_MATRIX_PATH)
    from tools.model_verify.nash.solve_nash_pushfold import combos
    cv = rc.manifest.contract_version
    cells = []
    for S in solved['stacks']:
        sol = solved['solutions'][str(S)]
        jam = sol['jam_btn']
        jam_mass = {h: combos(h) * f for h, f in jam.items() if f > _CLEAR_LO}
        tw = sum(jam_mass.values()) or 1e-9
        for h, f in sol['call_bb_vs_jam'].items():
            if _CLEAR_LO < f < _CLEAR_HI:
                continue
            nash = 'shove' if f >= _CLEAR_HI else 'fold'   # 'shove'==commit(call the jam)
            eq = sum(w * mat[h].get(j, 0.5) for j, w in jam_mass.items()) / tw
            # BB facing BTN jam, SB folded: pot = S (jam) + 1.5 blinds, price = S - 1.
            ctx = build_ctx(equity=eq, stack_bb=S, pot_bb=S + 1.5, call_bb=max(0.5, S - 1.0),
                            num_active_opp=1, position=2, street=0, contract_version=cv)
            cells.append((h, S, nash, ctx))
    return _score(rc, cells)

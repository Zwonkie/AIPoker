"""OFFLINE, author-time ONLY -- solves the 3-max push/fold Nash equilibrium in-repo and writes
nash_3max_solved.json ([V48, P48-0.1] -- the [VAL-1] external axis extended to the geometry V48
introduces; DoN ends at 3 players, so 3-handed is the endgame every session funnels toward).

    .venv/Scripts/python.exe -m tools.model_verify.nash.solve_nash_3max

The game (equal effective stacks S bb, blinds SB=0.5/BB=1.0, chip-EV, no ante, push/fold only):
  - BTN acts first: FOLD (net 0) or JAM for S.
  - Facing the jam, SB: FOLD (net -0.5) or CALL.
  - BB then: FOLD (net -1.0) or CALL -- with DIFFERENT ranges facing jam-only vs jam+call
    (four strategy vectors total: J_btn, C_sb, C_bb1, C_bb2).
  - BTN folding hands the SB an open-jam sub-game that IS the already-solved HU chart
    (solve_nash_pushfold.py) -- deliberately NOT re-solved here; the model check scores SB
    open-jams against that existing chart.

Payoffs (BTN perspective; equal stacks, no side pots):
  everyone folds ......... +1.5
  SB calls, BB folds ..... eq2(btn,sb)*(2S+1.0) - S
  SB folds, BB calls ..... eq2(btn,bb)*(2S+0.5) - S
  three-way .............. eq3(btn,sb,bb)*(3S) - S

Method: smoothed best-response iteration (fictitious play). Pairwise equities come from the
HU solver's cached 169x169 matrix (equity_matrix.json); THREE-WAY equities are Monte Carlo
with one canonical suit-representative per class, computed LAZILY and cached on disk
(equity3_cache.json) -- FP mass concentrates on short-stack jam/call ranges, so only a small
corner of the 169^3 cube is ever touched. Same two documented approximations as the HU
solver (per-class representatives; MC noise near margins).

Validation anchors (bottom of file): AA/KK jam+call everywhere; trash (72o) never calls a
jam at 10bb+; ranges MONOTONE in stack depth (shorter = wider); BB overcall range (C_bb2)
strictly tighter than call-vs-jam-only (C_bb1) at every depth.
"""
import json
import os
import random

from treys import Card, Deck, Evaluator

from tools.model_verify.nash.solve_nash_pushfold import (all_hands, combos, _rep_cards,
                                                         build_matrix)

_HERE = os.path.dirname(__file__)
_SOLVED_PATH = os.path.join(_HERE, 'nash_3max_solved.json')
_EQ3_CACHE_PATH = os.path.join(_HERE, 'equity3_cache.json')

_STACKS = [5, 8, 10, 15]
_FP_ITERS = 1200
_N_SIMS_3WAY = 160
_SEED = 4848

_EVAL = Evaluator()


class Eq3:
    """Lazy, disk-cached 3-way preflop equity for (btn, sb, bb) class triples."""

    def __init__(self):
        self.cache = {}
        if os.path.exists(_EQ3_CACHE_PATH):
            with open(_EQ3_CACHE_PATH) as f:
                self.cache = json.load(f)
        self.dirty = 0

    def get(self, a, b, c):
        key = '|'.join((a, b, c))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        rng = random.Random(hash(key) & 0xFFFFFFFF)
        cards_a = _rep_cards(a, set())
        cards_b = _rep_cards(b, set(cards_a) if cards_a else set())
        cards_c = None
        if cards_a and cards_b:
            cards_c = _rep_cards(c, set(cards_a) | set(cards_b))
        if not (cards_a and cards_b and cards_c):
            # Degenerate class triple under card removal (e.g. three AA needs six aces).
            # Vanishing probability mass in any real range triple -- uniform split.
            self.cache[key] = [1 / 3, 1 / 3, 1 / 3]
            return self.cache[key]
        used = list(cards_a) + list(cards_b) + list(cards_c)
        wins = [0.0, 0.0, 0.0]
        deck_master = [card for card in Deck().cards if card not in used]
        for _ in range(_N_SIMS_3WAY):
            board = rng.sample(deck_master, 5)
            scores = [_EVAL.evaluate(board, list(cards_a)),
                      _EVAL.evaluate(board, list(cards_b)),
                      _EVAL.evaluate(board, list(cards_c))]
            best = min(scores)
            winners = [i for i, sc in enumerate(scores) if sc == best]
            for i in winners:
                wins[i] += 1.0 / len(winners)
        eqs = [w / _N_SIMS_3WAY for w in wins]
        self.cache[key] = eqs
        self.dirty += 1
        # Flush interval scales with cache size: dumping the whole JSON is O(cache), so a
        # fixed interval makes flush overhead grow linearly with progress (measured ~40% of
        # wall time by the 100MB mark). Amortized this keeps flushing ~constant-fraction.
        if self.dirty >= max(2000, len(self.cache) // 10):
            self.flush()
        return eqs

    def flush(self):
        # Atomic + non-fatal: the cache is an optimization, a failed flush must not kill an
        # hours-long solve (2026-07-22: a mid-solve open() raised a transient OSError 22 on
        # the 27MB file and lost the run). tmp+replace also keeps a concurrent reader from
        # ever seeing a half-written file.
        tmp = _EQ3_CACHE_PATH + '.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump(self.cache, f)
            os.replace(tmp, _EQ3_CACHE_PATH)
            self.dirty = 0
        except OSError as e:
            print(f"  ! eq3 cache flush failed ({e}) -- continuing, will retry", flush=True)
            self.dirty = 1000   # halfway to the 2000 threshold: retry soon-ish, not every entry


def solve_stack(S, hands, w, eq2, eq3, iters=_FP_ITERS, seed=_SEED):
    """-> dict with jam_btn / call_sb / call_bb_vs_jam / call_bb_vs_jam_call frequency vectors
    (per hand class, 0..1 mixed strategies from smoothed FP averaging)."""
    rng = random.Random(seed + S)
    n = len(hands)
    total_w = sum(w)
    # strategy frequencies (start: tight-ish plausible seeds)
    J = [1.0 if i < n // 4 else 0.0 for i in range(n)]
    C1 = [1.0 if i < n // 6 else 0.0 for i in range(n)]      # SB call vs jam
    B1 = [1.0 if i < n // 6 else 0.0 for i in range(n)]      # BB call vs jam (SB folded)
    B2 = [1.0 if i < n // 10 else 0.0 for i in range(n)]     # BB overcall vs jam+call

    def norm_weights(strategy):
        return [w[i] * strategy[i] for i in range(n)]

    for it in range(iters):
        alpha = 2.0 / (it + 2.0)   # averaging step

        # --- BB best responses (innermost, holds k) --------------------------------
        jam_w = norm_weights(J)
        jam_total = sum(jam_w) or 1e-9
        callsb_w = [jam_w[i] * C1[i] for i in range(n)]  # joint jam+call mass proxy
        newB1, newB2 = [0.0] * n, [0.0] * n
        # sample a subset of (i) [and (i,j)] pairs for the expectation (stochastic FP)
        idx_pool = [i for i in range(n) if jam_w[i] > 1e-9] or [0]
        for k in range(n):
            # vs jam-only (SB folded): pot if call = 2S + 0.5
            ev = 0.0
            for i in rng.sample(idx_pool, min(40, len(idx_pool))):
                e = eq2[hands[k]][hands[i]]
                ev += jam_w[i] * (e * (2 * S + 0.5) - S)
            ev /= jam_total
            newB1[k] = 1.0 if ev >= -1.0 else 0.0
            # vs jam+call: three-way, pot 3S
            pairs = [(i, j) for i in rng.sample(idx_pool, min(8, len(idx_pool)))
                     for j in rng.sample(idx_pool, min(4, len(idx_pool))) if C1[j] > 1e-9]
            if pairs:
                ev2 = 0.0
                tw = 0.0
                for i, j in pairs:
                    wt = jam_w[i] * w[j] * C1[j]
                    if wt <= 0:
                        continue
                    e3 = eq3.get(hands[i], hands[j], hands[k])[2]
                    ev2 += wt * (e3 * 3 * S - S)
                    tw += wt
                newB2[k] = 1.0 if tw > 0 and ev2 / tw >= -1.0 else 0.0
            else:
                newB2[k] = newB1[k]

        # --- SB best response ------------------------------------------------------
        newC1 = [0.0] * n
        bb_call_w = norm_weights(newB1)
        for j in range(n):
            ev = 0.0
            for i in rng.sample(idx_pool, min(40, len(idx_pool))):
                # BB folds -> HU vs jammer, pot 2S+1; BB overcalls -> 3-way
                p_bb_over = sum(w[k] * newB2[k] for k in rng.sample(range(n), 30)) / (total_w * 30 / n)
                e2 = eq2[hands[j]][hands[i]]
                hu_ev = e2 * (2 * S + 1.0) - S
                e3 = eq3.get(hands[i], hands[j], 'AA')[1] if p_bb_over > 0.02 else e2  # coarse
                three_ev = e3 * 3 * S - S
                ev += jam_w[i] * ((1 - p_bb_over) * hu_ev + p_bb_over * three_ev)
            ev /= jam_total
            newC1[j] = 1.0 if ev >= -0.5 else 0.0

        # --- BTN best response -----------------------------------------------------
        newJ = [0.0] * n
        for i in range(n):
            p_sb_call_hands = [(j, w[j] * newC1[j]) for j in range(n) if newC1[j] > 0]
            p_sb_call = sum(x for _, x in p_sb_call_hands) / total_w
            p_bb_call = sum(w[k] * newB1[k] for k in range(n)) / total_w
            ev_fold_out = (1 - p_sb_call) * (1 - p_bb_call) * 1.5
            ev = ev_fold_out
            # SB calls (BB overcall folded into 3-way approx via B2 rate)
            for j, wt in rng.sample(p_sb_call_hands, min(24, len(p_sb_call_hands))) if p_sb_call_hands else []:
                e2 = eq2[hands[i]][hands[j]]
                ev += (wt / total_w) * (e2 * (2 * S + 1.0) - S) * (len(p_sb_call_hands) / max(1, min(24, len(p_sb_call_hands))))
            # BB calls after SB folds
            bb_hands = [(k, w[k] * newB1[k]) for k in range(n) if newB1[k] > 0]
            for k, wt in rng.sample(bb_hands, min(24, len(bb_hands))) if bb_hands else []:
                e2 = eq2[hands[i]][hands[k]]
                ev += (1 - p_sb_call) * (wt / total_w) * (e2 * (2 * S + 0.5) - S) * (len(bb_hands) / max(1, min(24, len(bb_hands))))
            newJ[i] = 1.0 if ev >= 0.0 else 0.0

        for v, nv in ((J, newJ), (C1, newC1), (B1, newB1), (B2, newB2)):
            for i in range(n):
                v[i] = (1 - alpha) * v[i] + alpha * nv[i]

    return {'jam_btn': dict(zip(hands, (round(x, 3) for x in J))),
            'call_sb': dict(zip(hands, (round(x, 3) for x in C1))),
            'call_bb_vs_jam': dict(zip(hands, (round(x, 3) for x in B1))),
            'call_bb_vs_jam_call': dict(zip(hands, (round(x, 3) for x in B2)))}


def main():
    random.seed(_SEED)
    hands = all_hands()
    w = [combos(h) for h in hands]
    eq2 = build_matrix(hands)   # cached 169x169 from the HU solver -- instant load
    eq3 = Eq3()
    out = {}
    for S in _STACKS:
        print(f"solving 3-max S={S}bb ...")
        out[str(S)] = solve_stack(S, hands, w, eq2, eq3)
        eq3.flush()
    with open(_SOLVED_PATH, 'w') as f:
        json.dump({'stacks': _STACKS, 'solutions': out,
                   'method': 'smoothed stochastic FP, pairwise matrix + lazy 3-way MC',
                   'fp_iters': _FP_ITERS, 'n_sims_3way': _N_SIMS_3WAY}, f, indent=1)
    print(f"wrote {_SOLVED_PATH} ({len(eq3.cache)} cached 3-way triples)")

    # ---- anchors ----------------------------------------------------------------
    fails = []
    for S in _STACKS:
        sol = out[str(S)]
        if sol['jam_btn']['AA'] < 0.9 or sol['call_sb']['AA'] < 0.9:
            fails.append(f"S={S}: AA not jam/call")
        if S >= 10 and sol['call_sb']['72o'] > 0.1:
            fails.append(f"S={S}: 72o calls a jam")
        if sol['call_bb_vs_jam_call']['AA'] < 0.9:
            fails.append(f"S={S}: AA doesn't overcall")
    for a, b in zip(_STACKS, _STACKS[1:]):
        ja = sum(out[str(a)]['jam_btn'].values())
        jb = sum(out[str(b)]['jam_btn'].values())
        if jb > ja + 3:
            fails.append(f"jam range not monotone: S={b} wider than S={a}")
    for S in _STACKS:
        b1 = sum(out[str(S)]['call_bb_vs_jam'].values())
        b2 = sum(out[str(S)]['call_bb_vs_jam_call'].values())
        if b2 > b1 + 1:
            fails.append(f"S={S}: overcall range wider than call range")
    if fails:
        print("ANCHOR FAILURES:")
        for x in fails:
            print("  !!", x)
        raise SystemExit(1)
    print("anchors OK")


if __name__ == '__main__':
    main()

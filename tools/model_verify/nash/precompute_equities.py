"""OFFLINE, author-time ONLY -- (re)builds nash_chart.json.

Run once when the curated cell list below changes:
    .venv/Scripts/python.exe -m tools.model_verify.nash.precompute_equities

It computes each hand's raw heads-up preflop equity vs ONE random opponent (an
assumption-free, range-neutral value -- the model is expected to supply the fold-equity /
push-fold reasoning itself from the stack/pot/call geometry, so the equity input is
deliberately NOT conditioned on any assumed calling range) and bakes it into the static
JSON. This is the ONLY file that touches core.evaluator; it never runs during a
model_verify invocation, so the runtime check has zero equity/evaluator dependency.

WHY A CURATED SUBSET (not the full 169-hand solved chart): every cell below is an
UNAMBIGUOUS Nash push/fold reference point -- a spot where the shove-or-fold answer is not
in dispute among any standard published HU Nash chart (0.5/1 blinds, no ante, chip-EV). We
deliberately avoid the near-indifference boundary hands whose exact BB threshold varies by
source, so a disagreement here means a real, gross divergence from GTO, not a rounding
argument. Extend toward a full sourced 169-hand table (with attribution) as a later Tier-B
pass if this axis proves useful.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from core.evaluator import PokerEvaluator

_HERE = os.path.dirname(__file__)
_CHART_PATH = os.path.join(_HERE, 'nash_chart.json')
_N_SIMS = 40000  # high, for stable baked equities -- this is offline, cost doesn't matter


# Each cell: (hand_code, effective_stack_bb, nash_action, note)
#   hand_code : 'AA'/'TT'/'22' (pair) | 'AKs'/'A5s' (suited) | 'AKo'/'A2o' (offsuit)
#   nash_action: 'shove' | 'fold'   (the Nash SB open-jam decision at that effective stack)
# Only high-confidence, non-controversial cells. See module docstring.
CELLS = [
    # --- Premiums: shove at any push/fold depth --------------------------------------
    ("AA", 15, "shove", "the nuts -- shoves at every push/fold depth"),
    ("KK", 15, "shove", "shoves at every push/fold depth"),
    ("QQ", 15, "shove", "shoves at every push/fold depth"),
    ("AKs", 15, "shove", "shoves at every push/fold depth"),
    ("AKo", 15, "shove", "shoves at every push/fold depth"),

    # --- Pocket pairs: all pairs are clear shoves at <=15bb HU ------------------------
    ("JJ", 15, "shove", "any pair shoves comfortably at 15bb HU"),
    ("TT", 15, "shove", "any pair shoves comfortably at 15bb HU"),
    ("88", 15, "shove", "any pair shoves comfortably at 15bb HU"),
    ("55", 15, "shove", "any pair shoves comfortably at 15bb HU"),
    ("22", 12, "shove", "even the worst pair shoves well past 12bb HU"),

    # --- Aces: suited any-ace shoves deep; offsuit aces shove short -------------------
    ("A2s", 15, "shove", "any suited ace shoves at 15bb HU"),
    ("A5s", 15, "shove", "any suited ace shoves at 15bb HU"),
    ("AJo", 12, "shove", "strong offsuit ace, clear shove at 12bb"),
    ("ATo", 12, "shove", "offsuit ace, clear shove at 12bb"),
    ("A9o", 10, "shove", "offsuit ace, clear shove at 10bb"),
    ("A2o", 7, "shove", "worst offsuit ace still shoves at 7bb HU"),

    # --- Suited broadway / strong suited kings: clear shoves at 10-12bb ---------------
    ("KQs", 12, "shove", "suited broadway, clear shove"),
    ("KJs", 12, "shove", "suited broadway, clear shove"),
    ("KTs", 12, "shove", "suited broadway, clear shove"),
    ("QJs", 10, "shove", "suited broadway, clear shove at 10bb"),
    ("JTs", 10, "shove", "suited connector broadway, clear shove at 10bb"),
    ("K9s", 10, "shove", "suited king, clear shove at 10bb"),
    ("KQo", 10, "shove", "offsuit broadway, clear shove at 10bb"),

    # --- Clear folds: bottom-tier offsuit trash folds well before 15bb ---------------
    ("72o", 15, "fold", "the single worst hand -- folds far below 15bb"),
    ("82o", 15, "fold", "offsuit trash, clear fold at 15bb"),
    ("83o", 15, "fold", "offsuit trash, clear fold at 15bb"),
    ("92o", 15, "fold", "offsuit trash, clear fold at 15bb"),
    ("93o", 15, "fold", "offsuit trash, clear fold at 15bb"),
    ("62o", 15, "fold", "offsuit trash, clear fold at 15bb"),
    ("73o", 15, "fold", "offsuit trash, clear fold at 15bb"),
    ("42o", 15, "fold", "offsuit trash, clear fold at 15bb"),
    ("32o", 15, "fold", "offsuit trash, clear fold at 15bb"),
    ("J2o", 15, "fold", "J-high offsuit trash, clear fold at 15bb"),
    ("Q2o", 15, "fold", "Q-high offsuit trash, clear fold at 15bb"),
    ("T2o", 15, "fold", "T-high offsuit trash, clear fold at 15bb"),

    # --- Below-training-floor (stack < 5bb): ATC shove regime, reported SEPARATELY ----
    # These are OOD for a model trained on 5bb+ stacks; kept as informational, NOT part
    # of the pass metric (the check flags them via in_training_range=False).
    ("72o", 2, "shove", "at 2bb HU, any two cards shove (below 5bb training floor -- OOD)"),
    ("32o", 2, "shove", "at 2bb HU, any two cards shove (below 5bb training floor -- OOD)"),
    ("92o", 2, "shove", "at 2bb HU, any two cards shove (below 5bb training floor -- OOD)"),
]

_RANKS = "23456789TJQKA"
_SUITS = "shdc"


def _cards_for(hand_code):
    """hand_code -> two concrete card strings (suits chosen only to realize suited/offsuit;
    value is suit-symmetric so any consistent choice yields the same equity)."""
    if len(hand_code) == 2:  # pair, e.g. "TT"
        r = hand_code[0]
        assert hand_code[0] == hand_code[1] and r in _RANKS, f"bad pair code {hand_code}"
        return [f"{r}{_SUITS[0]}", f"{r}{_SUITS[1]}"]
    assert len(hand_code) == 3, f"bad hand code {hand_code}"
    r1, r2, so = hand_code[0], hand_code[1], hand_code[2]
    assert r1 in _RANKS and r2 in _RANKS and r1 != r2, f"bad hand code {hand_code}"
    assert _RANKS.index(r1) > _RANKS.index(r2), f"hand code must be high-rank first: {hand_code}"
    if so == 's':
        return [f"{r1}{_SUITS[0]}", f"{r2}{_SUITS[0]}"]
    if so == 'o':
        return [f"{r1}{_SUITS[0]}", f"{r2}{_SUITS[1]}"]
    raise ValueError(f"suited/offsuit marker must be s or o: {hand_code}")


def main():
    ev = PokerEvaluator()
    # cache equity per unique hand_code (equity is independent of stack)
    eq_cache = {}
    out = []
    for hand_code, stack_bb, nash_action, note in CELLS:
        if hand_code not in eq_cache:
            cards = _cards_for(hand_code)
            equity, _log = ev.calculate_equity([], cards, num_opponents=1, num_simulations=_N_SIMS)
            eq_cache[hand_code] = round(float(equity), 4)
            print(f"  {hand_code:<4} equity_vs_random={eq_cache[hand_code]:.4f}")
        out.append({
            "hand": hand_code,
            "stack_bb": stack_bb,
            "nash_action": nash_action,
            "equity_vs_random": eq_cache[hand_code],
            "in_training_range": stack_bb >= 5,
            "note": note,
        })
    payload = {
        "_meta": {
            "description": "Curated unambiguous heads-up Nash push/fold reference cells (SB open-jam, "
                           "0.5/1 blinds, no ante, chip-EV). equity_vs_random is raw HU preflop equity "
                           "vs one random opponent, computed offline via core.evaluator.",
            "n_sims_per_hand": _N_SIMS,
            "n_cells": len(out),
        },
        "cells": out,
    }
    with open(_CHART_PATH, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {len(out)} cells -> {_CHART_PATH}")


if __name__ == '__main__':
    main()

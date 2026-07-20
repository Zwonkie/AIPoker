"""[BET-1] calibration probe: does making the opponent bots' "value raise regardless of price"
branch price-sensitive create a real, non-degenerate fold-equity gradient at the top of the
range, without overcorrecting?

Tests V22's (pre-fix) unmodified decide_postflop behavior as a BASELINE against a PATCHED variant
(size-aware value_bar, the same mechanism actually shipped in THIS version's opponent_bots.py) at
several candidate VALUE_PRICE_SENSITIVITY values, across a grid of (equity, pot_odds) for all 4
personalities. No simulator/training dependency -- pure direct calls into the bot logic, many
trials per cell to average out the randomized raise-vs-call choice.

This is the exact script used to calibrate VALUE_PRICE_SENSITIVITY=0.05 for V23 (see
opponent_bots.py's own docstring and versions/v23/SPECS.md for the full write-up of the results).
Kept as a reusable tool -- re-run this (adjusting POT_ODDS_GRID/EQUITY_GRID or the candidate list
in main()) if a future version wants to recalibrate this constant, e.g. after changing an
archetype's own thresholds.

Run: .venv/Scripts/python.exe -m versions.v24.self_play.calibrate_bet1
"""
import random

from versions.v22.self_play.opponent_bots import (
    TAG as TAG_BASELINE, LAG as LAG_BASELINE, NIT as NIT_BASELINE,
    CALLING_STATION as CS_BASELINE, STYLE_SHIFT_SCALE as STYLE_SHIFT_SCALE_BASELINE,
)
from versions.v24.self_play.opponent_bots import TAG, LAG, NIT, CALLING_STATION, STYLE_SHIFT_SCALE

BASELINE_PERSONALITIES = {'TAG': TAG_BASELINE, 'LAG': LAG_BASELINE, 'NIT': NIT_BASELINE, 'CALLING_STATION': CS_BASELINE}
PERSONALITIES = {'TAG': TAG, 'LAG': LAG, 'NIT': NIT, 'CALLING_STATION': CALLING_STATION}

# Bet sizes expressed as pot_odds (bet / (pot+bet)): 0.20≈¼-pot, 0.33≈½-pot, 0.50=pot,
# 0.67≈2x-pot overbet, 0.80≈shove-into-a-big-pot.
POT_ODDS_GRID = [0.20, 0.33, 0.50, 0.67, 0.80]
# Equities spanning "at or above a typical value threshold" (TAG/LAG/NIT/CS's flop need_for_value
# is 0.55-0.75) up to near-nuts.
EQUITY_GRID = [0.60, 0.70, 0.80, 0.90, 0.98]

N_TRIALS = 3000
STREET = 'flop'


def decide_postflop_patched(bot, equity, pot_odds, value_price_sensitivity):
    """Exact copy of FuzzyPlayerArchetype.decide_postflop's facing-a-bet branch, with ONE change:
    value_bar rises with pot_odds instead of being flat need_for_value. Everything else
    (continue_bar, bluff logic) is untouched -- isolates the effect of this one lever."""
    need_for_value = bot.current_value_threshold.get(STREET, 0.7)
    style_shift = (bot.current_fold_to_pressure - 0.5) * STYLE_SHIFT_SCALE
    continue_bar = min(0.95, max(0.02, pot_odds + style_shift))
    value_bar = min(0.98, need_for_value + value_price_sensitivity * pot_odds)

    if equity >= value_bar:
        if random.random() < bot.current_agg_freq * 2.0:
            return 'raise'
        return 'call'
    if equity >= continue_bar:
        if random.random() < bot.current_agg_freq * 1.0:
            return 'raise'
        return 'call'
    if random.random() < bot.current_bluff_freq and random.random() < bot.current_agg_freq * 1.5:
        return 'raise'
    return 'fold'


def decide_postflop_original(bot, equity, pot_odds):
    """Exact copy of the pre-fix (V22 and earlier) facing-a-bet branch, for baseline comparison."""
    need_for_value = bot.current_value_threshold.get(STREET, 0.7)
    style_shift = (bot.current_fold_to_pressure - 0.5) * STYLE_SHIFT_SCALE_BASELINE
    continue_bar = min(0.95, max(0.02, pot_odds + style_shift))

    if equity >= need_for_value:
        if random.random() < bot.current_agg_freq * 2.0:
            return 'raise'
        return 'call'
    if equity >= continue_bar:
        if random.random() < bot.current_agg_freq * 1.0:
            return 'raise'
        return 'call'
    if random.random() < bot.current_bluff_freq and random.random() < bot.current_agg_freq * 1.5:
        return 'raise'
    return 'fold'


def measure(decide_fn, bot, equity, pot_odds, **kw):
    folds = 0
    for _ in range(N_TRIALS):
        # start_new_hand() fuzzes traits with gaussian noise every hand in real play -- roll it
        # here too so the probe reflects the REAL fuzzed distribution, not one frozen draw.
        bot.start_new_hand()
        d = decide_fn(bot, equity, pot_odds, **kw)
        if d == 'fold':
            folds += 1
    return folds / N_TRIALS  # P(fold)


def main():
    print("=" * 100)
    print("BASELINE (V22 and earlier, pre-[BET-1]-fix): P(fold) at value-tier equities, by bet size")
    print("=" * 100)
    for pname, bot in BASELINE_PERSONALITIES.items():
        print(f"\n-- {pname} (flop need_for_value={bot.base_value_threshold['flop']:.2f}) --")
        header = "  equity | " + " | ".join(f"po={po:.2f}" for po in POT_ODDS_GRID)
        print(header)
        for eq in EQUITY_GRID:
            row = [measure(decide_postflop_original, bot, eq, po) for po in POT_ODDS_GRID]
            print(f"  {eq:.2f}   | " + " | ".join(f"{v:.3f} " for v in row))

    print()
    print("=" * 100)
    print("Baseline reading: P(fold) should be ~flat across bet sizes at a fixed equity if the")
    print("bug is real (no price sensitivity once past the value threshold).")
    print("=" * 100)

    for vps in (0.03, 0.05, 0.08, 0.10, 0.15, 0.25, 0.35, 0.50):
        print()
        print("=" * 100)
        print(f"PATCHED: VALUE_PRICE_SENSITIVITY = {vps}")
        print("=" * 100)
        for pname, bot in PERSONALITIES.items():
            print(f"\n-- {pname} --")
            header = "  equity | " + " | ".join(f"po={po:.2f}" for po in POT_ODDS_GRID)
            print(header)
            for eq in EQUITY_GRID:
                row = [measure(decide_postflop_patched, bot, eq, po, value_price_sensitivity=vps) for po in POT_ODDS_GRID]
                print(f"  {eq:.2f}   | " + " | ".join(f"{v:.3f} " for v in row))


if __name__ == '__main__':
    main()

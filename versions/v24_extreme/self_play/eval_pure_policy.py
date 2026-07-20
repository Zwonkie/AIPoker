"""
Pure-policy (deployed) winrate eval for V12-D.

Training BB/100 is dragged by exploration (5% random + heuristic anchor) which loses money.
This measures the DEPLOYED policy: model plays 100% of decisions (no exploration, no
bootstrap) against a fixed field, over many hands, reporting Hero BB/100 + VPIP/AGG.

Tests two fields:
  * the LOOSE field it trained on (fish/tag/nit) -> should win (value-bet stations)
  * the TIGHT nit+tag field -> confirms it generalizes, not just station-exploiting.

Run:  .venv/Scripts/python.exe -m versions.v24_extreme.self_play.eval_pure_policy
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import yaml
import torch
from versions.v24_extreme.core.model import PokerEVModelV4
from versions.v24_extreme.self_play.simulator import SixMaxSimulator
from versions.v24_extreme.core.manifest import MANIFEST
from shared.manifest import load_state_dict as load_ckpt_state

# Match training's equity mode so the deployed model sees the SAME equity it was trained on.
_CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'config.yaml')))['training']
RANGE_AWARE = bool(_CFG.get('range_aware_equity', False))


def run_field(model, label, pool, weights, live_players, n_hands=4000, equity_sims=120):
    sim = SixMaxSimulator(bb_size=10.0, equity_sims=equity_sims, hero_personality='main', bootstrap_alpha=0.0)
    sim.hero_model = model
    sim.opponent_pool_styles = pool
    sim.opponent_pool_weights = weights
    sim.live_players = live_players
    sim.fixed_stack_bb = 100.0
    sim.disable_exploration = True   # 100% model, deployed policy
    sim.range_aware_equity = RANGE_AWARE   # MUST match training (train/serve consistency)
    for i in range(n_hands):
        sim.simulate_hand(current_hand=200000 + i)
    h = sim.seat_histories[0]
    bb100 = (h['profit'] / 10.0) / max(1, n_hands) * 100.0
    vpip = h['vpip_acts'] / max(1, h['vpip_ops']) * 100.0
    agg = h['agg_acts'] / max(1, h['agg_ops']) * 100.0
    print(f"  {label:<28} | {n_hands:>5} hands | Hero {bb100:>+7.1f} BB/100 | VPIP {vpip:>4.1f}% | AGG {agg:>4.1f}%")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = os.path.join(os.path.dirname(__file__), '..', 'weights', 'expert_main.pth')
    model = PokerEVModelV4().to(device)
    model.load_state_dict(load_ckpt_state(ckpt, MANIFEST))
    model.eval()
    print(f"Pure-policy eval | {os.path.abspath(ckpt)} | device {device}\n")
    print("  field                        |  hands | winrate            | style")
    print("  " + "-" * 82)
    # 6-max fields to match live tables + training.
    run_field(model, "Loose (fish/tag/nit) 6max", ['fish', 'tag', 'nit'], [0.45, 0.30, 0.25], 6)
    run_field(model, "Tight (nit/tag) 6max",       ['nit', 'tag'],         [0.50, 0.50],        6)
    run_field(model, "Calling stations only 6max", ['fish'],              [1.0],               6)


if __name__ == '__main__':
    main()

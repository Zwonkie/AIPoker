"""
Policy-vs-Target divergence diagnostic for V12-D.

Bisects the core failure: is the ACTOR learning the target we give it, or not?

  * If the model's live policy MATCHES the counterfactual regret target but it still bleeds
    -> the TARGET is wrong (esp. postflop). Fix the target.
  * If the model's policy DIVERGES from the target (enters/spews where the target says fold)
    -> the actor is NOT learning what we teach -> optimization / self-play non-stationarity.

Part A  aggregate model P(fold/call/raise) vs TARGET, bucketed by STREET x equity, over live
        decision points. Plus mean KL(target||model) and entropies. (Postflop is the part we
        have never inspected -- the air-spew lives there.)
Part B  postflop SENSITIVITY: fixed (hole, board, equity) spots -> the model's policy, to see
        whether it responds to the board at all (air on a scary board should fold to a bet).

Run:  .venv/Scripts/python.exe -m versions.v16_foldregret.self_play.inspect_policy_vs_target
"""
import os
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import yaml
import torch

import versions.v16_foldregret.self_play.train as T
from versions.v16_foldregret.core.model import PokerEVModelV4
from versions.v16_foldregret.core.manifest import MANIFEST
from versions.v16_foldregret.self_play.simulator import SixMaxSimulator
from shared.manifest import load_state_dict as load_ckpt_state

STREETS = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}
EQ_BUCKETS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
# V14 6-action space labels (fold, call, raise 0.33/0.66/1.0 pot, all-in).
ACTION_LABELS = ['F', 'C', 'r33', 'r66', 'rP', 'AI']
K = len(ACTION_LABELS)


def fmt_dist(vals):
    return "/".join(f"{v:.2f}" for v in vals)


def argmax_label(vals):
    return ACTION_LABELS[max(range(len(vals)), key=lambda i: vals[i])]


def load_model(path, device, ablate_hole_cards=False):
    m = PokerEVModelV4().to(device)
    m.ablate_hole_cards = ablate_hole_cards
    m.load_state_dict(load_ckpt_state(path, MANIFEST))
    m.eval()
    return m


def make_sim(equity_sims=200):
    sim = SixMaxSimulator(bb_size=10.0, equity_sims=equity_sims)
    sim.opponent_pool_styles = ['nit', 'tag']
    sim.opponent_pool_weights = [0.5, 0.5]
    sim.live_players = 3
    sim.fixed_stack_bb = 100.0
    sim.disable_exploration = True
    import yaml
    _cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'config.yaml')))['training']
    sim.range_aware_equity = bool(_cfg.get('range_aware_equity', False))  # match training
    return sim


def kl(p, q):
    return sum(pi * math.log((pi + 1e-9) / (qi + 1e-9)) for pi, qi in zip(p, q))


def entropy(p):
    return -sum(pi * math.log(pi + 1e-9) for pi in p)


def part_a(model, device, n_hands=500, max_seq_len=20):
    """Compare the model's policy to the training target on the SAME live states."""
    sim = make_sim()
    # rows keyed by (street, eq_bucket_idx) -> accumulators
    agg = {}
    for h in range(n_hands):
        rec = sim.simulate_hand(current_hand=h)
        if not rec or not rec.decision_points:
            continue
        samples = T.vectorize_hand_samples(rec, max_seq_len=max_seq_len)
        if not samples:
            continue
        s = samples[0]
        hole, board, ctx, act = s[0], s[1], s[2], s[3]
        target_pi_seq = s[11]
        h_t = torch.tensor([hole], dtype=torch.long, device=device)
        b_t = torch.tensor([board], dtype=torch.long, device=device)
        c_t = torch.tensor([ctx], dtype=torch.float32, device=device)
        a_t = torch.tensor([act], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(h_t, b_t, c_t, a_t)['policy_logits'][0]     # [T,3]
            model_pi_seq = torch.softmax(logits, dim=-1).cpu().tolist()

        dps = rec.decision_points[-max_seq_len:]
        start = max_seq_len - len(dps)
        for i, dp in enumerate(dps):
            t = start + i
            street = STREETS.get(dp['street'], 'river')
            eq = dp['equity']
            bidx = next((k for k, (lo, hi) in enumerate(EQ_BUCKETS) if lo <= eq < hi), len(EQ_BUCKETS) - 1)
            key = (street, bidx)
            a = agg.setdefault(key, {'n': 0, 'm': [0.0] * K, 't': [0.0] * K, 'kl': 0.0,
                                     'mH': 0.0, 'tH': 0.0})
            mp = model_pi_seq[t]
            tp = target_pi_seq[t]
            a['n'] += 1
            for j in range(K):
                a['m'][j] += mp[j]
                a['t'][j] += tp[j]
            a['kl'] += kl(tp, mp)
            a['mH'] += entropy(mp)
            a['tH'] += entropy(tp)

    print("=" * 110)
    print(f"  PART A: MODEL policy vs TRAINING TARGET on live states  ({'/'.join(ACTION_LABELS)})")
    print(f"  target = {T.POLICY_TARGET_SOURCE} regret, temp={T.POLICY_TARGET_TEMP}")
    print("=" * 110)
    print(f"  {'street':<8} {'equity':<10} {'n':>5} | {'MODEL ' + '/'.join(ACTION_LABELS):>32} | {'TARGET ' + '/'.join(ACTION_LABELS):>32} | {'KL':>5} {'Hm':>5} {'Ht':>5}")
    for street in ['preflop', 'flop', 'turn', 'river']:
        for bidx, (lo, hi) in enumerate(EQ_BUCKETS):
            a = agg.get((street, bidx))
            if not a or a['n'] == 0:
                continue
            n = a['n']
            m = [x / n for x in a['m']]
            t = [x / n for x in a['t']]
            print(f"  {street:<8} {f'{lo:.1f}-{hi:.1f}':<10} {n:>5} | {fmt_dist(m):>32} | {fmt_dist(t):>32} | "
                  f"{a['kl']/n:>5.2f} {a['mH']/n:>5.2f} {a['tH']/n:>5.2f}")


def part_b(model, device):
    """Postflop sensitivity: does the model respond to the board?"""
    from core.board_state import BoardState, SeatState, HUDStats
    from versions.v16_foldregret.core.contract import ContractV12
    bridge = ContractV12(max_seq_len=20)
    print("\n" + "=" * 100)
    print("  PART B: POSTFLOP sensitivity -- model policy on fixed spots (facing a 20 into 40 pot)")
    print("=" * 100)
    print(f"  {'label':<34} {'equity':>6} | {'P(' + '/'.join(ACTION_LABELS) + ')':>32} | {'act(pi)':>7} | {'act(Q)':>7}")
    spots = [
        ("Air on wet board (7h2c / QsJsTs)", ["7h", "2c"], ["Qs", "Js", "Ts"], "Flop", 0.08),
        ("Weak pair (7h7d / AsKcQh)",         ["7h", "7d"], ["As", "Kc", "Qh"], "Flop", 0.22),
        ("Missed draw river (8h9h / Ac2d5sKcQd)", ["8h", "9h"], ["Ac", "2d", "5s", "Kc", "Qd"], "River", 0.04),
        ("Top pair (AhKd / Kc7h2s)",          ["Ah", "Kd"], ["Kc", "7h", "2s"], "Flop", 0.72),
        ("Set (7h7d / 7s Kc 2d)",             ["7h", "7d"], ["7s", "Kc", "2d"], "Flop", 0.88),
        ("Nut flush river (AhQh / 2h7hKh3s9d)", ["Ah", "Qh"], ["2h", "7h", "Kh", "3s", "9d"], "River", 0.95),
    ]
    for label, hole, board, street, eq in spots:
        state = BoardState(
            community_cards=board, hero_cards=hole, pot_size=40.0, hero_stack=1000.0,
            big_blind=10.0, call_amount=20.0, equity=eq, hero_position=2, street=street,
        )
        state.seats["seat_1"] = SeatState(name="Opp", stack=1000.0, is_active=True,
                                          hud=HUDStats(vpip_color="Blue", agg_color="Blue"))
        h_t, b_t, c_t, a_t = bridge.to_tensors(state, hero_actions=[6])
        with torch.no_grad():
            out = model(h_t.to(device), b_t.to(device), c_t.to(device), a_t.to(device))
            probs = torch.softmax(out['policy_logits'].squeeze(0)[-1], dim=-1).tolist()
            q = out['q_vals'].squeeze(0)[-1].tolist()
        print(f"  {label:<34} {eq:>6.2f} | {fmt_dist(probs):>32} | {argmax_label(probs):>7} | {argmax_label(q):>7}")


def part_c(model, device):
    """Equity ablation: hold hole cards + board FIXED, sweep the equity INPUT. If Q is flat,
    the model ignores equity (representation bug) OR the contract never feeds it (plumbing
    bug) -- printing the context tensor's equity slot distinguishes the two."""
    from core.board_state import BoardState, SeatState, HUDStats
    from versions.v16_foldregret.core.contract import ContractV12
    bridge = ContractV12(max_seq_len=20)
    print("\n" + "=" * 100)
    print("  PART C: equity ablation -- SAME hand/board (7h7d on 7s Kc 2d), sweep equity input")
    print("=" * 100)
    print(f"  {'equity_in':>9} {'ctx_eq':>7} | {'Q(' + '/'.join(ACTION_LABELS) + ')':>40} | {'P(' + '/'.join(ACTION_LABELS) + ')':>32}")
    for eq in [0.10, 0.30, 0.50, 0.70, 0.90]:
        state = BoardState(community_cards=["7s", "Kc", "2d"], hero_cards=["7h", "7d"],
                           pot_size=40.0, hero_stack=1000.0, big_blind=10.0, call_amount=20.0,
                           equity=eq, hero_position=2, street="Flop")
        state.seats["seat_1"] = SeatState(name="Opp", stack=1000.0, is_active=True,
                                          hud=HUDStats(vpip_color="Blue", agg_color="Blue"))
        h_t, b_t, c_t, a_t = bridge.to_tensors(state, hero_actions=[6])
        # state lands at the LAST row (left-padding); ctx[3] is the equity slot (contract L102)
        ctx_eq = c_t[0, -1, 3].item()
        with torch.no_grad():
            out = model(h_t.to(device), b_t.to(device), c_t.to(device), a_t.to(device))
            q = out['q_vals'].squeeze(0)[-1].tolist()
            p = torch.softmax(out['policy_logits'].squeeze(0)[-1], dim=-1).tolist()
        print(f"  {eq:>9.2f} {ctx_eq:>7.3f} | {fmt_dist(q):>40} | {fmt_dist(p):>32}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = os.path.join(os.path.dirname(__file__), '..', 'weights', 'expert_main.pth')
    print(f"Device: {device}  |  checkpoint: {os.path.abspath(ckpt)}")

    # Reproduce the target the model was TRAINED on, so the comparison is apples-to-apples.
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), 'config.yaml')))['training']
    T.POLICY_TARGET_SOURCE = str(cfg.get('policy_target_source', 'realized'))
    T.POLICY_TARGET_TEMP = float(cfg.get('policy_target_temperature', 1.0))
    if bool(cfg.get('disable_target_shaping', False)):
        T.TIGHTNESS_PENALTY_BB = 0.0
        T.COUNTERFACTUAL_WEIGHT = 0.0
    T.TARGET_CLIP_BB = float(cfg.get('target_clip_bb', 40.0))

    model = load_model(ckpt, device, ablate_hole_cards=bool(cfg.get('ablate_hole_cards', False)))
    part_a(model, device)
    part_b(model, device)
    part_c(model, device)


if __name__ == '__main__':
    main()

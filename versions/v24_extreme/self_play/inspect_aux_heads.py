"""
Aux-head rationality probe for V21_auxhead.

The bluff/strength/equity heads have existed in PokerEVModelV4 since early versions but always
trained at aux_loss_weight=0.0 -- a forward pass + loss computed every step, contributing exactly
zero gradient (genuinely inert since V14). V21_auxhead turns this weight on (0.05), warm-started
from V21's own converged checkpoint. This script checks whether the heads' predictions actually
track their own training labels once given real gradient, or stay indistinguishable from an
untrained head even with nonzero weight.

Method: simulate hands with the HEURISTIC pool only (hero_model=None) to get realistic, diverse
decision points and their TRUE labels already computed by the simulator (opp_bluff_prob,
opp_strength, dp['equity'] for self_equity). Reuse train.py's OWN vectorize_hand_samples on each
hand -- the exact featurization training uses -- to get context tensors AND the label sequences in
one shot, then forward-pass those tensors through the TRAINED model under test and read off
preds['bluff']/['strength']/['equity'] at the same timesteps.

The self_equity check is the cleanest signal: equity is ALSO a direct input feature (ctx[3]), so a
genuinely-wired head should learn to echo it back with near-trivial ease (correlation close to 1,
low error). If it doesn't, that's evidence of a broken gradient path, not just "a hard feature."
opp_bluff/opp_strength are noisier, cheaper proxy labels (see simulator.py's
_mc_target_evs_sized/_calculate_mc_target_evs) -- perfect accuracy isn't expected, but the
prediction should at minimum be DIRECTIONALLY correlated with the label, not flat/random.

Run:  .venv/Scripts/python.exe -m versions.v24_extreme.self_play.inspect_aux_heads --weights expert_main.pth
"""
import os
import sys
import argparse
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch

import versions.v24_extreme.self_play.train as T
from versions.v24_extreme.self_play.simulator import SixMaxSimulator
from versions.v24_extreme.core.model import PokerEVModelV4
from versions.v24_extreme.core.manifest import MANIFEST
from shared.manifest import load_state_dict as load_ckpt_state


def load_model(path, device):
    m = PokerEVModelV4().to(device)
    m.load_state_dict(load_ckpt_state(path, MANIFEST))
    m.eval()
    return m


def make_sim(equity_sims=200):
    """Heuristic-only rollout (hero_model=None) -- realistic diverse states, decoupled from
    whichever model we're probing. Mirrors the trained recipe's pool/curriculum settings closely
    enough to generate representative decision points (range_aware_equity ON, mixed stack depths)."""
    sim = SixMaxSimulator(bb_size=10.0, equity_sims=equity_sims)
    sim.opponent_pool_styles = ['tag', 'nit', 'maniac', 'fish']
    sim.opponent_pool_weights = [0.3, 0.25, 0.25, 0.2]
    sim.live_players = 6
    sim.stack_depth_mix = [[5, 14, 0.5], [14, 30, 0.3], [30, 50, 0.2]]
    sim.disable_exploration = True
    sim.range_aware_equity = True
    return sim


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float('nan')
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return float('nan')
    return cov / math.sqrt(vx * vy)


def mae(xs, ys):
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)


def collect(model, device, n_hands, max_seq_len=20):
    """Run heuristic-only hands, vectorize each with train.py's own function (so featurization is
    IDENTICAL to what training used), forward-pass the model under test, and pair up
    (prediction, label) for every real decision point (loss_mask==1)."""
    sim = make_sim()
    rows = {'bluff': [], 'strength': [], 'equity': []}
    n_collected = 0
    for h in range(n_hands):
        rec = sim.simulate_hand(current_hand=h)
        if not rec or not rec.decision_points:
            continue
        samples = T.vectorize_hand_samples(rec, max_seq_len=max_seq_len)
        if not samples:
            continue
        hole, board, ctx, act, _, _, loss_mask, opp_bluff, opp_strength, self_eq, _, _ = samples[0]
        h_t = torch.tensor([hole], dtype=torch.long, device=device)
        b_t = torch.tensor([board], dtype=torch.long, device=device)
        c_t = torch.tensor([ctx], dtype=torch.float32, device=device)
        a_t = torch.tensor([act], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(h_t, b_t, c_t, a_t)
            pred_bluff = out['bluff'][0].cpu().tolist()
            pred_strength = out['strength'][0].cpu().tolist()
            pred_equity = out['equity'][0].cpu().tolist()
        for t in range(len(loss_mask)):
            if loss_mask[t] < 0.5:
                continue
            rows['bluff'].append((pred_bluff[t], opp_bluff[t]))
            rows['strength'].append((pred_strength[t], opp_strength[t]))
            rows['equity'].append((pred_equity[t], self_eq[t]))
            n_collected += 1
    return rows, n_collected


def report(rows, n_collected):
    print("=" * 90)
    print(f"  AUX-HEAD RATIONALITY PROBE  ({n_collected} live decision points)")
    print("=" * 90)
    print(f"  {'head':<10} {'label':<16} {'n':>6} {'corr(r)':>9} {'MAE':>7}  {'pred mean/std':>16}  {'label mean/std':>16}")
    for head, label_name in (('equity', 'self_equity (ctx[3])'), ('strength', 'opp_strength'), ('bluff', 'opp_bluff_prob')):
        pairs = rows[head]
        if not pairs:
            print(f"  {head:<10} (no data)")
            continue
        preds = [p for p, _ in pairs]
        labels = [l for _, l in pairs]
        n = len(pairs)
        r = pearson(preds, labels)
        m = mae(preds, labels)
        pmean = sum(preds) / n
        pstd = math.sqrt(sum((x - pmean) ** 2 for x in preds) / n)
        lmean = sum(labels) / n
        lstd = math.sqrt(sum((x - lmean) ** 2 for x in labels) / n)
        print(f"  {head:<10} {label_name:<16} {n:>6} {r:>9.3f} {m:>7.3f}  {pmean:>7.3f}/{pstd:<7.3f}  {lmean:>7.3f}/{lstd:<7.3f}")
    print("-" * 90)
    print("  Reading this:")
    print("  * equity/self_equity is the cleanest test -- ctx[3] IS the label, so a genuinely")
    print("    wired head should show r close to 1.0 and MAE close to 0. If r is near 0 despite")
    print("    aux_loss_weight>0 and real training hands, the gradient path is likely broken, not")
    print("    just 'this is hard to learn'.")
    print("  * strength/bluff are noisier proxy labels -- perfect correlation isn't expected, but")
    print("    r meaningfully above 0 (not flat/near-zero) and a pred std NOT collapsed to ~0")
    print("    (i.e. not just predicting the label's mean for everything) is the bar for 'rational'.")


def main():
    ap = argparse.ArgumentParser(description="Aux-head rationality probe for V21_auxhead.")
    ap.add_argument('--weights', type=str, default='expert_main.pth')
    ap.add_argument('--n-hands', type=int, default=800)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    weights_dir = os.path.join(os.path.dirname(__file__), '..', 'weights')
    path = os.path.join(weights_dir, args.weights)
    print(f"Loading {path} ...")
    model = load_model(path, args.device)

    print(f"Collecting {args.n_hands} heuristic-only hands and forward-passing through the model...")
    rows, n = collect(model, args.device, args.n_hands)
    report(rows, n)


if __name__ == '__main__':
    main()

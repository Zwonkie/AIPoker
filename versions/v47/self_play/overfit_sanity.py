"""
Overfit sanity test for the V12-D training loop.

Purpose: verify the CORE learning path (vectorize_hand_samples -> model forward -> loss
-> backward -> optimizer) is wired correctly, INDEPENDENT of self-play dynamics,
curriculum, and target shaping. We repeatedly train on ONE tiny fixed batch and check
whether the loss collapses. A correctly wired loop must be able to memorize a handful of
examples; if it cannot, nothing downstream matters.

Two modes:
  A. SYNTHETIC targets -- a clean, deterministic function of the equity INPUT feature.
     The model is literally handed equity, so it MUST fit these to ~0 if the network has
     capacity and gradients flow end-to-end. Failure here == a hard plumbing/gradient bug
     (bad tensor shapes, detached graph, masking wrong, equity feature not connected...).
  B. REAL targets -- the batch's actual MC / counterfactual EV targets. How low the loss
     floors tells us whether the real targets are internally consistent (learnable) or
     noisy/contradictory (a DATA problem, not a wiring problem).

Run:  .venv/Scripts/python.exe -m versions.v47.self_play.overfit_sanity
"""
import os
import sys
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch
import torch.nn as nn

from versions.v47.core.model import PokerEVModelV4
from versions.v47.self_play.simulator import SixMaxSimulator
from versions.v47.self_play.train import vectorize_hand_samples

# Sample tuple layout produced by vectorize_hand_samples (see train.py):
#  0 hole  1 board  2 ctx  3 action_seq  4 action_taken  5 target_evs
#  6 loss_mask  7 bluff  8 strength  9 self_equity  10 target_w  11 policy_target
PASS_THRESHOLD_BB = 1.0   # synthetic-mode final mean|Q-target| under this == wiring OK


def build_fixed_batch(n_samples=64, equity_sims=100, seed=0):
    """Generate one small FIXED batch of vectorized samples from the simplest environment
    (heads-up vs a single static Nit). Seeded so the batch is identical every run."""
    random.seed(seed)
    sim = SixMaxSimulator(bb_size=10.0, equity_sims=equity_sims)
    sim.opponent_pool_styles = ['nit']
    sim.opponent_pool_weights = [1.0]
    sim.live_players = 2  # Hero + 1 opponent
    samples = []
    h = 0
    while len(samples) < n_samples and h < n_samples * 30:
        rec = sim.simulate_hand(current_hand=h)
        h += 1
        if rec and rec.decision_points:
            samples.extend(vectorize_hand_samples(rec))
    return samples[:n_samples]


def stack_tensors(samples, device):
    def T(idx, dtype):
        return torch.tensor([s[idx] for s in samples], dtype=dtype, device=device)
    return {
        'hole': T(0, torch.long),
        'board': T(1, torch.long),
        'ctx': T(2, torch.float32),
        'act': T(3, torch.long),
        'target_evs': T(5, torch.float32),
        'mask': T(6, torch.float32),
        'equity': T(9, torch.float32),
    }


def synthetic_targets(equity):
    """Clean monotonic EV as a function of equity only, over the V14 6-action space:
    fold=0, call ramps with equity, and each raise size ramps STEEPER as it gets bigger
    (bigger bet = more EV at high equity, more -EV at low). Because equity is a model INPUT,
    this is trivially fittable for a healthy network -- that is exactly the point."""
    fold_ev = torch.zeros_like(equity)
    call_ev = 20.0 * (equity - 0.50)
    r33_ev = 25.0 * (equity - 0.47)
    r66_ev = 30.0 * (equity - 0.45)
    rpot_ev = 35.0 * (equity - 0.43)
    allin_ev = 45.0 * (equity - 0.40)
    return torch.stack([fold_ev, call_ev, r33_ev, r66_ev, rpot_ev, allin_ev], dim=-1)


def overfit(mode, batch, steps=600, lr=1e-3):
    device = batch['hole'].device
    mask = batch['mask']
    m3 = mask.unsqueeze(-1)
    denom = m3.sum().clamp(min=1.0)

    target_q = synthetic_targets(batch['equity']) if mode == 'synthetic' else batch['target_evs']
    with torch.no_grad():
        target_pi = torch.softmax(target_q, dim=-1)

    torch.manual_seed(0)
    model = PokerEVModelV4().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    huber = nn.HuberLoss(reduction='none', delta=2.0)

    live = int(mask.sum().item())
    print(f"\n[{mode.upper()}] {batch['hole'].shape[0]} hands / {live} live decision steps")
    model.train()
    for step in range(steps):
        opt.zero_grad()
        out = model(batch['hole'], batch['board'], batch['ctx'], batch['act'])
        q = out['q_vals']
        logp = nn.functional.log_softmax(out['policy_logits'], dim=-1)
        loss_q = (huber(q, target_q) * m3).sum() / denom
        loss_pi = (-(target_pi * logp).sum(-1) * mask).sum() / mask.sum().clamp(min=1.0)
        loss = loss_q + loss_pi
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"  step {step:4d} | loss {loss.item():8.4f} | Q {loss_q.item():8.4f} | Pi {loss_pi.item():7.4f}")

    model.eval()
    with torch.no_grad():
        out = model(batch['hole'], batch['board'], batch['ctx'], batch['act'])
        q = out['q_vals']
        mae = ((q - target_q).abs() * m3).sum().item() / denom.item()
        # Policy fit: cross-entropy floors at H(target), NOT 0, so raw CE looks "stuck"
        # even at a perfect fit. Measure KL(target || pred) instead -- that goes to 0 iff
        # the actor actually matched the target distribution.
        pred_logp = nn.functional.log_softmax(out['policy_logits'], dim=-1)
        tgt_logp = torch.log(target_pi.clamp(min=1e-9))
        h_target = (-(target_pi * tgt_logp).sum(-1) * mask).sum().item() / mask.sum().clamp(min=1.0).item()
        kl = ((target_pi * (tgt_logp - pred_logp)).sum(-1) * mask).sum().item() / mask.sum().clamp(min=1.0).item()
    print(f"  FINAL mean|Q - target|      : {mae:.4f} bb")
    print(f"  FINAL policy KL(target||pred): {kl:.4f}  (target entropy floor H={h_target:.4f})")
    return mae, kl


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    samples = build_fixed_batch()
    if not samples:
        print("FAILED: could not build a batch (no decision points generated).")
        sys.exit(2)
    batch = stack_tensors(samples, device)

    syn_mae, syn_kl = overfit('synthetic', batch)
    real_mae, real_kl = overfit('real', batch)

    print("\n" + "=" * 60)
    print("  OVERFIT SANITY VERDICT")
    print("=" * 60)
    critic_ok = syn_mae < PASS_THRESHOLD_BB
    actor_ok = syn_kl < 0.05
    print(f"  Critic (Q) wiring : synth |Q-target| {syn_mae:.4f} bb  -> "
          f"{'PASS' if critic_ok else 'FAIL (plumbing/gradient bug)'}")
    print(f"  Actor (policy)    : synth KL(target||pred) {syn_kl:.4f}  -> "
          f"{'PASS (actor fits its target)' if actor_ok else 'FAIL (actor head not learning)'}")
    print(f"  Real targets      : |Q-target| {real_mae:.4f} bb, KL {real_kl:.4f}  -> "
          f"{'learnable' if real_mae < 3.0 else 'noisy/contradictory (data issue)'}")
    print("=" * 60)
    sys.exit(0 if (critic_ok and actor_ok) else 1)


if __name__ == '__main__':
    main()

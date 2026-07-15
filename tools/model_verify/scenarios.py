"""Synthetic scenario builders for the FAST model_verify checks.

Builds raw 35-feature context vectors directly at the index layout shared by every
sized-contract version (ContractV12: see versions/v16/core/contract.py:98-133 and
versions/v16/self_play/train.py:208-231 -- both must stay in lockstep with this layout).
Bypassing BoardState lets a check specify exactly the equity/stack/price it wants to test
without constructing a full game-state object graph.

CAVEAT: these checks drive the model through its CONTEXT features (equity, pot odds, stack,
opponent HUD) with placeholder (padding) hole/board cards, not real dealt cards. That's
sufficient to test the pathway that matters for these regressions (does the model respond
sanely to its own equity/stack/opponent-tightness inputs) but does NOT exercise the separate
raw-hole-card embedding the model also has access to. A future addition could layer in real
cards via the treys evaluator for higher-fidelity spot checks; not needed for what's below.
"""
import torch

MAX_SEQ_LEN = 20
PAD_CARD = 52

# Defaults for an inactive/unspecified opponent seat, matching ContractV12's own fallback
# (versions/v16/core/contract.py:122-124) so unfilled seats look like the training distribution's
# "no info" case rather than an out-of-distribution zero.
_DEFAULT_OPP_VPIP = 0.30
_DEFAULT_OPP_AGG = 0.40


def build_ctx(equity, stack_bb, pot_bb, call_bb, position=2, street=0,
              num_active_opp=2, opp_vpip=0.30, opp_agg=0.40,
              per_opp_vpip=None, per_opp_agg=None, opp_stack_bb=None):
    """One 35-length context vector, index layout identical to ContractV12.to_tensors."""
    pot_odds = call_bb / (pot_bb + call_bb) if (pot_bb + call_bb) > 0 else 0.0
    ctx = [
        position / 5.0,
        stack_bb / 400.0,
        pot_bb / 1000.0,
        equity,
        pot_odds,
        num_active_opp / 10.0,
        street / 3.0,
        opp_vpip, opp_agg,
        call_bb / 400.0,
    ]
    opp_stack_bb = stack_bb if opp_stack_bb is None else opp_stack_bb
    for j in range(5):
        active = 1.0 if j < num_active_opp else 0.0
        v = (per_opp_vpip[j] if per_opp_vpip and j < len(per_opp_vpip) else opp_vpip) if active else _DEFAULT_OPP_VPIP
        a = (per_opp_agg[j] if per_opp_agg and j < len(per_opp_agg) else opp_agg) if active else _DEFAULT_OPP_AGG
        pos_val = (float((j + 1 + position) % 6) / 5.0) if active else -1.0
        ctx.append(active)
        ctx.append(pos_val)
        ctx.append((opp_stack_bb / 400.0) if active else 0.0)
        ctx.append(v)
        ctx.append(a)
    assert len(ctx) == 35, f"context vector drifted from the 35-feature contract: len={len(ctx)}"
    return ctx


def build_tensors(ctx, hole=None, board=None, max_seq_len=MAX_SEQ_LEN):
    """Wrap a single decision's ctx (+ optional hole/board) into batch-first [1, seq, ...]
    tensors, LEFT-padded so the real decision sits at the final timestep -- the same
    convention ContractV12.to_tensors uses for a fresh (no-history) decision."""
    hole = hole or [PAD_CARD, PAD_CARD]
    board = board or [PAD_CARD] * 5
    board_seq = [[PAD_CARD] * 5 for _ in range(max_seq_len)]
    context_seq = [[0.0] * 35 for _ in range(max_seq_len)]
    board_seq[-1] = board
    context_seq[-1] = ctx
    act_ints = [0] * max_seq_len
    hole_t = torch.tensor([hole], dtype=torch.long)
    board_t = torch.tensor([board_seq], dtype=torch.long)
    ctx_t = torch.tensor([context_seq], dtype=torch.float32)
    act_t = torch.tensor([act_ints], dtype=torch.long)
    return hole_t, board_t, ctx_t, act_t


def run_policy(model, ctx, action_keys, hole=None, board=None, device="cpu"):
    """Forward pass -> (policy probs dict, q-value dict) keyed by action_keys."""
    hole_t, board_t, ctx_t, act_t = build_tensors(ctx, hole=hole, board=board)
    with torch.no_grad():
        out = model(hole_t.to(device), board_t.to(device), ctx_t.to(device), act_t.to(device))
    logits = out["policy_logits"][0, -1, :]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    q = out["q_vals"][0, -1, :].cpu().numpy()
    policy = {k: float(probs[i]) for i, k in enumerate(action_keys)}
    qvals = {k: float(q[i]) for i, k in enumerate(action_keys)}
    return policy, qvals

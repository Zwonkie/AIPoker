"""Synthetic scenario builders for the FAST model_verify checks.

Builds raw context vectors directly at the index layout shared by every sized-contract version
(ContractV12: see versions/v16/core/contract.py:98-133 and versions/v16/self_play/train.py:208-231
-- both must stay in lockstep with this layout). Bypassing BoardState lets a check specify exactly
the equity/stack/price it wants to test without constructing a full game-state object graph.

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


def _money_scale(contract_version):
    """Money-denominated feature (hero/opp stack, pot, call_amount) scale, keyed by
    contract_version -- MUST match whichever version is under test.

    contract_version <= 3 (v13-v19): legacy, uncapped /400 (stack, call) and /1000 (pot).
    contract_version >= 4 (V20+): stack/call clamped to 50bb, pot to 100bb, THEN /100
    (stack, call) or /250 (pot) -- see versions/v20/core/contract.py.

    FIX (2026-07-17, found while extending model_verify for v20_preflopEq): this function
    previously didn't exist -- build_ctx hardcoded the legacy /400,/1000 math unconditionally,
    so every FAST check run against V20 (contract_version=4, deployed live) had been feeding it
    systematically wrong-scale (~4x off) stack/pot/call inputs since V20 shipped, e.g.
    `deep_stack_ood_guard`'s stack_bb=15..40 sweep landed at ctx[1]=0.0375-0.10 when V20's own
    training/serving pipeline would compute min(stack_bb,50)/100=0.15-0.40 for those same real
    depths -- the check was silently exercising a different, out-of-distribution neighborhood
    than the one it claims to test. SLOW checks (which run the real simulator/contract.py) were
    NOT affected, only these FAST synthetic-context ones."""
    if contract_version >= 4:
        return dict(stack_ceil=50.0, pot_ceil=100.0, call_ceil=50.0,
                    stack_scale=100.0, pot_scale=250.0, call_scale=100.0)
    return dict(stack_ceil=None, pot_ceil=None, call_ceil=None,
                stack_scale=400.0, pot_scale=1000.0, call_scale=400.0)


def _scaled(value_bb, ceil, scale):
    v = value_bb if ceil is None else min(value_bb, ceil)
    return v / scale


def build_ctx(equity, stack_bb, pot_bb, call_bb, position=2, street=0,
              num_active_opp=2, opp_vpip=0.30, opp_agg=0.40,
              per_opp_vpip=None, per_opp_agg=None, opp_stack_bb=None,
              contract_version=3, hand_strength=None, equity_edge=None):
    """One context vector (35-length for contract_version<=4, 37-length for >=5), index layout
    identical to ContractV12.to_tensors for the matching contract_version.

    `contract_version`: MUST be the manifest.contract_version of the model under test (see
    `_money_scale`) -- defaults to 3 (the legacy scale used by v13-v19) ONLY for backward
    compatibility with call sites that predate this parameter; every FAST_CHECKS call site now
    passes `rc.manifest.contract_version` explicitly.

    `hand_strength`/`equity_edge`: only meaningful (appended) when contract_version>=5
    (V20_preflopEq's 37-feature contract -- see versions/v20_preflopEq/core/contract.py).
    `equity_edge` auto-derives from `equity*(num_active_opp+1)` (its real-play formula) unless
    explicitly overridden; `hand_strength` defaults to neutral (0.5) unless explicitly overridden
    -- pass an explicit value to probe the model's sensitivity to it in isolation (a synthetic
    ablation: in real play it's never decoupled from equity, but perturbing it alone here tests
    whether the network actually reads that ctx slot)."""
    scale = _money_scale(contract_version)
    pot_odds = call_bb / (pot_bb + call_bb) if (pot_bb + call_bb) > 0 else 0.0
    ctx = [
        position / 5.0,
        _scaled(stack_bb, scale['stack_ceil'], scale['stack_scale']),
        _scaled(pot_bb, scale['pot_ceil'], scale['pot_scale']),
        equity,
        pot_odds,
        num_active_opp / 10.0,
        street / 3.0,
        opp_vpip, opp_agg,
        _scaled(call_bb, scale['call_ceil'], scale['call_scale']),
    ]
    opp_stack_bb = stack_bb if opp_stack_bb is None else opp_stack_bb
    for j in range(5):
        active = 1.0 if j < num_active_opp else 0.0
        v = (per_opp_vpip[j] if per_opp_vpip and j < len(per_opp_vpip) else opp_vpip) if active else _DEFAULT_OPP_VPIP
        a = (per_opp_agg[j] if per_opp_agg and j < len(per_opp_agg) else opp_agg) if active else _DEFAULT_OPP_AGG
        pos_val = (float((j + 1 + position) % 6) / 5.0) if active else -1.0
        ctx.append(active)
        ctx.append(pos_val)
        ctx.append(_scaled(opp_stack_bb, scale['stack_ceil'], scale['stack_scale']) if active else 0.0)
        ctx.append(v)
        ctx.append(a)

    if contract_version >= 5:
        eff_equity_edge = equity * (num_active_opp + 1) if equity_edge is None else equity_edge
        eff_hand_strength = 0.5 if hand_strength is None else hand_strength
        ctx.append(eff_equity_edge)
        ctx.append(eff_hand_strength)

    expected_len = 37 if contract_version >= 5 else 35
    assert len(ctx) == expected_len, f"context vector drifted from the contract: len={len(ctx)}, expected {expected_len}"
    return ctx


def build_tensors(ctx, hole=None, board=None, max_seq_len=MAX_SEQ_LEN):
    """Wrap a single decision's ctx (+ optional hole/board) into batch-first [1, seq, ...]
    tensors, LEFT-padded so the real decision sits at the final timestep -- the same
    convention ContractV12.to_tensors uses for a fresh (no-history) decision."""
    hole = hole or [PAD_CARD, PAD_CARD]
    board = board or [PAD_CARD] * 5
    context_dim = len(ctx)
    board_seq = [[PAD_CARD] * 5 for _ in range(max_seq_len)]
    context_seq = [[0.0] * context_dim for _ in range(max_seq_len)]
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

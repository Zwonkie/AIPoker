"""
Self-Play Reinforcement Learning Training Script with Live Dashboard for V8.
Implements Diversity-Based League Training, Stack Curriculum Learning,
Heuristic Bootstrap Decay, and Corrected Preflop EV Target Calculations.
"""
import os
import sys
import time
import csv
import argparse
import random
from multiprocessing import Pool

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import yaml

class Tee:
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from versions.v21_auxhead.core.model import PokerEVModelV4
from versions.v21_auxhead.self_play.simulator import SixMaxSimulator, OPPONENT_POOL_STYLES, OPPONENT_POOL_WEIGHTS, STYLE_SLOT
from versions.v21_auxhead.core.manifest import MANIFEST
from versions.v21_auxhead.self_play.opponents import build_opponent_pool, describe_pool_entry
from shared.manifest import save_checkpoint, load_state_dict as load_ckpt_state
# [V20_preflopEq bug fix] vectorize_hand_samples builds the ACTUAL gradient-training context
# tensors and, before this fix, never received V20's own /100,/250 rescale (it kept a separate,
# stale /400,/1000 copy of the same math) -- see MANIFEST's docstring. Importing the scale
# helpers from contract.py (the single source ALL other paths already use: self-play rollout
# decisions via simulator.py, and live serving via core/decision.py) instead of re-deriving them
# here removes the duplication that let that drift happen in the first place.
from versions.v21_auxhead.core.contract import (
    scaled_stack_bb, scaled_pot_bb, scaled_call_bb, equity_edge_feature,
)

# [V18] Fallback pool config if config.yaml's `opponents.pool` isn't declarative-format (a plain
# list of style-name strings, the pre-V18 format) or is omitted entirely -- pure heuristic, no
# models, matching the simulator's own long-standing module-level defaults.
DEFAULT_OPPONENT_POOL_CONFIG = [
    {'style': s, 'weight': w} for s, w in zip(OPPONENT_POOL_STYLES, OPPONENT_POOL_WEIGHTS)
]

# Must match ml_bridge.py vocabulary
VOCAB = {'<PAD>': 0, 'B': 1, 'b': 2, 'c': 3, 'k': 4, 'K': 5, 'r': 6, 'f': 7, 'A': 8, 'Q': 9}

def card_to_int(card_str):
    if not card_str or len(card_str) != 2:
        return 52
    rank, suit = card_str[0], card_str[1]
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    try:
        r = ranks.index(rank.upper())
        s = suits.index(suit.lower())
        return s * 13 + r
    except ValueError:
        return 52

def map_vpip_to_midpoint(v):
    if v < 0.18: return 0.10     # Blue
    elif v < 0.26: return 0.22   # Green
    elif v < 0.35: return 0.30   # Yellow
    else: return 0.45            # Red

def map_agg_to_midpoint(a):
    if a < 0.36: return 0.18     # Blue
    elif a < 0.56: return 0.46   # Green
    elif a < 0.71: return 0.63   # Yellow
    else: return 0.85            # Red

# --- Q-target shaping constants (anti loose-collapse) ---
# TARGET_CLIP_BB: hard clip on the go-forward MC target (was 100bb). A tighter clip
#   damps the fat right tail (occasionally stacking a fish for 100bb+) that biased the
#   "enter" Q-values upward and drove the VPIP ratchet.
TARGET_CLIP_BB = 40.0
# [V21] The critic-side preflop tightness prior (TIGHTNESS_PENALTY_BB/ENTRY_EQUITY_MARGIN, gated
# by config's `disable_target_shaping`) has been DELETED outright, not just left disabled --
# superseded by range-aware equity (dp['equity'] is already VPIP-conditioned once
# range_aware_equity=true, which every recipe since V15/V16 runs), and validated dormant
# (disable_target_shaping: true) across V17/V17_gauntlet/V19/V20/V20_preflopEq/V20_preflopEq_AI.
# See versions/v21/SPECS.md item 1.
# Weight applied to the UNTAKEN actions' counterfactual (model-free MC) EV targets.
# The taken action (realized return) and the fold baseline keep full weight 1.0; the
# untaken heads get a fractional weight since their targets are estimates, not ground
# truth. This supplies the "raising/calling air is -EV" signal that a taken-action-only
# loss omits (which caused the 200k model to hallucinate positive Raise EV everywhere).
COUNTERFACTUAL_WEIGHT = 0.5
# Weight on the actor (policy) loss relative to the critic (Q) loss. The policy head is
# what actually selects actions at play time, so it carries a full-strength gradient.
POLICY_LOSS_WEIGHT = 1.0
# Actor-sharpening (v12d). The regret-matching policy target is intrinsically high-entropy,
# so with the anti-ratchet priors off the actor stays near-uniform and enters ~70% of hands
# even though its argmax is directionally correct. POLICY_TARGET_TEMP < 1.0 sharpens the
# TARGET distribution before the cross-entropy loss (temp=1.0 is a no-op). The complementary
# entropy-penalty lever (coefficient on +H(pred) in the actor loss) is a run_training arg.
POLICY_TARGET_TEMP = 1.0
# Source of the values fed to the regret-matching POLICY target:
#  'realized'      -> taken action uses its REALIZED go-forward return (legacy). Against a
#                     folding field this reinforces weak entries that won uncontested (the
#                     fold-equity ratchet: inspect_ev_targets Part 2 showed +7.82bb realized
#                     vs -4.04bb counterfactual for weak-hand raises).
#  'counterfactual'-> use the well-calibrated all-action counterfactual EVs (fold=0, weak
#                     entries correctly negative). Removes the on-policy survivorship bias.
# NOTE: the CRITIC always keeps the realized return on its taken-action head; this only
# changes what the ACTOR regresses toward.
POLICY_TARGET_SOURCE = 'realized'
# Realization discount on the counterfactual POLICY target. The counterfactual EVs use all-in
# equity (no multi-street realization), so they overvalue speculative entries -> the model
# learns a structurally loose style (loses to tight fields, VPIP 60%+). This penalizes the
# call/raise policy-target values for sub-pivot-equity hands, folding edges that won't be
# realized. bb units; 0 disables. Only affects the ACTOR target, not the critic.
POLICY_TIGHTNESS_BB = 0.0
POLICY_TIGHTNESS_PIVOT = 0.45


def _realization_discount(equity, pivot, magnitude):
    """[V21] SINGLE shared formula for the actor's realization discount -- how much to subtract
    from a voluntary (non-fold) action's target when hero's equity is below `pivot`. Previously
    implemented twice independently (inline in `vectorize_hand_samples` for the scalar/per-hand
    pre-cutover path, and again inline in `regret_match_policy_torch` for the batched/tensor
    post-cutover path), with only a comment asserting the two matched -- if `POLICY_TIGHTNESS_BB`/
    `POLICY_TIGHTNESS_PIVOT` were ever retuned, nothing would stop them silently drifting apart.
    Both call sites now call this one function; it dispatches on whether `equity` is a plain float
    (scalar path) or a tensor (batched path) so the same formula genuinely runs both places rather
    than being duplicated. See versions/v21/SPECS.md item 5.
    """
    if magnitude <= 0.0:
        return torch.zeros_like(equity) if torch.is_tensor(equity) else 0.0
    if torch.is_tensor(equity):
        below_pivot = (pivot - equity).clamp(min=0.0)
    else:
        below_pivot = max(0.0, pivot - equity)
    return magnitude * below_pivot / pivot


def sharpen_distribution(probs, temp):
    """Raise each prob to 1/temp and renormalize. temp<1 sharpens (peaks the argmax),
    temp>1 flattens, temp==1 is identity. Preserves the action ordering."""
    if temp == 1.0:
        return probs
    powed = [p ** (1.0 / temp) for p in probs]
    s = sum(powed)
    if s <= 1e-12:
        return probs
    return [p / s for p in powed]


def regret_match_policy(action_values):
    """One-step regret-matching policy target over [fold, call, raise] action values.

    Regret of an action = how much its value beats the uniform-strategy value (the mean
    of the action values). The target policy is proportional to POSITIVE regret; if no
    action beats the mean (a degenerate tie) it falls back to uniform. This is the CFR
    building block, and unlike argmax(Q) it yields a normalized distribution — the fix
    for the V11 raise-/call-everything collapse.
    """
    mean_v = sum(action_values) / len(action_values)
    regrets = [max(v - mean_v, 0.0) for v in action_values]
    total = sum(regrets)
    if total <= 1e-9:
        n = len(action_values)
        return [1.0 / n] * n
    return [r / total for r in regrets]


# [V17] Actor-critic cutover: same milestone bootstrap_alpha already uses to finish its decay
# (see run_training's `elif hands_done < 20000` block) -- reused deliberately, not a new tuned
# constant, so the actor switches value sources exactly when the critic stops being propped up
# by the heuristic-anchored rollout and has had a full bootstrap-decay's worth of training.
# [V21] was 30000; shifted 10k earlier alongside the bootstrap decay (5k-20k, was 10k-30k).
ACTOR_CRITIC_CUTOVER_HANDS = 20000


def regret_match_policy_torch(action_values, equity=None, baseline_mode='mean'):
    """Batched torch equivalent of `regret_match_policy`, for use INSIDE the training loop against
    the model's own (detached) critic q_vals instead of the dataset's precomputed per-hand target.

    [V17 round 2] `baseline_mode` selects the regret baseline:
      - 'mean' (round 1, matches the dataset-time `regret_match_policy` / every prior version):
        regret measured against the mean of all K action values. DIAGNOSED (2026-07-15, comparing
        the 25,668- and 51,579-hand checkpoints) to structurally dilute FOLD's share even against a
        well-calibrated critic: one steeply-negative outlier action (ALLIN at weak equity) drags the
        shared mean down far enough that CALL/small-raise -- objectively worse than FOLD -- still
        clears that diluted mean and keeps real probability mass. Confirmed the critic itself was
        NOT the problem (it ranked FOLD correctly, increasingly confidently, over training) --
        the MEAN-BASELINE FORMULA is.
      - 'fold': regret measured against action_values[...,0] (FOLD, always ~0 by construction, same
        idea as v16_foldregret). Tested directly against the SAME 51,579-hand checkpoint's critic
        Q-values offline before building this: gave a clean 100% FOLD at weak equity, vs ~29% under
        'mean' on the identical Q-vector. v16_foldregret tried this baseline on NOISY raw per-hand
        `mc_evs` and it broke deep-stack style-adaptation (over-corrected, general fold-bias) --
        the hypothesis this round tests is that it was the NOISE being fold-anchored that caused the
        overcorrection, not the fold-relative baseline itself; a DENOISED critic Q should be safe to
        fold-anchor without that side effect. Used ONLY post-cutover (see below) -- pre-cutover still
        uses 'mean' via the dataset's b_pol, exactly as V16/every prior version did, so the noisy-raw
        regime is untouched and this stays a single-variable change vs the V17-round-1 mechanism.

    [V21] Applies the SAME realization discount `vectorize_hand_samples` applies to the
    dataset-time actor target (via the `equity` context feature) through the single shared
    `_realization_discount` helper -- so baseline_mode is the only thing this changes relative to
    round 1, not a second simultaneous change, and the two paths can't drift out of sync.

    action_values: [B, T, K] tensor (expected DETACHED by the caller -- this function does not
        detach; the actor's cross-entropy loss must not backprop into the critic's own weights
        through this path).
    equity: optional [B, T] tensor (the ctx[3] equity feature) to apply the realization discount.
    Returns: [B, T, K] normalized target distribution (uniform fallback where nothing beats fold).
    """
    K = action_values.shape[-1]
    values = action_values.clone()
    if equity is not None:
        pen = _realization_discount(equity, POLICY_TIGHTNESS_PIVOT, POLICY_TIGHTNESS_BB)  # [B, T]
        values[..., 1:] = values[..., 1:] - pen.unsqueeze(-1)
    if baseline_mode == 'fold':
        baseline = values[..., 0:1]
    else:
        baseline = values.mean(dim=-1, keepdim=True)
    regrets = (values - baseline).clamp(min=0.0)
    total = regrets.sum(dim=-1, keepdim=True)
    matched = regrets / total.clamp(min=1e-9)
    if baseline_mode == 'fold':
        # Degenerate tie (nothing beats folding): fold outright, matching v16_foldregret's scalar
        # fallback -- mixing uniformly across options that are all at-or-below the zero-risk
        # baseline made no sense there and doesn't here either.
        fallback = torch.zeros_like(values)
        fallback[..., 0] = 1.0
    else:
        fallback = torch.full_like(values, 1.0 / K)
    return torch.where(total > 1e-9, matched, fallback)


def _bluff_pos_weight(b_bluff, b_m, cap=20.0):
    """[V21_auxhead, 2026-07-17] `opp_bluff_prob` is a genuinely rare positive label after the
    same-day fix (gated on a specific last-raiser's own equity, not "any weak opponent at the
    table") -- observed positive rate ~2-4% across both eval runs. Plain MSE trivially minimizes
    by predicting near-zero for everything, which is exactly what Phase 2 showed (pred std
    collapsed to ~0.02 vs the label's own ~0.14-0.16, correlation r=0.08, barely above the
    Phase-1-with-the-OLD-broken-label result of r=0.13 despite a full 100k-hand run).

    [Phase 3 finding] The FULL inverse-frequency ratio ((1-p)/p, ~49x at a 2% positive rate)
    overcorrected: prediction std stopped collapsing (0.02 -> 0.25, no longer just predicting
    near-zero) but the model swung to systematically OVER-predicting instead (mean 0.30 vs the
    true ~2% base rate) -- fixed the collapse, broke calibration. Phase 4 dampens via sqrt of the
    ratio instead of the raw ratio (~7x rather than ~49x at 2%) -- a standard adjustment for
    exactly this "correct but too strong" overshoot, still computed PER BATCH from the batch's own
    observed positive rate rather than a hand-picked constant.

    Returns a per-timestep weight tensor: 1.0 where the label is 0, `pos_weight` where it's 1."""
    with torch.no_grad():
        n = b_m.sum().clamp(min=1.0)
        pos_rate = ((b_bluff * b_m).sum() / n).clamp(min=1e-3, max=0.5)
        pos_weight = torch.sqrt((1.0 - pos_rate) / pos_rate).clamp(max=cap)
    return 1.0 + b_bluff * (pos_weight - 1.0)


def vectorize_hand_samples(record, max_seq_len=20, compute_policy_target=True):
    """Convert a HandRecordV4 into sequence tensors for V8.

    [V21] `compute_policy_target=False` skips the regret-matching actor-target computation below
    (`policy_target_seq` stays at its uniform-fallback default) -- pass False once
    `hands_done >= ACTOR_CRITIC_CUTOVER_HANDS`, since past that point the training loop computes
    the actor target live from the critic (`regret_match_policy_torch`) and never reads this
    dataset-time value. Previously this was computed (real per-hand regret-matching + sharpening)
    and then silently discarded for ~87% of a typical 150k-hand run. See SPECS.md item 5.
    """
    dps = record.decision_points
    if not dps:
        return []
        
    hole_ints = [card_to_int(c) for c in record.hero_cards]
    while len(hole_ints) < 2:
        hole_ints.append(52)
        
    # V14: K-action space [fold, call, raise_0..raise_{K-3}]. Inferred from the sim's per-size EV
    # target so it stays in lockstep with raise_pot_fractions (fallback 6 = 3 raise sizes + all-in).
    K = len(dps[0].get('target_evs') or []) or 6
    # Action-HISTORY tokens stay COARSE (the 'act' input vocab is unchanged; only the model OUTPUT
    # widened): fold->7, call->3, ANY raise size->6.
    act_map = {0: 7, 1: 3}
    for k in range(2, K):
        act_map[k] = 6

    board_seq = [[52]*5 for _ in range(max_seq_len)]
    context_seq = [[0.0]*37 for _ in range(max_seq_len)]
    action_seq = [0] * max_seq_len
    action_taken_seq = [0] * max_seq_len
    target_evs_seq = [[0.0]*K for _ in range(max_seq_len)]
    # Per-action loss weights: which of the K Q-heads get a gradient at each step. We always
    # train the fold baseline + the action actually taken (Fix 1).
    target_w_seq = [[0.0]*K for _ in range(max_seq_len)]
    # V12 actor target: a regret-matching policy distribution over the K actions (uniform default).
    policy_target_seq = [[1.0/K]*K for _ in range(max_seq_len)]
    loss_mask = [0.0] * max_seq_len
    
    # Aux labels
    opp_bluff_seq = [0.0] * max_seq_len
    opp_strength_seq = [0.0] * max_seq_len
    self_equity_seq = [0.0] * max_seq_len
    
    # Truncate if hand is too long
    dps = dps[-max_seq_len:]
    
    bb = dps[0]['big_blind'] if dps else 10.0
    final_profit = record.final_hero_profit
    
    start_idx = max_seq_len - len(dps)
    
    for i, dp in enumerate(dps):
        idx = start_idx + i
        b_ints = [card_to_int(c) for c in dp['board']]
        while len(b_ints) < 5:
            b_ints.append(52)
        board_seq[idx] = b_ints
        
        pot_odds = dp['call_amount'] / (dp['pot_size'] + dp['call_amount']) if (dp['pot_size'] + dp['call_amount']) > 0 else 0.0
        
        active_opps_count = sum(dp['active_opponents_mask'])
        if active_opps_count > 0:
            sum_vpip = 0.0
            sum_agg = 0.0
            for j in range(5):
                if dp['active_opponents_mask'][j] == 1.0:
                    seat_key = f"seat_{j+1}"
                    prof = record.opponents_profiles.get(seat_key, {'vpip': 0.3, 'agg': 0.4})
                    sum_vpip += map_vpip_to_midpoint(prof.get('vpip', 0.3))
                    sum_agg += map_agg_to_midpoint(prof.get('agg', 0.4))
            global_vpip = sum_vpip / active_opps_count
            global_agg = sum_agg / active_opps_count
        else:
            global_vpip = 0.3
            global_agg = 0.4

        ctx = [
            dp['hero_position'] / 5.0,
            # [V20_preflopEq bug fix] These 3 (+ opp_stack below) now share contract.py's exact
            # /100,/250-scale + clamp helpers instead of a separate, stale /400,/1000 copy of the
            # same math -- see this file's import comment + MANIFEST's docstring. Previously the
            # model was gradient-trained on ONE scale here while every inference path (self-play
            # rollout, live serving) fed it a DIFFERENT scale for the identical real chip amount.
            scaled_stack_bb(dp['hero_stack'], bb),
            scaled_pot_bb(dp['pot_size'], bb),
            dp['equity'],
            pot_odds,
            active_opps_count / 10.0,
            dp['street'] / 3.0,
            global_vpip, global_agg,
            scaled_call_bb(dp['call_amount'], bb),
        ]

        for j in range(5):
            seat_key = f"seat_{j+1}"
            prof = record.opponents_profiles.get(seat_key, {'vpip': 0.3, 'agg': 0.4})

            opp_pos = (j + 1 + dp['hero_position']) % 6
            pos_val = float(opp_pos) / 5.0 if dp['active_opponents_mask'][j] == 1.0 else -1.0

            ctx.append(float(dp['active_opponents_mask'][j]))
            ctx.append(pos_val)
            ctx.append(scaled_stack_bb(dp['opponents_stacks'][j], bb))
            ctx.append(map_vpip_to_midpoint(prof.get('vpip', 0.3)))
            ctx.append(map_agg_to_midpoint(prof.get('agg', 0.4)))

        # [V20_preflopEq] Two new appended features (indices 35, 36 -- everything above is
        # untouched V20, now scale-corrected). Mirrors contract.py's to_tensors exactly so the
        # gradient-training tensors and every inference path agree on what these mean.
        ctx.append(equity_edge_feature(dp['equity'], active_opps_count))
        ctx.append(float(dp.get('hand_strength', 0.5)))

        context_seq[idx] = ctx
        action_seq[idx] = act_map.get(dp['action'], 0)
        action_taken_seq[idx] = dp['action']
        
        # Go-forward Monte-Carlo return for the action actually taken (excludes sunk cost).
        action_taken = dp['action']  # 0=fold, 1=call, 2..K-1=raise sizes
        mc_return = (final_profit + dp['committed_before']) / bb

        # [V21] The preflop-entry tightness-prior adjustment that used to sit here
        # (TIGHTNESS_PENALTY_BB/ENTRY_EQUITY_MARGIN) has been deleted -- see the constants-block
        # comment above.

        # Tighter clip (was +/-100bb) to damp the fat tail that biased "enter" upward.
        def _clip(ev):
            return max(-TARGET_CLIP_BB, min(TARGET_CLIP_BB, ev))
        mc_return = _clip(mc_return)

        # Model-free counterfactual EVs from the simulator's true-equity Monte Carlo
        # (`_calculate_mc_target_evs`), scaled chips -> BB and clipped. These are correctly
        # NEGATIVE for weak hands (e.g. calling/raising 0-equity air), which is exactly the
        # signal a taken-action-only loss omitted.
        mc_evs = [_clip(ev / bb) for ev in dp.get('target_evs', [0.0]*K)]

        # --- All-action targets with realized-return override ---
        # Fold head -> 0 (exact go-forward baseline, full weight).
        # Taken action -> its realized MC return (ground truth, full weight).
        # Untaken actions (incl. EVERY raise size) -> model-free counterfactual EV (fractional weight).
        t_evs = list(mc_evs)
        t_evs[0] = 0.0
        t_evs[action_taken] = mc_return
        t_w = [COUNTERFACTUAL_WEIGHT] * K
        t_w[0] = 1.0                 # fold baseline always full weight
        t_w[action_taken] = 1.0      # taken action always full weight

        target_evs_seq[idx] = t_evs
        target_w_seq[idx] = t_w

        # --- V12 actor target: regret-matching policy over the action values ---
        # [V21] Skipped entirely once compute_policy_target=False (post-cutover callers) --
        # see this function's docstring.
        if compute_policy_target:
            # One-step regret matching from the uniform strategy: the value of the uniform
            # strategy is the mean action value; an action's regret is how much it beats that
            # mean. The target policy is proportional to POSITIVE regret (uniform if none).
            # POLICY_TARGET_SOURCE selects whether the taken action carries its realized return
            # (legacy) or the counterfactual EV (removes the fold-equity survivorship ratchet).
            if POLICY_TARGET_SOURCE == 'counterfactual':
                p_evs = list(mc_evs)
                p_evs[0] = 0.0
                # Realization discount: the all-in-equity counterfactual overvalues speculative
                # entries. Penalize VOLUNTARY entries (call + EVERY raise size) for sub-pivot
                # equity so the ACTOR folds edges that won't be realized multi-street. [V21] now
                # the single shared `_realization_discount` helper (see its definition) instead of
                # a second independent inline copy of this formula.
                pen = _realization_discount(dp['equity'], POLICY_TIGHTNESS_PIVOT, POLICY_TIGHTNESS_BB)
                if pen > 0.0:
                    for j in range(1, K):
                        p_evs[j] -= pen
            else:
                p_evs = t_evs
            policy_target_seq[idx] = sharpen_distribution(regret_match_policy(p_evs), POLICY_TARGET_TEMP)

        loss_mask[idx] = 1.0
        
        opp_bluff_seq[idx] = float(dp.get('opp_bluff_prob', 0.0))
        opp_strength_seq[idx] = float(dp.get('opp_strength', 0.0))
        self_equity_seq[idx] = float(dp['equity'])
        
    return [(hole_ints, board_seq, context_seq, action_seq, action_taken_seq, target_evs_seq, loss_mask, opp_bluff_seq, opp_strength_seq, self_equity_seq, target_w_seq, policy_target_seq)]

def simulate_worker(current_hand, bb_size, equity_sims, num_hands, hero_personality,
                    active_model_path=None, opp_pool_config=None,
                    bootstrap_alpha=0.0, focus_archetype=None,
                    live_players=6,
                    disable_extreme_stacks=False, fixed_stack_bb=None, disable_exploration=False,
                    ablate_hole_cards=False, range_aware_equity=False, stack_depth_mix=None):
    """Headless simulation worker process for V8.

    [V18] `opp_pool_config`: list of {'style', 'weight', 'model' (absolute path or None),
    'forced' (bool, default False)} dicts -- the SINGLE declarative source for both which
    styles get sampled into opponent seats (style+weight) and which model, if any, plays each
    style (model+forced). Replaces the pre-V18 opp_pool/opp_weights + five separate
    `*_model_path` params -- adding, removing, or swapping an opponent (including which frozen
    checkpoint plays which style, and whether that style's decisions get archetype-forced) is
    now a config change, not a new positional worker arg. See opponents.py.
    """
    sim = SixMaxSimulator(
        bb_size=bb_size, equity_sims=equity_sims, hero_personality=hero_personality,
        bootstrap_alpha=bootstrap_alpha
    )
    sim.focus_archetype = focus_archetype
    sim.live_players = live_players
    sim.disable_extreme_stacks = disable_extreme_stacks
    sim.fixed_stack_bb = fixed_stack_bb
    sim.stack_depth_mix = stack_depth_mix
    sim.disable_exploration = disable_exploration
    sim.range_aware_equity = range_aware_equity

    def _load_worker_model(path, required):
        """Load a self-describing checkpoint into a fresh model, FAIL-LOUD (P1).

        `required=True` (the active hero model) RAISES on failure: training on random
        weights was the exact silent bug that disabled the NN for whole V11 runs. Optional
        league models surface a clear warning but let the sim fall back to heuristics.
        """
        if not (path and os.path.exists(path)):
            if required:
                raise FileNotFoundError(f"Active model weights not found at {path}")
            return None
        try:
            m = PokerEVModelV4()
            m.ablate_hole_cards = ablate_hole_cards
            m.load_state_dict(load_ckpt_state(path, MANIFEST))
            m.eval()
            return m
        except Exception as e:
            if required:
                raise RuntimeError(f"FATAL: could not load active model {path}: {e}") from e
            print(f"WARNING: could not load league model {path}: {e}")
            return None

    # Active (hero) model MUST load — a failure here means training on garbage.
    sim.hero_model = _load_worker_model(active_model_path, required=True)

    # [V18] Config-driven opponent lineup + per-style model loading, uniformly.
    pool_config = opp_pool_config or DEFAULT_OPPONENT_POOL_CONFIG
    sim.opponent_pool_styles = [e['style'] for e in pool_config]
    sim.opponent_pool_weights = [e.get('weight', 1.0) for e in pool_config]
    heuristic_bots = {
        'fish': sim.fish_heuristic, 'maniac': sim.maniac_heuristic,
        'nit': sim.nit_heuristic, 'tag': sim.tag_heuristic, 'past': sim.tag_heuristic,
    }
    sim.opponent_pool = build_opponent_pool(
        pool_config, heuristic_bots,
        query_fn=sim._query_model_decide, error_fn=sim._note_query_error,
        load_model_fn=lambda path: _load_worker_model(path, required=False),
    )

    records = []
    for i in range(num_hands):
        rec = sim.simulate_hand(current_hand=current_hand + i)
        if rec and rec.decision_points:
            records.append(rec)
            
    return records, sim.seat_histories, sim.global_metrics, sim.global_exploitation_net

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def print_dashboard(hands_done, total_hands, elapsed, hands_per_sec, train_loss, val_loss,
                    seat_profits, seat_vpips, seat_aggs, epoch, personality, bootstrap_alpha,
                    training_samples, total_hands_simulated, seat_extra_stats, global_metrics,
                    telemetry=None, global_exploitation_net=None,
                    train_loss_q=0.0, train_loss_pi=0.0, train_loss_bluff=0.0, train_loss_str=0.0, train_loss_eq=0.0,
                    seat_labels=None):
    """Prints a clear, V8 multi-personality training dashboard with telemetry."""
    pct = (hands_done / max(1, total_hands)) * 100
    eta = ((total_hands - hands_done) / max(1, hands_per_sec)) if hands_per_sec > 0 else 0  # Calculate Phase
    # [V21] These labels now name the mechanisms actually gating behavior in THIS recipe (bootstrap
    # decay / actor-critic cutover / dynamic table sizes), not the old stack-curriculum phases --
    # those are bypassed whenever `stack_depth_mix` is set (as it always is here; see config.yaml),
    # so the old "Phase 1: 100BB Static" / "Phase 3: Extreme Stacks" labels were describing a
    # curriculum that wasn't actually running. Thresholds match run_training's own bootstrap decay
    # (5k-20k), ACTOR_CRITIC_CUTOVER_HANDS (20k), and the Phase-4/5 hand-count gates below (was
    # 10k/30k/50k/75k, shifted 10k earlier).
    if hands_done < 5000:
        phase = "Phase 1: Heuristic Bootstrap"
    elif hands_done < 20000:
        phase = "Phase 2: Bootstrap Decay -> Actor-Critic Cutover"
    elif hands_done < 40000:
        phase = "Phase 3: Pure Actor-Critic"
    elif hands_done < 65000:
        phase = "Phase 4: Dynamic Table Sizes"
    else:
        phase = "Phase 5: Dynamic Tables + Focus Rounds (if enabled)"
        
    # [V18] seat_labels is now passed in (built once in run_training from the actual opp_pool_config
    # via describe_pool_entry), so a slot shows what's REALLY loaded there (e.g. "Nit: V15") instead
    # of a static archetype name that can silently drift from reality -- exactly the class of gap
    # that hid v17_gauntlet's broken 'tag' seat. Falls back to the classic hardcoded labels if the
    # caller doesn't provide one (e.g. an older/simpler call site).
    if seat_labels is None:
        seat_labels = ["Hero (Main)", "Opp 1 (Maniac)", "Opp 2 (Nit)", "Opp 3 (Sticky)", "Opp 4 (Past Self)", "Opp 5 (TAG Bot)"]
    # [V18] Label width is now dynamic (labels can carry a loaded model's name, e.g. "Past Self:
    # Lagged-Self", which the old fixed 16-char padding was never sized for) so columns still align.
    _label_width = max(16, max(len(lbl) for lbl in seat_labels))
    seat_lines = []
    for s in range(6):
        profit = seat_profits[s]
        bb100 = (profit / max(1.0, total_hands_simulated)) * 10.0
        vpip = seat_vpips[s] * 100.0
        agg = seat_aggs[s] * 100.0
        extra = seat_extra_stats.get(s, {'raises': 0, 'folds': 0, 'all_ins': 0})
        r, f, a = extra['raises'], extra['folds'], extra['all_ins']
        seat_lines.append(f"|  - Seat {s} {seat_labels[s]:<{_label_width}}: {bb100:>+6.1f} BB/100 (VPIP:{vpip:>4.1f}% AGG:{agg:>4.1f}%) [R:{r:<5} F:{f:<5} AI:{a:<4}]|")
        
    avg_players = global_metrics.get('flop_players', 0) / max(1, global_metrics.get('flop_count', 0))
        
    lines = [
        "+========================================================================================+",
        "|  SELF-PLAY V8 MULTI-PERSONALITY LEAGUE SYSTEM (RTX 4080 GPU)                           |",
        "+========================================================================================+",
        f"|  Active Personality: {personality.upper():<12}                                                       |",
        f"|  Hands Simulated:    {hands_done:>10,} / {total_hands:>10,}   ({pct:>5.1f}%)                            |",
        f"|  Training Epoch:     {epoch:>4}                                                              |",
        f"|  Bootstrap Alpha:    {bootstrap_alpha:>6.2f}                                                            |",
        f"|  Curriculum Stage:   {phase:<24}                                      |",
        f"|  Elapsed Time:       {format_time(elapsed):<12}                                                     |",
        f"|  Sim Speed:          {hands_per_sec:>6.0f} hands/sec                                                 |",
        f"|  ETA:                {format_time(eta):<12}                                                     |",
        f"|  Training Samples:   {training_samples:>10,}                                                        |",
        "+----------------------------------------------------------------------------------------+",
        "|  CUMULATIVE PERFORMANCE BY SEAT:                                                       |",
    ]
    for s_line in seat_lines:
        lines.append(s_line)
    lines.extend([
        "+----------------------------------------------------------------------------------------+",
        f"|  Global Post-Flop Avg Active Players: {avg_players:<4.2f}                                             |",
        "+----------------------------------------------------------------------------------------+"
    ])
    
    if global_exploitation_net is not None:
        lines.extend([
            "|  EXPLOITATION SCOREBOARD (Net BB/100 Matrix):                                          |",
            "|  [Winner \\ Loser] | Hero  | S1    | S2    | S3    | S4    | S5    |                    |",
        ])
        for i in range(6):
            row_str = f"|  {seat_labels[i]:<16} |"
            for j in range(6):
                if i == j:
                    row_str += "   -   |"
                else:
                    net = global_exploitation_net.get(i, {}).get(j, 0.0)
                    bb100_net = (net / max(1.0, total_hands_simulated)) * 10.0
                    row_str += f" {bb100_net:>+5.1f} |"
            lines.append(row_str + "                    |")
        lines.append("+----------------------------------------------------------------------------------------+")
    
    if telemetry is not None:
        stats_dict = telemetry.get_matrix_stats()
        ent = telemetry.get_average_entropy()
        lines.extend([
            f"|  Action Entropy:     {ent:>8.4f}                                                        |",
            "+------------------------------------------------------------------------------------------------------------------------+",
            "|  Equity Matrix (FACING A BET ONLY -- free checks tallied separately, see Free col)                                    |",
            "|  Equity Bucket     | Fold  | Call  | r33   | r66   | rPot  | All-In | N Hands | Free  | Avg St | Net Chips | Won    | Lost  |",
            "+------------------------------------------------------------------------------------------------------------------------+"
        ])
        labels = {
            '<20': '<20% (Pure Air)',
            '20-40': '20-40% (Draws)',
            '40-60': '40-60% (Marginal)',
            '60-80': '60-80% (Strong)',
            '>80': '>80% (Nuts)'
        }
        for b in ['<20', '20-40', '40-60', '60-80', '>80']:
            s = stats_dict[b]
            f = s['f_pct'] * 100
            c = s['c_pct'] * 100
            r33 = s['r33_pct'] * 100
            r66 = s['r66_pct'] * 100
            rpot = s['rpot_pct'] * 100
            ai = s['ai_pct'] * 100
            avg_st = s['avg_street']
            net = s['total_chips']
            won = s.get('won_chips', 0.0)
            lost = s.get('lost_chips', 0.0)
            n_hands = s.get('total_hands', 0)
            free = s.get('free_checks', 0)

            lines.append(f"|  {labels[b]:<17} | {f:>4.1f}% | {c:>4.1f}% | {r33:>4.1f}% | {r66:>4.1f}% | {rpot:>4.1f}% | {ai:>4.1f}%  | {n_hands:>7,} | {free:>5,} | {avg_st:>5.1f}  | {net:>+7.1f}   | {won:>+6.0f} | {lost:>+6.0f} |")

        lines.append("+------------------------------------------------------------------------------------------------------------------------+")

        # --- V14 Action Usage & Opponent Adaptation --------------------------------------------
        usage = telemetry.get_action_usage()           # 6 fractions [Fold, Call, r33, r66, rPot, All-In]
        aw, an = telemetry.get_allin_winrate()
        jam = telemetry.get_jam_by_color()             # {color: (jam_freq, n)}
        u = [x * 100 for x in usage]
        lines.append(f"|  ACTION USAGE (all decisions) | Fold {u[0]:>4.1f}% | Call {u[1]:>4.1f}% | r33 {u[2]:>4.1f}% | r66 {u[3]:>4.1f}% | rPot {u[4]:>4.1f}% | All-In {u[5]:>4.1f}% |")
        jam_str = " ".join(f"{c} {jam[c][0]*100:>4.1f}%" for c in ['Blue', 'Green', 'Yellow', 'Red'])
        lines.append(f"|  ALL-IN WinRate {aw*100:>4.1f}% (n={an}) | JAM by Opp-Color: {jam_str} |")
        lines.append("+------------------------------------------------------------------------------------------------------------------------+")

    lines.extend([
        f"|  Train Loss:         {train_loss:>8.4f}  |  Val Loss: {val_loss:>8.4f}                                |",
        f"|  Loss Q: {train_loss_q:>6.4f} | Pi: {train_loss_pi:>6.4f} | Bluff: {train_loss_bluff:>6.4f} | Str: {train_loss_str:>6.4f} | Eq: {train_loss_eq:>6.4f}        |",
        "+========================================================================================+"
    ])
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n".join(lines))
    sys.stdout.flush()

def run_intermediate_sensitivity_check(model, step_label, device):
    """Validation range check inside training."""
    model.eval()
    print(f"\n==================================================")
    preflop_hands = [
        ("7d 2s (Garbage)", ["7d", "2s"], 0.30),
        ("Jh Ts (Medium)", ["Jh", "Ts"], 0.46),
        ("Ad Qo (Strong)", ["Ad", "Qo"], 0.60),
        ("Qd Qs (Premium)", ["Qd", "Qs"], 0.78),
        ("Ah As (Nuts)", ["Ah", "As"], 0.85)
    ]
    
    from core.board_state import BoardState, SeatState, HUDStats
    from versions.v21_auxhead.core.contract import ContractV12, preflop_hand_strength
    bridge = ContractV12(max_seq_len=20)

    print(f"--- INTERMEDIATE SENSITIVITY CHECK AT {step_label} ---")
    print(f"| Hand | Equity | P(Fold) | P(Call) | P(Raise) | Action |")
    print(f"| :--- | :---: | :---: | :---: | :---: | :---: |")

    for label, cards, equity in preflop_hands:
        state = BoardState(
            community_cards=[],
            hero_cards=cards,
            pot_size=30.0,
            hero_stack=1000.0,
            big_blind=10.0,
            call_amount=20.0,
            equity=equity,
            hero_position=0,
            street="Preflop",
            hand_strength=preflop_hand_strength(cards[0], cards[1]),
        )
        state.seats["seat_1"] = SeatState(
            name="Opponent 1",
            stack=1000.0,
            is_active=True,
            hud=HUDStats(vpip_color="Green", agg_color="Green")
        )
        
        h_t, b_t, c_t, a_t = bridge.to_tensors(state, hero_actions=[6])
        
        with torch.no_grad():
            preds = model(h_t.to(device), b_t.to(device), c_t.to(device), a_t.to(device))
            # V12: grade the ACTOR (policy) head — the head that actually chooses actions.
            logits = preds['policy_logits'].squeeze(0)[0]
            probs = torch.softmax(logits, dim=-1)

        p_f, p_c, p_r = probs[0].item(), probs[1].item(), probs[2].item()
        best_act = "RAISE" if p_r >= p_c and p_r >= p_f else "CALL" if p_c >= p_f else "FOLD"
        print(f"| {label:<20} | {equity:.2f} | {p_f:>6.2f} | {p_c:>6.2f} | {p_r:>7.2f} | **{best_act}** |")
    print("==================================================\n")
    sys.stdout.flush()

def run_training(personality, num_hands=100000, batch_size=256, epochs_per_batch=3,
                 sim_batch_size=2000, lr=1e-3, equity_sims=200,
                 save_name=None, resume_path=None, initial_hands_done=0,
                 mid_flight_diagnostics_interval=10000,
                 checkpoint_dump_interval=25000,
                 opp_pool_config=None, live_players=6, disable_focus_rounds=False,
                 disable_extreme_stacks=False, fixed_stack_bb=None, disable_exploration=False,
                 disable_bootstrap=False, aux_loss_weight=10.0,
                 aux_loss_weight_bluff=None, aux_loss_weight_strength=None, aux_loss_weight_equity=None,
                 target_clip_bb=None, policy_target_temp=1.0, policy_entropy_penalty=0.0,
                 policy_target_source='realized', ablate_hole_cards=False,
                 policy_tightness_bb=0.0, range_aware_equity=False,
                 stack_depth_mix=None):
    # [V21_auxhead Phase 6] Per-head aux weights, defaulting to the shared `aux_loss_weight` for
    # backward compat. Motivated by Phases 3-5: sharing ONE scalar across bluff/strength/equity's
    # summed loss let bluff's now-properly-reweighted (larger) loss claim a growing share of the
    # shared gradient -- strength's correlation declined MONOTONICALLY across every subsequent
    # phase (0.151->0.065->0.047->0.033), never recovering with more hands, unlike equity's mostly-
    # transient dip. Decoupling the weights lets strength get its own (typically higher) budget
    # instead of being crowded out by bluff. See SPECS.md Phase 6.
    w_bluff = aux_loss_weight if aux_loss_weight_bluff is None else aux_loss_weight_bluff
    w_strength = aux_loss_weight if aux_loss_weight_strength is None else aux_loss_weight_strength
    w_equity = aux_loss_weight if aux_loss_weight_equity is None else aux_loss_weight_equity
    # Overrides the module globals that vectorize_hand_samples reads at call time (it runs
    # in THIS process, not the workers). Two SEPARATE concerns:
    #  * target_clip_bb = VARIANCE control (load-bearing). Unclipped realized returns are
    #    fat-tailed (+/-100bb when a stack goes in); the same state then gets wildly varying
    #    targets each visit, the critic diverges, and the regret-matching actor collapses to
    #    uniform. Keep this ON. Set to a large number only to deliberately test instability.
    #  * [V21] The second concern this comment used to describe (disable_target_shaping, the
    #    preflop-entry bias prior) has been deleted outright -- see versions/v21/SPECS.md item 1.
    global TARGET_CLIP_BB
    global POLICY_TARGET_TEMP, POLICY_TARGET_SOURCE
    if target_clip_bb is not None:
        TARGET_CLIP_BB = target_clip_bb
    POLICY_TARGET_TEMP = policy_target_temp      # actor-target sharpening (read by vectorize)
    POLICY_TARGET_SOURCE = policy_target_source  # 'realized' | 'counterfactual'
    global POLICY_TIGHTNESS_BB
    POLICY_TIGHTNESS_BB = policy_tightness_bb     # realization discount on counterfactual actor target
    if save_name is None:
        save_name = f"expert_{personality}.pth"
        
    print("=" * 60)
    print("  SELF-PLAY RL TRAINING SYSTEM (PLURIBUS V8)")
    print("=" * 60)
    print(f"  Target Hands:    {num_hands:,}")
    print(f"  Personality:     {personality.upper()}")
    print(f"  Sim Batch Size:  {sim_batch_size:,}")
    print(f"  Epochs/Batch:    {epochs_per_batch}")
    print(f"  Learning Rate:   {lr}")
    print(f"  Save File Name:  {save_name}")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PokerEVModelV4().to(device)
    model.ablate_hole_cards = ablate_hole_cards
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss(reduction='none', delta=2.0)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Checkpoint directory: this version's OWN weights folder (versions/v12/weights)
    weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'weights'))
    os.makedirs(weights_dir, exist_ok=True)
    
    # 1. Initialize weights from V7 baseline model to avoid starting from scratch
    v7_baseline_path = os.path.join(weights_dir, 'expert_v7_selfplay.pth')
    if resume_path and os.path.exists(resume_path):
        checkpoint_path = resume_path
    else:
        checkpoint_path = v7_baseline_path
        
    if os.path.exists(checkpoint_path):
        print(f"Initializing V8 weights from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])        # self-describing (v12+)
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])  # legacy dict
            else:
                model.load_state_dict(checkpoint)                      # bare state_dict
        except Exception as e:
            print(f"WARNING: Could not load V7 starting weights ({e}). Initializing random weights.")
    else:
        print("WARNING: V7 baseline weights not found. Initializing random weights.")
        
    # [V18] The 'past' seat's resolution mode comes from its OWN pool-config entry now, not
    # separate freeze_past_self/frozen_past_filename params: `lagged_self: true` -> a fresh
    # snapshot of the CURRENT run, refreshed every 5k hands; `model: <filename>` -> a STATIC
    # frozen expert (never overwritten); absent entirely -> the 'past' style just isn't sampled.
    _past_entry = next((e for e in (opp_pool_config or []) if e.get('style') == 'past'), None)
    past_lagged = bool(_past_entry and _past_entry.get('lagged_self'))
    if past_lagged:
        past_path = os.path.join(weights_dir, 'past_checkpoint.pth')
    elif _past_entry and _past_entry.get('model'):
        past_path = os.path.join(weights_dir, _past_entry['model'])
        if not os.path.exists(past_path):
            print(f"WARNING: past seat model {past_path} missing — Past-Self seat falls back to TAG heuristic.")
    else:
        past_path = None
    # [V18] Dynamic seat labels for the dashboard -- "{BotName} ({Heuristic|NN})", derived from the
    # SAME opp_pool_config that actually drives what plays each seat (opponents.describe_pool_entry),
    # so a label can never silently drift from reality. Deliberately drops the archetype-SLOT name
    # ("Nit"/"TAG Bot") from the display entirely -- that's internal stat-bucket bookkeeping, not a
    # useful thing to show the user, and it was never fully accurate anyway (the 'maniac' slot's
    # real bot is LAG, not "Maniac"). Naming the actual bot instead of the slot it occupies is also
    # what would have made v17_gauntlet's broken 'tag' seat immediately visible (it would have kept
    # showing "TAG (Heuristic)" instead of the intended "V16 (NN)").
    _slot_style = {v: k for k, v in STYLE_SLOT.items() if k not in ('hero', 'main')}
    _entry_by_style = {e['style']: e for e in (opp_pool_config or [])}
    seat_labels = ["Hero (Main)"]
    for _slot_idx in range(1, 6):
        _style = _slot_style.get(_slot_idx)
        _entry = _entry_by_style.get(_style)
        if _entry:
            _name, _kind = describe_pool_entry(_entry)
        else:
            _name, _kind = (_style or '?').capitalize(), 'Heuristic'
        seat_labels.append(f"Opp {_slot_idx}: {_name} ({_kind})")

    # Fresh run: drop any stale LAGGED past-self checkpoint (not a static frozen file) so "Past
    # Self" only ever plays a snapshot of the CURRENT run (not leftovers from a previous training).
    if past_lagged and initial_hands_done == 0 and past_path and os.path.exists(past_path):
        os.remove(past_path)

    # CSV logger
    global_metrics = {'flop_players': 0, 'flop_count': 0}
    global_exploitation_net = {i: {j: 0.0 for j in range(6)} for i in range(6)}
    
    log_dir = os.path.dirname(__file__)
    log_path = os.path.join(log_dir, f'training_log_{personality}.csv')
    mode = 'a' if initial_hands_done > 0 else 'w'
    log_file = open(log_path, mode, newline='')
    csv_writer = csv.writer(log_file)
    if initial_hands_done == 0:
        csv_writer.writerow([
            'timestamp', 'hands_done', 'training_samples', 'train_loss', 'val_loss',
            'train_loss_q', 'train_loss_bluff', 'train_loss_str', 'train_loss_eq',
            'hero_bb100', 'hands_per_sec', 'elapsed_sec'
        ])
    
    hands_done = initial_hands_done
    batch_records = []
    total_training_samples = 0
    train_loss = 0.0
    val_loss = 0.0
    train_loss_q = 0.0
    train_loss_pi = 0.0
    train_loss_bluff = 0.0
    train_loss_str = 0.0
    train_loss_eq = 0.0
    seat_cumulative_profits = [0.0] * 6
    current_seat_vpips = [0.30] * 6
    current_seat_aggs = [0.40] * 6
    global_vpip_ops = [0] * 6
    global_vpip_acts = [0] * 6
    global_agg_ops = [0] * 6
    global_agg_acts = [0] * 6
    seat_extra_stats = {s: {'raises': 0, 'folds': 0, 'all_ins': 0} for s in range(6)}
    global_metrics = {'flop_players': 0, 'flop_count': 0}
    total_hands_simulated = 0
    t_start = time.time()
    total_epochs_completed = 0
    
    # Flag flags to execute intermediate checks
    # [V21] shifted 10k earlier to match the new bootstrap-decay milestones (5k/20k, was 10k/30k).
    check_5k_done = False
    check_20k_done = False
    
    from versions.v21_auxhead.self_play.telemetry import TrainingTelemetry
    telemetry = TrainingTelemetry()
    
    print(f"\nLaunching simulation batch workers...")

    # V13: precompute the preflop-range ranking ONCE in the main process (cached to disk) so
    # the spawned workers all load it instead of racing to build it.
    if range_aware_equity:
        from versions.v21_auxhead.self_play.simulator import _get_preflop_ranked
        print("  Building preflop range ranking (range-aware equity)...")
        _get_preflop_ranked()

    num_workers = min(os.cpu_count(), 8)
    pool = Pool(num_workers)
    try:
        while hands_done < num_hands:
            # Check intermediate check stops
            if hands_done >= 5000 and not check_5k_done:
                run_intermediate_sensitivity_check(model, "5k Hands", device)
                check_5k_done = True

            if hands_done >= 20000 and not check_20k_done:
                run_intermediate_sensitivity_check(model, "20k Hands", device)
                check_20k_done = True
                
            # Automated Mid-Flight Diagnostics
            if hasattr(run_training, "last_diag") is False:
                run_training.last_diag = initial_hands_done
                
            if hands_done - run_training.last_diag >= mid_flight_diagnostics_interval:
                print(f"\n[DIAGNOSTICS] Mid-flight checkpoint at {hands_done} hands...")
                # Save a self-describing checkpoint to this version's weights dir.
                # NOTE: the old subprocess diagnostic used the legacy engine registry
                # (v8-v11 only) and is intentionally not run for v12 — evaluation goes
                # through shared.registry / the unified harness instead.
                checkpoint_path = os.path.join(weights_dir, save_name)
                save_checkpoint(model.state_dict(), checkpoint_path, MANIFEST, hands_trained=hands_done)
                run_training.last_diag = hands_done

            # Periodic restore-point dump: a DISTINCT, never-overwritten checkpoint every
            # checkpoint_dump_interval hands (default 25k), so there is always some earlier
            # model to fall back to or evaluate -- unlike the rolling checkpoints above
            # (checkpoint_path / active_model_path), which get overwritten each time.
            if hasattr(run_training, "last_dump") is False:
                run_training.last_dump = initial_hands_done

            if hands_done - run_training.last_dump >= checkpoint_dump_interval:
                dump_dir = os.path.join(weights_dir, "checkpoints")
                os.makedirs(dump_dir, exist_ok=True)
                dump_path = os.path.join(dump_dir, f"{personality}_hands{hands_done}.pth")
                save_checkpoint(model.state_dict(), dump_path, MANIFEST, hands_trained=hands_done)
                print(f"\n[CHECKPOINT] Restore-point dump saved: {dump_path}")
                run_training.last_dump = hands_done


            # Decay bootstrap alpha (verify mode: no heuristic warmup -> pure model from hand 0)
            if disable_bootstrap:
                bootstrap_alpha = 0.0
            elif hands_done < 5000:
                bootstrap_alpha = 1.0
            elif hands_done < 20000:
                # [V21] decay window shifted 10k earlier: 5k-20k (was 10k-30k).
                bootstrap_alpha = 1.0 - (hands_done - 5000) / 15000.0
            else:
                bootstrap_alpha = 0.0
                
            batch_hands = min(sim_batch_size, num_hands - hands_done)
            
            # Save active model weights temporarily for workers to read
            active_model_path = os.path.join(log_dir, f'temp_active_model_{personality}.pth')
            save_checkpoint(model.state_dict(), active_model_path, MANIFEST, hands_trained=hands_done)
            
            # Save past self checkpoint every 5,000 hands -- ONLY when the pool's 'past' entry
            # asked for a true lagged mirror (`lagged_self: true`); a static frozen file (or no
            # 'past' entry at all) must never be overwritten by this.
            if past_lagged and hands_done > 0 and (hands_done // 5000) > ((hands_done - len(batch_records)) // 5000):
                save_checkpoint(model.state_dict(), past_path, MANIFEST, hands_trained=hands_done)

            # Worker args
            hands_per_worker = max(1, batch_hands // num_workers)

            # [V18] Resolve this batch's opponent pool config: every entry's model path is
            # STATIC (resolved once against weights_dir) EXCEPT 'past' when lagged_self is set,
            # which needs the just-refreshed snapshot path re-checked every batch.
            resolved_pool_config = []
            for entry in (opp_pool_config or []):
                e = dict(entry)
                if e.get('style') == 'past':
                    e['model'] = past_path if (past_path and os.path.exists(past_path)) else None
                    # NOTE: 'lagged_self' is intentionally KEPT (not popped) so display_name
                    # derivation (opponents.describe_pool_entry) can still render "Lagged-Self"
                    # instead of the raw 'past_checkpoint' filename stem.
                elif e.get('model'):
                    e['model'] = os.path.join(weights_dir, e['model'])
                resolved_pool_config.append(e)

            # Phase 5: Focus Rounds Logic (disabled for diagnostic runs so the field stays
            # constant; a shifting focus swarm would confound the VPIP attribution).
            focus_archetype = None
            if (not disable_focus_rounds) and hands_done >= 65000:  # [V21] was 75000
                focus_archetype = random.choice(['maniac', 'nit', 'fish'])

            args = [
                (hands_done, 10.0, equity_sims, hands_per_worker, personality,
                 active_model_path, resolved_pool_config, bootstrap_alpha, focus_archetype,
                 live_players, disable_extreme_stacks,
                 fixed_stack_bb, disable_exploration, ablate_hole_cards, range_aware_equity,
                 stack_depth_mix)
                for _ in range(num_workers)
            ]
            
            results = pool.starmap(simulate_worker, args)
                
            batch_records = []
            recent_vpip_ops = {s: 0 for s in range(6)}
            recent_vpip_acts = {s: 0 for s in range(6)}
            recent_agg_ops = {s: 0 for s in range(6)}
            recent_agg_acts = {s: 0 for s in range(6)}
            
            for res, worker_hist, worker_extra, worker_exploitation in results:
                batch_records.extend(res)
                global_metrics['flop_players'] += worker_extra.get('flop_players', 0)
                global_metrics['flop_count'] += worker_extra.get('flop_count', 0)
                for i in range(6):
                    for j in range(6):
                        global_exploitation_net[i][j] += worker_exploitation.get(i, {}).get(j, 0.0)
                for s in range(6):
                    seat_cumulative_profits[s] += worker_hist[s]['profit']
                    recent_vpip_ops[s] += worker_hist[s]['vpip_ops']
                    recent_vpip_acts[s] += worker_hist[s]['vpip_acts']
                    recent_agg_ops[s] += worker_hist[s]['agg_ops']
                    recent_agg_acts[s] += worker_hist[s]['agg_acts']
                    seat_extra_stats[s]['raises'] += worker_hist[s].get('raises', 0)
                    seat_extra_stats[s]['folds'] += worker_hist[s].get('folds', 0)
                    seat_extra_stats[s]['all_ins'] += worker_hist[s].get('all_ins', 0)
                    
            for s in range(6):
                global_vpip_ops[s] += recent_vpip_ops[s]
                global_vpip_acts[s] += recent_vpip_acts[s]
                global_agg_ops[s] += recent_agg_ops[s]
                global_agg_acts[s] += recent_agg_acts[s]

                if hands_done < 1000:
                    if global_vpip_ops[s] > 0:
                        current_seat_vpips[s] = global_vpip_acts[s] / global_vpip_ops[s]
                    if global_agg_ops[s] > 0:
                        current_seat_aggs[s] = global_agg_acts[s] / global_agg_ops[s]
                else:
                    # Calculate batch VPIP/AGG
                    batch_vpip = recent_vpip_acts[s] / recent_vpip_ops[s] if recent_vpip_ops[s] > 0 else current_seat_vpips[s]
                    batch_agg = recent_agg_acts[s] / recent_agg_ops[s] if recent_agg_ops[s] > 0 else current_seat_aggs[s]
                    
                    # Apply Exponential Moving Average (EMA) (alpha = 0.2)
                    current_seat_vpips[s] = 0.8 * current_seat_vpips[s] + 0.2 * batch_vpip
                    current_seat_aggs[s] = 0.8 * current_seat_aggs[s] + 0.2 * batch_agg
                
            hands_done += len(batch_records)
            total_hands_simulated += len(batch_records)
                
            # Vectorize simulated hands
            X_hole, X_board, X_ctx, X_act, X_sa, Y_mc, X_mask = [], [], [], [], [], [], []
            Y_bluff, Y_strength, Y_equity, Y_w, Y_pol = [], [], [], [], []
            # [V21] hands_done has already been incremented for this batch (above) -- the same
            # value the loss-computation switch below uses for its own cutover check, so this
            # stays batch-level consistent with which regime actually trains on this data.
            _compute_pol = hands_done < ACTOR_CRITIC_CUTOVER_HANDS
            for rec in batch_records:
                samples = vectorize_hand_samples(rec, compute_policy_target=_compute_pol)
                if rec.decision_points:
                    # Record terminal state of the hand
                    final_dp = rec.decision_points[-1]
                    eq = final_dp['equity']
                    street = final_dp['street']
                    action = final_dp['action']
                    call_amount = final_dp['call_amount']
                    is_all_in = final_dp.get('is_all_in', False)
                    net_profit = rec.final_hero_profit
                    telemetry.record_hand_terminal_state(eq, street, action, call_amount, is_all_in, net_profit)
                    # Per-decision usage: size-selection histogram + jam-by-opponent-colour.
                    for dp in rec.decision_points:
                        telemetry.record_decision(dp['action'], dp.get('is_all_in', False),
                                                  dp.get('opp_vpip_color'))

                for h, b, c, a, sa, mc, mask, opp_b, opp_s, self_e, tw, pol in samples:
                    X_hole.append(h)
                    X_board.append(b)
                    X_ctx.append(c)
                    X_act.append(a)
                    X_sa.append(sa)
                    Y_mc.append(mc)
                    X_mask.append(mask)
                    Y_bluff.append(opp_b)
                    Y_strength.append(opp_s)
                    Y_equity.append(self_e)
                    Y_w.append(tw)
                    Y_pol.append(pol)
                    
            if not X_hole:
                continue
                
            hole_t = torch.tensor(X_hole, dtype=torch.long)
            board_t = torch.tensor(X_board, dtype=torch.long)
            ctx_t = torch.tensor(X_ctx, dtype=torch.float32)
            act_t = torch.tensor(X_act, dtype=torch.long)
            sa_t = torch.tensor(X_sa, dtype=torch.long)
            mc_t = torch.tensor(Y_mc, dtype=torch.float32)
            mask_t = torch.tensor(X_mask, dtype=torch.float32)
            bluff_t = torch.tensor(Y_bluff, dtype=torch.float32)
            str_t = torch.tensor(Y_strength, dtype=torch.float32)
            eq_t = torch.tensor(Y_equity, dtype=torch.float32)
            w_t = torch.tensor(Y_w, dtype=torch.float32)
            pol_t = torch.tensor(Y_pol, dtype=torch.float32)

            total_training_samples += len(X_hole)

            # Split train/val
            dataset = TensorDataset(hole_t, board_t, ctx_t, act_t, sa_t, mc_t, mask_t, bluff_t, str_t, eq_t, w_t, pol_t)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            if train_size == 0 or val_size == 0:
                continue
                
            train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
            
            # GPU Training Epochs
            for epoch in range(epochs_per_batch):
                model.train()
                epoch_loss = 0.0
                epoch_loss_q = 0.0
                epoch_loss_pi = 0.0
                epoch_loss_bluff = 0.0
                epoch_loss_str = 0.0
                epoch_loss_eq = 0.0
                total_items = 0
                for b_h, b_b, b_c, b_a, b_sa, b_mc, b_m, b_bluff, b_str, b_eq, b_w, b_pol in train_loader:
                    b_h = b_h.to(device, non_blocking=True)
                    b_b = b_b.to(device, non_blocking=True)
                    b_c = b_c.to(device, non_blocking=True)
                    b_a = b_a.to(device, non_blocking=True)
                    b_sa = b_sa.to(device, non_blocking=True)
                    b_mc = b_mc.to(device, non_blocking=True)
                    b_m = b_m.to(device, non_blocking=True)
                    b_bluff = b_bluff.to(device, non_blocking=True)
                    b_str = b_str.to(device, non_blocking=True)
                    b_eq = b_eq.to(device, non_blocking=True)
                    b_w = b_w.to(device, non_blocking=True)
                    b_pol = b_pol.to(device, non_blocking=True)

                    optimizer.zero_grad()

                    with torch.cuda.amp.autocast():
                        out = model(b_h, b_b, b_c, b_a)
                        preds = out['q_vals']            # critic
                        pred_policy = out['policy_logits']  # actor
                        pred_bluff = out['bluff']
                        pred_str = out['strength']
                        pred_eq = out['equity']

                        with torch.no_grad():
                            # Entropy of the ACTOR (the head that actually chooses actions).
                            probs = torch.softmax(pred_policy, dim=-1)
                            batch_entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()
                            telemetry.record_entropy_value(batch_entropy)

                        # Critic loss: per-action counterfactual EV regression. b_w is
                        # [B, T, 3] weights (fold baseline + taken action full; untaken
                        # counterfactuals fractional). This feeds the actor's regret target.
                        step_w = b_w * b_m.unsqueeze(-1)
                        loss_q = criterion(preds, b_mc) * step_w
                        final_loss_q = loss_q.sum() / step_w.sum().clamp(min=1.0)

                        # Actor loss: cross-entropy toward a regret-matching policy target.
                        # [V17 round 2] BEFORE the critic has had a full bootstrap-decay's worth of
                        # training (hands_done < ACTOR_CRITIC_CUTOVER_HANDS -- the same 30k
                        # milestone bootstrap_alpha itself uses), keep the dataset's precomputed
                        # b_pol (MEAN-baseline regret over the raw single-hand simulator EV, as
                        # every prior version did -- untouched, so the noisy-raw regime stays
                        # exactly as validated). AFTER that, regret-match over the CRITIC's own
                        # (detached) q_vals with a FOLD-relative baseline -- diagnosed round 1:
                        # the critic is well-calibrated (correctly ranks FOLD at weak equity, more
                        # confidently over training) but the MEAN baseline dilutes its share
                        # regardless, because one steeply-negative outlier action (ALLIN) drags the
                        # shared mean down far enough that worse-than-fold actions still clear it.
                        # Fold-relative on this denoised critic gave 100% FOLD on the exact
                        # Q-vectors where mean-baseline gave ~29% (tested offline before this).
                        if hands_done >= ACTOR_CRITIC_CUTOVER_HANDS:
                            policy_target = regret_match_policy_torch(preds.detach(), equity=b_c[:, :, 3], baseline_mode='fold')
                        else:
                            policy_target = b_pol
                        log_policy = nn.functional.log_softmax(pred_policy, dim=-1)
                        loss_pi_step = -(policy_target * log_policy).sum(dim=-1) * b_m
                        final_loss_pi = loss_pi_step.sum() / b_m.sum().clamp(min=1.0)

                        # Actor-sharpening lever: penalize predicted-policy entropy so the
                        # actor commits to decisions instead of sitting at the near-uniform
                        # regret-matching floor (adds +beta*H(pred); minimizing -> lower H).
                        if policy_entropy_penalty > 0.0:
                            ent_step = -(torch.softmax(pred_policy, dim=-1) * log_policy).sum(dim=-1) * b_m
                            final_loss_pi = final_loss_pi + policy_entropy_penalty * (ent_step.sum() / b_m.sum().clamp(min=1.0))

                        # Interpretable Auxiliary Heads Loss
                        # [V21_auxhead] bluff_w: per-batch inverse-frequency reweighting so the
                        # rare positive label actually gets gradient weight instead of being
                        # trivially outvoted by the ~96-98% negative examples -- see
                        # _bluff_pos_weight.
                        bluff_w = _bluff_pos_weight(b_bluff, b_m)
                        loss_bluff = nn.functional.mse_loss(pred_bluff, b_bluff, reduction='none') * b_m * bluff_w
                        loss_str = nn.functional.mse_loss(pred_str, b_str, reduction='none') * b_m
                        loss_eq = nn.functional.mse_loss(pred_eq, b_eq, reduction='none') * b_m

                        sc_bluff = loss_bluff.sum() / (b_m * bluff_w).sum().clamp(min=1.0)
                        sc_str = loss_str.sum() / b_m.sum().clamp(min=1.0)
                        sc_eq = loss_eq.sum() / b_m.sum().clamp(min=1.0)

                        final_loss = (final_loss_q + POLICY_LOSS_WEIGHT * final_loss_pi
                                      + w_bluff * sc_bluff + w_strength * sc_str + w_equity * sc_eq)

                    scaler.scale(final_loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    epoch_loss += float(final_loss) * len(b_h)
                    epoch_loss_q += float(final_loss_q) * len(b_h)
                    epoch_loss_pi += float(final_loss_pi) * len(b_h)
                    epoch_loss_bluff += float(sc_bluff) * len(b_h)
                    epoch_loss_str += float(sc_str) * len(b_h)
                    epoch_loss_eq += float(sc_eq) * len(b_h)
                    total_items += len(b_h)

                total_epochs_completed += 1
                train_loss = epoch_loss / total_items if total_items > 0 else 0.0
                train_loss_q = epoch_loss_q / total_items if total_items > 0 else 0.0
                train_loss_pi = epoch_loss_pi / total_items if total_items > 0 else 0.0
                train_loss_bluff = epoch_loss_bluff / total_items if total_items > 0 else 0.0
                train_loss_str = epoch_loss_str / total_items if total_items > 0 else 0.0
                train_loss_eq = epoch_loss_eq / total_items if total_items > 0 else 0.0
                
            scheduler.step()
            
            # Validation Epoch
            model.eval()
            val_epoch_loss = 0.0
            val_items = 0
            with torch.no_grad():
                for b_h, b_b, b_c, b_a, b_sa, b_mc, b_m, b_bluff, b_str, b_eq, b_w, b_pol in val_loader:
                    b_h = b_h.to(device)
                    b_b = b_b.to(device)
                    b_c = b_c.to(device)
                    b_a = b_a.to(device)
                    b_sa = b_sa.to(device)
                    b_mc = b_mc.to(device)
                    b_m = b_m.to(device)
                    b_bluff = b_bluff.to(device)
                    b_str = b_str.to(device)
                    b_eq = b_eq.to(device)
                    b_w = b_w.to(device)
                    b_pol = b_pol.to(device)

                    with torch.cuda.amp.autocast():
                        out = model(b_h, b_b, b_c, b_a)
                        preds = out['q_vals']
                        pred_policy = out['policy_logits']
                        pred_bluff = out['bluff']
                        pred_str = out['strength']
                        pred_eq = out['equity']

                        step_w = b_w * b_m.unsqueeze(-1)
                        loss_q = criterion(preds, b_mc) * step_w
                        final_loss_q = loss_q.sum() / step_w.sum().clamp(min=1.0)

                        # [V17 round 2] Same hard cutover + fold-relative baseline as the training
                        # loop, for a like-for-like val_loss (already under torch.no_grad() here,
                        # so preds needs no explicit .detach()).
                        if hands_done >= ACTOR_CRITIC_CUTOVER_HANDS:
                            policy_target = regret_match_policy_torch(preds, equity=b_c[:, :, 3], baseline_mode='fold')
                        else:
                            policy_target = b_pol
                        log_policy = nn.functional.log_softmax(pred_policy, dim=-1)
                        loss_pi_step = -(policy_target * log_policy).sum(dim=-1) * b_m
                        final_loss_pi = loss_pi_step.sum() / b_m.sum().clamp(min=1.0)
                        if policy_entropy_penalty > 0.0:
                            ent_step = -(torch.softmax(pred_policy, dim=-1) * log_policy).sum(dim=-1) * b_m
                            final_loss_pi = final_loss_pi + policy_entropy_penalty * (ent_step.sum() / b_m.sum().clamp(min=1.0))

                        bluff_w = _bluff_pos_weight(b_bluff, b_m)
                        loss_bluff = nn.functional.mse_loss(pred_bluff, b_bluff, reduction='none') * b_m * bluff_w
                        loss_str = nn.functional.mse_loss(pred_str, b_str, reduction='none') * b_m
                        loss_eq = nn.functional.mse_loss(pred_eq, b_eq, reduction='none') * b_m
                        sc_bluff_v = loss_bluff.sum() / (b_m * bluff_w).sum().clamp(min=1.0)
                        sc_str_v = loss_str.sum() / b_m.sum().clamp(min=1.0)
                        sc_eq_v = loss_eq.sum() / b_m.sum().clamp(min=1.0)

                        final_loss = (final_loss_q + POLICY_LOSS_WEIGHT * final_loss_pi
                                      + w_bluff * sc_bluff_v + w_strength * sc_str_v + w_equity * sc_eq_v)

                    val_epoch_loss += final_loss.item() * len(b_h)
                    val_items += len(b_h)
                    
            val_loss = val_epoch_loss / val_items if val_items > 0 else 0.0
            
            # Logging metrics
            elapsed = time.time() - t_start
            hands_per_sec = total_hands_simulated / max(1, elapsed)
            hero_bb100 = (seat_cumulative_profits[0] / 10.0) / max(1, total_hands_simulated) * 100.0
            
            csv_writer.writerow([
                time.strftime('%Y-%m-%d %H:%M:%S'), hands_done, total_training_samples,
                f"{train_loss:.6f}", f"{val_loss:.6f}", f"{train_loss_q:.6f}",
                f"{train_loss_bluff:.6f}", f"{train_loss_str:.6f}", f"{train_loss_eq:.6f}",
                f"{hero_bb100:.2f}", f"{hands_per_sec:.2f}", f"{elapsed:.2f}"
            ])
            log_file.flush()
            
            # Refresh live dashboard
            print_dashboard(
                hands_done=hands_done, total_hands=num_hands, elapsed=elapsed,
                hands_per_sec=hands_per_sec, train_loss=train_loss, val_loss=val_loss,
                seat_profits=seat_cumulative_profits, seat_vpips=current_seat_vpips,
                seat_aggs=current_seat_aggs, epoch=total_epochs_completed,
                personality=personality, bootstrap_alpha=bootstrap_alpha,
                training_samples=total_training_samples,
                total_hands_simulated=total_hands_simulated,
                seat_extra_stats=seat_extra_stats, global_metrics=global_metrics,
                telemetry=telemetry, global_exploitation_net=global_exploitation_net,
                train_loss_q=train_loss_q, train_loss_pi=train_loss_pi,
                train_loss_bluff=train_loss_bluff,
                train_loss_str=train_loss_str, train_loss_eq=train_loss_eq,
                seat_labels=seat_labels
            )
            
        # Save final personality model checkpoint
        final_save_path = os.path.join(weights_dir, save_name)
        save_checkpoint(model.state_dict(), final_save_path, MANIFEST, hands_trained=hands_done)
        print(f"\nTraining completed successfully! Saved final weights to: {final_save_path}")
        
    finally:
        pool.close()
        pool.join()
        log_file.close()
        # Clean up temp active weight files
        temp_model_path = os.path.join(log_dir, f'temp_active_model_{personality}.pth')
        if os.path.exists(temp_model_path):
            try:
                os.remove(temp_model_path)
            except Exception:
                pass

if __name__ == '__main__':
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    active_log_path = os.path.join(repo_root, 'active_training.log')
    sys.stdout = Tee(active_log_path, 'w')

    parser = argparse.ArgumentParser(description="Herocules V11 self-play training loop.")
    parser.add_argument('--personality', type=str, default='main', choices=['main', 'maniac', 'nit', 'sticky'],
                        help="The playing personality to optimize (shapes the EV loss rewards).")
    parser.add_argument('--resume_path', type=str, default=None, help="Resume from an existing checkpoint.")
    parser.add_argument('--save_name', type=str, default=None, help="Output weights filename.")
    parser.add_argument('--hands_done', type=int, default=0, help="Initial hand offset to resume tracking from.")
    parser.add_argument('--num_hands', type=int, default=None,
                        help="Target TOTAL hands to train to (overrides config target_hands). "
                             "To continue an existing run for +100k, pass e.g. "
                             "--resume_path <ckpt> --hands_done 100000 --num_hands 200000.")

    args = parser.parse_args()
    
    # Load YAML config
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    t_cfg = config.get('training', {})
    o_cfg = config.get('opponents', {}) or {}
    c_cfg = config.get('curriculum', {}) or {}

    target_hands = args.num_hands if args.num_hands is not None else t_cfg.get('target_hands', 100000)

    disable_extreme_stacks = bool(c_cfg.get('disable_extreme_stacks', False))
    fixed_stack_bb = c_cfg.get('fixed_stack_bb', None)  # e.g. 100.0 -> flat 100bb all run
    stack_depth_mix = c_cfg.get('stack_depth_mix', None)  # V15 DoN-shaped [[lo,hi,w],...] per-hand mixture

    # Verify-mode training knobs (isolate the core learning loop).
    disable_bootstrap = bool(t_cfg.get('disable_bootstrap', False))
    disable_exploration = bool(t_cfg.get('disable_exploration', False))
    aux_loss_weight = float(t_cfg.get('aux_loss_weight', 10.0))
    # [V21_auxhead Phase 6] per-head overrides; None (unset in config) falls back to aux_loss_weight.
    _alw_bluff = t_cfg.get('aux_loss_weight_bluff')
    _alw_strength = t_cfg.get('aux_loss_weight_strength')
    _alw_equity = t_cfg.get('aux_loss_weight_equity')
    aux_loss_weight_bluff = float(_alw_bluff) if _alw_bluff is not None else None
    aux_loss_weight_strength = float(_alw_strength) if _alw_strength is not None else None
    aux_loss_weight_equity = float(_alw_equity) if _alw_equity is not None else None
    target_clip_bb = float(t_cfg.get('target_clip_bb', 40.0))  # variance clip; keep ON
    policy_target_temp = float(t_cfg.get('policy_target_temperature', 1.0))  # <1 sharpens actor target
    policy_entropy_penalty = float(t_cfg.get('policy_entropy_penalty', 0.0))  # +beta*H(pred) in actor loss
    policy_target_source = str(t_cfg.get('policy_target_source', 'realized'))  # 'realized' | 'counterfactual'
    ablate_hole_cards = bool(t_cfg.get('ablate_hole_cards', False))  # zero hole embed -> force equity use
    policy_tightness_bb = float(t_cfg.get('policy_tightness_bb', 0.0))  # realization discount on actor target
    range_aware_equity = bool(t_cfg.get('range_aware_equity', False))  # V13: hero equity vs opp VPIP ranges

    # [V18] Opponent lineup: `opponents.pool` is now a DECLARATIVE list of
    # {style, weight, model?, forced?, lagged_self?} dicts -- the single source for both which
    # styles get sampled (style+weight) and which model (if any) plays each style (model+forced/
    # lagged_self). Replaces the pre-V18 opp_pool/opp_weights/disable_past_self/freeze_past_self/
    # frozen_past_filename/nit_model_filename/tag_model_filename params. A legacy list-of-strings
    # `pool` (pre-V18 format) is upgraded to bare heuristic entries for backward compatibility.
    if isinstance(o_cfg, dict):
        _raw_pool = o_cfg.get('pool')
        if _raw_pool and isinstance(_raw_pool[0], str):
            _raw_weights = o_cfg.get('weights') or [1.0] * len(_raw_pool)
            opp_pool_config = [{'style': s, 'weight': w} for s, w in zip(_raw_pool, _raw_weights)]
        else:
            opp_pool_config = _raw_pool
        live_players = o_cfg.get('live_players', 6)
        disable_focus_rounds = bool(o_cfg.get('disable_focus_rounds', False))
    else:
        opp_pool_config = None
        live_players = 6
        disable_focus_rounds = False

    if opp_pool_config:
        print("  Opponent Pool:")
        for _e in opp_pool_config:
            _desc = (f"FROZEN {_e['model']} (forced={_e.get('forced', False)})" if _e.get('model')
                     else ("lagged self-play mirror" if _e.get('lagged_self') else "heuristic archetype"))
            print(f"    {_e['style']:<8} weight={_e.get('weight', 1.0):<5} {_desc}")
        print(f"  Live Players:    {live_players}  (Hero + {max(0, live_players - 1)} opponents)")
        print(f"  Focus rounds:    {'DISABLED' if disable_focus_rounds else 'enabled'}")
    print(f"  Extreme stacks:  {'DISABLED (flat moderate band)' if disable_extreme_stacks else 'enabled (Phase 3: 10-300bb)'}")
    if stack_depth_mix is not None:
        print(f"  Stack depth mix: {stack_depth_mix} (V15 DoN-shaped per-hand bands)")
    if fixed_stack_bb is not None:
        print(f"  Fixed stacks:    {fixed_stack_bb} bb (all curriculum stack sizing OFF)")
    print(f"  Aux loss weight: {aux_loss_weight}{'  (aux heads OFF)' if aux_loss_weight == 0 else ''}"
          f"  [bluff={aux_loss_weight_bluff if aux_loss_weight_bluff is not None else aux_loss_weight}, "
          f"strength={aux_loss_weight_strength if aux_loss_weight_strength is not None else aux_loss_weight}, "
          f"equity={aux_loss_weight_equity if aux_loss_weight_equity is not None else aux_loss_weight}]")
    print(f"  Target clip:     {target_clip_bb} bb")
    print(f"  Actor sharpen:   target_temp={policy_target_temp}  entropy_penalty={policy_entropy_penalty}")
    print(f"  Policy target:   {policy_target_source}  (actor regresses toward this action-value source)")
    print(f"  Policy tightness:{policy_tightness_bb} bb (realization discount below eq {POLICY_TIGHTNESS_PIVOT})")
    print(f"  Range-aware eq:  {'ON (hero equity vs opp VPIP-color ranges)' if range_aware_equity else 'off (vs random)'}")
    if ablate_hole_cards:
        print(f"  Hole cards:      ABLATED (zeroed) -- forcing equity/board reliance")
    print(f"  Bootstrap:       {'DISABLED (pure model)' if disable_bootstrap else 'enabled'}   "
          f"Exploration: {'DISABLED' if disable_exploration else 'enabled'}")
    print(f"  Checkpoint dump: every {t_cfg.get('checkpoint_dump_interval', 25000):,} hands "
          f"(restore points in weights/checkpoints/)")

    run_training(
        personality=args.personality,
        num_hands=target_hands,
        batch_size=t_cfg.get('batch_size', 256),
        epochs_per_batch=t_cfg.get('epochs_per_batch', 3),
        sim_batch_size=t_cfg.get('sim_batch_size', 2000),
        lr=float(t_cfg.get('learning_rate', 1e-3)),
        equity_sims=t_cfg.get('equity_sims', 200),
        save_name=args.save_name,
        resume_path=args.resume_path,
        initial_hands_done=args.hands_done,
        mid_flight_diagnostics_interval=t_cfg.get('mid_flight_diagnostics_interval', 10000),
        checkpoint_dump_interval=t_cfg.get('checkpoint_dump_interval', 25000),
        opp_pool_config=opp_pool_config, live_players=live_players,
        disable_focus_rounds=disable_focus_rounds,
        disable_extreme_stacks=disable_extreme_stacks, fixed_stack_bb=fixed_stack_bb,
        disable_exploration=disable_exploration, disable_bootstrap=disable_bootstrap,
        aux_loss_weight=aux_loss_weight,
        aux_loss_weight_bluff=aux_loss_weight_bluff, aux_loss_weight_strength=aux_loss_weight_strength,
        aux_loss_weight_equity=aux_loss_weight_equity,
        target_clip_bb=target_clip_bb, policy_target_temp=policy_target_temp,
        policy_entropy_penalty=policy_entropy_penalty, policy_target_source=policy_target_source,
        ablate_hole_cards=ablate_hole_cards, policy_tightness_bb=policy_tightness_bb,
        range_aware_equity=range_aware_equity,
        stack_depth_mix=stack_depth_mix,
    )

import os
import random
from core.models.engine import ModelEngine
from core.models.v13_engine import V13ModelEngine
from core.models.v14_engine import V14ModelEngine, V14_ACTION_KEYS
from core.models.v15_engine import V15ModelEngine
from core.models.v17_engine import V17ModelEngine
from core.models.v17_gauntlet_engine import V17GauntletModelEngine
from core.models.v19_engine import V19ModelEngine
from core.models.v20_engine import V20ModelEngine
from core.models.v20_preflopEq_engine import V20PreflopEqModelEngine
from core.models.v20_preflopEq_AI_engine import V20PreflopEqAIModelEngine

# Live action selection: SAMPLE from the actor policy (matching training/eval, which sample rather
# than argmax) but SHARPEN with a temperature < 1 so genuine mixing survives on close spots while
# rare low-probability actions ("spew") are suppressed. Lower = closer to argmax. Tune here.
# REVERTED (2026-07-16): briefly tried 0.4 after a user request to sharpen "a bit" following a live
# board review. Live data straddling that change showed a much bigger, one-sided effect than
# intended -- rough VPIP dropped from ~23% to ~8% across the boards immediately before/after
# (sharpening pushes any already-fold-leaning marginal spot much more decisively to fold; it isn't
# symmetric, and only one example -- where CALL was the argmax -- was checked before shipping it).
# Reverted to the value `tools/model_verify --full` actually validated. Any future retune should be
# checked against model_verify AT the candidate temperature before going live again, not iterated
# blind on live tables.
LIVE_POLICY_TEMPERATURE = 0.5

# Stack-scaled sampling temperature [V16 P2]: at short stacks (<= SHORT_STACK_BB) the correct
# strategy is near-pure push/fold, so sampling at the base temperature occasionally fires a
# dominated action ("spew" -- observed live: folded 50% eq HU, spew-raised 2-14% air). Ease from
# near-argmax at short stacks up to the base LIVE_POLICY_TEMPERATURE by DEEP_STACK_BB, where
# genuine mixing has real value.
SHORT_STACK_BB = 8.0
DEEP_STACK_BB = 20.0
SHORT_STACK_TEMPERATURE = 0.2  # reverted alongside LIVE_POLICY_TEMPERATURE above


def _stack_scaled_temperature(board_state):
    bb = float(getattr(board_state, 'big_blind', 10.0) or 10.0)
    stack = float(getattr(board_state, 'hero_stack', 0.0) or 0.0)
    stack_bb = (stack / bb) if bb > 0 else DEEP_STACK_BB
    if stack_bb <= SHORT_STACK_BB:
        return SHORT_STACK_TEMPERATURE
    if stack_bb >= DEEP_STACK_BB:
        return LIVE_POLICY_TEMPERATURE
    t = (stack_bb - SHORT_STACK_BB) / (DEEP_STACK_BB - SHORT_STACK_BB)
    return SHORT_STACK_TEMPERATURE + t * (LIVE_POLICY_TEMPERATURE - SHORT_STACK_TEMPERATURE)

# V14 raise buckets -> pot-fraction (None = all-in). MUST match versions/v14 config
# raise_pot_fractions [0.33, 0.66, 1.0, null] and simulator._raise_size_for_fraction.
V14_RAISE_FRAC = {'RAISE_33': 0.33, 'RAISE_66': 0.66, 'RAISE_POT': 1.0, 'ALLIN': None}


def _fmt_dist(d):
    """Compact 2-decimal rendering of a policy/EV dict for the decision log."""
    try:
        return "{" + ", ".join(f"{k} {float(v):.2f}" for k, v in d.items()) + "}"
    except Exception:
        return str(d)


# Equity bands used to narrate the model's own "thinking" for the live HUD/log. Not model
# outputs -- these read the SAME real, trained-on numbers the decision already used (equity
# fed into the model as an input feature, plus the chosen action) rather than the model's
# 'bluff'/'strength'/'equity' aux heads, which currently train against OPPONENT read labels
# (opp_bluff_prob/opp_strength), not a hero self-assessment, and have aux_loss_weight=0.0 in
# every active config (v15/v16*) -- untrained, so their live outputs would be pure noise.
_THINKING_BANDS = (0.30, 0.45, 0.60, 0.80)


def _narrate_thinking(action, board_state, evs):
    """Human-readable read of WHY, built from equity (real input feature) + the chosen action.
    Deliberately not sourced from the aux heads -- see comment above _THINKING_BANDS."""
    try:
        eq = float(getattr(board_state, 'equity', 0.0) or 0.0)
    except Exception:
        return None
    act = (action or '').upper()
    is_aggro = act.startswith('RAISE') or act == 'ALLIN' or act == 'ALL_IN' or act == 'BET'
    is_fold = act == 'FOLD'
    is_call = act in ('CALL', 'CHECK')

    if is_fold:
        return f"Thinking: equity too low ({eq:.0%}) to continue profitably -- folding."
    if is_aggro:
        if eq < _THINKING_BANDS[0]:
            return f"Thinking: weak hand ({eq:.0%} equity) -- bluffing, betting on fold equity rather than hand strength."
        if eq < _THINKING_BANDS[1]:
            return f"Thinking: marginal hand ({eq:.0%} equity) -- semi-bluffing, betting for fold equity plus backup outs."
        if eq < _THINKING_BANDS[2]:
            return f"Thinking: showdown-value hand ({eq:.0%} equity) -- betting for value/protection while ahead or close."
        if eq < _THINKING_BANDS[3]:
            return f"Thinking: strong hand ({eq:.0%} equity) -- value betting to get called by worse."
        return f"Thinking: near-nuts ({eq:.0%} equity) -- betting big for maximum value."
    if is_call:
        if eq < _THINKING_BANDS[1]:
            return f"Thinking: {eq:.0%} equity -- speculative call, playing for draws/implied odds rather than a made hand."
        if eq < _THINKING_BANDS[2]:
            return f"Thinking: {eq:.0%} equity -- calling to see the next card, not strong enough to raise."
        return f"Thinking: {eq:.0%} equity -- flat-calling for value/deception, keeping weaker hands in."
    return None

from core.bridge.contract_v8_v9 import ContractV8V9
from core.bridge.v11.contract_v11 import ContractV8V9 as ContractV11
from versions.v13.core.contract import ContractV12
from versions.v20.core.contract import ContractV12 as ContractV12_v20
from versions.v20_preflopEq.core.contract import ContractV12 as ContractV12_v20PreflopEq
from core.board_state import BoardState
from core.action import PokerAction

class PokerDecisionEngine:
    def __init__(self, game_type="limit"):
        self.game_type = game_type
        # LIVE MODEL REGISTRY. Only models that actually LOAD belong here — a model that fails
        # to load outputs random actions, which is dangerous at a real table. The legacy
        # v8/v9/v10 weights are missing from core/weights/ and the v11 checkpoints are the old
        # 159-feature contract (mismatch vs the 163-feature architecture), so all of them were
        # loading garbage. They are pruned; their weight files remain on disk for reproducibility.
        # To re-enable a legacy model, first fix its weights/contract, THEN re-add it here.
        self.models = {
            # V20_preflopEq_AI: IDENTICAL architecture/tensor schema to V20_preflopEq (context_dim=37,
            # contract_version=5) -- this version only changed the training opponent pool (shifted
            # toward real NN opponents: a lagged self-play mirror + V20_preflopEq's own 25k/50k
            # checkpoints, testing whether that reduces the shove-preference traced to the heuristic
            # bots' price-insensitive value-branch -- see versions/v20_preflopEq_AI/SPECS.md). Shares
            # bridge_v20_preflopEq (same contract class, no new bridge needed), gated by
            # is_v20_preflopEq_AI_model below (checked BEFORE is_v20_preflopEq_model -- same substring
            # trap as v20/v20_preflopeq: 'v20_preflopeq' is contained in 'v20_preflopeq_ai'). Loads
            # `expert_main.pth`, the 150k-hand final checkpoint. model_verify --full @ 150k: 12 PASS/
            # 1 WARN/1 FAIL/0 SKIP -- the sizing-diversity hypothesis did NOT pan out (action_diversity
            # stayed allin-dominant, deep_stack_ood_guard still FAILs, same as V20_preflopEq), but the
            # model is a clear overall improvement: beats_frozen_predecessor actually RAN this time
            # (same architecture as V20_preflopEq, no scale-mismatch skip) and PASSED at +53.5 BB/100
            # vs a field including frozen V20_preflopEq -- the first real validated predecessor win
            # this lineage has managed. Also beats V20_preflopEq's own bb100_vs_standard_fields
            # baseline in all 4 fields and shows stronger vpip_adapts_to_style deltas. Deployed for
            # user testing per explicit request (2026-07-17). V20_preflopEq and V20 both stay fully
            # intact below as rollback options. ACTIVE.
            'Herocules (v20_preflopEq_AI)': V20PreflopEqAIModelEngine(weight_name="expert_main.pth"),
            # V20_preflopEq: same PokerEVModelV4 arch + 6-action contract as V20, but WIDER context
            # (context_dim 35->37, contract_version 4->5) -- two new appended features (equity_edge,
            # hand_strength) plus a fix to the shared range-aware equity function (hero's opponents
            # now split front/already-acted [guaranteed in] vs after/still-to-act [normal VPIP roll],
            # instead of one flat roll for everyone). DIFFERENT input SCALE+WIDTH than every other
            # model here -- uses its OWN bridge (self.bridge_v20_preflopEq, versions.v20_preflopEq
            # .core.contract), gated by is_v20_preflopEq below. Loads `expert_main.pth`, the 75k-hand
            # final checkpoint from this version's first production run (see
            # versions/v20_preflopEq/SPECS.md). model_verify --full @ 75k: 11 PASS/1 WARN/1 FAIL/
            # 1 SKIP -- vpip_adapts_to_style PASS (short +6.6pt, deep +7.1pt, the metric most directly
            # downstream of the equity fix), bb100_vs_standard_fields PASS positive across all 4
            # fields, beats_offformula_stress PASS, both new-feature sensitivity checks PASS
            # (confirmed load-bearing). deep_stack_ood_guard FAIL / free_check_low_fold WARN are the
            # SAME long-standing soft spots V19/V20 also carry, not new. beats_frozen_predecessor
            # SKIPs (no cross-scale-compatible frozen checkpoint -- same limitation V20 itself hit) --
            # NO direct head-to-head number exists against V20 specifically. Deployed anyway per
            # explicit user decision (2026-07-17), accepting that gap for the strong broad evidence
            # above. V20 (below) stays fully intact as the rollback. ACTIVE.
            'Herocules (v20_preflopEq)': V20PreflopEqModelEngine(weight_name="expert_main.pth"),
            # V20: rescaled context-feature resolution (stack/pot/call_amount ctx[1]/ctx[2]/ctx[9]
            # + 5x opp_stack, /400(/1000) -> /100(/250)) to fit the actual 5-50bb training range --
            # DIFFERENT input SCALE than every other model here (contract_version 3->4), so it uses
            # its OWN bridge (self.bridge_v20, versions.v20.core.contract), gated by is_v20_model
            # below -- NOT the shared bridge_v13 every other sized model was trained on. Loads
            # `expert_main_200k.pth`, a preserved snapshot cloned at the end of the 120k->200k
            # continuation (see versions/v20/SPECS.md) so live weights stay fixed regardless of any
            # further training. LIVE-SAFETY CLAMP applied before deployment: the rescale's
            # resolution gain trades away headroom past the 50bb training ceiling -- contract.py
            # clamps stack/pot/call_amount-derived features to that ceiling so real 80-150bb+
            # tables stay in-distribution (verified: set-of-aces fold rate no longer climbs with
            # real depth). model_verify --full @ 120k: 9 PASS/1 WARN/1 FAIL/1 SKIP. @ 200k:
            # 8 PASS/2 WARN/1 FAIL/1 SKIP -- a real tradeoff, not a strict improvement:
            # deep_stack_ood_guard's failure narrowed sharply (1 failing cell vs 5 uniformly-jamming
            # cells at 120k) but short_stack_polarization flipped PASS->WARN ([P3] shove-or-fold
            # call-mass roughly doubled, 0.12->0.25, still open). Deployed at 200k per explicit user
            # decision (accepting the short-stack regression for the deep-stack gain). ACTIVE.
            'Herocules (v20)': V20ModelEngine(weight_name="expert_main_200k.pth"),
            # V19: three targeted content fixes on top of v18's opponent-architecture refactor
            # (same 6-action sized contract, same PokerEVModelV4 arch): [P0] a size-aware preflop
            # opponent fold-bar (targets the deep-stack trash-jam target-EV inflation), a real
            # button-relative hero_position fed to every training-time query (Hero's own AND every
            # opponent's -- previously silently defaulted to Button for all of them), and a
            # Past-Self VPIP mystery investigation (documented, not fixed -- see
            # versions/v19/SPECS.md). model_verify --full: 10 PASS/1 WARN/1 FAIL --
            # deep_stack_ood_guard STILL FAILS (NOT fixed by [P0]; the failure grid is roughly
            # FLAT across stack depth, eq>=0.43 argmax-jams 13/25 cells regardless of stack --
            # doesn't match the stack-scaling hypothesis [P0] targeted, points instead at a
            # `policy_tightness_bb` threshold effect near eq 0.45, not yet investigated). Deployed
            # anyway per explicit user decision (2026-07-16): every other gate passes strongly --
            # vpip_adapts_to_style short +9.7pt/deep +6.8pt, bb100_vs_standard_fields positive
            # across all 4 fields, beats_frozen_predecessor +56.8 BB/100 vs the v17_gauntlet field,
            # beats_offformula_stress PASS. deep_stack_ood_guard carried forward as backlog. ACTIVE.
            'Herocules (v19)': V19ModelEngine(weight_name="expert_main.pth"),
            # V17_gauntlet: same actor-critic/fold-relative recipe as V17, opponent pool widened
            # to frozen V15 (nit seat) and a true lagged self-play mirror (past seat) -- the tag
            # seat was INTENDED to load frozen V16 but a wiring bug silently nullified it (see
            # versions/v17_gauntlet/SPECS.md "CORRECTION"); this checkpoint actually trained
            # against the TAG heuristic there, not frozen V16. Still a real, valid improvement --
            # per-seat action-forcing bypassed for the real models that DID load correctly.
            # Beats V17 on 3/4 bb100_vs_standard_fields (loose_short +28.9->+32.5, tight_short
            # +18.4->+26.8, tight_deep +32.6->+35.4; loose_deep +90.3->+69.8 -- still strongly
            # positive, reads as more balanced not a regression) and more than doubles V17's
            # deep-stack vpip_adapts_to_style delta (+5.8pt->+12.3pt). model_verify: 10 PASS/1
            # WARN/1 FAIL (the FAIL is the same pre-existing deep-stack OOD defect every version
            # in this line carries, tracked as V18 [P0]). Beats frozen-V17 by +84.3 BB/100. ACTIVE.
            'Herocules (v17_gauntlet)': V17GauntletModelEngine(weight_name="expert_main.pth"),
            # V17: same 6-action sized contract as V14/V15, actor regret-matching routed through
            # the critic's own (detached) Q-values past 30k hands with a fold-relative baseline.
            # Fixes the air/draws overcontinuation V16 had (air_folds_mostly 0.62->1.00) WITHOUT
            # v16_foldregret's style-flip regression (loose_deep BB/100 +62.1->+90.3, not a
            # collapse). Superseded by V17_gauntlet above; kept as fallback.
            'Herocules (v17 Actor-Critic)': V17ModelEngine(weight_name="expert_main.pth"),
            # V15: same 6-action sized contract as V14, retrained on a DoN-shaped stack mixture
            # (5-50bb) + a frozen-V14 expert opponent. Fixes v14's deep-stack OOD; loose-aggressive
            # style that crushes loose/station fields (the live population). Kept as fallback.
            'Herocules (v15 DoN)': V15ModelEngine(weight_name="expert_main.pth"),
            # V14: discretized bet-size action space; short-stack winner. Kept as fallback.
            'Herocules (v14 Sized)': V14ModelEngine(weight_name="expert_main.pth"),
            # V13: equity-primary + range-aware equity. Kept as the tagged MILESTONE fallback.
            'Herocules (v13 Range-Aware)': V13ModelEngine(weight_name="expert_main.pth"),
        }
        self.active_model_name = 'Herocules (v20_preflopEq_AI)'
        self.bridge_v9 = ContractV8V9()
        self.bridge_v11 = ContractV11()
        self.bridge_v13 = ContractV12(max_seq_len=20)
        # V20 uses a DIFFERENT context-feature scale (contract_version 4, see the registry
        # comment above) -- it CANNOT share bridge_v13 with every other sized model, or it would
        # get fed the wrong-scale stack/pot/call_amount values it was never trained on.
        self.bridge_v20 = ContractV12_v20(max_seq_len=20)
        # V20_preflopEq uses a DIFFERENT scale+width again (contract_version 5, context_dim 37,
        # see the registry comment above) -- needs its own bridge, can't share bridge_v20 either.
        self.bridge_v20_preflopEq = ContractV12_v20PreflopEq(max_seq_len=20)
        self.hand_history_buffer = []
        self._last_street = None
        self._last_hole_cards = None

    def set_active_model(self, model_name: str, tree_file: str = None):
        if model_name in self.models:
            self.active_model_name = model_name
        else:
            print(f"Warning: {model_name} is not loaded or supported. Falling back to V20.")
            self.active_model_name = 'Herocules (v20)'

    def _v14_size_to_slider(self, frac, board_state):
        """Translate a V14 pot-fraction raise bucket into (raise_size_chips, slider_fraction).

        raise_size mirrors simulator._raise_size_for_fraction EXACTLY (train/serve consistency):
        frac None -> all-in (full stack); else min(pot*frac, stack) floored at a legal min-raise
        (to_call + BB) and capped at the stack. That chip amount is then mapped to the live client's
        raise SLIDER, whose 0.0 end = 1 small blind and 1.0 end = the whole stack (all-in) — the
        same convention core/models/heuristic.py already uses to drive the slider live.
        """
        pot = float(board_state.pot_size or 0.0)
        to_call = float(getattr(board_state, 'call_amount', 0.0) or 0.0)
        hero_stack = float(getattr(board_state, 'hero_stack', 0.0) or 0.0)
        bb = float(getattr(board_state, 'big_blind', 10.0) or 10.0)

        if frac is None:
            raise_size = hero_stack                       # all-in
        else:
            raise_size = min(pot * frac, hero_stack)
            raise_size = max(raise_size, to_call + bb)    # >= min-raise
            raise_size = min(raise_size, hero_stack)

        sb = max(1.0, bb / 2.0)                            # slider 0.0 == 1 small blind
        lo, hi = sb, hero_stack
        if frac is None or hi <= lo:
            slider = 1.0                                   # all-in -> slam slider to the right
        else:
            slider = max(0.0, min(1.0, (raise_size - lo) / (hi - lo)))
        return raise_size, slider

    def make_decision(self, board_state: BoardState,
                      use_preflop_chart: bool = True,
                      use_math_engine: bool = True,
                      use_bluff_engine: bool = True,
                      use_dynamic_sizing: bool = True,
                      bet_raise_available: bool = True,
                      check_call_available: bool = True,
                      action_history_raw: list = None):
        
        active_model = self.models.get(self.active_model_name)
        if not active_model:
            return 'FOLD', "Model not found", 0.0, {}

        # Round Reset Logic (Sandbox tracking)
        street_order = {'Preflop': 0, 'Flop': 1, 'Turn': 2, 'River': 3}
        current_street_val = street_order.get(board_state.street, 0)
        last_street_val = street_order.get(self._last_street, -1)
        
        current_hole_cards = sorted([str(c) for c in board_state.hero_cards]) if board_state.hero_cards else []
        # Reset if hole cards changed or street went backwards (e.g., new hand started)
        if current_hole_cards != self._last_hole_cards or current_street_val < last_street_val:
            self.hand_history_buffer = []
            
        self._last_street = board_state.street
        self._last_hole_cards = current_hole_cards
        
        self.hand_history_buffer.append(board_state)

        is_v13_model = getattr(active_model, 'is_v13', False) or 'v13' in self.active_model_name.lower()
        is_v14_model = getattr(active_model, 'is_v14', False) or 'v14' in self.active_model_name.lower()
        is_v15_model = getattr(active_model, 'is_v15', False) or 'v15' in self.active_model_name.lower()
        is_v17_gauntlet_model = getattr(active_model, 'is_v17_gauntlet', False) or 'v17_gauntlet' in self.active_model_name.lower()
        is_v17_model = (not is_v17_gauntlet_model) and (getattr(active_model, 'is_v17', False) or 'v17' in self.active_model_name.lower())
        is_v19_model = getattr(active_model, 'is_v19', False) or 'v19' in self.active_model_name.lower()
        # NOTE: resolution order matters -- 'v20_preflopeq' is a substring of 'v20_preflopeq_ai',
        # and 'v20' is a substring of both, so each must be checked BEFORE the shorter name it
        # contains or the naive name-based fallback misfires true for the wrong model.
        is_v20_preflopEq_AI_model = getattr(active_model, 'is_v20_preflopEq_AI', False) or 'v20_preflopeq_ai' in self.active_model_name.lower()
        is_v20_preflopEq_model = (not is_v20_preflopEq_AI_model) and (getattr(active_model, 'is_v20_preflopEq', False) or 'v20_preflopeq' in self.active_model_name.lower())
        is_v20_model = (not is_v20_preflopEq_model) and (not is_v20_preflopEq_AI_model) and (getattr(active_model, 'is_v20', False) or 'v20' in self.active_model_name.lower())
        is_v11_model = getattr(active_model, 'is_v11', False) or 'v11' in self.active_model_name.lower()
        # V14/V15/V17/V17_gauntlet/V19/V20/V20_preflopEq/V20_preflopEq_AI share the IDENTICAL
        # 6-action sized contract -> one live path (selection + slider sizing). Anything gated on
        # "the sized model" uses this flag.
        is_sized_model = is_v14_model or is_v15_model or is_v17_model or is_v17_gauntlet_model or is_v19_model or is_v20_model or is_v20_preflopEq_model or is_v20_preflopEq_AI_model
        try:
            if is_v20_preflopEq_model or is_v20_preflopEq_AI_model:
                # V20_preflopEq / V20_preflopEq_AI (identical architecture, different training
                # opponent pool only) were trained on a DIFFERENT context-feature WIDTH+scale
                # (contract_version 5, context_dim 37) -- MUST use their shared bridge, not
                # bridge_v13 or bridge_v20, or the two appended features (equity_edge,
                # hand_strength) would be missing/misaligned (see
                # versions/v20_preflopEq/core/contract.py).
                hole, board, ctx, act = self.bridge_v20_preflopEq.to_tensors(self.hand_history_buffer, action_history_raw)
            elif is_v20_model:
                # V20 was trained on a DIFFERENT context-feature scale (contract_version 4) --
                # MUST use its own bridge, not bridge_v13, or stack/pot/call_amount would be fed
                # at the wrong scale (see versions/v20/core/contract.py, versions/v20/SPECS.md).
                hole, board, ctx, act = self.bridge_v20.to_tensors(self.hand_history_buffer, action_history_raw)
            elif is_v13_model or is_sized_model:
                # V13/V14/V15/V17 share the sequence contract (ContractV12, 35-dim ctx) and the
                # ACTOR policy head; V14/V15/V17 only differ in head WIDTH (6 vs 3), applied after
                # inference.
                hole, board, ctx, act = self.bridge_v13.to_tensors(self.hand_history_buffer, action_history_raw)
            elif getattr(active_model, 'is_v11', False) or 'v11' in self.active_model_name.lower():
                hole, board, ctx, act = self.bridge_v11.to_tensors(self.hand_history_buffer, action_history_raw)
            else:
                hole, board, ctx, act = self.bridge_v9.to_tensors(board_state, action_history_raw)
            evs = active_model.predict_ev(hole, board, ctx, act)
        except Exception as e:
            return 'FOLD', f"Fatal decision engine crash: {e}", 0.0, {}

        # 1. Base Model Decision.
        # Actor-policy models (v13/v11) output a probability distribution over {FOLD,CALL,RAISE}.
        # We SAMPLE from it (matching training/eval, which sample rather than argmax) with
        # temperature sharpening, preserving the model's game-theoretic mixing while suppressing
        # rare low-prob 'spew'. Legacy critic-only checkpoints keep argmax over Q.
        # Train/serve consistency: FOLD is strictly dominated when checking is free (call_amount==0),
        # so the sim zeroes it before selecting (simulator._select_action) — we mirror that here.
        is_actor_policy = is_v13_model or is_v11_model or is_sized_model
        # call_amount can be None on a parse miss -> treat unknown as "facing a bet" (do NOT mask
        # fold), which is the safe default (never auto-check/call into an unknown price).
        _ca = getattr(board_state, 'call_amount', 0)
        free_check = _ca is not None and _ca <= 0
        argmax_action = max(evs, key=evs.get)
        free_fold_overridden = free_check and argmax_action == 'FOLD'

        bet_size = 0.0
        sampled_probs = None   # normalized post-temperature distribution the sampler actually drew from
        temp = _stack_scaled_temperature(board_state)
        if is_sized_model:
            # V14/V15 6-way policy {FOLD,CALL,RAISE_33,RAISE_66,RAISE_POT,ALLIN}. Sharpen+sample as
            # for v13, then TRANSLATE the chosen raise bucket into a slider-sized live action so the
            # client actually raises-to-X / shoves (the P1c fix v13 lacked).
            keys = list(V14_ACTION_KEYS)
            probs = {a: max(0.0, float(evs.get(a, 0.0))) for a in keys}
            if free_check:
                probs['FOLD'] = 0.0   # never fold a free option
            if not bet_raise_available:
                for rk in V14_RAISE_FRAC:   # raise button gone -> only fold/call/check are legal
                    probs[rk] = 0.0
            sharp = {a: (v ** (1.0 / temp)) for a, v in probs.items()}
            names = [a for a in keys if sharp[a] > 0.0]
            sharp_total = sum(sharp[a] for a in names) or 1.0
            sampled_probs = {a: (sharp[a] / sharp_total if a in names else 0.0) for a in keys}
            choice = random.choices(names, weights=[sharp[a] for a in names], k=1)[0] if names \
                else ('CALL' if free_check or not bet_raise_available else 'FOLD')

            chosen_key = choice   # the raw policy bucket the sampler picked, e.g. RAISE_66 or ALLIN
            if choice in V14_RAISE_FRAC:
                raise_size, slider = self._v14_size_to_slider(V14_RAISE_FRAC[choice], board_state)
                action = f"RAISE_SLIDER_{slider:.2f}"
                bet_size = raise_size
                size_note = f"{choice} -> raise-to {raise_size:.0f} (slider {slider:.2f})"
            else:
                action = choice   # FOLD / CALL
                size_note = choice
            _tag = "V20_preflopEq_AI" if is_v20_preflopEq_AI_model else ("V20_preflopEq" if is_v20_preflopEq_model else ("V20" if is_v20_model else ("V19" if is_v19_model else ("V17_gauntlet" if is_v17_gauntlet_model else ("V17" if is_v17_model else ("V15" if is_v15_model else "V14"))))))
            reason = (f"{_tag} sampled (temp={temp:.2f}"
                      + (", free-check fold-masked" if free_check else "")
                      + (", raise-unavail" if not bet_raise_available else "")
                      + f") -> {size_note}: {_fmt_dist(evs)}")
        elif is_actor_policy:
            probs = {a: max(0.0, float(evs.get(a, 0.0))) for a in ('FOLD', 'CALL', 'RAISE')}
            if free_check:
                probs['FOLD'] = 0.0   # never fold a free option
            sharp = {a: (v ** (1.0 / temp)) for a, v in probs.items()}
            names = [a for a in ('FOLD', 'CALL', 'RAISE') if sharp[a] > 0.0]
            sharp_total = sum(sharp[a] for a in names) or 1.0
            sampled_probs = {a: (sharp[a] / sharp_total if a in names else 0.0) for a in ('FOLD', 'CALL', 'RAISE')}
            if names:
                action = random.choices(names, weights=[sharp[a] for a in names], k=1)[0]
            else:
                action = 'CALL' if free_check else 'FOLD'   # degenerate safety net
            chosen_key = action
            reason = (f"Sampled policy (temp={temp:.2f}"
                      + (", free-check fold-masked" if free_fold_overridden else "")
                      + f") -> {action}: {_fmt_dist(evs)}")
        else:
            action = argmax_action
            if free_fold_overridden:
                non_fold = {k: v for k, v in evs.items() if k in ('CALL', 'RAISE')}
                action = max(non_fold, key=non_fold.get) if non_fold else action
            chosen_key = action
            reason = f"Model Output: {_fmt_dist(evs)}"

        # Determine basic bet size (legacy 3-way RAISE only; V14/V15 already sized their raise above).
        if action == 'RAISE' and not is_sized_model:
            if use_dynamic_sizing:
                bet_size = board_state.pot_size * 0.75
            else:
                bet_size = board_state.big_blind * 3

        # 2. V9 River Air Guardrail
        if board_state.street == 'River' and board_state.equity < 0.35 and board_state.call_amount > 0:
             if self.active_model_name == 'Herocules (v9 Main)' and action in ('RAISE', 'ALL_IN'):
                  action = 'FOLD'
                  reason = "Guardrail: V9 River Air Defense (Equity too low to bluff shove)"
                  bet_size = 0.0

        # 3. Apply Post-flop Math Engine Guardrail (Pot Odds Check)
        math_engine_status = "Passed"
        math_engine_details = "Math checks out OK"
        # Bypass math engine for V11, V13, V14 AND V15 models to preserve their trained-policy
        # behavior (they already reason about pot odds + opponent range via range-aware equity). The
        # math override also can't parse the sized RAISE_SLIDER_x action string anyway.
        is_v11_model = getattr(active_model, 'is_v11', False) or 'v11' in self.active_model_name.lower()
        if use_math_engine and board_state.street != 'Preflop' and board_state.call_amount > 0 and not is_v11_model and not is_v13_model and not is_sized_model:
            pot_odds = board_state.call_amount / (board_state.pot_size + board_state.call_amount)
            
            buffer_offset = -0.05
            profile_desc = "Default"
            
            # Very simplistic HUD aggregation for the math buffer
            is_maniac = any(s.hud.agg_color == 'Red' or s.hud.vpip_color == 'Red' for s in board_state.seats.values() if s.is_active)
            is_loose = any(s.hud.agg_color == 'Yellow' or s.hud.vpip_color == 'Yellow' for s in board_state.seats.values() if s.is_active)
            is_nit = any(s.hud.vpip_color == 'Blue' and s.hud.agg_color in ('Blue', None) for s in board_state.seats.values() if s.is_active)
            is_tight = any(s.hud.vpip_color == 'Blue' for s in board_state.seats.values() if s.is_active)
            
            if is_maniac:
                buffer_offset = -0.12
                profile_desc = "Maniac"
            elif is_loose:
                buffer_offset = -0.09
                profile_desc = "Loose"
            elif is_nit:
                buffer_offset = 0.02
                profile_desc = "Nit"
            elif is_tight:
                buffer_offset = 0.00
                profile_desc = "Tight"
            
            if board_state.equity < pot_odds + buffer_offset:
                if action not in ('FOLD', 'CHECK'):
                    action = 'FOLD'
                    reason = f"Math Override ({profile_desc} HUD): Equity ({board_state.equity:.1%}) < Pot Odds ({pot_odds:.1%}) + Buffer ({buffer_offset:+.1%}). Overriding to FOLD."
                    bet_size = 0.0
                    math_engine_status = "Triggered (Active)"
                    math_engine_details = f"Overridden to FOLD: Equity < Pot Odds + Buffer"

        # Construct decision path for UI
        decision_path = {
            'preflop_chart': {'status': "Bypassed", 'details': "Deprecated in V8/V9"},
            'active_model': {'status': "Active", 'details': f"Loaded {self.active_model_name}"},
            'bluff_engine': {'status': "Passed", 'details': "Handled natively by V8/V9"},
            'math_engine': {'status': math_engine_status, 'details': math_engine_details}
        }
        
        ev_dict = evs.copy()
        ev_dict['decision_path'] = decision_path
        ev_dict['thinking'] = _narrate_thinking(action, board_state, evs)
        # The raw policy bucket the sampler ("dice roll") actually picked, e.g. RAISE_66 or ALLIN --
        # distinct from `action`, which may be a translated slider string (RAISE_SLIDER_x) or later
        # overridden by a safety guardrail. Lets the UI highlight the sampled bar even when the
        # executed action differs. Read by PHPHelp.py's action-distribution panel.
        ev_dict['chosen_key'] = chosen_key
        # The temperature-sharpened, RENORMALIZED distribution the sampler actually drew `chosen_key`
        # from -- distinct from the raw actor probabilities already at the top level of ev_dict.
        # None for the legacy argmax/non-actor-critic path (no sampling happens there). Lets the UI
        # show both the model's raw output and what the live temperature scaling actually did to it.
        ev_dict['sampled_probs'] = sampled_probs
        ev_dict['temperature'] = temp
        # Diagnostics only (added AFTER argmax so it can't affect the chosen action): the critic's
        # per-action Q (EV vs fold, ~BB). None for non-actor-critic models. Read by F12 diagnostics.
        ev_dict['q_vals'] = getattr(active_model, 'last_q_vals', None)
        # Replay payload: the exact model-input tensors for this decision, so a future replay engine
        # can reload and re-run the forward pass faithfully (no re-derivation from raw state needed).
        try:
            ev_dict['model_input'] = {
                'hole': hole.tolist(), 'board': board.tolist(),
                'ctx': ctx.tolist(), 'act': act.tolist(),
            }
        except Exception:
            ev_dict['model_input'] = None

        return action, reason, bet_size, ev_dict

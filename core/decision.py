import os
import random
from core.models.engine import ModelEngine
from core.models.v13_engine import V13ModelEngine
from core.models.v14_engine import V14ModelEngine, V14_ACTION_KEYS
from core.models.v15_engine import V15ModelEngine

# Live action selection: SAMPLE from the actor policy (matching training/eval, which sample rather
# than argmax) but SHARPEN with a temperature < 1 so genuine mixing survives on close spots while
# rare low-probability actions ("spew") are suppressed. Lower = closer to argmax. Tune here.
LIVE_POLICY_TEMPERATURE = 0.5

# Stack-scaled sampling temperature [V16 P2]: at short stacks (<= SHORT_STACK_BB) the correct
# strategy is near-pure push/fold, so sampling at the base temperature occasionally fires a
# dominated action ("spew" -- observed live: folded 50% eq HU, spew-raised 2-14% air). Ease from
# near-argmax at short stacks up to the base LIVE_POLICY_TEMPERATURE by DEEP_STACK_BB, where
# genuine mixing has real value.
SHORT_STACK_BB = 8.0
DEEP_STACK_BB = 20.0
SHORT_STACK_TEMPERATURE = 0.2


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

from core.bridge.contract_v8_v9 import ContractV8V9
from core.bridge.v11.contract_v11 import ContractV8V9 as ContractV11
from versions.v13.core.contract import ContractV12
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
            # V15: same 6-action sized contract as V14, retrained on a DoN-shaped stack mixture
            # (5-50bb) + a frozen-V14 expert opponent. Fixes v14's deep-stack OOD; loose-aggressive
            # style that crushes loose/station fields (the live population). ACTIVE.
            'Herocules (v15 DoN)': V15ModelEngine(weight_name="expert_main.pth"),
            # V14: discretized bet-size action space; short-stack winner. Kept as fallback.
            'Herocules (v14 Sized)': V14ModelEngine(weight_name="expert_main.pth"),
            # V13: equity-primary + range-aware equity. Kept as the tagged MILESTONE fallback.
            'Herocules (v13 Range-Aware)': V13ModelEngine(weight_name="expert_main.pth"),
        }
        self.active_model_name = 'Herocules (v15 DoN)'
        self.bridge_v9 = ContractV8V9()
        self.bridge_v11 = ContractV11()
        self.bridge_v13 = ContractV12(max_seq_len=20)
        self.hand_history_buffer = []
        self._last_street = None
        self._last_hole_cards = None

    def set_active_model(self, model_name: str, tree_file: str = None):
        if model_name in self.models:
            self.active_model_name = model_name
        else:
            print(f"Warning: {model_name} is not loaded or supported. Falling back to V15.")
            self.active_model_name = 'Herocules (v15 DoN)'

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
        is_v11_model = getattr(active_model, 'is_v11', False) or 'v11' in self.active_model_name.lower()
        # V14 and V15 share the IDENTICAL 6-action sized contract -> one live path (selection +
        # slider sizing). Anything gated on "the sized model" uses this combined flag.
        is_sized_model = is_v14_model or is_v15_model
        try:
            if is_v13_model or is_sized_model:
                # V13/V14/V15 share the sequence contract (ContractV12, 35-dim ctx) and the ACTOR
                # policy head; V14/V15 only differ in head WIDTH (6 vs 3), applied after inference.
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
            choice = random.choices(names, weights=[sharp[a] for a in names], k=1)[0] if names \
                else ('CALL' if free_check or not bet_raise_available else 'FOLD')

            if choice in V14_RAISE_FRAC:
                raise_size, slider = self._v14_size_to_slider(V14_RAISE_FRAC[choice], board_state)
                action = f"RAISE_SLIDER_{slider:.2f}"
                bet_size = raise_size
                size_note = f"{choice} -> raise-to {raise_size:.0f} (slider {slider:.2f})"
            else:
                action = choice   # FOLD / CALL
                size_note = choice
            _tag = "V15" if is_v15_model else "V14"
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
            if names:
                action = random.choices(names, weights=[sharp[a] for a in names], k=1)[0]
            else:
                action = 'CALL' if free_check else 'FOLD'   # degenerate safety net
            reason = (f"Sampled policy (temp={temp:.2f}"
                      + (", free-check fold-masked" if free_fold_overridden else "")
                      + f") -> {action}: {_fmt_dist(evs)}")
        else:
            action = argmax_action
            if free_fold_overridden:
                non_fold = {k: v for k, v in evs.items() if k in ('CALL', 'RAISE')}
                action = max(non_fold, key=non_fold.get) if non_fold else action
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

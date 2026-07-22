"""The model-side half of the live handover: LiveObservation in, executable decision out.

[V45_liveHandover] Everything VERSION-SPECIFIC about turning a table snapshot into a model
decision lives behind this layer, keyed off what the active engine itself declares
(`live_features()` / `make_bridge()` / `is_sized`, resolved by core/decision.py). The dashboard
(PHPHelp.py) builds a LiveObservation and calls `PokerDecisionEngine.decide(obs, ...)` -- it no
longer computes equity, hand_strength, front/after splits, or effective_field itself, and it
never needs editing when a new version ships.

Division of labour:

  TableState.to_observation()   RAW facts, model-agnostic, rarely changes   (live layer)
  BaseLiveAdapter.decide()      version-owned interpretation of those facts (this file)
  PokerDecisionEngine.make_decision()  tensor encoding + policy transforms   (shared, unchanged)

`BaseLiveAdapter` reproduces the exact pipeline the PHPHelp call site ran before this refactor
(V42_liveFixes semantics, byte-for-byte -- see versions/v45_liveHandover/verify_handover.py), and
is parameterized entirely by the engine's own declarations. An engine that needs something the
base pipeline can't express declares `make_live_adapter(decision_engine)` and returns its own
adapter (subclass this class); none of the currently registered engines need to.
"""

from dataclasses import dataclass, field
from typing import Optional

from core.live_observation import LiveObservation, SEAT_ORDER_CLOCKWISE


# ===================================================================== #
#  Front/after classification (pure function over the observation)
# ===================================================================== #
# Port of PHPHelp._classify_opponents_by_action_order (V42_liveFixes round-2 semantics: chips in
# the pot are the criterion, position only breaks the postflop "this street?" tie). PHPHelp's
# method now delegates HERE so the two copies cannot drift. Every behavioural quirk is kept
# deliberately -- parity over improvement, this refactor moves code, not semantics:
#   - the preflop blind exemption compares BOTH blinds against the BIG blind (an SB completing to
#     exactly 1bb therefore stays 'after' -- conservative, matches the shipped V42 behaviour);
#   - first-to-act preflop is a full-ring approximation (button+3) that doesn't adjust for
#     already-folded blinds;
#   - reopened action within a street is still not detectable ([OPP-4], open).

def classify_front_after(obs: LiveObservation):
    """Split ACTIVE opponents' HUD colors into (front, after): 'front' = positively committed to
    this pot (no VPIP fold-roll in the equity model), 'after' = may still fold. Returns
    (front_colors, after_colors), or (None, None) when no ordering can be established (button not
    on a known seat). Unknown HUD colors map to 'Yellow' ([V20_preflopEq] Finding 1)."""
    if obs.dealer_idx not in range(0, 6):
        return None, None

    button_seat = 'hero' if obs.dealer_idx == 0 else f'seat_{obs.dealer_idx}'
    if button_seat not in SEAT_ORDER_CLOCKWISE:
        return None, None
    order = list(SEAT_ORDER_CLOCKWISE)
    start = order.index(button_seat)
    order = order[start:] + order[:start]              # button first, then clockwise

    is_preflop = obs.is_preflop
    first_to_act_offset = 3 if is_preflop else 1       # past the blinds preflop
    order = order[first_to_act_offset:] + order[:first_to_act_offset]

    if 'hero' not in order:
        return None, None
    hero_pos = order.index('hero')
    before_hero = order[:hero_pos]
    after_hero = order[hero_pos + 1:]

    active_by_key = {s.seat_key: s for s in obs.seats if s.occupied and s.is_active}

    def is_in_pot(seat_key):
        if seat_key == 'hero':
            committed = obs.hero_committed
            raised_this_street = obs.hero_raised_this_street
            is_blind = obs.hero_is_small_blind or obs.hero_is_big_blind
        else:
            seat = active_by_key.get(seat_key) or obs.seat(seat_key)
            if seat is None or not seat.occupied:
                return False
            committed = seat.committed
            raised_this_street = seat.raised_this_street
            is_blind = seat.is_small_blind or seat.is_big_blind

        if committed <= 0:
            return False
        # A seat recorded as raising THIS street is in, wherever it sits (a 3-bet from behind
        # hero reopens action -- position cannot see that; chips can).
        if raised_this_street:
            return True
        if is_preflop:
            # Preflop, committed chips ARE this street's chips -- except a posted blind is
            # involuntary and only counts once it exceeds the forced amount.
            if is_blind:
                return committed > obs.big_blind + 1e-9
            return True
        # Postflop, `committed` spans earlier streets; the positional read answers "acted this
        # street", gated on chips-in-hand filtering folded/phantom seats.
        return seat_key in before_hero

    def colors_for(seat_list):
        colors = []
        for seat_key in seat_list:
            seat = active_by_key.get(seat_key)         # opponents only; hero never yields a color
            if seat is not None:
                colors.append(seat.vpip_color or 'Yellow')
        return colors

    contesting = before_hero + after_hero
    confirmed_in = [s for s in contesting if is_in_pot(s)]
    may_still_fold = [s for s in contesting if not is_in_pot(s)]
    return colors_for(confirmed_in), colors_for(may_still_fold)


# ===================================================================== #
#  Observation -> BoardState (pure, mirrors TableState.to_board_state)
# ===================================================================== #

def observation_to_board_state(obs: LiveObservation, equity: float = 0.0,
                               hand_strength: float = 0.5, effective_field: float = 0.0):
    """Build the BoardState the tensor bridges consume, from the observation alone. Field-for-field
    identical to TableState.to_board_state() + the caller-side hand_strength/effective_field
    threading it replaces (verified by versions/v45_liveHandover/verify_handover.py)."""
    from core.board_state import BoardState, SeatState, HUDStats

    bs = BoardState(
        community_cards=list(obs.community_cards),
        hero_cards=list(obs.hero_cards),
        pot_size=obs.pot_size,
        hero_stack=obs.hero_stack,
        active_buttons=list(obs.active_buttons),
        dealer_idx=obs.dealer_idx,
        hero_position=obs.hero_position,
        street=obs.street,
        # [2026-07-22, flagged turn_10 JJ-fold] An OCR-missed call amount (None) must NOT
        # encode as 0.0: price-slot zeros read as a FREE CHECK to the net, whose known
        # free_check_low_fold pathology then folds ~0.9+, while the FOLD mask correctly
        # treats unknown != free (V42 #13) and does not intervene -- two layers disagreeing
        # about one number folded pocket jacks preflop. Unknown price floors at one big
        # blind (the minimum real price when facing action); a TRUE free check still
        # arrives as call_amount=0.0, never None.
        call_amount=(float(obs.call_amount) if obs.call_amount is not None
                     else float(obs.big_blind or 0.0)),
        equity=equity,
        big_blind=obs.big_blind,
        hero_committed=obs.hero_committed,
        pot_type=min(2, obs.raise_count),
    )
    bs.hand_strength = hand_strength
    bs.effective_field = float(effective_field or 0.0)

    for seat in obs.seats:
        if not seat.occupied:
            continue
        slot = seat.contract_slot or seat.seat_key     # [V42_liveFixes C3] true-position slot
        bs.seats[slot] = SeatState(
            name=seat.name,
            stack=seat.stack,
            is_active=seat.is_active,
            state_label=seat.state_label,
            hud=HUDStats(
                # [V42_liveFixes C1] unknown == average (Yellow/Green), matching training's
                # absent-profile default -- the observation carries the raw None.
                vpip_color=seat.vpip_color or 'Yellow',
                agg_color=seat.agg_color or 'Green',
            ),
            committed=seat.committed,
            raised_this_hand=seat.raised_this_hand,
            raised_this_street=seat.raised_this_street,
        )
    return bs


# ===================================================================== #
#  The decision object handed back to the dashboard
# ===================================================================== #

@dataclass
class LiveDecision:
    """Everything the dashboard needs from one decision -- the action to execute plus the
    diagnostics the HUD renders, so the dashboard never recomputes model-side values."""
    action: str = 'FOLD'
    reason: str = ''
    bet_size: float = 0.0
    ev_dict: Optional[dict] = None          # full model output (policy + Q + decision path)
    equity: float = 0.0
    sim_msg: str = ''
    equity_meta: dict = field(default_factory=dict)
    hand_strength: float = 0.5
    board_state: object = None              # the exact BoardState the tensors were built from
    observation: Optional[LiveObservation] = None

    def as_tuple(self):
        """The legacy (action, reason, bet_size, ev_dict) shape older call sites expect."""
        return self.action, self.reason, self.bet_size, self.ev_dict


# ===================================================================== #
#  BaseLiveAdapter -- the version-owned interpretation pipeline
# ===================================================================== #

class BaseLiveAdapter:
    """Turns a LiveObservation into a LiveDecision using ONLY what the engine declares.

    The pipeline (identical to the pre-refactor PHPHelp call site, V42_liveFixes semantics):
      1. resolve this version's own feature implementations (live_feature_providers)
      2. range-aware equity with the front/after split, vs-random fallback
      3. hand_strength (only if this version's contract reads it)
      4. effective_field (only if this version's contract exposes it, V44+)
      5. BoardState assembly -> make_decision (shared tensor/policy machinery, unchanged)
    """

    def __init__(self, decision_engine, model_name: str):
        self.decision_engine = decision_engine
        self.model_name = model_name

    # -- overridable seams ------------------------------------------------
    def _evaluator(self, evaluator=None):
        if evaluator is not None:
            return evaluator
        if not hasattr(self, '_own_evaluator'):
            from core.evaluator import PokerEvaluator
            self._own_evaluator = PokerEvaluator()
        return self._own_evaluator

    # -- the pipeline ------------------------------------------------------
    def decide(self, obs: LiveObservation, *, evaluator=None,
               fallback_sims: int = 2000, fallback_num_opponents: int = None,
               log_fn=None) -> LiveDecision:
        # [v46_legacySweep] The use_* layer-toggle kwargs are gone -- every override layer they
        # gated was removed from make_decision (dead code for sized models).
        log = log_fn or (lambda msg: None)
        engine = self.decision_engine
        evaluator = self._evaluator(evaluator)

        active = obs.active_seats
        detected = len(active)
        num_opponents = detected if detected > 0 else int(fallback_num_opponents or 1)

        # 1. This version's own live-feature implementations (engine-declared; 'unresolved' is a
        #    real problem the log must surface -- V42_liveFixes' founding bug).
        providers = engine.live_feature_providers(self.model_name)
        equity = None
        sim_msg = None
        equity_meta = {"method": "vs-random", "opp_colors": None, "num_opponents": num_opponents,
                       "feature_source": providers.get('source')}
        if providers.get('error'):
            log(f"[Equity] ERROR resolving live features: {providers['error']}")

        # 2. Equity, the way THIS version was trained to see it.
        compute_range_aware_equity = providers.get('equity_fn')
        front_colors = after_colors = None
        if compute_range_aware_equity is None:
            log(f"[Equity] WARNING: no range-aware equity implementation for "
                f"'{self.model_name}' (source={providers.get('source')}) -- falling back to "
                f"VS-RANDOM equity, which the model was NOT trained on.")
        else:
            try:
                opp_colors = [s.vpip_color or 'Yellow' for s in active]
                equity_meta["opp_colors"] = opp_colors
                front_colors, after_colors = classify_front_after(obs)
                equity_meta["opp_colors_in_pot"] = front_colors
                equity_meta["opp_colors_still_to_act"] = after_colors
                # sims=250 live vs training's 150: same estimator, less noise -- see the original
                # call site note (2026-07-16).
                if providers.get('use_front_colors') and front_colors is not None:
                    ra = compute_range_aware_equity(
                        list(obs.hero_cards), list(obs.community_cards),
                        after_colors, sims=250, front_colors=front_colors)
                else:
                    ra = compute_range_aware_equity(
                        list(obs.hero_cards), list(obs.community_cards),
                        opp_colors, sims=250)
                if ra is not None:
                    equity = ra
                    equity_meta["method"] = "range-aware"
                    sim_msg = f"Range-aware equity vs {opp_colors or 'random'}: {equity:.2f}"
                else:
                    equity_meta["fallback_reason"] = "range-aware returned None (no HUD colors?)"
            except Exception as e:
                equity_meta["fallback_reason"] = f"range-aware raised: {e}"
                log(f"[Equity] range-aware failed ({e}); vs-random fallback")

        if equity is None:
            equity, sim_msg = evaluator.calculate_equity(
                list(obs.community_cards), list(obs.hero_cards),
                num_opponents=num_opponents, num_simulations=fallback_sims)
        equity_meta["value"] = equity
        equity_meta["equity_edge"] = equity * (num_opponents + 1)   # display-only, contract derives its own

        # 3. hand_strength -- only when this version's contract actually reads it.
        hand_strength = 0.5
        preflop_hand_strength = providers.get('hand_strength_fn')
        if preflop_hand_strength is not None:
            try:
                if len(obs.community_cards) == 0:
                    hand_strength = preflop_hand_strength(obs.hero_cards[0], obs.hero_cards[1])
                else:
                    hand_strength, _ = evaluator.calculate_equity(
                        list(obs.community_cards), list(obs.hero_cards),
                        num_opponents=1, num_simulations=200)
            except Exception as e:
                log(f"[Equity] hand_strength computation failed ({e}); using neutral 0.5")
        equity_meta["hand_strength"] = hand_strength

        # 4. effective_field (V44+ contracts only; nominal count == V43 behaviour otherwise).
        effective_field = 0.0
        eff_fn = providers.get('effective_field_fn')
        if eff_fn is not None:
            try:
                if len(obs.community_cards) == 0:
                    if front_colors is None and after_colors is None:
                        effective_field = float(detected)   # button unread -- flat count, no guess
                    else:
                        effective_field = float(eff_fn(front_colors or [], after_colors or []))
                else:
                    effective_field = float(detected)
            except Exception as e:
                log(f"[V44] effective_field computation failed ({e}); falling back to nominal count.")
                effective_field = float(detected)

        # 5. Assemble the BoardState and hand off to the shared tensor/policy machinery.
        board_state = observation_to_board_state(
            obs, equity=equity, hand_strength=hand_strength, effective_field=effective_field)
        result = engine.make_decision(
            board_state,
            bet_raise_available=obs.bet_raise_available,
            check_call_available=obs.check_call_available,
            call_amount_known=obs.call_amount_known,
        )
        action, reason, bet_size = result[0], result[1], result[2]
        ev_dict = result[3] if len(result) > 3 else None

        return LiveDecision(
            action=action, reason=reason, bet_size=bet_size, ev_dict=ev_dict,
            equity=equity, sim_msg=sim_msg or '', equity_meta=equity_meta,
            hand_strength=hand_strength, board_state=board_state, observation=obs,
        )

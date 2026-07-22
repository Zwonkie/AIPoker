import os
import random
from core.models.engine import ModelEngine
from core.models.v14_engine import V14_ACTION_KEYS
from core.models.v29_engine import V29ModelEngine
from core.models.v40_engine import V40ModelEngine
from core.models.v41_engine import V41ModelEngine
from core.models.v43_engine import V43ModelEngine
from core.models.v44_engine import V44ModelEngine

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

# [TEST FLAG — 2026-07-20] Critic-argmax action mode. When ON, live selection picks the action with
# the highest critic Q (EV-vs-fold, ~BB) instead of SAMPLING the actor-policy distribution. Purpose:
# probe the actor/critic short-stack divergence logged as [STACK-3] in OFK
# known-shortcomings-backlog.md (actor folds jams its own critic prefers). Flip the default below, or
# toggle without editing code via env var:  HEROCULES_CRITIC_ARGMAX=1  (on) / 0 (off, default).
# Safe no-op for any model without a critic Q head (last_q_vals is None) and honors the exact same
# free-check / raise-availability masking the sampler applies. Diagnostic only — remove when done.
USE_CRITIC_ARGMAX_ACTION = os.environ.get('HEROCULES_CRITIC_ARGMAX', '0').strip().lower() \
    not in ('0', '', 'false', 'no', 'off')


def _critic_argmax_action(q_vals, legal_names):
    """[TEST] Highest-Q action among `legal_names`, or None if no critic Q is available (so callers
    fall through to the normal sampled choice). `legal_names` is the already-masked candidate set
    (free-check FOLD removed, unavailable raises removed), so the critic pick obeys the same legality
    constraints the sampler does."""
    if not q_vals or not legal_names:
        return None
    scored = [(a, q_vals[a]) for a in legal_names if a in q_vals]
    if not scored:
        return None
    return max(scored, key=lambda kv: kv[1])[0]


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


# [Fable review #14/H3] Hero's own past-action tokens. Training and every model_verify rollout feed
# these (simulator.py appends 7/3/6 to `hero_actions_histories[0]` after each hero decision and
# passes the list into `bridge.to_tensors`), but the LIVE path never passed `action_history_raw`, so
# `to_tensors` filled `act_ints` with 20 PADs. The transformer was trained and validated with its
# own line populated and served with it blank -- anything it learned to condition on its own past
# actions (barreling after raising, giving up after checking) was silently unavailable live, and no
# eval reproduced that input. Same integer vocabulary as the simulator; raises are size-blind (every
# bucket is 6), which is the known [OPP-3] limitation, not a new one.
HERO_ACTION_FOLD, HERO_ACTION_CALL, HERO_ACTION_RAISE = 7, 3, 6


def _hero_action_token(action: str) -> int:
    """Map a returned live action string to the simulator's own action-history token."""
    a = (action or '').upper()
    if a == 'FOLD':
        return HERO_ACTION_FOLD
    if a in ('CALL', 'CHECK'):
        return HERO_ACTION_CALL
    if a.startswith('RAISE') or a.startswith('BET') or a in ('ALLIN', 'ALL_IN', 'ALL-IN'):
        return HERO_ACTION_RAISE
    return HERO_ACTION_CALL   # unknown/passive default -- never PAD, which would mean "no action"


# ======================================================================= #
#  [V42_liveFixes / Fable review #16-H4 remainder + #6-CE] Live FEATURE providers.
# ======================================================================= #
# PHPHelp.py used to carry two more hand-maintained `is_vN` substring ladders -- one selecting the
# version's `compute_range_aware_equity`, one its `preflop_hand_strength` -- and both stopped at
# 'v29'. V40 and V41 were deployed live without being added, so for the whole time V41 was the
# active model it was served **vs-random equity** (its single most load-bearing input feature, and
# the one thing every train/serve invariant list names first) and a **constant hand_strength=0.5**.
# Nothing threw; the tensors were shape-valid. That is exactly the silent-degradation failure mode
# the review's #16 describes, realised on the primary feature.
#
# The fix is the same one `make_bridge()` applied to the tensor ladder: the VERSION declares what it
# needs, the shared layer just asks. An engine implements `live_features()` and is immune to the
# ladder forever; engines that don't are resolved through the legacy table below, which is now the
# ONLY copy of that mapping. A model matching neither is reported loudly (see
# `live_feature_providers`) instead of quietly dropping to vs-random.
#
# Order matters in this table exactly as it did in the ladders it replaces: each entry must precede
# any entry whose key it contains as a substring ('v20_preflopeq_ai' > 'v20_preflopeq' > 'v20',
# 'v17_gauntlet' > 'v17').
_LEGACY_LIVE_FEATURES = [
    # (name substring, version package, range-aware equity?, front/after split?, hand_strength?,
    #  fallback (stack, pot, call) money-feature scales)
    #
    # The scales column is a FALLBACK ONLY, for contracts that bake the divisors inline instead of
    # exporting them. `context_scales()` prefers the version contract's own STACK_SCALE/POT_SCALE/
    # CALL_SCALE module constants whenever they exist, so for V20_preflopEq and everything after it
    # this column is never consulted and cannot drift. Pre-V20 contracts all use the original
    # bridge_v13 scale (/400 stack+call, /1000 pot); V20 itself was the rescale but writes 100/250/
    # 100 inline (versions/v20/core/contract.py L128-138).
    ('v29',              'versions.v29',              True,  True,  True,  (100.0, 250.0, 100.0)),
    ('v28',              'versions.v28',              True,  True,  True,  (100.0, 250.0, 100.0)),
    ('v26',              'versions.v26',              True,  True,  True,  (100.0, 250.0, 100.0)),
    ('v25',              'versions.v25',              True,  True,  True,  (100.0, 250.0, 100.0)),
    ('v21_auxhead',      'versions.v21_auxhead',      True,  True,  True,  (100.0, 250.0, 100.0)),
    ('v20_preflopeq_ai', 'versions.v20_preflopEq_AI', True,  True,  True,  (100.0, 250.0, 100.0)),
    ('v20_preflopeq',    'versions.v20_preflopEq',    True,  True,  True,  (100.0, 250.0, 100.0)),
    ('v20',              'versions.v20',              True,  False, False, (100.0, 250.0, 100.0)),
    ('v19',              'versions.v19',              True,  False, False, (400.0, 1000.0, 400.0)),
    ('v17_gauntlet',     'versions.v17_gauntlet',     True,  False, False, (400.0, 1000.0, 400.0)),
    ('v17',              'versions.v17',              True,  False, False, (400.0, 1000.0, 400.0)),
    ('v15',              'versions.v15',              True,  False, False, (400.0, 1000.0, 400.0)),
    ('v14',              'versions.v14',              True,  False, False, (400.0, 1000.0, 400.0)),
    ('v13',              'versions.v13',              True,  False, False, (400.0, 1000.0, 400.0)),
]


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
# every model EXCEPT v21_auxhead/V25 (inherited unchanged through V22-V25 -- see
# _narrate_opponent_read below, gated on is_v21_auxhead_model or is_v25_model) -- for every other
# model here, the heads are still untrained noise.
_THINKING_BANDS = (0.30, 0.45, 0.60, 0.80)

# [V21_auxhead/V25 only] aux-head opponent-read narration. Correlations are real (inspect_aux_heads.py
# @ Phase 8: self_equity r=0.922, opp_strength r=0.171, opp_bluff r=0.091) but modest -- opp_strength
# in particular has a predicted std (0.034) far narrower than its label's own (0.142), i.e. the head
# barely swings from its mean. Deliberately reported as raw numbers with an explicit confidence
# caveat, NOT confident categorical bands ("strong range", "clearly bluffing") -- inventing bands
# out of a weak-correlation, compressed-range signal would overstate its reliability. opp_bluff is
# only meaningful facing a real bet (its training label is last_raiser-gated -- see
# versions/v21_auxhead/simulator.py::_mc_target_evs_sized), so it's omitted on free checks.
def _narrate_opponent_read(aux, board_state):
    if not aux:
        return None
    try:
        strength = float(aux.get('opp_strength', 0.0))
        bluff = float(aux.get('opp_bluff', 0.0))
        facing_bet = float(getattr(board_state, 'call_amount', 0) or 0) > 0
    except Exception:
        return None
    parts = [f"strength-read {strength:.0%}"]
    if facing_bet:
        parts.append(f"bluff-read {bluff:.0%}")
    return "Opponent read (experimental, weak signal): " + ", ".join(parts) + "."


def _narrate_thinking(action, board_state, evs, aux=None):
    """Human-readable read of WHY, built from equity (real input feature) + the chosen action,
    plus an optional aux-head opponent-read line (v21_auxhead only, see _narrate_opponent_read)."""
    try:
        eq = float(getattr(board_state, 'equity', 0.0) or 0.0)
    except Exception:
        return None
    act = (action or '').upper()
    is_aggro = act.startswith('RAISE') or act == 'ALLIN' or act == 'ALL_IN' or act == 'BET'
    is_fold = act == 'FOLD'
    is_call = act in ('CALL', 'CHECK')

    if is_fold:
        base = f"Thinking: equity too low ({eq:.0%}) to continue profitably -- folding."
    elif is_aggro:
        if eq < _THINKING_BANDS[0]:
            base = f"Thinking: weak hand ({eq:.0%} equity) -- bluffing, betting on fold equity rather than hand strength."
        elif eq < _THINKING_BANDS[1]:
            base = f"Thinking: marginal hand ({eq:.0%} equity) -- semi-bluffing, betting for fold equity plus backup outs."
        elif eq < _THINKING_BANDS[2]:
            base = f"Thinking: showdown-value hand ({eq:.0%} equity) -- betting for value/protection while ahead or close."
        elif eq < _THINKING_BANDS[3]:
            base = f"Thinking: strong hand ({eq:.0%} equity) -- value betting to get called by worse."
        else:
            base = f"Thinking: near-nuts ({eq:.0%} equity) -- betting big for maximum value."
    elif is_call:
        if eq < _THINKING_BANDS[1]:
            base = f"Thinking: {eq:.0%} equity -- speculative call, playing for draws/implied odds rather than a made hand."
        elif eq < _THINKING_BANDS[2]:
            base = f"Thinking: {eq:.0%} equity -- calling to see the next card, not strong enough to raise."
        else:
            base = f"Thinking: {eq:.0%} equity -- flat-calling for value/deception, keeping weaker hands in."
    else:
        base = None

    read = _narrate_opponent_read(aux, board_state)
    if base and read:
        return base + " " + read
    return base or read

from core.bridge.contract_v8_v9 import ContractV8V9
from core.bridge.v11.contract_v11 import ContractV8V9 as ContractV11
from versions.v13.core.contract import ContractV12
from versions.v20.core.contract import ContractV12 as ContractV12_v20
from versions.v20_preflopEq.core.contract import ContractV12 as ContractV12_v20PreflopEq
from versions.v25.core.contract import ContractV12 as ContractV12_v25
from versions.v29.core.contract import ContractV12 as ContractV12_v29
from versions.v41.core.contract import ContractV12 as ContractV12_v41
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
            # V29: NEW contract (context_dim=54, contract_version=8 -- NOT V25/V26/V27/V28's shared
            # 44/7) -- own bridge (self.bridge_v29 below), gated by is_v29_model, checked BEFORE the
            # v25/v26/v28 combined branch. Two changes, both by explicit user direction (2026-07-20):
            # [OPP-2] ten new appended per-opponent-seat raise-attribution features, now REAL and
            # LIVE-FUNCTIONAL (2026-07-20) -- core/table_state.py gained per-seat raise/call
            # classification (stack-drop diffs compared against the bet level actually faced, not
            # just "any drop"), `committed`/`hero_committed` (start-of-hand stack minus current),
            # and `pot_type` (live whole-hand raise-event counter), all previously silently inert
            # since V22/V23 for `committed`/`hero_committed`/`pot_type` -- see
            # `.agents/skills/OFK/references/known-shortcomings-backlog.md` [OPP-2]. Also fixed, as
            # a direct byproduct: an opponent's stack going to a genuine 0 (all-in) wasn't updating
            # (core/vision.py already emits a reliable state='All-In' text-match signal for
            # opponents that table_state.py wasn't using); mirrored the same fix for HERO's own
            # all-in by adding the equivalent 'ALL'/'IN' text-match to vision.py's hero-stack OCR
            # (which previously lacked it, so a genuine hero all-in and a failed OCR read were
            # indistinguishable). Second change: a critic-consistency filter + risk_aversion_
            # coefficient bump (0.10->0.15) on the training target (training-loop-only, no live-
            # serving impact). See versions/v29/SPECS.md for the full derivation, calibration, and
            # model_verify --full results (21 PASS/2 WARN/0 FAIL/1 SKIP -- the cleanest scorecard in
            # this lineage; deep_stack_ood_guard [STACK-1] PASSED for the first time since V22).
            # DEPLOYED LIVE (2026-07-20) per explicit user request, on the strength of this result.
            # V41: SAME contract as V29/V40 (context_dim=54, contract_version=8) -- nothing about
            # the tensor schema changed in either version, so this reuses V29's live game-state
            # work ([OPP-2] per-seat raise tracking in core/table_state.py and the
            # committed/hero_committed/pot_type + all-in stack-tracking byproduct fixes) unchanged.
            # It still gets its OWN bridge (self.bridge_v41) rather than sharing bridge_v29, so a
            # future divergence can't silently misalign -- that is exactly how the V20 drift class
            # and [OPP-7]'s tensor-boundary bug both happened.
            # Lineage V29 -> V40 (BET-3 package: betting round no longer ends on a check; CALL no
            # longer exempt from the variance penalty / continuation credit; ALLIN veto rescoped)
            # -> V41 (simulation realism: dead blinds, NN opponents playing a degraded self,
            # asymmetric stacks + the min-raise and reopen-action rule bugs that exposed, and
            # [OPP-7] finally fixed AT THE TENSOR BOUNDARY -- V27's remap wrote the real hero to a
            # `seat_0` key ContractV12.to_tensors never reads, so hero was invisible to 128 of 128
            # NN-opponent queries). Each fix carries a measured before/after -- see
            # versions/v41/SPECS.md and
            # .agents/skills/OFK/references/fable-review-resolution-log.md.
            # V43: SAME contract as V29/V40/V41 (context_dim=54, contract_version=8) -- every change
            # is training-side, so this reuses all live game-state work unchanged. Declares
            # make_bridge(), live_features() AND is_sized, so no ladder in this file or in
            # PHPHelp.py can misroute it (the three failure modes that bit the V40 deploy).
            # Corrective-prior cleanup: the V12 realization discount and the V29 ALLIN veto are
            # REMOVED (entry width and jam frequency are learned from the inputs V40/V41 fixed, not
            # imposed); the variance penalty is KEPT and re-scaled 0.15->0.20 alongside
            # TARGET_CLIP_BB 40->100, because the old clip was an undeclared deep-stack all-in
            # dampener. See versions/v43/SPECS.md.
            # DEPLOYED LIVE 2026-07-21 by explicit user decision on a MIXED scorecard, BEFORE
            # beats_frozen_predecessor finished. Known regressions at deploy time:
            # vpip_adapts_to_style FAIL (deep delta +4.2pts vs the >=5pt gate; V41 passed at
            # short +5.9/deep +7.2) and nash_bbcall_vs_jam 47% (V41 passed) -- both the predicted
            # cost of removing the realization discount, i.e. hero enters and calls jams wider.
            # Genuinely better than V41: allin_vs_nextbest_qgap negative at EVERY cell,
            # opponent_style_sweep WARN->PASS, action_diversity genuinely mixed, [BET-3] resolved.
            # ROLLBACK: set active_model_name back to 'Herocules (v41)' (still registered below).
            # V44: effective-contested-field `equity_edge` (contract_version 8->9; ctx[35]
            # keeps its width but changes meaning). Own bridge (make_bridge -> versions.v44),
            # live_features(), is_sized/display_tag/has_aux all declared -- no ladder here or in
            # PHPHelp can misroute it. model_verify --full 2026-07-22: 21 PASS / 6 WARN / 0 FAIL;
            # vpip_adapts_to_style [P4] PASS for the FIRST time (short +6.1/deep +5.9),
            # beats_frozen_predecessor +91.7 BB/100 vs frozen V43. Cost: committed_sensitivity /
            # pot_type_sensitivity dropped to WARN. Its live edge denominator needs
            # BoardState.effective_field, supplied by live_feature_providers()['effective_field_fn']
            # and set in PHPHelp (falls back to nominal -> V43 behaviour if absent). DEPLOYED LIVE
            # 2026-07-22 by explicit user decision on the 0-FAIL scorecard + first-ever [P4] pass.
            # ROLLBACK: set active_model_name back to 'Herocules (v43)'. See versions/v44/SPECS.md.
            'Herocules (v44)': V44ModelEngine(weight_name="expert_main.pth"),
            'Herocules (v43)': V43ModelEngine(weight_name="expert_main.pth"),
            'Herocules (v41)': V41ModelEngine(weight_name="expert_main.pth"),
            # V40: the [BET-3] package -- same 54/8 contract as V29/V41, so it reuses V29's live
            # game-state work untouched and declares its own bridge (no ladder entry needed).
            # Trained 100k hands 2026-07-20 (hero +63.7 BB/100 vs field). The multiway-aggression
            # collapse that motivated the package went from 6/6 short-stack cells to 3/6, with
            # 3-way aggression at eq 0.65 rising 0.01 -> ~0.5. DEPLOYED LIVE 2026-07-21 as an
            # INTERIM model by explicit user request, for play-testing while V41 trains -- note its
            # model_verify --full was only partially completed (all FAST checks plus
            # vpip_adapts_to_style and beats_offformula_stress passed; bb100_vs_standard_fields and
            # beats_frozen_predecessor were cut short to free CPU). See versions/v40/SPECS.md.
            'Herocules (v40)': V40ModelEngine(weight_name="expert_main.pth"),
            'Herocules (v29)': V29ModelEngine(weight_name="expert_main.pth"),
            # --- Live dropdown pruned 2026-07-22 (user request: keep the 5 newest iterations) ---
            # Registered live models are now v44 (active), v43, v41, v40, v29. Versions v28, v26,
            # v25, v21_auxhead, v20_preflopEq_AI, v20_preflopEq, v20, v19, v17_gauntlet, v17, v15,
            # v14 and v13 were REMOVED FROM THE SELECTOR ONLY -- their version slices, weights,
            # engine files (core/models/vNN_engine.py) and bridge branches all remain on disk, so
            # re-registering any of them is a one-line add here (deprecate-not-delete, guardrails
            # Rule 6). v13 was the tagged MILESTONE and v41 the other; both weights are preserved.
        }
        # DEPLOYED 2026-07-21: V41 replaces V40 (which was live for a few hours as an interim
        # play-test model). Trained 100,000 hands; model_verify --full 22 PASS / 5 WARN / 0 FAIL /
        # 0 SKIP -- the cleanest scorecard of any version (V29's was 21/2/0/1), and the first with
        # ZERO skips. Headline: multiway_shortstack_aggression PASSES outright (V29 collapsed all
        # 6 short-stack cells, V40 fixed 3/6, V41 holds 3-way aggression at 0.81 where V29 gave
        # 0.01) -- [BET-3], the live symptom that started this whole line of work, is resolved.
        # beats_frozen_predecessor is a REAL head-to-head for the first time since the V18 refactor
        # (frozen V40 seated as an NNOpponent): +64.3 BB/100 over 4000 hands.
        # V25-V40 remain registered as rollback options -- switching back is a one-line change
        # here (or set_active_model at runtime).
        # DEPLOYED 2026-07-21: V43 replaces V41 by explicit user decision, on a MIXED scorecard and
        # before beats_frozen_predecessor finished -- see the registry entry above for exactly what
        # was known at deploy time. V41 (the MILESTONE) stays registered as the one-line rollback.
        self.active_model_name = 'Herocules (v44)'  # V43 registered above as rollback
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
        # V25 uses a DIFFERENT scale+width again (contract_version 7, context_dim 44 -- V22's
        # entry-sizing features + V23's pot_type, inherited unchanged) -- needs its own bridge.
        # V26/V28 are architecturally IDENTICAL to V25 (only the training opponent pool / target
        # formula differ) -- share this same bridge, no separate bridge_v26/bridge_v28 needed.
        self.bridge_v25 = ContractV12_v25(max_seq_len=20)
        # [V29] NEW contract (context_dim=54, contract_version=8) -- own bridge, not shared with
        # bridge_v25 above (V25/V26/V27/V28 all stay on the 44/7 contract untouched).
        self.bridge_v29 = ContractV12_v29(max_seq_len=20)
        # [V41] Same 54/8 contract as V29, but its own bridge instance -- see the registry note.
        self.bridge_v41 = ContractV12_v41(max_seq_len=20)
        # [Fable review #16/H4] ENGINE-OWNED BRIDGES. The dispatch below is three hand-synchronised
        # substring ladders (here, plus two more in PHPHelp.py) whose failure mode is silent: a new
        # engine added to the registry but missed in a ladder matches no `is_vN` flag, falls through
        # to `bridge_v9`, throws, is caught, and FOLDS EVERY HAND while play continues. An engine
        # that declares `make_bridge()` short-circuits the whole ladder with its OWN contract, so a
        # new version needs no ladder edit and cannot be silently misrouted. Substring collisions
        # (a future 'v29b' matching 'v29') are sidestepped for the same reason. Existing engines
        # without `make_bridge` keep the ladder exactly as it was -- this is additive, not a rewrite.
        self._engine_bridges = {}
        for _name, _engine in self.models.items():
            _factory = getattr(_engine, 'make_bridge', None)
            if callable(_factory):
                try:
                    self._engine_bridges[_name] = _factory()
                except Exception as _e:
                    print(f"WARNING: {_name} declares make_bridge() but it raised: {_e!r}. "
                          f"Falling back to the version ladder for this model.")
        self._report_engine_health()

        self.hand_history_buffer = []
        # [Fable review #14/H3] One token per entry in hand_history_buffer, appended in the SAME
        # call that appended the state, so the two can never drift. Holds the actions taken at all
        # PRIOR states: len(hero_action_buffer) == len(hand_history_buffer) - 1 while a decision is
        # in flight, which is exactly the alignment `to_tensors` expects (it leaves the current
        # step's slot as PAD, and the transformer shifts by one internally).
        self.hero_action_buffer = []
        self._last_street = None
        self._last_hole_cards = None

    def _report_engine_health(self):
        """[Fable review #15+#16] One startup line per engine: did its weights load, and how will
        its tensors be built. Previously a failed load was a single WARNING scrolling past among
        many, and bridge routing was invisible until something folded every hand at a real table."""
        for name, engine in self.models.items():
            loaded = getattr(engine, 'loaded', True)
            route = 'own bridge' if name in self._engine_bridges else 'version ladder'
            status = 'OK' if loaded else 'FAILED TO LOAD -- will refuse to act'
            marker = '*' if name == self.active_model_name else ' '
            print(f"  [engine]{marker} {name:<34} weights={status:<38} tensors={route}")
        # [V42_liveFixes / Fable review live-L3] Surface the diagnostic action-selection mode at
        # startup. It used to be visible only in per-decision reason text, so a leftover
        # HEROCULES_CRITIC_ARGMAX could put live play on an eval-unvalidated selector unnoticed.
        if USE_CRITIC_ARGMAX_ACTION:
            print("  [engine]! ACTION SELECTION = CRITIC-ARGMAX(Q) -- DIAGNOSTIC MODE, NOT the "
                  "sampled actor policy and NOT validated by model_verify. Set "
                  "CRITIC_ARGMAX_MODE = False in PHPHelp.py to turn it off.")

    def set_active_model(self, model_name: str, tree_file: str = None):
        if model_name in self.models:
            self.active_model_name = model_name
            return
        # [Fable review #16/H4] Was: silently switch to 'Herocules (v20)' -- nine versions stale, a
        # different CONTRACT, and indistinguishable at the HUD from the model you asked for. Keep
        # serving whatever is already active (a known-good model) and say loudly that the switch
        # did not happen, rather than quietly swapping in a different one.
        print(f"ERROR: '{model_name}' is not in the live registry -- KEEPING "
              f"'{self.active_model_name}' active. Registered: {sorted(self.models)}")

    def _resolve_live_spec(self, name: str):
        """Which version package serves `name`, and what it declares -- engine first, name-ladder
        second. Shared by `live_feature_providers()` and `context_scales()` so the two can never
        disagree about which version a live model actually is.

        Returns (spec | None, source, error | None).
        """
        engine = self.models.get(name)

        declare = getattr(engine, 'live_features', None)
        if callable(declare):
            try:
                spec = declare() or None
                if spec:
                    return spec, 'engine', None
            except Exception as e:
                return None, 'unresolved', f"{name}'s live_features() raised: {e!r}"

        lowered = name.lower()
        for key, package, range_aware, front, hand_strength, scales in _LEGACY_LIVE_FEATURES:
            if key in lowered:
                return ({'version_package': package, 'range_aware_equity': range_aware,
                         'front_colors': front, 'hand_strength': hand_strength,
                         'context_scales': scales},
                        'name-ladder', None)

        return None, 'unresolved', (
            f"'{name}' matches no entry in _LEGACY_LIVE_FEATURES and its engine declares no "
            f"live_features() -- live equity would silently fall back to vs-random. Add "
            f"live_features() to its engine.")

    def context_scales(self, model_name: str = None):
        """The money-feature divisors the ACTIVE model's contract uses for ctx[1] hero_stack,
        ctx[2] pot and ctx[9] call_amount -- so a diagnostic can turn a recorded input tensor back
        into big blinds.

        Read from the version contract's OWN `STACK_SCALE`/`POT_SCALE`/`CALL_SCALE` module
        constants wherever they exist, which is the only copy that can't go stale: it is literally
        the constant the encoder divided by. `_LEGACY_LIVE_FEATURES`' scales column is consulted
        only for the pre-V20_preflopEq contracts that bake the divisor inline.

        This replaced a hand-maintained `is_v20_family` substring check in `PHPHelp.py` that
        stopped at 'v29', so V40/V41/V43 turns were decoded with the pre-V20 /400,/1000 constants:
        history/Turbo_1171580052/flagged/turn_2_20260721_201440 rendered a real 75BB/1.5BB/1.0BB
        node as 300BB/6.0BB/4.0BB and then raised a bogus "MODEL-INPUT vs RAW-OCR MISMATCH ->
        BRIDGE issue" banner against a bridge that was correct. The same decode also feeds
        `_build_turn_record`, so every recorded turn's `to_call`/`pot_odds` inherited the error.
        Note this is the THIRD time this specific decoder has been wrong for the live model (see
        its own docstring's V20_preflopEq_AI case) -- hence sourcing it from the contract itself.

        Returns {'stack': float, 'pot': float, 'call': float, 'source': str, 'error': str | None}.
        """
        import importlib

        name = model_name or self.active_model_name
        spec, source, resolve_error = self._resolve_live_spec(name)

        # Pre-V20 bridge_v13 scale -- the oldest contract in the registry, so the least-surprising
        # answer if we genuinely cannot tell. Always paired with a non-None `error`.
        fallback = (400.0, 1000.0, 400.0)

        if spec is None:
            stack, pot, call = fallback
            return {'stack': stack, 'pot': pot, 'call': call, 'source': 'unresolved',
                    'error': resolve_error}

        package = spec.get('version_package')
        try:
            contract = importlib.import_module(f"{package}.core.contract")
        except Exception as e:
            stack, pot, call = spec.get('context_scales') or fallback
            return {'stack': stack, 'pot': pot, 'call': call, 'source': f'{source}:declared',
                    'error': f"importing {package}.core.contract for '{name}' failed: {e!r}"}

        stack = getattr(contract, 'STACK_SCALE', None)
        pot = getattr(contract, 'POT_SCALE', None)
        call = getattr(contract, 'CALL_SCALE', None)
        if None not in (stack, pot, call):
            return {'stack': float(stack), 'pot': float(pot), 'call': float(call),
                    'source': f'{source}:contract', 'error': None}

        declared = spec.get('context_scales')
        if declared:
            stack, pot, call = declared
            return {'stack': float(stack), 'pot': float(pot), 'call': float(call),
                    'source': f'{source}:declared', 'error': None}

        stack, pot, call = fallback
        return {'stack': stack, 'pot': pot, 'call': call, 'source': f'{source}:fallback',
                'error': (f"{package}.core.contract exports no STACK_SCALE/POT_SCALE/CALL_SCALE and "
                          f"'{name}' declares no context_scales -- decoded BB values for this model "
                          f"are a GUESS. Export the constants from that contract.")}

    def live_feature_providers(self, model_name: str = None):
        """[V42_liveFixes] The equity / hand_strength implementations the ACTIVE model was trained
        with, resolved once per decision by the shared layer instead of by a substring ladder in
        `PHPHelp.py`. See `_LEGACY_LIVE_FEATURES` above for why this exists.

        Returns a dict:
          `equity_fn`         -- that version's own `compute_range_aware_equity`, or None
          `hand_strength_fn`  -- that version's own `preflop_hand_strength`, or None
          `use_front_colors`  -- whether its equity fn takes the front/after split (V20_preflopEq+)
          `source`            -- 'engine' | 'name-ladder' | 'unresolved', for the live log
          `error`             -- a human-readable reason when something is missing

        `source == 'unresolved'` is a REAL PROBLEM, not a shrug: it means the model will be served
        vs-random equity while having been trained on range-aware equity. The caller is expected to
        surface it. A new version avoids the whole question by declaring `live_features()` on its
        engine (see `core/models/v41_engine.py`).
        """
        import importlib

        name = model_name or self.active_model_name
        spec, source, resolve_error = self._resolve_live_spec(name)

        if spec is None:
            return {'equity_fn': None, 'hand_strength_fn': None, 'use_front_colors': False,
                    'source': 'unresolved', 'error': resolve_error}

        package = spec.get('version_package')
        equity_fn = hand_strength_fn = effective_field_fn = None
        error = None
        try:
            simulator = importlib.import_module(f"{package}.self_play.simulator")
            contract = importlib.import_module(f"{package}.core.contract")
            if spec.get('range_aware_equity', True):
                equity_fn = getattr(simulator, 'compute_range_aware_equity', None)
            if spec.get('hand_strength', False):
                hand_strength_fn = getattr(contract, 'preflop_hand_strength', None)
            # [V44] If this version's contract exposes `effective_contested_field` (contract_version
            # >= 9), give the live caller a closure that turns the SAME front/after HUD-colour split
            # PHPHelp already computes for the equity call into the effective-field denominator for
            # ctx[35]. Built from the version's OWN `effective_contested_field` and its OWN colour->
            # VPIP map (`_COLOR_TO_VPIP`), so live and training use one implementation -- feeding the
            # nominal count here instead would be a silent train/serve mismatch on exactly the
            # feature V44 exists to fix. Absent (older contracts) -> None, and the caller leaves
            # BoardState.effective_field at 0.0, which every pre-V44 contract ignores anyway.
            _eff = getattr(contract, 'effective_contested_field', None)
            _vpip_map = getattr(simulator, '_COLOR_TO_VPIP', None)
            if _eff is not None and _vpip_map is not None:
                def effective_field_fn(front_colors, after_colors, _eff=_eff, _vm=_vpip_map):
                    after_vpips = [_vm.get(c, 0.30) for c in (after_colors or [])]
                    return _eff(after_vpips, n_front=len(front_colors or []))
        except Exception as e:
            error = f"importing {package} for '{name}' failed: {e!r}"

        return {'equity_fn': equity_fn, 'hand_strength_fn': hand_strength_fn,
                'effective_field_fn': effective_field_fn,
                'use_front_colors': bool(spec.get('front_colors', False)),
                'source': source, 'error': error}

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
                      action_history_raw: list = None,
                      call_amount_known: bool = True):
        
        active_model = self.models.get(self.active_model_name)
        if not active_model:
            return 'FOLD', "Model not found", 0.0, {}
        # [Fable review #15] An engine whose weights failed to load holds RANDOM weights and will
        # happily emit confident-looking nonsense at a real table -- the failure was previously
        # swallowed in the engine's constructor and never checked here. Every engine sets `.loaded`;
        # anything that doesn't is treated as loaded (legacy engines predate the flag). Fold rather
        # than act on garbage, and say why.
        if getattr(active_model, 'loaded', True) is False:
            err = getattr(active_model, 'load_error', 'unknown error')
            return 'FOLD', f"Model '{self.active_model_name}' failed to load ({err}) - refusing to act", 0.0, {}

        # Round Reset Logic (Sandbox tracking)
        street_order = {'Preflop': 0, 'Flop': 1, 'Turn': 2, 'River': 3}
        current_street_val = street_order.get(board_state.street, 0)
        last_street_val = street_order.get(self._last_street, -1)
        
        current_hole_cards = sorted([str(c) for c in board_state.hero_cards]) if board_state.hero_cards else []
        # Reset if hole cards changed or street went backwards (e.g., new hand started)
        if current_hole_cards != self._last_hole_cards or current_street_val < last_street_val:
            self.hand_history_buffer = []
            self.hero_action_buffer = []   # [H3] reset together -- see the buffer's own note
            
        self._last_street = board_state.street
        self._last_hole_cards = current_hole_cards
        
        self.hand_history_buffer.append(board_state)
        # [Fable review #14/H3] Serve the model its own action history. An explicit
        # `action_history_raw` from the caller still wins (replay/debug harnesses pass one); with
        # none, use the buffer we maintain ourselves, which is guaranteed aligned by construction.
        if action_history_raw is None:
            action_history_raw = list(self.hero_action_buffer)

        is_v13_model = getattr(active_model, 'is_v13', False) or 'v13' in self.active_model_name.lower()
        is_v14_model = getattr(active_model, 'is_v14', False) or 'v14' in self.active_model_name.lower()
        is_v15_model = getattr(active_model, 'is_v15', False) or 'v15' in self.active_model_name.lower()
        is_v17_gauntlet_model = getattr(active_model, 'is_v17_gauntlet', False) or 'v17_gauntlet' in self.active_model_name.lower()
        is_v17_model = (not is_v17_gauntlet_model) and (getattr(active_model, 'is_v17', False) or 'v17' in self.active_model_name.lower())
        is_v19_model = getattr(active_model, 'is_v19', False) or 'v19' in self.active_model_name.lower()
        # NOTE: resolution order matters -- 'v20_preflopeq' is a substring of 'v20_preflopeq_ai',
        # and 'v20' is a substring of both, so each must be checked BEFORE the shorter name it
        # contains or the naive name-based fallback misfires true for the wrong model.
        # 'v25'/'v26'/'v28'/'v29' are substrings of nothing among each other, so order doesn't
        # matter for THESE flags, but is_v29_model gets its OWN bridge branch below (checked
        # BEFORE the v25/v26/v28 combined branch, since v29's contract is a different width/scale).
        is_v41_model = getattr(active_model, 'is_v41', False) or 'v41' in self.active_model_name.lower()
        is_v40_model = getattr(active_model, 'is_v40', False) or 'v40' in self.active_model_name.lower()
        is_v29_model = getattr(active_model, 'is_v29', False) or 'v29' in self.active_model_name.lower()
        is_v28_model = getattr(active_model, 'is_v28', False) or 'v28' in self.active_model_name.lower()
        is_v26_model = getattr(active_model, 'is_v26', False) or 'v26' in self.active_model_name.lower()
        is_v25_model = getattr(active_model, 'is_v25', False) or 'v25' in self.active_model_name.lower()
        is_v21_auxhead_model = getattr(active_model, 'is_v21_auxhead', False) or 'v21_auxhead' in self.active_model_name.lower()
        is_v20_preflopEq_AI_model = getattr(active_model, 'is_v20_preflopEq_AI', False) or 'v20_preflopeq_ai' in self.active_model_name.lower()
        is_v20_preflopEq_model = (not is_v20_preflopEq_AI_model) and (getattr(active_model, 'is_v20_preflopEq', False) or 'v20_preflopeq' in self.active_model_name.lower())
        is_v20_model = (not is_v20_preflopEq_model) and (not is_v20_preflopEq_AI_model) and (getattr(active_model, 'is_v20', False) or 'v20' in self.active_model_name.lower())
        is_v11_model = getattr(active_model, 'is_v11', False) or 'v11' in self.active_model_name.lower()
        # V14/V15/V17/V17_gauntlet/V19/V20/V20_preflopEq/V20_preflopEq_AI/V21_auxhead/V25 share the
        # IDENTICAL 6-action sized contract -> one live path (selection + slider sizing). Anything
        # gated on "the sized model" uses this flag.
        # [V43 / Fable review #16-H4 remainder] An engine may now DECLARE `is_sized` instead of
        # being recognised by this OR-chain. That chain is the ladder that silently bit the V40
        # deploy: its bridge was fine (V40 declares make_bridge()), but `is_sized_model` did not
        # know the name, so live emitted a bare `RAISE_POT` with size=0.0 -- a raise the executor
        # cannot size -- instead of an executable `RAISE_SLIDER_x`. Engines without the flag keep
        # the ladder exactly as it was; this is additive.
        _declared_sized = getattr(active_model, 'is_sized', None)
        is_sized_model = bool(_declared_sized) if _declared_sized is not None else (
            is_v14_model or is_v15_model or is_v17_model or is_v17_gauntlet_model or is_v19_model or is_v20_model or is_v20_preflopEq_model or is_v20_preflopEq_AI_model or is_v21_auxhead_model or is_v25_model or is_v26_model or is_v28_model or is_v29_model or is_v40_model or is_v41_model)
        try:
            own_bridge = self._engine_bridges.get(self.active_model_name)
            if own_bridge is not None:
                # [Fable review #16/H4] The engine declared its own contract -- no ladder involved.
                hole, board, ctx, act = own_bridge.to_tensors(self.hand_history_buffer, action_history_raw)
            elif is_v41_model:
                # [V41] Same 54/8 tensor schema as V29 -- separate bridge on purpose (see registry).
                hole, board, ctx, act = self.bridge_v41.to_tensors(self.hand_history_buffer, action_history_raw)
            elif is_v29_model:
                # [V29] NEW contract (context_dim=54, contract_version=8) -- own bridge, NOT
                # bridge_v25 (V25/V26/V28's shared 44/7 contract would silently misalign the 10
                # new appended per-opponent-seat raise features).
                hole, board, ctx, act = self.bridge_v29.to_tensors(self.hand_history_buffer, action_history_raw)
            elif is_v25_model or is_v26_model or is_v28_model:
                # V25/V26/V28 use a DIFFERENT context-feature WIDTH+scale again (contract_version 7,
                # context_dim 44) -- MUST use their shared bridge, not bridge_v20_preflopEq or
                # bridge_v13, or the entry-sizing (V22) + pot_type (V23) features would be
                # missing/misaligned. V26/V28 are architecturally identical to V25 (only the
                # training opponent pool / target formula differ), so they reuse this same bridge.
                hole, board, ctx, act = self.bridge_v25.to_tensors(self.hand_history_buffer, action_history_raw)
            elif is_v20_preflopEq_model or is_v20_preflopEq_AI_model or is_v21_auxhead_model:
                # V20_preflopEq / V20_preflopEq_AI / V21_auxhead (identical architecture -- only the
                # training opponent pool / aux-head loss differ) were trained on a DIFFERENT
                # context-feature WIDTH+scale (contract_version 5, context_dim 37) -- MUST use their
                # shared bridge, not bridge_v13 or bridge_v20, or the two appended features
                # (equity_edge, hand_strength) would be missing/misaligned (see
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
            elif getattr(active_model, 'is_v9', False) or 'v9' in self.active_model_name.lower():
                hole, board, ctx, act = self.bridge_v9.to_tensors(board_state, action_history_raw)
            else:
                # [Fable review #16/H4] Previously this WAS the bridge_v9 branch: any model the
                # ladder didn't recognise got the v9 contract, threw, and folded every hand behind a
                # generic "Fatal decision engine crash". Name the actual problem and the actual fix.
                raise RuntimeError(
                    f"no tensor bridge resolved for active model '{self.active_model_name}'. "
                    f"Add a make_bridge() method to its engine (preferred -- see _engine_bridges), "
                    f"or add it to the version ladder in make_decision.")
            evs = active_model.predict_ev(hole, board, ctx, act)
        except Exception as e:
            # [H3] The state was already appended; record the action we actually return so the
            # buffers stay in lockstep even on the crash path.
            self.hero_action_buffer.append(HERO_ACTION_FOLD)
            return 'FOLD', f"Fatal decision engine crash: {e}", 0.0, {}

        # 1. Base Model Decision.
        # Actor-policy models (v13/v11) output a probability distribution over {FOLD,CALL,RAISE}.
        # We SAMPLE from it (matching training/eval, which sample rather than argmax) with
        # temperature sharpening, preserving the model's game-theoretic mixing while suppressing
        # rare low-prob 'spew'. Legacy critic-only checkpoints keep argmax over Q.
        # Train/serve consistency: FOLD is strictly dominated when checking is free (call_amount==0),
        # so the sim zeroes it before selecting (simulator._select_action) — we mirror that here.
        is_actor_policy = is_v13_model or is_v11_model or is_sized_model
        # [V42_liveFixes / Fable review #13] FOLD is masked ONLY when checking is positively known to
        # be free. The old test was `call_amount is None or <= 0`, and the live caller could not
        # produce None -- an unreadable Check/Call button left `call_amount` at its 0.0 initialiser,
        # so a vision failure while facing a real bet made FOLD illegal. `call_amount_known=False`
        # now says "this number is an estimate, not a reading": the estimate still drives the tensor
        # (pot_odds, scaled_call), but it can never take FOLD away.
        _ca = getattr(board_state, 'call_amount', 0)
        free_check = bool(call_amount_known) and _ca is not None and _ca <= 0
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
            # [V42_liveFixes / Fable review M4] `check_call_available` was accepted by this method
            # and never read. With no Check/Call button on screen (the all-in case PHPHelp detects
            # by button brightness), the sampler could still return CALL and the executor would
            # click where no button exists. Mask it, exactly as the raise buckets are masked when
            # the raise button is gone.
            if not check_call_available:
                probs['CALL'] = 0.0
            sharp = {a: (v ** (1.0 / temp)) for a, v in probs.items()}
            names = [a for a in keys if sharp[a] > 0.0]
            sharp_total = sum(sharp[a] for a in names) or 1.0
            sampled_probs = {a: (sharp[a] / sharp_total if a in names else 0.0) for a in keys}
            choice = random.choices(names, weights=[sharp[a] for a in names], k=1)[0] if names \
                else (('CALL' if check_call_available else 'FOLD')
                      if (free_check or not bet_raise_available) else 'FOLD')

            # [TEST FLAG] Optionally replace the sampled actor choice with the critic's argmax-Q pick
            # over the SAME masked candidate set (see USE_CRITIC_ARGMAX_ACTION). No-op if off / no Q.
            _critic_pick = _critic_argmax_action(getattr(active_model, 'last_q_vals', None), names) \
                if USE_CRITIC_ARGMAX_ACTION else None
            if _critic_pick is not None:
                choice = _critic_pick

            chosen_key = choice   # the raw policy bucket the sampler picked, e.g. RAISE_66 or ALLIN
            if choice in V14_RAISE_FRAC:
                raise_size, slider = self._v14_size_to_slider(V14_RAISE_FRAC[choice], board_state)
                action = f"RAISE_SLIDER_{slider:.2f}"
                bet_size = raise_size
                size_note = f"{choice} -> raise-to {raise_size:.0f} (slider {slider:.2f})"
            else:
                action = choice   # FOLD / CALL
                size_note = choice
            # [V43] An engine may declare `display_tag`; otherwise fall back to the 13-deep nested
            # ternary below. Without this a new version silently renders under the FINAL else
            # ("V14") in the live reason line -- the HUD naming a different model than the one
            # acting, which is exactly the class of mislabelling the review's #16/H4 is about.
            _tag = getattr(active_model, 'display_tag', None) or (
                   "V41" if is_v41_model else ("V40" if is_v40_model else ("V29" if is_v29_model else ("V28" if is_v28_model else ("V26" if is_v26_model else ("V25" if is_v25_model else ("V21_auxhead" if is_v21_auxhead_model else ("V20_preflopEq_AI" if is_v20_preflopEq_AI_model else ("V20_preflopEq" if is_v20_preflopEq_model else ("V20" if is_v20_model else ("V19" if is_v19_model else ("V17_gauntlet" if is_v17_gauntlet_model else ("V17" if is_v17_model else ("V15" if is_v15_model else "V14"))))))))))))))
            _mode = "CRITIC-ARGMAX(Q)" if _critic_pick is not None else f"sampled (temp={temp:.2f}"
            _modeclose = "" if _critic_pick is not None else ")"
            reason = (f"{_tag} {_mode}"
                      + (", free-check fold-masked" if free_check else "")
                      + (", raise-unavail" if not bet_raise_available else "")
                      + (", call-unavail" if not check_call_available else "")
                      + (", price-estimated" if not call_amount_known else "")
                      + f"{_modeclose} -> {size_note}: {_fmt_dist(evs)}")
        elif is_actor_policy:
            probs = {a: max(0.0, float(evs.get(a, 0.0))) for a in ('FOLD', 'CALL', 'RAISE')}
            if free_check:
                probs['FOLD'] = 0.0   # never fold a free option
            if not check_call_available:
                probs['CALL'] = 0.0   # [V42_liveFixes] no Check/Call button -- see the sized path
            sharp = {a: (v ** (1.0 / temp)) for a, v in probs.items()}
            names = [a for a in ('FOLD', 'CALL', 'RAISE') if sharp[a] > 0.0]
            sharp_total = sum(sharp[a] for a in names) or 1.0
            sampled_probs = {a: (sharp[a] / sharp_total if a in names else 0.0) for a in ('FOLD', 'CALL', 'RAISE')}
            if names:
                action = random.choices(names, weights=[sharp[a] for a in names], k=1)[0]
            else:
                action = 'CALL' if (free_check and check_call_available) else 'FOLD'   # degenerate safety net
            # [TEST FLAG] Optional critic-argmax override (see USE_CRITIC_ARGMAX_ACTION); no-op off /
            # for models without a Q head.
            _critic_pick = _critic_argmax_action(getattr(active_model, 'last_q_vals', None), names) \
                if USE_CRITIC_ARGMAX_ACTION else None
            if _critic_pick is not None:
                action = _critic_pick
            chosen_key = action
            _mode = "CRITIC-ARGMAX(Q)" if _critic_pick is not None else f"Sampled policy (temp={temp:.2f}"
            _modeclose = "" if _critic_pick is not None else ")"
            reason = (f"{_mode}"
                      + (", free-check fold-masked" if free_fold_overridden else "")
                      + f"{_modeclose} -> {action}: {_fmt_dist(evs)}")
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
        # Aux-head opponent read (v21_auxhead only -- see _narrate_opponent_read; every other
        # model's aux heads are untrained noise at aux_loss_weight=0.0).
        # [V43] `has_aux` is engine-declarable, same reason as is_sized/display_tag above: a new
        # version with the identical aux-head architecture would otherwise silently lose its HUD
        # opponent-read line by not being named in this OR-chain.
        _declared_aux = getattr(active_model, 'has_aux', None)
        _aux_ok = bool(_declared_aux) if _declared_aux is not None else (
            is_v21_auxhead_model or is_v25_model or is_v26_model or is_v28_model or is_v29_model or is_v40_model or is_v41_model)
        aux_read = getattr(active_model, 'last_aux', None) if _aux_ok else None
        ev_dict['aux_read'] = aux_read
        ev_dict['thinking'] = _narrate_thinking(action, board_state, evs, aux=aux_read)
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

        # [Fable review #14/H3] Record hero's own action for this state, so the NEXT decision this
        # hand serves the model a real action history instead of all-PAD.
        self.hero_action_buffer.append(_hero_action_token(action))

        return action, reason, bet_size, ev_dict

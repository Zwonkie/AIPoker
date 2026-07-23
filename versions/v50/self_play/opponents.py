"""Uniform Opponent interface for SixMaxSimulator's 5 opponent seats (V18).

Replaces the ad-hoc wiring every prior version used: five separate `self.<style>_model`
attributes on the simulator, a hardcoded style->model `elif` chain in the seat-assignment loop,
and per-style ACTION-FORCING logic duplicated across the preflop/postflop branches of
`_opponent_decide`. That shape is what let a stray leftover line silently nullify the `tag` seat's
model load during v17_gauntlet's training run (see versions/v17_gauntlet/SPECS.md "CORRECTION") --
there was no structural reason a new `elif style == 'x':` branch couldn't be typo'd or left
half-edited, and no test surface that would have caught it.

Every opponent seat is now an `Opponent` instance built by `build_opponent_pool(...)` from a
DECLARATIVE pool config (a list of `{style, weight, model, forced}` dicts -- see config.yaml's
`opponents.pool`), not bespoke per-style code. Adding a new frozen checkpoint to a seat, or
changing whether a seat's decisions get archetype-forced, is a config change.

Two concrete kinds:
  - `HeuristicOpponent`: wraps a scripted archetype bot (`opponent_bots.py`).
  - `NNOpponent`: wraps a loaded checkpoint, decided via an injected query function (the
    simulator's own `_query_model_decide`, passed in rather than imported, so this module has no
    dependency on SixMaxSimulator itself).
Both share one `.decide_preflop(...)` / `.decide_postflop(...)` / `.apply_forcing_*(...)` surface,
so `_opponent_decide` in simulator.py no longer needs to know which kind it's talking to.
"""
import os
import random

from versions.v50.self_play.tree_opponent import TreeOpponent


# --- Per-archetype stat-forcing rules (2026-07-15, V17_gauntlet's forcing-bypass discovery) -----
# Nudges a HEURISTIC opponent's realized decision toward its target archetype's stats (e.g. nit:
# fold more once realized VPIP drifts above 15%). Deliberately ONLY ever attached to
# HeuristicOpponent -- forcing a genuine trained network's decision toward a scripted stereotype
# would mostly replace its learned judgment with the forcing rule (see
# versions/v17_gauntlet/SPECS.md). Exact same rules/probabilities as every prior version; this is
# a pure refactor, not a behavior change.

def _force_preflop_maniac(decision, vpip, agg):
    if agg < 0.60 and random.random() < 0.50:
        decision = 'raise'
    if vpip < 0.65 and random.random() < 0.50:
        decision = random.choice(['call', 'raise'])
    return decision


def _force_preflop_nit(decision, vpip, agg):
    if vpip > 0.15 and random.random() < 0.80:
        decision = 'fold'
    return decision


def _force_preflop_fish(decision, vpip, agg):
    if vpip < 0.50 and random.random() < 0.60:
        decision = 'call'
    if agg > 0.20 and random.random() < 0.80 and decision.startswith('raise'):
        decision = 'call'
    return decision


def _force_postflop_maniac(decision, vpip, agg):
    if agg < 0.60 and random.random() < 0.50:
        decision = 'raise'
    return decision


def _force_postflop_fish(decision, vpip, agg):
    if agg > 0.20 and random.random() < 0.80 and decision.startswith('raise'):
        decision = 'call'
    return decision


_FORCE_PREFLOP = {'maniac': _force_preflop_maniac, 'nit': _force_preflop_nit, 'fish': _force_preflop_fish}
_FORCE_POSTFLOP = {'maniac': _force_postflop_maniac, 'fish': _force_postflop_fish}
# 'past'/'tag' never had forcing rules in any prior version -- absent from both dicts, matching
# that (a no-op via .get(..., default) below), not a gap introduced by this refactor.

# The REAL archetype identity behind each style slot's default heuristic (matches each bot's own
# `.name` in opponent_bots.py exactly -- NOT the style key, which was never a reliable display name
# even for heuristics: the 'maniac' slot's bot is LAG, 'fish' is CALLING_STATION ("Calling
# Station"), neither literally named after their slot). Single source for `describe_pool_entry`'s
# heuristic-fallback label, used both when building the real Opponent and when the dashboard
# builds its labels straight from config (no bot instance available there).
_HEURISTIC_ARCHETYPE_NAMES = {'maniac': 'LAG', 'nit': 'Nit', 'fish': 'Calling Station', 'tag': 'TAG'}
# 'past' has no archetype of its own -- it borrows the TAG heuristic as ITS fallback too (see
# simulator.py's heuristic_bots mapping in train.py), so an un-modeled 'past' entry correctly
# resolves to 'TAG' here as well, not a separate 'Past' pseudo-archetype.


class Opponent:
    """Base interface. `recording_bot` is used ONLY for HUD VPIP/AGG stat recording (every
    opponent, heuristic or NN, records through a bot instance for dashboard telemetry) and as the
    fallback decision-maker on a model-query failure or during bootstrap heuristic-anchoring.

    `style` is an INTERNAL bookkeeping key only (which fixed stat-bucket this seat's realized
    VPIP/AGG accumulate into, and which forcing rule applies) -- it is NEVER shown to the user.
    `display_name` + `kind` ("Heuristic" | "NN") identify WHAT is actually playing this seat, and
    ARE what telemetry/dashboards show, as "{display_name} ({kind})" (`.label` below) -- e.g.
    "V15 (NN)", "TAG (Heuristic)". Previously the dashboard displayed the archetype SLOT name
    ("Nit", "TAG Bot") regardless of what was actually loaded there, which is what let
    v17_gauntlet's broken `tag` seat go unnoticed for an entire training run, and was never fully
    accurate even for heuristics (the 'maniac' slot's real bot is LAG, not "Maniac"). Dropped
    entirely in favor of naming the actual bot, not the slot it happens to occupy."""
    def __init__(self, style, recording_bot, forced, display_name=None):
        self.style = style
        self.recording_bot = recording_bot
        self.forced = forced
        self.display_name = display_name or style.capitalize()
        self.kind = 'Opponent'  # overridden by concrete subclasses

    @property
    def label(self):
        return f"{self.display_name} ({self.kind})"

    def decide_preflop(self, equity, pot_odds, pot_size, stack, num_opps, cards,
                        table_state_dict, model_state_history, hero_actions_history,
                        force_heuristic=False):
        raise NotImplementedError

    def decide_postflop(self, equity, pot_odds, pot_size, stack, street_idx, num_opps, cards,
                         table_state_dict, model_state_history, hero_actions_history,
                         force_heuristic=False):
        raise NotImplementedError

    def apply_forcing_preflop(self, decision, opp_vpip, opp_agg):
        return decision

    def apply_forcing_postflop(self, decision, opp_vpip, opp_agg):
        return decision


class HeuristicOpponent(Opponent):
    """A pure scripted archetype bot (fish/maniac/nit/tag heuristics)."""
    def __init__(self, style, bot, forced=True, display_name=None):
        display_name = display_name or _HEURISTIC_ARCHETYPE_NAMES.get(style, style.capitalize())
        super().__init__(style, recording_bot=bot, forced=forced, display_name=display_name)
        self.bot = bot
        self.kind = 'Heuristic'

    def decide_preflop(self, equity, pot_odds, *_a, force_heuristic=False, **_kw):
        return self.bot.decide_preflop(equity, pot_odds)

    def decide_postflop(self, equity, pot_odds, pot_size, stack, street_idx, *_a,
                         force_heuristic=False, **_kw):
        return self.bot.decide_postflop(equity, pot_odds, pot_size, stack, street_idx)

    def apply_forcing_preflop(self, decision, opp_vpip, opp_agg):
        if not self.forced:
            return decision
        return _FORCE_PREFLOP.get(self.style, lambda d, v, a: d)(decision, opp_vpip, opp_agg)

    def apply_forcing_postflop(self, decision, opp_vpip, opp_agg):
        if not self.forced:
            return decision
        return _FORCE_POSTFLOP.get(self.style, lambda d, v, a: d)(decision, opp_vpip, opp_agg)


def _call_amount_from_pot_odds(pot_odds, pot_size):
    """[V41, review #8] Recover the REAL chips-to-call from (pot_odds, pot_size).

    Every NN opponent query used to pass `pot_odds * pot_size` as the model's `call_amount`
    feature. Since the caller defines `pot_odds = to_call / (pot + to_call)`, that expression is
    `to_call * pot / (pot + to_call)` -- NOT `to_call`. A pot-sized bet arrived at the network as
    HALF its real size, and the error grows with bet size, so the lagged-self mirror and every
    frozen checkpoint played a systematically price-distorted version of themselves (which also
    quietly flattered every "beats frozen VX" head-to-head).

    Inverting the caller's own definition is exact:
        pot_odds = t / (p + t)  =>  t = pot_odds * p / (1 - pot_odds)
    The `1 - pot_odds` denominator only approaches zero when to_call dwarfs the pot; clamp it so a
    huge overbet can't produce an infinity, and fall back to the pot itself as a sane ceiling.
    """
    if pot_odds <= 0.0:
        return 0.0
    if pot_odds >= 0.999:
        return pot_size * 999.0
    return pot_odds * pot_size / (1.0 - pot_odds)


class NNOpponent(Opponent):
    """A loaded checkpoint, queried via an injected `query_fn` (the simulator's own
    `_query_model_decide`, bound and passed in at construction). `forced` defaults to False --
    unlike a heuristic, a genuine trained network already has its own tendencies; forcing it
    toward an archetype stereotype is an explicit opt-in, not the default, per the v17_gauntlet
    finding (versions/v17_gauntlet/SPECS.md)."""
    def __init__(self, style, model, query_fn, error_fn, recording_bot, forced=False, display_name=None):
        super().__init__(style, recording_bot=recording_bot, forced=forced, display_name=display_name)
        self.model = model
        self._query_fn = query_fn
        self._error_fn = error_fn
        self.kind = 'NN'

    def decide_preflop(self, equity, pot_odds, pot_size, stack, num_opps, cards,
                        table_state_dict, model_state_history, hero_actions_history,
                        force_heuristic=False):
        if force_heuristic:
            return self.recording_bot.decide_preflop(equity, pot_odds)
        try:
            return self._query_fn(self.model, cards, equity, pot_size,
                                   _call_amount_from_pot_odds(pot_odds, pot_size),
                                   stack, num_opps, table_state_dict, model_state_history,
                                   hero_actions_history)
        except Exception as e:
            self._error_fn("_opponent_decide/preflop", e)
            return self.recording_bot.decide_preflop(equity, pot_odds)

    def decide_postflop(self, equity, pot_odds, pot_size, stack, street_idx, num_opps, cards,
                         table_state_dict, model_state_history, hero_actions_history,
                         force_heuristic=False):
        if force_heuristic:
            return self.recording_bot.decide_postflop(equity, pot_odds, pot_size, stack, street_idx)
        try:
            return self._query_fn(self.model, cards, equity, pot_size,
                                   _call_amount_from_pot_odds(pot_odds, pot_size),
                                   stack, num_opps, table_state_dict, model_state_history,
                                   hero_actions_history)
        except Exception as e:
            self._error_fn("_opponent_decide/postflop", e)
            return self.recording_bot.decide_postflop(equity, pot_odds, pot_size, stack, street_idx)
    # apply_forcing_preflop/postflop intentionally inherited as no-ops from Opponent -- an
    # NNOpponent's `forced` flag is stored (so config CAN opt in later) but this refactor doesn't
    # add new behavior; the base class no-op matches "forced=False" exactly, and no version has
    # ever set an NN opponent's forced=True, so this is a pure equivalence refactor for now.


def describe_pool_entry(entry):
    """(display_name, kind) for a pool config entry -- e.g. ('V15', 'NN') for a
    `model: frozen_v15.pth` entry, ('Lagged-Self', 'NN') for a `lagged_self: true` entry, or
    ('Nit', 'Heuristic') / ('TAG', 'Heuristic') / etc for a bare heuristic entry (no override --
    real archetype identity from `_HEURISTIC_ARCHETYPE_NAMES`, NOT the style/slot key). The SINGLE
    source both `Opponent.display_name`/`.kind` and dashboard seat labels (`train.py
    print_dashboard`) derive from, so a bot's name/version/type can never drift between "what's
    actually loaded" and "what the telemetry shows" -- exactly the kind of mismatch that went
    unnoticed for v17_gauntlet's silently-broken `tag` seat. Callers render `f"{name} ({kind})"`.

    `entry['model']`, if present, may be a bare filename OR a full resolved path (the caller
    passes whichever it has -- only the basename is used here); a `frozen_` prefix and the
    extension are stripped and the remainder upper-cased ('frozen_v15.pth' -> 'V15').
    `lagged_self: true` wins over a `model` path if both are somehow present (the 'past' seat's
    resolved entry carries both -- see train.py's resolved_pool_config construction).
    """
    if entry.get('lagged_self'):
        return 'Lagged-Self', 'NN'
    if entry.get('tree_cluster') is not None:
        return f"RealPlay-{entry['tree_cluster']}", 'Tree'
    model_path = entry.get('model')
    if model_path:
        stem = os.path.splitext(os.path.basename(model_path))[0]
        if stem.startswith('frozen_'):
            stem = stem[len('frozen_'):]
        return stem.upper(), 'NN'
    style = str(entry.get('style', '?'))
    return _HEURISTIC_ARCHETYPE_NAMES.get(style, style.capitalize()), 'Heuristic'


def build_opponent_pool(pool_config, heuristic_bots, query_fn, error_fn, load_model_fn):
    """Build {style: Opponent} from a declarative pool config.

    pool_config: list of dicts, each `{'style': str, 'weight': float, 'model': str|None,
        'forced': bool|None, 'lagged_self': bool|None}`. `model`, when present, is an ABSOLUTE
        path to a checkpoint file (resolution -- which frozen file, or this batch's lagged-self
        snapshot -- happens in the caller, same as every prior version; this factory only loads
        whatever path it's given).
    heuristic_bots: {style: bot_instance} fallback/recording bot for each style (also used as the
        sole decision-maker for any style with no 'model' entry, or when a specified model fails
        to load -- e.g. 'past' before the first lagged snapshot exists).
    query_fn / error_fn: the simulator's own `_query_model_decide` / `_note_query_error` bound
        methods, injected so this module has no dependency on SixMaxSimulator.
    load_model_fn: callable(path) -> loaded model or None (the caller's own `_load_worker_model`,
        already handling fail-loud-for-required / warn-and-None-for-optional).

    Returns {style: Opponent}. A style whose model path is absent or fails to load falls back to
    HeuristicOpponent(forced=True) -- matching the exact "model is None -> forcing eligible,
    heuristic decision-maker" semantics every prior version had, just now explicit instead of an
    implicit `if model is None` check scattered across two call sites. Each built Opponent's
    `.display_name` is derived via `describe_pool_entry` -- a fallback-to-heuristic still gets a
    plain archetype-style label (the model that failed to load isn't what's actually playing).
    """
    pool = {}
    for entry in pool_config:
        style = entry['style']
        bot = heuristic_bots[style]
        tree_cluster = entry.get('tree_cluster')
        model_path = entry.get('model')
        model = load_model_fn(model_path) if model_path else None
        if tree_cluster is not None:
            # [V25] Real-play-fitted opponent (Pluribus/WSOP full-info hands, see
            # tree_opponent.py) -- takes priority over `model`/heuristic fallback for this entry
            # since a tree_cluster key is an explicit, unambiguous request for this specific kind.
            forced = bool(entry.get('forced', False))
            pool[style] = TreeOpponent(tree_cluster, recording_bot=bot, style=style,
                                        forced=forced, display_name=f"RealPlay-{tree_cluster}")
        elif model is not None:
            forced = bool(entry.get('forced', False))
            name, _kind = describe_pool_entry(entry)
            pool[style] = NNOpponent(style, model, query_fn, error_fn, recording_bot=bot,
                                      forced=forced, display_name=name)
        else:
            # Fell back to heuristic (no model configured, or the configured one failed to load)
            # -- HeuristicOpponent's own constructor already derives the correct real archetype
            # name (_HEURISTIC_ARCHETYPE_NAMES) when display_name isn't passed, so no need to
            # call describe_pool_entry(entry) here (it would be wrong anyway if `entry` requested
            # a model that just failed -- this seat is NOT playing that model).
            pool[style] = HeuristicOpponent(style, bot, forced=True)
    return pool

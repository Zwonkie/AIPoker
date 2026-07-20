"""V24 manifest -- decouples the [BET-1] fix's target-EV computation from live opponent behavior,
and adds a "show of strength" mechanism for non-all-in raises. Same context/contract as V23
(context_dim=44, contract_version=7, `pot_type` unchanged) -- this version touches ONLY
`simulator.py`'s `_mc_target_evs_sized` (hero's own training-target computation) and
`opponent_bots.py` (a new per-personality trait), not the network's input features.

**Why**: V23 applied the [BET-1] price-sensitivity fix (`VALUE_PRICE_SENSITIVITY`) directly inside
`opponent_bots.py`'s `decide_postflop`/`decide_preflop` -- the SAME functions `_mc_target_evs_sized`
calls to sample opponent fold rates for computing hero's own per-size EV target. Result (see
versions/v23/SPECS.md): `action_diversity`/`deep_stack_ood_guard` REGRESSED, not improved. Root
cause, confirmed by code inspection: making bots fold more to oversized bets doesn't just describe
more realistic live play -- it ALSO mechanically inflates hero's own ALLIN training target, since
`p_all_fold * pot` is credited straight into that size's counterfactual EV. The fix's two effects
(opponents demand more to continue vs. hero gets more fold-equity credit for shoving) pointed in
opposite directions, and the wrong one won.

**Fix 1 -- decoupling**: `_mc_target_evs_sized` no longer calls `bot.decide_preflop`/
`decide_postflop` directly. It now uses a dedicated `_ev_target_fold_decision` (new,
`simulator.py`) that reverts to the PRE-BET-1 flat `need_for_value` value branch for the
target-EV's fold-sampling, while keeping the original (validated, pre-dates BET-1) P1b
`continue_bar` price-sensitivity. Live self-play opponents (`_opponent_decide`) still use the
BET-1-fixed, price-sensitive functions in `opponent_bots.py` unchanged -- so self-play itself stays
realistic, but hero's own regression target no longer gets inflated by it.

**Fix 2 -- "show of strength" for raises (not all-in)**: within that same decoupled fold model, a
new per-personality trait `bot_bluff_perc` (added to `FuzzyPlayerArchetype`, `opponent_bots.py`)
drives a raise-only bonus: for non-all-in raise sizes, with probability `1.0 -
opponent.bot_bluff_perc`, the opponent "respects" the raise as committed value and folds more than
raw price alone would justify (`need_for_value` boosted by `RAISE_RESPECT_BOOST`). All-in gets
NONE of this bonus -- priced honestly on raw pot odds alone. This directly targets the no-middle-
gear problem: today fold-equity scales monotonically with size (bigger bet -> more folds), which
makes all-in always dominate by construction; giving raises a categorical, non-price-scaling fold-
equity source that shoves don't get breaks that monotonicity, creating a genuine incentive to use a
raise over a shove in some spots.

`bot_bluff_perc` is a bot's own trait (how often ITS raises are "for show," not backed by real
strength) reused, when that bot is the one FACING a raise, as an inverse read of how skeptical it
is of others' raises -- a personality that bluffs a lot itself assumes others do too (low respect,
harder to fold via a raise); one that rarely bluffs trusts raises more (high respect, easier to
fold via a raise). This is deliberately a NEW, general-purpose per-bot trait (not derived from the
existing `base_bluff_freq`, which only governs a bot's own below-the-bar bluff-raise frequency) --
scoped to this one mechanism for now, but intended for reuse in other bluff-related calculations
later (e.g. preflop bluff-raise frequency) per this version's own discussion.

Calibration (values, EV-arithmetic checks) documented in `opponent_bots.py`'s own docstring and
`versions/v24/SPECS.md`.

Base: copied from `versions/v23` (`pot_type`, deep-stack curriculum, entry-sizing all inherited
unchanged). Opponent pool unchanged from V22/V23 (maniac/fish as plain heuristics).

See: versions/v23/SPECS.md (root-cause finding this version fixes) |
.agents/skills/OFK/references/known-shortcomings-backlog.md [BET-1] | versions/v24/SPECS.md (full
detail, calibration data)
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v24",
    context_dim=44,                  # UNCHANGED from V23 -- no context/feature change
    contract_version=7,              # UNCHANGED from V23 -- target-EV computation only, no contract change
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v24.core.model:PokerEVModelV4",
    contract_class="versions.v24.core.contract:ContractV12",
    weights_dir="versions/v24/weights",
    status="training",
    milestone=False,
)

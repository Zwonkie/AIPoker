"""V23 manifest -- two bundled additions on top of V22's foundation (contract_version 7,
context_dim 43->44), both scoped in the same "let's discuss BET-1 / clean up backlog" session that
produced V22:

1. **[BET-1] opponent price-sensitivity fix** (`versions/v23/self_play/opponent_bots.py`): the
   long-standing root cause of hero's shove-preference -- `FuzzyPlayerArchetype`'s "value raise
   regardless of price" branch treated a min-bet and an all-in shove identically once equity
   cleared a FLAT `need_for_value` threshold, so the critic never learned any downside to sizing
   up. Fixed by making the value bar itself rise with bet size (`value_bar = need_for_value +
   VALUE_PRICE_SENSITIVITY * pot_odds`), the same mechanism the existing marginal `continue_bar`
   already had, just extended to the top of the equity range instead of stopping short of it.
   Applied to both `decide_postflop` and a restructured `decide_preflop` (which previously checked
   the value threshold BEFORE even looking at `pot_odds`/`facing_bet`).

   `VALUE_PRICE_SENSITIVITY = 0.05`, calibrated via a standalone probe (not checked into the repo;
   results captured in `opponent_bots.py`'s own docstring and `versions/v23/SPECS.md`) that
   measured P(fold) at value-tier equities across a pot_odds grid for all 4 archetypes (TAG, LAG,
   NIT, CALLING_STATION). Key finding: a single global constant compounds UNEVENLY across
   archetypes -- NIT (tightest, already highest need_for_value+style_shift) overcorrected into
   folding 95%+ of 80%-equity hands to a shove at 0.15, while TAG only reached 58% at the same
   value. 0.05 was chosen as the largest value that still produced a real, non-degenerate
   fold-equity gradient across all four archetypes without any of them collapsing into "always
   folds a strong hand."

2. **`pot_type` feature** (deferred from V22, now built): whole-hand raise count so far (any
   street, any actor), bucketed 0=limped/unraised, 1=single-raised, 2=3-bet+, one new APPENDED
   global context feature (ctx[43] -- every existing index 0-42 stays stable). Distinguishes a pot
   that's seen one raise from one that's been 3-bet+, which `committed`/`call_amount` alone don't
   cleanly capture (a big call_amount can come from one big bet OR a raise war -- different
   situations). Sourced from a new `raise_count` counter in `simulator.py` (mirrors how
   `committed[]` already existed for the entry-sizing feature), threaded through
   `core/board_state.py`'s new `BoardState.pot_type` field the same way `hero_committed`/
   `hand_strength` are.

Base: copied from `versions/v22` (the current live-candidate foundation), inheriting its deeper
stack curriculum and entry-sizing features unchanged. Opponent pool inherited unchanged from V22
(maniac/fish as plain heuristics, since frozen cross-version NN checkpoints remain architecture-
incompatible with this contract).

See: versions/v22/SPECS.md (base) | .agents/skills/OFK/references/known-shortcomings-backlog.md
[BET-1] (root cause + prior investigation) | versions/v23/SPECS.md (full detail, calibration data)
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v23",
    context_dim=44,                  # 43 (V22) + 1 pot_type
    contract_version=7,              # bumped: 1 new appended feature (pot_type)
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v23.core.model:PokerEVModelV4",
    contract_class="versions.v23.core.contract:ContractV12",
    weights_dir="versions/v23/weights",
    status="training",
    milestone=False,
)

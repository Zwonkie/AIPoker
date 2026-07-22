"""V44 manifest -- `equity_edge` normalized by the EFFECTIVE contested field.

Base: cloned from `versions/v43` (fresh weights, not resumed -- per [VAL-5]).

CONTRACT CHANGE (contract_version 8 -> 9, context_dim unchanged at 54): ctx[35] `equity_edge` keeps
its index and its width but changes MEANING, so V43 and earlier checkpoints are NOT
behaviourally compatible with this contract even though they would load. The bump is what stops
that from happening silently.

## Why

`equity_edge` exists to say "this hand is strong FOR THIS FIELD SIZE" -- and no model in this
lineage has ever used it. Measured on V43 (AKs preflop, equity computed exactly as training
computes it, `hand_strength` constant at 0.661 throughout):

    opp   equity   equity_edge   P(FOLD)   P(raise)   chosen
      1    0.670          1.34     0.000      0.826    RAISE_POT
      3    0.600          2.40     0.008      0.674    CALL
      5    0.520          3.12     0.907      0.005    FOLD

The edge feature climbs 1.34 -> 3.12 exactly as designed while P(raise) collapses 0.826 -> 0.005:
a top-5 starting hand folded 91% of the time at the precise moment its edge peaks. A threshold
sweep confirms the model gates on near-constant ABSOLUTE equity (eq* ~ 0.51 from 2 opponents up),
so edge* rises linearly with field size (0.79/1.47/2.04/2.58/3.08). If the model were using the
edge, edge* would be FLAT.

## The root cause: the two halves counted different things

`equity` is measured against the EFFECTIVE contested field -- preflop, every still-to-act opponent
is rolled at their VPIP and all-fold samples are SKIPPED (see `compute_range_aware_equity`), so
5 Yellow opponents is really E[k|k>=1] = 1.80 expected contesting opponents, fair share 0.357.
But `equity_edge = equity * (num_active + 1)` normalized by the NOMINAL field, fair share 0.167.
Different denominators, diverging as the field grows -- so the feature was never the clean
"equity vs fair share" ratio its docstring claimed, which is the likely reason it was never learned.

## The change

`n` in the `n+1` denominator is now the effective contested field, closed-form off the same
`_COLOR_TO_VPIP` the equity roll already uses:

    E[k | k>=1] = (|front| + sum(p_after)) / (1 - prod(1 - p_after))

Front opponents (already committed this round) are p=1 and force the denominator to 1, since
someone is then guaranteed in and no conditioning is needed. POSTFLOP THIS DEGENERATES TO THE
NOMINAL COUNT -- there is no fold-roll postflop -- so this is a PREFLOP-ONLY change and postflop
semantics are untouched.

Measured effect on the feature itself (AKs): a 2.4x field-size swing becomes flat (1.32 / 1.37 /
1.46, spread 0.19), while it still separates hands, which is the entire point of it:
AA 1.70-2.16, AKs 1.30-1.49, JTs 0.88-1.01, 94o 0.64-0.67, 72o 0.59-0.62. The residual upward
drift on AA is real signal -- a monster's share genuinely does outgrow fair share as the field
widens -- not noise.

Computed at the CALLER and carried on `BoardState.effective_field`, the same pattern `equity` and
`hand_strength` already use: the contract only receives seat state and cannot know the front/after
split, while the simulator and the live path both build front/after immediately before their
equity call. `ctx[5] num_active` deliberately stays NOMINAL -- the model should still know how many
players are seated; only the edge denominator changes.

Chosen over the other half of the fork (dropping the fold-roll from `equity`), which would have
moved ctx[3] -- the most load-bearing feature in an equity-primary architecture -- and destroyed the
conditional-on-contested property that stops 72o and AA both reading ~0.9. This touches only
ctx[35], a feature the model demonstrably ignores today, so there is almost nothing to unlearn.

Everything else is V43 unchanged: realization discount and ALLIN veto removed, TARGET_CLIP_BB 100,
risk_aversion_coefficient 0.20. See versions/v44/SPECS.md and [BET-3] in
.agents/skills/OFK/references/known-shortcomings-backlog.md.
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v44",
    context_dim=54,                  # unchanged -- ctx[35] changes MEANING, not width
    contract_version=9,              # BUMPED from 8: ctx[35] semantics changed, V43 weights are
                                     # not behaviourally compatible even though they would load
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v44.core.model:PokerEVModelV4",
    contract_class="versions.v44.core.contract:ContractV12",
    weights_dir="versions/v44/weights",
    status="active",                 # NOT TRAINED YET -- V43 is still the live model
    milestone=False,
)

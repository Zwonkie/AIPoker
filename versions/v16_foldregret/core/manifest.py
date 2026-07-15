"""V16_foldregret manifest — clones V16 (versions/v16), same main-run recipe/config, ONE isolated
change: the actor's regret-matching baseline.

`regret_match_policy` (train.py) used to measure each action's regret against the MEAN of all
action values. That let a bluff-raise's fold-equity (a legitimately positive EV from folding out
opponents) drag the mean up enough that OTHER, independently-negative actions (e.g. calling with
air) could still show positive regret and retain real probability mass -- diagnosed 2026-07-15
from the training dashboard's Equity Action Matrix (<20%/20-40% equity buckets net-losing chips
despite ~50% continue rates). Fix: regret is now measured against FOLD's value directly (always
0), so every action has to independently beat "just fold" rather than beat a mean that includes
other bad options. See versions/v16_foldregret/SPECS.md for the full trace + math.

Same 6-action contract, same context_dim=35, same opponent pool/config as the main V16 run --
this is a single-variable comparison against versions/v16, trained FRESH (no warm-start), exactly
like V16 itself was.

OUTCOME (2026-07-15): trained to completion (100k hands), NOT deployed. tools/model_verify --full
found the air/draws fix worked as intended (Fold% up sharply in both buckets, held stable to
completion) but introduced a real regression on `vpip_adapts_to_style` at deep stacks (V16 PASSES
+8.4pt delta; this version FAILS at +2.0pt) -- the fold-relative regret baseline appears to zero
out the small style-conditioned edge V16's [P4] fix relies on at deep stacks, alongside the bad
air/draws actions it was targeting. Kept as a validated experiment result / reference for a future
blended-baseline follow-up, not promoted over versions/v16. Full trace: SPECS.md "Outcome" section.

See: .agents/skills/OFK/references/versioned-architecture-guardrails.md
     versions/v16_foldregret/SPECS.md   |   versions/v16/SPECS.md (parent line)
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v16_foldregret",
    context_dim=35,                 # input schema UNCHANGED (35-feature context)
    contract_version=3,             # UNCHANGED from v15: same discretized 6-action space
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v16_foldregret.core.model:PokerEVModelV4",
    contract_class="versions.v16_foldregret.core.contract:ContractV12",
    weights_dir="versions/v16_foldregret/weights",
    status="active",
    milestone=False,                # v13 remains the kept milestone/fallback
)

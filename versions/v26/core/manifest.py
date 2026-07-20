"""V26 manifest -- SAME architecture/contract/target-EV mechanism as V25 (context_dim=44,
contract_version=7, the multi-street `_rollout_continuation_ev` fix -- see versions/v25/SPECS.md
for that mechanism's own derivation, calibration, and results). V26 changes ONLY the training
OPPONENT POOL: 2 of 5 seats (previously `maniac`/`nit` heuristics) are replaced with `TreeOpponent`
instances -- real decision models fit directly on human hand-history data (Pluribus/WSOP
full-information hands), not hand-designed rule formulas and not anything trained inside this
simulator's own self-play loop. See `versions/v25/self_play/tree_opponent.py` and
`versions/v25/SPECS.md`'s "New infrastructure" section for the full pipeline (download -> pokerkit
replay -> identity-agnostic behavior clustering -> per-cluster XGBoost -> Opponent-interface
integration) and its own honest caveat: the 4 clusters found in the Pluribus data are only weakly
differentiated from each other (all elite, similarly-skilled professionals, not a diverse
recreational population) -- this experiment tests whether even that modest, but genuinely
non-heuristic, source of behavioral diversity changes what hero learns, not whether it fixes
[BET-1] specifically.

**Why this experiment**: 2026-07-18 discussion -- everything in the current opponent pool, heuristic
archetypes or the lagged-self NN mirror, is either hand-designed or trained inside this same
simulator's self-play loop, so it shares whatever representational ceiling that loop already has
(lagged-self, in particular, inherits the SAME blind spots hero has, since it grew up against the
same heuristic population). A model fit directly on real, external human data is the one source of
behavior in this training population NOT shaped by this codebase's own assumptions.

**Not attempted here**: carrying a real predicted bet SIZE through to the two TreeOpponent seats
(they still get the same fixed 0.75x-pot sizing every heuristic opponent's 'raise' decision gets --
see tree_opponent.py's own docstring) -- a real follow-up requiring an Opponent-interface extension,
not a training-recipe change.

Base: copied from `versions/v25` (multi-street EV fix, entry-sizing, deep-stack curriculum,
`pot_type`, `bot_bluff_perc` all inherited unchanged). Fresh weights, no `--resume_path`, per
[VAL-5] -- a real opponent-population change needs its own from-scratch run to evaluate cleanly.

See: versions/v25/SPECS.md (the EV-fix mechanism + full TreeOpponent pipeline this version's pool
change relies on) | .agents/skills/OFK/references/known-shortcomings-backlog.md
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v26",
    context_dim=44,                  # UNCHANGED from V25 -- no context/feature change
    contract_version=7,              # UNCHANGED from V25 -- opponent pool only, no contract change
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v26.core.model:PokerEVModelV4",
    contract_class="versions.v26.core.contract:ContractV12",
    weights_dir="versions/v26/weights",
    status="training",
    milestone=False,
)

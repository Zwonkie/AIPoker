"""V20_preflopEq_AI manifest — clone of V20_preflopEq (versions/v20_preflopEq). IDENTICAL
architecture/tensor schema (context_dim=37, contract_version=5, same PokerEVModelV4 arch) --
this is a training-RECIPE experiment, not a contract change.

Motivation (2026-07-17, from a live-session discussion of why the model shoves all-in so
readily): traced the mechanism in `opponent_bots.py`'s `FuzzyPlayerArchetype` -- once an
opponent's equity clears its `need_for_value` threshold, it calls/raises "regardless of price"
(price-insensitive by design), and the continue-bar saturates at 0.95, so bigger bets barely
induce more folds beyond a point. Against that specific population, shoving captures the same
fold-equity as a smaller bet while roughly doubling the value extracted when called -- confirmed
directly via the trained critic's own Q-values (clean, monotonic ALLIN > RAISE_POT > RAISE_66 >
RAISE_33 > CALL at every equity/stack combo tested, no smooth EV gradient favoring the middle
sizes). The suspected root cause: the training population is dominated by these deterministic,
price-insensitive heuristic bots, which don't punish overbets the way a real opponent (or a
genuine learned policy) would.

**This version's experiment**: shift the opponent pool toward real NN opponents (learned, more
continuous decision boundaries) instead of relying mainly on the heuristic bots, without needing
the (still unbuilt) per-model contract-selection mechanism -- every opponent here is natively
context_dim=37/contract_version=5, either a frozen restore-point checkpoint from V20_preflopEq's
own 75k run or a lagged self-play mirror of THIS run, so no cross-version scale mismatch is
possible. Pool: `past` (lagged self-play, 0.25), `frozen_50k` (V20_preflopEq's 50k checkpoint,
0.20), `frozen_25k` (V20_preflopEq's 25k checkpoint, 0.15), `tag` (heuristic anchor, 0.25), `nit`
(heuristic, short-stack push/fold discipline, 0.15) -- see config.yaml.

Honest caveat (documented before running, not after): `frozen_50k`/`frozen_25k`/`past` are all
the SAME lineage, trained under the same heuristic-dominated conditions this experiment is
questioning -- they may share the shove-bias rather than punish it. This tests "does self-play
diversity alone fix it"; if it doesn't move `check_action_diversity`'s raise-bucket usage or the
critic's mid-size Q-values, the more direct lever (making the heuristic bots' value-branch
price-sensitive) is the next thing to try, not iterating further on pool composition.

Target: 150k hands, checkpoints/sanity checks every 35k (config.yaml
checkpoint_dump_interval=35000). Also carries frozen_v20_preflopEq.pth (this version's own
75k-hand final model) as its `beats_frozen_predecessor` benchmark -- SAME architecture as this
version (unlike v20_preflopEq's own attempt against frozen V20, which SKIPped on a scale
mismatch), so this comparison should actually run.

See: versions/v20_preflopEq/SPECS.md (parent) | versions/v20_preflopEq_AI/SPECS.md (this version,
to be written once results are in)
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v20_preflopEq_AI",
    context_dim=37,                 # UNCHANGED from v20_preflopEq -- same tensor schema
    contract_version=5,              # UNCHANGED -- opponent-pool/training-recipe experiment only
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v20_preflopEq_AI.core.model:PokerEVModelV4",
    contract_class="versions.v20_preflopEq_AI.core.contract:ContractV12",
    weights_dir="versions/v20_preflopEq_AI/weights",
    status="active",
    milestone=False,
)

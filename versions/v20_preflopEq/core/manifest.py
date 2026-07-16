"""V20_preflopEq manifest — clone of V20 (versions/v20). Implements the range-aware equity
investigation in versions/v20_preflopEq/SPECS.md:

  1. [Finding 2] Front/after positional split in the hero equity call (`compute_range_aware_equity`
     / `_calculate_range_aware_equity`, shared by training self-play and live serving). Opponents
     who have ALREADY acted and committed this betting round are now guaranteed in (no VPIP
     fold-roll); opponents still to act keep the existing roll. Previously every active opponent
     got the identical flat roll regardless of the simulator's own (correct) ground-truth action
     order -- see SPECS.md for the quantified impact.

  2. [Bug found while implementing, fixed here] `train.py::vectorize_hand_samples` never actually
     received V20's own /100,/250 context-feature rescale -- it kept building `ctx[1]`/`ctx[2]`/
     `ctx[9]`/opp_stack on the OLD /400,/1000 scale while every inference path (self-play rollout
     via `_query_model_decide`, live serving via `core/decision.py`) already used the new /100,/250
     scale via `contract.py`. That meant V20's deployed weights were gradient-fit to one scale but
     acted-on at a different one -- a real, unintentional train/rollout+live mismatch, not a
     deliberate scope choice (V20's own manifest/SPECS.md describe the rescale as covering "every
     context feature" via contract.py alone). Fixed here by having `vectorize_hand_samples` share
     the exact same per-feature scale/clamp logic as `contract.py` (factored into one place so this
     class of drift can't recur -- see contract.py's `context_feature_scale` helpers).

  3. Two new context features (both appended, not inserted, so every existing index is unchanged):
     `equity_edge` (equity's edge over the field-size fair share, `equity * (num_active + 1)`) and
     `hand_strength` (field-independent card quality -- preflop an O(1) lookup into
     `preflop_equities.csv`'s 169-hand vs-1-random equity table, postflop a cheap live vs-1-random
     MC call). context_dim 35 -> 37, contract_version 4 -> 5 (NOT interchangeable with any v20
     checkpoint despite being the same lineage -- fresh training required, same as every prior
     context_dim bump in this line).

See: versions/v20_preflopEq/SPECS.md | versions/v20/SPECS.md (parent)
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v20_preflopEq",
    context_dim=37,                 # 35 (v20) + equity_edge + hand_strength
    contract_version=5,             # BUMPED: new features appended + the vectorize_hand_samples
                                     # scale fix -- not interchangeable with contract_version=4 (v20).
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v20_preflopEq.core.model:PokerEVModelV4",
    contract_class="versions.v20_preflopEq.core.contract:ContractV12",
    weights_dir="versions/v20_preflopEq/weights",
    status="active",
    milestone=False,                # v13 remains the kept milestone/fallback
)

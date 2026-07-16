"""V20 manifest — clones V19 (versions/v19, DEPLOYED LIVE, the P0/hero_position pass). V20 targets
the sharpest lead V19 turned up for `deep_stack_ood_guard`, the 6-version-old bug: NOT the
`policy_tightness_bb` threshold near eq 0.45 that V19's SPECS.md flagged as the top suspect (that
hypothesis was RE-EXAMINED and DISPROVEN here -- see SPECS.md, the pivot math actually runs
BACKWARDS from what would explain the failure grid). Investigating further found the real
candidate: a systemic context-feature RESOLUTION problem.

**Rescaled context features (ctx[1]=hero_stack, ctx[2]=pot_size, ctx[9]=call_amount, + the 5x
opp_stack slots) in `core/contract.py`.** All were normalized against a hypothetical ~400bb range
(`/400`, `/1000`) dating back through the whole model line, but `stack_depth_mix` has capped real
training stacks at 5-50bb since V15 -- meaningfully different stacks (15bb vs 40bb) landed only
0.0625 apart on a nominal [0,1] feature, over 85% of each feature's representable range never
touched by any training example. Same class of problem [P5] already flagged for call_amount alone;
turns out it's systemic across every money-denominated feature, not just bet-size. Rescaled to
/100 (stack, call_amount) and /250 (pot) -- same context_dim=35 (no width change, in-place value
rescale), 4x more resolution in the actually-trained band. NOT backward-compatible with any
older-scale checkpoint's learned weights -- requires fresh training, and (until a per-model
contract-selection mechanism exists) can't safely use any PRIOR version's frozen checkpoint as a
live opponent through this version's own simulator, since it would silently receive the wrong
scale. Scope decision: 'nit'/'tag' reverted to plain heuristic archetypes for this version instead
(see config.yaml) rather than building that mechanism now -- isolates the rescale as the one true
variable.

Same 6-action contract, same opponent-architecture refactor as V18/V19; only the context-feature
SCALE changed, not width or the plumbing layer.

See: .agents/skills/OFK/references/versioned-architecture-guardrails.md
     versions/v20/SPECS.md   |   versions/v19/SPECS.md (parent, deployed live)
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v20",
    context_dim=35,                 # input schema width UNCHANGED (35-feature context)
    contract_version=4,             # BUMPED: same width, but ctx[1]/ctx[2]/ctx[9]/opp_stack now
                                     # carry a different SCALE -- not interchangeable with contract_version=3
                                     # checkpoints even though context_dim matches (see contract.py).
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v20.core.model:PokerEVModelV4",
    contract_class="versions.v20.core.contract:ContractV12",
    weights_dir="versions/v20/weights",
    status="active",
    milestone=False,                # v13 remains the kept milestone/fallback
)

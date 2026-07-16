"""V18 manifest — clones V17_gauntlet (versions/v17_gauntlet, deployed live), IDENTICAL training
recipe and config (same opponent pool composition: fish/maniac heuristic, nit=frozen V15,
tag=frozen V16, past=true lagged self-play mirror). The ONLY change is architectural, not
behavioral: the ad-hoc opponent wiring (five separate `*_model` attributes, a hardcoded
style->model elif chain, per-style forcing logic scattered across two call sites, a new positional
`simulate_worker` arg + config key required for every new opponent slot) is replaced with a
uniform `Opponent` interface (`self_play/opponents.py`):

- `HeuristicOpponent(bot, style)` -- wraps an existing scripted archetype bot.
- `NNOpponent(model, style)` -- wraps a loaded checkpoint, querying it via `_query_model_decide`.
- `ForcedOpponent(inner, style)` -- wraps EITHER of the above, applying the archetype's
  stat-forcing rules (previously a hardcoded "only if style is heuristic" implicit rule during
  v17_gauntlet's build -- now an explicit, composable choice per pool entry).

Built from a DECLARATIVE pool config (`opponents.pool` entries are now `{style, weight, model,
forced}` dicts instead of bare style-name strings + a growing set of bespoke `*_model_filename`
config keys) via `self_play/opponents.py`'s `build_opponent_pool(...)` factory. Adding or swapping
an opponent (a new frozen checkpoint, a different forcing choice) is now a config change, not a
code change across `simulate_worker`'s signature + the worker-args tuple + a new `elif` branch.

Motivation: building v17_gauntlet's `tag` slot (a model-loading option that never existed for that
style before) required six separate touches across simulator.py/train.py, including a fragile
POSITIONAL worker-args tuple that silently breaks if misordered. This refactor is a pure plumbing
change -- verified behaviorally equivalent to v17_gauntlet via a 20k-hand sanity run before any
further training investment, not a new training experiment itself. See SPECS.md.

Same 6-action contract, same context_dim=35, same weights_dir layout as every version in this line.

See: .agents/skills/OFK/references/versioned-architecture-guardrails.md
     versions/v18/SPECS.md   |   versions/v17_gauntlet/SPECS.md (parent line, identical recipe)
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v18",
    context_dim=35,                 # input schema UNCHANGED (35-feature context)
    contract_version=3,             # UNCHANGED: same discretized 6-action space
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v18.core.model:PokerEVModelV4",
    contract_class="versions.v18.core.contract:ContractV12",
    weights_dir="versions/v18/weights",
    status="active",
    milestone=False,                # v13 remains the kept milestone/fallback
)

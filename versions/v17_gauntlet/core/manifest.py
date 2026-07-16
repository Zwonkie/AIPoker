"""V17_gauntlet manifest — clones V17 (versions/v17, deployed live), same 6-action contract/arch/
training recipe. ONE INTENDED change: the opponent pool. Pulls the V18 backlog item "widen the
frozen-opponent pool" forward -- instead of scripted heuristics + a single frozen predecessor, the
hero was INTENDED to face THREE genuinely skilled frozen/lagged opponents in three of its five
opponent seats:

- `nit` slot -> frozen V15 (`frozen_v15.pth`), UNFORCED (see below) -- WORKED AS INTENDED.
- `tag` slot -> frozen V16 (`frozen_v16.pth`), UNFORCED -- NEW model-loading wiring, this slot was
  previously hardcoded to always use the pure heuristic bot with no model option at all.
  **CORRECTION (found 2026-07-16 while planning V18's opponent-architecture refactor): this wiring
  had a bug (`opp_model = self.tag_model` immediately followed by a leftover `opp_model = None`)
  that silently nullified it. The already-trained `v17_gauntlet` checkpoint actually trained
  against the TAG heuristic bot in this seat, NOT frozen V16.** See versions/v17_gauntlet/SPECS.md
  "CORRECTION" section. Fixed going forward in versions/v18 (not retroactive to this checkpoint).
- `past` slot -> TRUE lagged self-play mirror of V17_gauntlet's own training history
  (`freeze_past_self: false`, the standard mechanism -- NOT a static frozen file) -- WORKED AS
  INTENDED.
- `fish`/`maniac` stay pure scripted heuristics, for continued diversity against non-NN opponents

Prerequisite fix required: `maniac`/`nit`/`fish` style slots have STYLE-FORCING logic in
`_opponent_decide` (`simulator.py`) that probabilistically overrides the seat's actual decision to
push its realized stats toward a target archetype (e.g. nit: 80% chance to force-fold whenever
VPIP>15%, regardless of what the model wanted). That's appropriate for a scripted heuristic hitting
its target stats, but would badly distort a genuine trained network's judgment -- mostly overriding
V15's actual read rather than letting it play itself. Fixed: forcing is now bypassed whenever a real
model is loaded for that seat (`opponent.get('model') is not None`); scripted heuristic seats are
unaffected, forcing still applies to them exactly as before.

Also reverted `equity_sims` 5000->2000 (see `versions/v18/SPECS.md` "MC equity_sims budget" --
measured 2.46x wall-clock cost for a ~0.3-percentage-point noise reduction that's small next to the
critic's own much more powerful denoising; not worth the cost for this run).

200k hands (up from V17's 100k test budget), 6-max (`live_players: 6`). Curriculum (stack_depth_mix,
phase settings) deliberately UNCHANGED from V17's validated recipe -- the opponent pool is already
the one variable this run is testing; changing the curriculum at the same time would confound it.

See: .agents/skills/OFK/references/versioned-architecture-guardrails.md
     versions/v17_gauntlet/SPECS.md   |   versions/v17/SPECS.md (parent line)   |   versions/v18/SPECS.md (backlog item this pulls forward)
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v17_gauntlet",
    context_dim=35,                 # input schema UNCHANGED (35-feature context)
    contract_version=3,             # UNCHANGED: same discretized 6-action space
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v17_gauntlet.core.model:PokerEVModelV4",
    contract_class="versions.v17_gauntlet.core.contract:ContractV12",
    weights_dir="versions/v17_gauntlet/weights",
    status="active",
    milestone=False,                # v13 remains the kept milestone/fallback
)

"""V22 manifest -- two structural additions on top of V21_auxhead's foundation (contract_version
6, context_dim 37->43), bundled together since both touch the same context vector and need only
ONE retrain:

1. **Deeper stack curriculum** ([STACK-2] in the OFK backlog, deferred from V21 item 9):
   `STACK_CEIL_BB` 50->100 (POT_CEIL_BB 100->200, CALL_CEIL_BB 50->100 alongside it, since 100bb
   pots/calls are now reachable). The clamp-then-scale formula meant the old ceiling only ever used
   HALF the normalized [0,1] range (50/100=0.5) -- raising the ceiling to match the existing scale
   uses the full range at the same per-bb resolution, no rescale needed. `stack_depth_mix` widened
   to `[5-14bb:0.40, 14-30bb:0.30, 30-60bb:0.20, 10-100bb:0.10]` -- the last band OVERLAPS the
   others (uniform-sampled 10-100bb) rather than being a disjoint 60-100bb bucket, so there's no
   density cliff at the old 60bb seam between bands.

2. **Per-opponent/hero entry-sizing** ([OPP-2]/[OPP-3]-adjacent, deferred from V21 item 8, deliberately
   scoped CHEAP -- not the full per-street action-token architecture change those backlog items
   describe): `simulator.py` already tracks a per-seat `committed[]` array (hand-total chips in,
   used for side-pot math) but never surfaced it into the context. Two new APPENDED features (every
   existing index 0-36 stays stable, matching this codebase's append-only contract discipline):
   `opp_committed_this_hand_bb` per opponent slot (ctx[37:42], one per seat) and
   `hero_committed_this_hand_bb` (ctx[42], global). Distinguishes two opponents with identical
   remaining stack/VPIP/AGG color who got there via very different lines this hand (e.g. limped-in
   vs 3-bet) -- something no existing feature can tell apart, since `opp_stack` is a remaining-stack
   snapshot, not a this-hand action signal. Does NOT resolve full [OPP-2]/[OPP-3] (still no
   per-street action-type sequence) -- a single cumulative-committed scalar can't distinguish HOW
   the money went in, only how much. A `pot_type` (limped/single-raised/3-bet+) feature was
   considered as a companion but deliberately deferred to backlog (see known-shortcomings-backlog.md)
   rather than bundled here.

Base: copied from `versions/v21_auxhead` (the live foundation this branches from), inheriting its
validated aux-head configuration (corrected `opp_bluff_prob` label, sqrt-dampened reweighting,
per-head weights bluff=0.05/strength=0.10/equity=0.05) unchanged -- V22 is not another aux-head
experiment, that thread is closed.

See: versions/v21/SPECS.md items 8/9 (original scoping) | versions/v22/SPECS.md (full detail)
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v22",
    context_dim=43,                  # 37 (V21_auxhead) + 5 opp_committed + 1 hero_committed
    contract_version=6,              # bumped: STACK/POT/CALL_CEIL_BB raised + 2 new appended features
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v22.core.model:PokerEVModelV4",
    contract_class="versions.v22.core.contract:ContractV12",
    weights_dir="versions/v22/weights",
    status="training",
    milestone=False,
)

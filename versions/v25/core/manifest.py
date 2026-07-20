"""V25 manifest -- attacks [BET-1] (all-in dominates by construction) from a structurally different
angle than V23/V24/V24_extreme. Same context/contract as V24 (context_dim=44, contract_version=7,
`pot_type` unchanged) -- this version touches ONLY `simulator.py`'s `_mc_target_evs_sized` (hero's
own training-target computation), not the network's input features. Inherits V24's decoupled fold
model + `bot_bluff_perc` "show of strength" mechanism UNCHANGED (calibrated values kept) -- this is
additive on top of that, not a replacement.

**Why**: V23/V24/V24_extreme all tried to fix the all-in-dominates problem by shaping OPPONENT
fold behavior (make raises "respected" as much as or more than shoves). V24_extreme showed that
lever genuinely CAN move the needle, pushed to an extreme, but at the cost of a new
`vpip_adapts_to_style` regression, and the result was confounded across five simultaneous changes
(see versions/v24_extreme/SPECS.md). Independently, a structural flaw was identified (2026-07-18
discussion) in `_mc_target_evs_sized` itself: `ev_if_called = true_equity * (pot + 2*raise_size -
to_call) - raise_size` treats a called (non-all-in) raise as a TERMINAL, single-street outcome --
there is no representation anywhere of the additional money that realistically goes in on FUTURE
streets (implied odds when hero improves or an opponent pays off later, continued fold equity from
further aggression) if the hand keeps going instead of jamming right now. All-in forecloses that
option entirely and its EV is (correctly) computed as terminal; a smaller raise's real advantage
over all-in is exactly the future-streets value this formula has never modeled. Per the user's own
framing: prioritize a fix that produces genuinely EMERGENT/learned behavior (the value of keeping
betting alive should be discovered by simulating what actually happens next, not hand-tuned into
opponent fold thresholds) over more heuristic opponent-response calibration.

**Fix -- one-street-deep MC rollout continuation value**: new `_rollout_continuation_ev` in
`simulator.py`, called from `_mc_target_evs_sized` for every non-all-in raise size on any
non-river street (river has no next street -- the existing single-street formula is already exactly
correct there, left untouched). For each of a few trials: deals ONLY the cards needed to reach the
next decision point (3 for preflop->flop, 1 for flop->turn, 1 for turn->river), recomputes a cheap
MC equity at that point (still correctly integrating any further undealt cards), applies a FIXED
hero continuation policy (bet ~2/3 pot if the new equity clears a threshold, else check -- not the
live NN, to avoid a recursive self-referential target), and -- if hero bets -- asks the opponent's
own REAL (BET-1-price-sensitive) `decide_postflop` whether it folds. The realized value of that one
extra street, averaged across trials and compared against what the existing formula already
assumed, becomes an additive `continuation_ev` correction. Pure Monte-Carlo/counterfactual in style
(matches every other target computation in this codebase already) -- no bootstrapping from the
model's own in-training critic, which was considered and explicitly deferred as a higher-risk,
harder-to-calibrate paradigm shift for this codebase (see OFK known-shortcomings-backlog [BET-1]).

Calibration (isolated, pre-retrain comparison of corrected vs uncorrected EV across representative
scenarios) documented in `self_play/calibrate_multistreet_ev.py` and this version's own SPECS.md.

Base: copied from `versions/v24` (decoupled fold model, `bot_bluff_perc`, entry-sizing, deep-stack
curriculum, `pot_type` all inherited unchanged). Opponent pool unchanged from V22-V24 (maniac/fish
as plain heuristics).

See: versions/v24_extreme/SPECS.md (the diagnostic that prompted this structural pivot) |
.agents/skills/OFK/references/known-shortcomings-backlog.md [BET-1] | versions/v25/SPECS.md (full
detail, calibration data)
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v25",
    context_dim=44,                  # UNCHANGED from V23/V24 -- no context/feature change
    contract_version=7,              # UNCHANGED from V23/V24 -- target-EV computation only, no contract change
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v25.core.model:PokerEVModelV4",
    contract_class="versions.v25.core.contract:ContractV12",
    weights_dir="versions/v25/weights",
    status="training",
    milestone=False,
)

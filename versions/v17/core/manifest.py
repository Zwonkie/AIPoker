"""V17 manifest — clones V16 (versions/v16), same 6-action contract/arch/config, ONE experiment:
routing the ACTOR's regret-matching target through the CRITIC's own learned Q-values instead of a
fresh single-hand simulator Monte Carlo sample, once the critic has had enough training to be
trustworthy (hard cutover at hands_done>=30000, reusing the SAME milestone already used for
bootstrap_alpha's decay endpoint -- not a new tuned constant).

Root cause this targets: the actor's per-hand target (`policy_target_seq` in train.py) was always
computed from `regret_match_policy(p_evs)` where `p_evs` is a ONE-SHOT, noisy simulator EV estimate
for that single hand -- independent of the model's own weights or how many hands it has already
seen. The critic (`q_vals`), by contrast, IS trained via MSE regression across every hand and epoch
-- exactly what regression asymptotically estimates: the smoothed, population-average value. V17
wires the actor to consume that accumulated signal instead of a fresh noisy sample each time.

Required prerequisite discovered while implementing: V16/foldregret both ran with
`disable_target_shaping: true`, which also zeroed `COUNTERFACTUAL_WEIGHT` -- meaning the critic's
own loss only had real gradient for FOLD (always 0) and whichever action was actually taken; every
UNTAKEN action's counterfactual target got weight 0. Routing the actor through a critic starved of
supervision on 4-5 of its 6 heads per sample would trade a noisy-but-complete value source for a
smoothed-but-sparse one. Fixed in train.py: `COUNTERFACTUAL_WEIGHT` is now restored to its original
default (0.5) independent of `disable_target_shaping` (which still only zeros
`TIGHTNESS_PENALTY_BB`, left out of scope for this experiment). This is a deliberate, explained
prerequisite, not a silent reversion -- see SPECS.md.

Round 1 (mean-baseline critic routing) was trained to 55,296 hands and STOPPED: air/draws Fold%
declined steadily post-cutover instead of improving. Diagnosis (comparing checkpoints directly)
found the critic itself well-calibrated -- the MEAN-baseline regret-matching formula dilutes FOLD's
share regardless, because one steeply-negative outlier action (ALLIN) drags the shared mean down
far enough that worse-than-fold actions still clear it. Round 2 (current): fold-relative baseline
applied ONLY to the post-cutover critic-routed target (pre-cutover untouched, still mean-baseline
over raw mc_evs as in V16). This is the same baseline `v16_foldregret` used, but on a DENOISED
critic Q instead of a noisy single-hand sample -- testing whether it was the noise being
fold-anchored that caused foldregret's deep-stack/style regression, not the fold-relative baseline
itself. Full trace: SPECS.md "Round 1 result" / "Round 2".

Same 6-action contract, same context_dim=35, same opponent pool/config as V16 -- single-variable
experiment, trained fresh, exactly like every version in this line.

See: .agents/skills/OFK/references/versioned-architecture-guardrails.md
     versions/v17/SPECS.md   |   versions/v16/SPECS.md (parent line)   |   versions/v16_foldregret/SPECS.md (superseded alternative)
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v17",
    context_dim=35,                 # input schema UNCHANGED (35-feature context)
    contract_version=3,             # UNCHANGED: same discretized 6-action space
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v17.core.model:PokerEVModelV4",
    contract_class="versions.v17.core.contract:ContractV12",
    weights_dir="versions/v17/weights",
    status="active",
    milestone=False,                # v13 remains the kept milestone/fallback
)

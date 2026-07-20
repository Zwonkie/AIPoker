"""V27 manifest -- SAME architecture/contract/target-EV mechanism/opponent pool as V26
(context_dim=44, contract_version=7 -- see versions/v25/SPECS.md for the multi-street
`_rollout_continuation_ev` mechanism, versions/v26/SPECS.md for the TreeOpponent real-data pool).

V27 is a small-fixes/cleanup version, not a new capability -- bundling two tracked
`model_verify` WARNs the way V23 bundled its own small fixes:

1. **[VAL-3] `free_check_low_fold` fix**: `regret_match_policy_torch`'s fold-relative baseline
   (`baseline_mode='fold'`) measures every action's regret against the model's OWN (noisy) Q_fold
   output. Q_fold's own critic training target is a hardcoded constant 0.0 in every state
   (`_mc_target_evs_sized`'s `evs[0] = 0.0`), so the network has every incentive to learn Q_fold as
   a near-constant near-zero function -- and Q_call is ALSO naturally near zero exactly when
   equity is low AND checking is free (call EV = equity*pot - 0, tiny when both factors are small).
   Whenever critic noise nudges Q_fold >= every other action in that corner, the degenerate-tie
   fallback (line ~236 of train.py) hands the actor a literal FOLD=1.0 supervised label -- even
   though checking for free is never wrong. Every OTHER place in this codebase already masks fold
   to zero when call_amount==0 (`_query_model_decide`'s self-play action selection, `core/
   decision.py`'s live inference) -- this was the one remaining place that didn't, so the training
   TARGET itself could still teach the network a "sometimes fold a free option" signal. Fixed by
   threading the same free-check mask into `regret_match_policy_torch` (see train.py).
2. **`pot_type_sensitivity` investigated, NOT changed**: checked `raise_count`'s wiring in
   simulator.py (increments correctly for both hero's own raises and every opponent's, feeds
   `pot_type_val = min(2, raise_count)` correctly) -- no bug found. The WARN's own conclusion
   ("pot_type may be redundant with call_amount/pot_size/committed") appears to be genuinely true:
   a 3-bet pot naturally has much larger pot_size/call_amount than a limped one, so this feature
   carries little INCREMENTAL information the network doesn't already have another way to see.
   Left in the contract unchanged -- removing a already-shipped feature is a contract-version churn
   for a feature that isn't broken, just low-marginal-value, not what a cleanup version is for.
3. **[OPP-7] fix**: `_query_model_decide`'s opponent-seat block was hardcoded to seats 1-5
   regardless of who was querying -- correct for hero's own query, but wrong for every other NN
   opponent (e.g. Lagged-Self), which saw ITSELF as one of its own opponents and never saw the real
   hero. Fixed via a seat-relative remap (`other_seats = [s for s in range(6) if s != actor_seat]`)
   -- verified byte-identical for hero's own path, and verified the real hero now appears (with a
   live-computed VPIP/AGG read) for non-hero queries, with no self-referential entry. See
   versions/v27/SPECS.md item 3 for the full derivation and direct verification.

Base: copied from `versions/v26` (all TreeOpponent/real-data-pool infrastructure, deep-stack
curriculum, entry-sizing, `pot_type`, multi-street EV fix all inherited unchanged). Fresh weights,
no `--resume_path`, per [VAL-5] -- even a training-loop-only change needs its own from-scratch run
per this lineage's own convention.

See: versions/v26/SPECS.md (opponent pool) | versions/v25/SPECS.md (EV-fix mechanism) |
.agents/skills/OFK/references/known-shortcomings-backlog.md ([VAL-3])
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v27",
    context_dim=44,                  # UNCHANGED from V25/V26 -- no context/feature change
    contract_version=7,              # UNCHANGED from V25/V26 -- training-loop-only change
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v27.core.model:PokerEVModelV4",
    contract_class="versions.v27.core.contract:ContractV12",
    weights_dir="versions/v27/weights",
    status="training",
    milestone=False,
)

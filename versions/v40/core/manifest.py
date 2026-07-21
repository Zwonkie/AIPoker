"""V40 manifest -- the [BET-3] package from the 2026-07-20 Fable full-stack review of V29.

Base: cloned from `versions/v29` (fresh weights, not resumed -- per [VAL-5]). SAME contract
(context_dim=54, contract_version=8, [OPP-2] per-seat raise attribution), SAME opponent pool,
deep-stack curriculum, risk-adjusted target and critic-consistency filter as V29 -- see
versions/v29/SPECS.md for all of that; none of it is re-litigated here.

Contract is BYTE-IDENTICAL to V29's on purpose: every change below is a SIMULATION or
TRAINING-TARGET change, no tensor-schema change, so V29 and V40 checkpoints are contract-compatible
and the live bridge needs no new wiring.

Three changes, all from the review's "Explains live behavior already logged in the backlog" tier
(findings #1, #2, #3 of `.agents/skills/OFK/references/fable-review-consolidated.md`):

## Change 1 [BET-3, review finding #1]: the betting round no longer ends on a single check

`simulate_hand`'s betting loop terminated on `all_matched and (last_raiser == -1 or current_actor
== last_raiser)`. Postflop `highest_bet` starts at 0, so `all_matched` is true from the street's
first instant and the round closed as soon as the OPENING seat acted -- or immediately, if that
seat was already folded. The reviewer confirmed this empirically over 1000 instrumented hands:
0 of 849 postflop checks were followed by any other player acting, and the BB never once received
its limped-pot preflop option. Check-behind, check-raise, "checked to me in position", delayed
c-bet and BB-option nodes had literally never appeared in training data -- a strong root-cause
candidate for [BET-3] (V29 collapses to call/fold with 3+ opponents live, won't raise at 90%
equity).

Fix uses the `acted_this_round` array the simulator ALREADY maintained (for the V20_preflopEq
front/after range-aware-equity split, including its reset-on-raise that re-opens action) and simply
never consulted for termination: a round ends when every still-live seat WITH CHIPS has both acted
this round and matched the current bet. The companion early `break` (`to_call == 0.0 and not
first_round and last_raiser == -1`) is now additionally gated on `acted_this_round[current_actor]`,
demoting it from de-facto terminator to safety net.

Also hardened alongside it (needed to make the longer rounds safe, review M5): both raise branches
now do `highest_bet = max(highest_bet, street_committed[actor])` instead of a bare assignment -- a
stack-capped "raise" below `to_call` used to LOWER the bet level, which yields negative `to_call`
and an `all_matched` that can never become true, i.e. a hung hand. Previously unreachable only
because all six stacks start equal.

## Change 2 [BET-3, review finding #2]: CALL is no longer exempt from the risk penalty or the
## multi-street continuation credit

`_mc_target_evs_sized` gave every SIZED action two corrections that CALL never got:
  - the [V25] `_rollout_continuation_ev` implied-odds/fold-equity credit (non-all-in raises only),
    so calling with a draw was scored as single-street terminal while raising at the identical node
    had its future-street value credited -- a structural pro-aggression/anti-call tilt; and
  - the [V28/V29] `risk_aversion_coefficient * sqrt(Var[X])` penalty, so every raise was docked
    variance RELATIVE to a risk-free CALL. That penalty scales with pot size, i.e. it bit hardest
    in exactly the multiway/high-equity spots where V29 refuses to raise live.
Both are now applied to CALL through the same shared helpers (`_outcome_variance` instantiated for
CALL's own 2-point mixture: p_fold=0, base pot = pot+to_call, "raise_size" = to_call). One
deliberate exception: the variance penalty is NOT applied when `to_call == 0` -- a free check risks
no chips and FOLD (the actor's regret baseline) is a flat 0.0 with no penalty of its own, so
penalizing a free check would tilt the target toward folding for free, the corner
`free_check_low_fold` already tracks.

NOT included from finding #2's "Related:" clause (explicit user scope decision, 2026-07-20): the
multiway `ev_if_called` / single-caller-geometry fix, and "raise EV never models being re-raised".

## Change 3 [STACK-3, review finding #3]: ALLIN critic-consistency veto rescoped

`regret_match_policy_torch`'s ALLIN veto compared ALLIN against `values[..., :-1]` (every non-ALLIN
action, FOLD included). Now `values[..., 1:-1]` -- non-fold alternatives only -- so it can never
fire merely because FOLD outranks ALLIN. Documented honestly in that function: under the
`baseline_mode='fold'` both call sites use, this is a PROVABLE NO-OP (when FOLD is the argmax,
ALLIN's regret is already clamped to 0 before the veto runs), so the review's stated feedback loop
does not hold as written. Kept as correct-by-construction for a 'mean' baseline. The genuinely open
half of finding #3 -- margin 0.15 possibly below critic noise, four risk-dampeners stacked with no
joint calibration -- is a CALIBRATION question, deliberately NOT changed blind.

See: `.agents/skills/OFK/references/fable-review-resolution-log.md` (per-finding resolution status)
| versions/v29/SPECS.md (contract, [OPP-2], critic-consistency filter, risk coefficient)
| .agents/skills/OFK/references/known-shortcomings-backlog.md ([BET-3], [STACK-3])
"""
from shared.manifest import VersionManifest

MANIFEST = VersionManifest(
    version_id="v40",
    context_dim=54,                  # unchanged from V29 -- no contract change in this version
    contract_version=8,              # unchanged from V29 -- V29/V40 checkpoints are contract-compatible
    action_space=("fold", "call", "raise_33", "raise_66", "raise_pot", "allin"),
    model_class="versions.v40.core.model:PokerEVModelV4",
    contract_class="versions.v40.core.contract:ContractV12",
    weights_dir="versions/v40/weights",
    status="training",
    milestone=False,
)

# V27 SPECS

Branches from `versions/v26` (fresh weights, not resumed -- per [VAL-5]). SAME architecture,
contract, target-EV mechanism, and opponent pool as V26 (context_dim=44, contract_version=7, the
multi-street `_rollout_continuation_ev` fix, the two TreeOpponent real-data seats -- see
`versions/v25/SPECS.md` and `versions/v26/SPECS.md` for those, not re-litigated here). V27 is a
small-fixes/cleanup version: bundling two of the tracked `model_verify` WARNs every version since
V21_auxhead has carried, the same way V23 bundled its own small fixes.

## What changed

### 1. [VAL-3] `free_check_low_fold` fix (real fix, in `train.py`)

**Root cause found**: `regret_match_policy_torch`'s fold-relative baseline (`baseline_mode='fold'`,
active post-cutover) measures every action's regret against the model's OWN Q_fold output. Q_fold's
own critic training target is a hardcoded constant `0.0` in EVERY state
(`_mc_target_evs_sized`'s `evs = [0.0, ...]`), so the critic has every incentive to learn Q_fold as
a near-constant, near-zero function. Q_call is ALSO naturally near zero exactly when equity is low
AND checking is free (`call EV = equity*pot - 0`, tiny when both factors are small) -- so this is
precisely the corner where ordinary critic noise can nudge Q_fold above Q_call. When that happens,
the "nothing beats folding" degenerate-tie fallback hands the actor a literal **FOLD=1.0 supervised
label** for a state where folding a free option is never correct.

Every OTHER place in this codebase already masks fold to zero when `call_amount==0`
(`_query_model_decide`'s self-play action selection at `simulator.py:636-640`, `core/decision.py`'s
live inference) -- this was the one remaining gap, in the training TARGET itself, upstream of both
of those. Fixed by threading a `free_check_mask` (`b_c[:, :, 9] <= 1e-6`, the scaled `call_amount`
context feature) into `regret_match_policy_torch`, applied at both its call sites (train-loop and
val-loop). Zeros fold and renormalizes over the remaining actions -- with one subtlety caught by a
unit test before shipping: when the ORIGINAL result was already the degenerate all-fold fallback
(nothing left to renormalize, `0/0`), it now defaults to CALL, exactly matching
`simulator.py:640`'s own fallback (`else torch.tensor([0.0, 1.0, 0.0], ...)`) for the identical
situation at inference time.

### 2. `pot_type_sensitivity` investigated, NOT changed

Checked `raise_count`'s wiring in `simulator.py`: it increments correctly for both hero's own
raises (line ~1469) and every opponent's (line ~1528), and `pot_type_val = min(2, raise_count)`
buckets it correctly (0=limped/unraised, 1=single-raised, 2=3-bet+) into `contract.py`'s `ctx[43]`.
No bug found. The WARN's own conclusion ("pot_type may be redundant with call_amount/pot_size/
committed") appears to be genuinely true: a 3-bet pot naturally has much larger pot_size/
call_amount than a limped one, so this feature carries little INCREMENTAL information the network
doesn't already have another way to see. Left in the contract unchanged -- removing an
already-shipped feature is a contract-version churn for a feature that isn't broken, just
low-marginal-value, not what a cleanup version is for.

### 3. [OPP-7] Self-referential/hero-blind NN-opponent queries (real fix, in `simulator.py`)

**Root cause found**: `_query_model_decide` (the shared query path both hero's own decisions AND
every NN opponent's decisions funnel through) built its 5-slot opponent-seat block with a
hardcoded `seat_key = f"seat_{idx+1}"` loop -- i.e. always literal absolute seats 1-5, regardless
of who was actually querying. Correct for hero's own query (`actor_seat==0`, whose real opponents
genuinely are seats 1-5) but wrong for every other NN opponent's query: a non-hero actor like
Lagged-Self at seat 4 saw ITSELF as one of its own opponents (a phantom self-referential entry
using its own VPIP/AGG/committed values), and never saw the real hero (seat 0) at all, since
`opponents_profiles` only has entries for seats 1-5.

**Fix**: `other_seats = [s for s in range(6) if s != actor_seat]`, then index into it by slot
instead of the hardcoded `idx+1`. For `actor_seat==0` this list is EXACTLY `[1,2,3,4,5]` --
byte-identical to the old hardcoding, verified directly (same VPIP/AGG colors, same committed
values, same ordering, for an identical synthetic input). For any other actor, the real hero now
appears (reading hero's own live accumulated VPIP/AGG from `self.seat_histories[0]`, the same
`acts/ops` computation every other seat's profile already uses) and the querying seat itself never
appears. Verified directly: constructed a synthetic `actor_seat=4` query and inspected the actual
`SeatState` objects built -- confirmed no "seat 4" entry exists and "seat 0" (hero) appears with
the correct live-computed VPIP/AGG color and committed amount.

Only affects the NN opponents' own training data quality (what Lagged-Self perceives and therefore
learns from) -- hero's own live decisions are unaffected, and hero's own query path is provably
unchanged.

## Verification (pre-training)

- Unit-tested the `free_check_mask` fix in isolation against a synthetic degenerate-tie corner
  (Q_fold=0.01 slightly beating Q_call=0.005 and everything else, at a free-check timestep):
  correctly redistributes to CALL=1.0 instead of the invalid `[0,0,0,0,0,0]` an earlier version of
  the fix produced (caught and fixed before running any real training). Also verified a legitimate
  paid-decision fold-is-correct timestep is left untouched, and a free-check timestep where call
  already had real proportional regret against fold renormalizes correctly across the surviving
  actions.
- Unit-tested the [OPP-7] seat-remap directly: a synthetic `actor_seat=0` (hero) query reproduces
  the exact old seat_1..seat_5 ordering/values (byte-identical, confirming zero behavior change for
  hero's own path); a synthetic `actor_seat=4` (non-hero) query confirms no "seat 4" self-entry
  exists and "seat 0" (real hero) appears with a live-computed VPIP/AGG color and correct committed
  amount.
- Smoke-tested the full `train.py` entry point end-to-end TWICE (300 hands before the [OPP-7] fix,
  500 hands after, 0 crashes either time) -- this only exercises import/plumbing at this hand
  count, since both stay in the pure-heuristic bootstrap phase, below `ACTOR_CRITIC_CUTOVER_HANDS`
  (30k) where the fold-relative baseline (and the [VAL-3] fix) actually engages; the direct unit
  tests above are what validate each fix's own logic. The 500-hand run does genuinely exercise
  [OPP-7]'s new code path live, since Lagged-Self (an `NNOpponent`) queries the model every hand.
- `target_hands: 100000` (matches V26's own confirmatory run scale), fresh weights, no
  `--resume_path`.

## Results (2026-07-19, `expert_main.pth`, 100k hands complete)

**`model_verify --full`: 18 PASS, 5 WARN, 1 FAIL, 0 SKIP** (`tools/model_verify/results/v27__expert_main.json`).

- **`beats_frozen_predecessor`: PASS, +40.2 BB/100** over 4000 hands vs a field including a frozen
  V26 snapshot (`versions/v27/weights/frozen_v26.pth`) -- a real, direct win over V26.
- **`opponent_style_sweep` genuinely improved**: 0.041 -> 0.165 -- Lagged-Self now seeing the real
  hero (instead of a self-referential phantom entry) plausibly contributed to a richer, more
  differentiated training population, exactly the kind of effect [OPP-7]'s own "worth confirming"
  caveat was watching for.
- **BUT several checks got materially worse, not better, alongside the win**:
  - VPIP roughly DOUBLED (V26 ~16-26% short/deep-tight-loose -> V27 ~40-48%) -- hero now enters
    vastly more hands.
  - `action_diversity` narrowed from 3 actions (`fold`/`call`/`allin`) to 2 (`fold`/`allin`) --
    call disappeared from the test grid's argmax entirely.
  - `stack_full_sweep`'s argmax path flipped from all-`call` (V26) to all-`allin` (V27) across the
    ENTIRE 5-180bb sweep at the fixed marginal test spot.
  - `position_sweep` newly WARNs: table position barely moves the policy anymore (spread 0.378 ->
    0.022) -- a real, new regression, not present in V26.
  - `allin_vs_nextbest_qgap` [BET-1] got WORSE at every stack depth (e.g. 40bb: +0.55 -> +0.61) and
    every archetype -- the shove-preference problem this whole V28 diagnostic effort is targeting
    is measurably worse in V27, not better.
  - `beats_offformula_stress` deep-stack dropped sharply: +75.2 -> +5.6 BB/100.
  - `bb100_vs_standard_fields` loose_deep dropped: +53.6 -> +17.3 BB/100.

**Verdict**: V27 wins head-to-head against a frozen V26, but the mechanism looks like "wins via
much higher aggression/looser entry," not "wins via a more refined strategy" -- the same
win-rate-up-but-nuance-down pattern this lineage has seen before (V24_extreme). Both fixes
([VAL-3], [OPP-7]) were verified correct in isolation via direct unit tests before this run (not
guesses), and per explicit user decision, V27 -- not V26 -- is the base for V28, since the fixes
themselves were for well-established, diagnosed reasons; whether OPP-7's genuine change to
Lagged-Self's training population is the specific cause of the broader shift (vs. VAL-3, vs. plain
run-to-run variance) has not been isolated and is flagged as an open question, not resolved here.

See `versions/v26/SPECS.md` (opponent pool) | `versions/v25/SPECS.md` (EV-fix mechanism) |
`.agents/skills/OFK/references/known-shortcomings-backlog.md` ([VAL-3], [OPP-7], [BET-1])

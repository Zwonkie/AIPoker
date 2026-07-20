# V23 SPECS

Branches from `versions/v22` (`expert_main.pth`, the current live-candidate foundation) with two
additions, discussed and scoped in the session that produced V22's own zero-FAIL result:

1. **[BET-1] opponent price-sensitivity fix** -- the single biggest, most-cited open issue in the
   whole model lineage.
2. **`pot_type` feature** -- deferred from V22's own entry-sizing work.

## 1. [BET-1]: opponent value-branch price-sensitivity

**Root cause** (already known going into this version, see the OFK backlog's `[BET-1]` entry):
`versions/v23/self_play/opponent_bots.py`'s `FuzzyPlayerArchetype.decide_postflop` had a "value
raise" branch --

```python
if equity >= need_for_value:
    # Value raise (strong hands raise regardless of price)
    ...
```

-- where `need_for_value` was a FLAT per-street constant, never consulting `pot_odds`. A min-bet
and an all-in shove got IDENTICAL continuation once a bot's equity cleared it, so hero's critic
saw no downside to sizing up -- bigger was always weakly better in expectation, which is the
direct mechanism behind `action_diversity`'s persistent "no middle gear" pattern (RAISE_33/66/POT
essentially never winning argmax anywhere in the equity x stack grid, in every version checked).
`decide_preflop` had the same bug in a slightly different shape: the value-threshold check ran
BEFORE `pot_odds`/`facing_bet` was even computed, so it was unconditional regardless of price.

**Fix**: extend the SAME price-sensitivity mechanism the existing marginal `continue_bar` already
had (`continue_bar = pot_odds + style_shift`, the V14 P1b fix) up to the value bar too, instead of
leaving it flat:

```python
value_bar = min(0.98, need_for_value + VALUE_PRICE_SENSITIVITY * pot_odds)
if equity >= value_bar:
    ...raise/slowplay as before...
elif equity >= continue_bar:
    ...as before...
```

Applied to both `decide_postflop` (its facing-a-bet branch) and a restructured `decide_preflop`
(moved the `facing_bet` check first; the no-bet-yet RFI/open branch is untouched -- there's no
price to be sensitive to when opening the pot).

### Calibration

`VALUE_PRICE_SENSITIVITY` is a genuinely new tunable constant, and unlike the V21_auxhead aux-head
weights, this isn't testable via a cheap warm-start continuation -- it changes what the OPPONENT
POOL does, so any candidate needs a real retrain to see the effect on hero's learned behavior. To
avoid guessing blind (and to avoid the same overcorrection risk the aux-head bluff-reweighting work
hit earlier), a standalone probe (`versions/v23/self_play/calibrate_bet1.py`, kept as a reusable
tool for future recalibration) directly measured `decide_postflop`'s P(fold) at value-tier equities
(0.60-0.98) across a pot_odds grid (0.20/0.33/0.50/0.67/0.80 -- roughly 1/4-pot up to a
shove-into-a-big-pot), for all 4 archetypes (TAG, LAG, NIT, CALLING_STATION), 3000 trials per cell
(re-fuzzing traits via `start_new_hand()` every trial to reflect the real fuzzed distribution).

**Baseline (unmodified) confirmed the bug**: at equity clearly above an archetype's own value
threshold, P(fold) stayed ~flat regardless of bet size -- e.g. TAG at equity=0.70: 0.003 (po=0.67)
vs 0.004 (po=0.80).

**Key finding -- a single global constant compounds UNEVENLY across archetypes**: NIT (tightest,
already has both the highest `need_for_value`=0.75 AND the steepest `style_shift`) responds far
more steeply than TAG/LAG at the identical constant. At `VALUE_PRICE_SENSITIVITY=0.15`, NIT's
P(fold) at equity=0.80 facing a shove jumped from a 9.9% baseline to 95%+ -- an unrealistic
"always folds a big favorite" collapse -- while TAG only reached 58% fold at that same value.
Every value from 0.03 up produced a REAL, meaningful gradient for every archetype; the tuning
question was purely "how far before it overcorrects."

Values tested: 0.03, 0.05, 0.08, 0.10, 0.15, 0.25, 0.35, 0.50. NIT's P(fold) at equity=0.80,
po=0.80 (the most sensitive cell found) by value: 0.03->24%, 0.05->40%, 0.08->62%, 0.10->77%,
0.15->95%+. Diminishing returns set in above ~0.25 for the other archetypes (their responses
largely saturate against the `min(0.98, ...)` clamp).

**Chosen: `VALUE_PRICE_SENSITIVITY = 0.05`**. At this value:
- TAG: real gradient at equity=0.70 (1.8%->5.4% fold across the tested bet sizes), stays at 0% for
  equity>=0.80 (never folds a clearly strong hand).
- LAG: 24-26% fold at equity=0.60 against a big bet, near-zero above that.
- NIT: strongest response (up to 40% fold at equity=0.80 vs. a shove) -- fitting, since NIT is the
  tightest archetype; thematically correct, not judged an overcorrection at this level.
- CALLING_STATION: modest increase (18.7%->30.3% at equity=0.70 vs. a shove).
- Every archetype NEVER folds a near-nuts hand (equity>=0.90) regardless of bet size at this
  value -- the `min(0.98, ...)` clamp keeps monster hands safe; they can only move from the
  auto-raise-for-value branch into the (still-continuing, price-sensitive) marginal-continue
  branch, never all the way to fold.

Full per-archetype/per-equity/per-bet-size tables from the calibration run are in this version's
session transcript; not reproduced in full here to keep this doc scannable -- the summary above
captures every number that drove the decision.

## 2. `pot_type` feature

Deferred from V22's own entry-sizing work (see `versions/v22/SPECS.md` item 2's note, and the OFK
backlog). A single cumulative `committed_this_hand` scalar (V22's own addition) can't distinguish
HOW money went in -- one big raise reads identically to limp-then-call-a-raise if the total
happens to match. `pot_type` adds the missing structural signal: whole-hand raise count so far
(any street, any actor), bucketed 0=limped/unraised, 1=single-raised, 2=3-bet-or-more, one new
APPENDED global context feature (`ctx[43]`, normalized `/2.0`) -- every existing index 0-42 stays
stable.

**Wiring**:
- `versions/v23/self_play/simulator.py`: new `raise_count` counter, reset once per hand alongside
  `committed`, incremented at BOTH raise sites (hero's own raise and any opponent's raise, any
  street) -- mirrors how `committed[]` already existed for the entry-sizing feature.
- `core/board_state.py`: new `BoardState.pot_type` field (int, default 0) -- additive/optional,
  inert for every earlier version's contract, same pattern as `hand_strength`/`hero_committed`.
- `_query_model_decide`: reads `raise_count` from `table_state_dict` (a hand-level property, not
  actor-relative like `committed`) and bucket-clamps it (`min(2, raise_count)`) into
  `BoardState.pot_type` for every query.
- `versions/v23/core/contract.py`: appends `float(pot_type) / 2.0` at `ctx[43]`.
- `versions/v23/self_play/train.py`'s `vectorize_hand_samples`: mirrors `contract.py` exactly
  (same bucketing/normalization), sourced from a new `raise_count_before` field on
  `decision_points`.
- `tools/model_verify/scenarios.py`/`checks.py`: `build_ctx` supports `contract_version>=7`
  (44-dim, `pot_type` param); new `pot_type_sensitivity` FAST check (isolate-one-slot ablation,
  mirrors `committed_sensitivity`'s pattern) added.

## Verification (pre-training)

- `PokerEVModelV4()` forward pass with a 44-dim context tensor -- correct output shapes.
- `shared.registry.get_manifest('v23')` auto-discovers the new manifest (context_dim=44,
  contract_version=7).
- `tools/model_verify/run.py --version v23` against throwaway random-init weights -- runs to
  completion with no crashes (same plumbing-only check as V22's own pre-training smoke test).
- 40 real simulated hands via `SixMaxSimulator` + `vectorize_hand_samples` end to end -- confirmed
  `pot_type` shows real, correct bucket values (0.0/0.5/1.0, i.e. 0/1/2 normalized /2.0) across
  vectorized hands, context tensor 44-wide at every timestep, and the BET-1 opponent restructure
  didn't crash or otherwise break real simulated play.

## Training setup

- `target_hands: 150000` (raised from V22's 100000) -- the BET-1 fix reshapes the ENTIRE opponent
  pool's behavior for the whole run (not just a new input feature), so more hands gives the hero's
  critic more exposure to a population with genuine size-scaled fold equity before trusting the
  result. Chosen from the session's own offered range (100k/150k/200k) as a middle ground between
  thoroughness and total runtime.
- `checkpoint_dump_interval: 20000` -- same cadence as V21/V22.
- Aux-head config, actor-critic cutover, bootstrap/exploration schedule, policy target source,
  range-aware equity, realization discount, deep-stack curriculum -- all inherited unchanged from
  V22.
- Opponent pool: unchanged from V22 (`past` lagged-self 0.25 / `maniac` heuristic 0.20 / `fish`
  heuristic 0.15 / `tag` heuristic 0.25 / `nit` heuristic 0.15).
- Fresh from-scratch run (no `--resume_path`) -- matches [VAL-5]'s own finding (warm-started
  continuations degrade action diversity independent of whatever else is being changed).

## Results (2026-07-18, `expert_main.pth`, 150k hands)

Training completed cleanly, no NaN/crashes. Final dashboard: Hero +48.0 BB/100, all six actions
represented in cumulative `ACTION USAGE` (Fold 45.9% / Call 18.8% / r33 8.2% / r66 8.0% / rPot
10.3% / All-In 8.8%) -- a reasonably balanced cumulative spread. **This cumulative-average
appearance turned out to be misleading** -- see below.

**`inspect_aux_heads.py`: all three aux-head correlations hit NEW HIGHS** across the whole
V21_auxhead/V22/V23 lineage, despite V23 not touching the aux-head config at all (inherited
unchanged): equity r=0.963 (prior best 0.922), strength r=0.271 (prior best 0.171), bluff r=0.143
(prior best 0.130). Plausible explanation: the BET-1 fix produces a more informative, less
degenerate training distribution (opponents no longer collapse into predictable "always continue"
behavior at the top of their range), giving the aux heads richer signal. A genuine, unplanned side
benefit.

**`model_verify --full`: 16 PASS / 4 WARN / 1 FAIL / 1 SKIP.**

- **`committed_sensitivity` improved further** (0.077 in V22 -> 0.109 here) -- the entry-sizing
  feature keeps getting more load-bearing with more training exposure.
- **`pot_type_sensitivity`: WARN** (TV=0.004, essentially flat). The new feature did NOT show
  meaningful uptake in this run -- may need more training exposure, or the network may already be
  deriving what it needs from `call_amount`/`committed`. Not yet confirmed load-bearing.
- **[BET-1]: NOT resolved, and `action_diversity`/`deep_stack_ood_guard` actively REGRESSED**
  relative to V22:
  - `action_diversity`: `{'fold':9, 'allin':12}` (2 actions) vs V22's `{'fold':9, 'call':3,
    'allin':9}` (3 actions) -- WORSE, not better.
  - `stack_full_sweep`'s argmax path: `allin` at every one of the 9 stack points (5-180bb) vs
    V22's `call`-dominant path -- a clear regression.
  - `deep_stack_ood_guard`: **FAILED** (`eq=0.55, stack=40bb -> ALLIN@0.40`) -- V22 had PASSED this
    gate for the first time ever in the whole lineage; V23 lost that.
- Direct Q-value inspection at the failing cell (`eq=0.55, stack=40bb, num_active_opp=1`) makes
  the regression concrete: V22's Q-values were `call=0.52, raise_33=0.63, raise_66=0.64 (the actual
  max), raise_pot=0.56, allin=0.49` -- a healthy, non-degenerate gradient where ALLIN wasn't even
  the top action. V23's Q-values at the IDENTICAL cell: `call=0.76, raise_33=1.03, raise_66=1.10,
  raise_pot=1.24, allin=2.99` -- ALLIN now more than double the next-best action, a STRONGER shove
  preference than V22 had, not a weaker one.
- Everything else stayed healthy: `vpip_adapts_to_style` PASS (deltas 14.7-14.8pts, the best of
  any version so far), `beats_offformula_stress` PASS, `bb100_vs_standard_fields` PASS across all
  4 fields. `opponent_style_sweep`/`position_sweep` WARNs are the same pre-existing, untouched
  issues ([OPP-5] and its pre-existing counterpart).

### Root cause of the regression (found via direct code inspection, not speculation)

`simulator.py`'s `_mc_target_evs_sized` -- the function that computes HERO's own per-size
counterfactual EV target used for training -- calls the *exact same* `bot.decide_preflop`/
`decide_postflop` functions the BET-1 fix patched, to sample how often each opponent folds to each
of hero's HYPOTHETICAL raise sizes:

```python
for _ in range(10):
    d = (bot.decide_preflop(oeq, size_pot_odds) if street_idx == 0
         else bot.decide_postflop(oeq, size_pot_odds, new_pot, opp['stack'], street_idx))
    if d == 'fold':
        f += 1
p_all_fold *= f / 10.0
...
evs.append(p_all_fold * pot + (1.0 - p_all_fold) * ev_if_called)
```

Making the bots fold MORE to oversized bets doesn't only describe more realistic LIVE opponent
behavior -- it ALSO directly and mechanically INFLATES hero's own computed EV target for
shoving, since `p_all_fold * pot` is a certain pot-win credited straight into the ALLIN target's
counterfactual EV. The same fix that was meant to discourage hero from overbetting (by making
opponents demand more equity to continue against a bigger bet) simultaneously rewards hero's
shoves with MORE fold-equity credit in its own training signal, because the identical
price-sensitive decision function feeds BOTH roles (live opponent play AND hero's own
counterfactual-EV computation) at once. These two effects point in opposite directions for the
shove-preference goal -- and per the observed Q-value shift, the fold-equity-credit effect won out
over the intended overbetting-discouragement effect, RE-inforcing the shove preference instead of
reducing it.

**This is the key finding of this experiment** -- more valuable than a simple "the fix didn't
work": it identifies precisely why a fix at this specific lever backfires, and redirects what a
future [BET-1] attempt needs to do differently (see Suggestion in the OFK backlog entry).

### Verdict

**V23 is NOT recommended for live deployment as-is.** V22 remains the better-validated candidate
(16 PASS/4 WARN/0 FAIL/1 SKIP) for live use. V23 is a genuine, informative experiment with a real
negative result for the specific BET-1 lever tested (rules out fixing opponent fold behavior via
the shared bot-decision function, given how `_mc_target_evs_sized` reuses it), plus two confirmed
positive side effects (aux-head correlations at new highs, `committed_sensitivity` improved
further) and one still-open feature (`pot_type`, not yet shown load-bearing). Full model_verify
report: see `tools/model_verify/results/v23__expert_main.json` (also rendered to `.html`).

**Recommended next steps for a future BET-1 attempt** (not undertaken here -- a further retrain
requires explicit direction): decouple `_mc_target_evs_sized`'s fold-sampling from the LIVE
opponent-decision function (so a future fix to bot behavior doesn't automatically alter hero's own
training target the same way), or attack the shove-preference from a different angle entirely
(e.g. an explicit overbet EV discount in the target computation itself, independent of simulated
opponent folding).

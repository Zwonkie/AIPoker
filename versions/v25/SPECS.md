# V25 SPECS

Branches from `versions/v24` (fresh weights, not resumed -- per [VAL-5]). Same context/contract as
V24 (context_dim=44, contract_version=7, `pot_type` unchanged) -- V25 touches ONLY `simulator.py`'s
`_mc_target_evs_sized` (hero's own training-target computation), adding a new correction on top of
V24's already-decoupled fold model + `bot_bluff_perc` "show of strength" mechanism (both inherited
unchanged). No network input features change.

## Motivation: a structural pivot, not another calibration pass

V23, V24, and V24_extreme all attacked [BET-1] (all-in dominates the counterfactual EV target by
construction) from the same angle: shape OPPONENT fold behavior so raises earn fold-equity
competitive with shoves. V24_extreme (see `versions/v24_extreme/SPECS.md`) showed that lever
genuinely CAN move the needle -- pushed to an extreme, non-production setting, `raise_pot` won
argmax cells for the first time in the whole V20-V24 lineage -- but at the cost of a new
`vpip_adapts_to_style` regression, and the result was confounded across five simultaneous changes.

Independently (2026-07-18 discussion), a structural flaw was identified directly in
`_mc_target_evs_sized` itself:

```python
ev_if_called = true_equity * (pot + 2.0 * raise_size - to_call) - raise_size
```

This treats a CALLED, non-all-in raise as a terminal, right-now showdown for the pot as it stands.
There is no representation anywhere of the extra money that realistically goes in on FUTURE
streets -- implied odds when hero improves or an opponent pays off later, continued fold equity
from further aggression -- if the hand keeps going instead of jamming. All-in forecloses all of
that and its EV is (correctly) computed as terminal; a smaller raise's real advantage over
shoving is EXACTLY the future-streets value this formula has never modeled. Per the user's own
framing (2026-07-18): prioritize a fix that produces genuinely EMERGENT/learned behavior -- the
value of keeping betting alive should be discovered by simulating what actually happens next, not
hand-tuned into opponent fold thresholds -- over more heuristic opponent-response calibration.

Two implementations were considered (put to the user as an explicit fork, since they carry very
different risk/engineering profiles):

1. **TD bootstrap via the model's own in-training critic** -- the textbook "true" fix for
   terminal-value myopia (semi-gradient TD), maximally "emergent" in principle, but a genuine
   paradigm shift for this codebase (every target here is pure MC/counterfactual today, never
   self-referential), with real instability risk (this codebase already needed a variance clip,
   `target_clip_bb`, to keep the critic from diverging) and much harder to sanity-check in
   isolation before a costly retrain.
2. **One-street-deep MC rollout with a fixed continuation policy** -- deal forward, simulate one
   more (simplified) round of betting using the EXISTING bot decision logic, average over trials.
   Same MC/counterfactual paradigm already used everywhere else in this codebase; lower risk;
   directly calibratable in isolation like every other fix this session.

**User chose option 2.** This SPECS.md covers its design and calibration.

## The fix: `_rollout_continuation_ev`

New method on `SixMaxSimulator` (`simulator.py`), called from `_mc_target_evs_sized` for every
non-all-in raise size on a non-river street (river has no next street -- the existing formula is
already exactly correct there; all-in is never routed through this at all, gated at the call site).

Per trial (averaged over `CONTINUATION_ROLLOUT_TRIALS=4`):
1. Deal ONLY the cards needed to reach the next decision point (3 for preflop->flop, 1 for
   flop->turn, 1 for turn->river) from a deck excluding every already-known card (hero's, the real
   board's, every active opponent's oracle hand -- same "known cards" convention the rest of the
   simulator's training-time target computation already uses).
2. Recompute a cheap MC equity (`CONTINUATION_EQUITY_SIMS=150`, mirroring `_hand_strength`'s own
   existing 200-sim cheap-equity budget) at that new, possibly still-incomplete board -- still
   correctly integrates any further undealt card, exactly like `_calculate_equity` does elsewhere.
3. Apply hero's FIXED continuation policy: bet `HERO_CBET_POT_FRACTION=0.66` of the pot if the new
   equity clears `HERO_CBET_EQUITY_THRESHOLD=0.55`, else check. Deliberately NOT the live NN --
   this keeps the target computation from bootstrapping off the very model it's training (the
   paradigm-shift risk of option 1 above, avoided).
4. If hero bets, ask each active opponent's own REAL `decide_postflop` (BET-1-price-sensitive,
   unlike the decoupled `_ev_target_fold_decision` V24 uses for the raw fold-vs-continue sample --
   this is modeling genuine continued play, not the fold-equity-credit-for-this-raise-size
   computation V24 had to decouple from) whether it folds.
5. The realized value of that one extra street, minus what the existing formula already assumed
   (`true_equity * base_pot`), is the trial's delta. Averaged across trials, this becomes an
   ADDITIVE correction to `ev_if_called` -- it does not replace the base term.

**Known approximations** (acceptable for a first pass, documented in the method's own docstring):
opponent equity at the next street is approximated as `1 - new_equity` (ignores ties, avoids a
second MC call per trial); if ANY active opponent doesn't fold, the pot is treated as contested by
the whole remaining field at the N-way equity (doesn't model some folding while others call --
same multiway simplification `_mc_target_evs_sized`'s own `true_equity` already makes); a checking
hero being donk-bet into by an opponent is not modeled.

## Calibration (before committing to a retrain)

`self_play/calibrate_multistreet_ev.py` -- direct calls into `_rollout_continuation_ev`, no
simulator/training loop, five checks:

**Methodology note**: the first version of this script hardcoded an eyeballed `true_equity` per
scenario instead of computing it from the actual cards. That produced spurious large deltas
(including consistently NEGATIVE ones on the "should be positive" flush-draw case) that were
purely an artifact of the guessed baseline not matching the real equity of the dealt cards --
caught by inspection before drawing any conclusion, not by a retrain. Fixed by computing
`true_equity` the same way the real caller does (oracle MC equity vs the scenario's own known
opponent cards) before comparing. Re-run after the fix:

| check | result |
|---|---|
| River / all-in control | delta = 0.0 exactly, as required (no next street either way) |
| Shallow stack (400bb -> 0bb remaining) | magnitude shrinks from ~5.7 toward <1.2 as stack empties -- noisy (large spread relative to a small mean at only 160 samples/cell) but directionally consistent with "smaller raise and all-in converge once there's no room for another street" |
| Deep-stack flop flush draw (core hypothesis test) | **positive for all 4 archetypes** (+3.3 to +15.8, ~2-10% of base_pot) -- a real, correctly-signed implied-odds signal |
| Deep-stack turn, hero already well ahead | **positive for all 4 archetypes** (+2.5 to +55.7, ~1-19% of base_pot) -- LARGER on average than the flush-draw case, which makes sense on reflection: an equity favorite clears the c-bet threshold on nearly every trial and gets paid nearly every time, while a draw only benefits in the minority of trials that actually hit -- both effects are real, but the "keep betting while ahead" case fires far more reliably |
| Preflop -> flop (3-card deal) | no crash, sane magnitude (+10.4, ~17% of a small pot) |

All five read as expected: the mechanism adds real, correctly-signed value exactly where the
hypothesis predicts it should (draws, favorites continuing to bet), is inert exactly where it
should be (river, all-in), and its magnitude is a modest correction relative to the base pot, not
a multiple of it -- it nudges the per-size EV ranking, it doesn't dominate it.

## Cost

Smoke-tested 300 real hands via `SixMaxSimulator.simulate_hand` end to end (heuristic-only
opponents, no hero model needed to exercise the target-EV path) against an identical v24 run as a
baseline:

| version | hands/sec | hero-decisions/sec |
|---|---|---|
| v24 (no rollout) | 18.81 | 28.41 |
| v25 (with rollout) | 7.90 | 11.91 |

**~2.4x slower.** Expected: up to 3 non-all-in sizes x 4 trials x a 150-sim equity call = up to
1800 extra MC simulations per hero decision, roughly comparable in order of magnitude to the
existing oracle-equity call (2000 sims) this simulator already pays once per decision. No
crashes/exceptions across 300 hands in either version.

At v24's own observed training rate, a 150k-hand run took a little under 2h; at v25's ~2.4x
slower rate, the same hand count would cost roughly 4.5-5h. `CONTINUATION_ROLLOUT_TRIALS`/
`CONTINUATION_EQUITY_SIMS` are the two knobs to cut this if needed (both module-level constants in
`simulator.py`), at the cost of noisier per-decision corrections -- not yet tuned down, pending a
decision on run size/budget.

## Results (2026-07-18, `expert_main.pth`, 50k-hand FAST DIAGNOSTIC)

User chose the fast-diagnostic path (50k hands, ~1.7h, same pattern as V24_extreme but at REALISTIC
settings -- no extreme parameter push, standard curriculum/bootstrap unchanged) before committing to
a full run. Training completed cleanly (Val Loss 0.90, Train Loss 4.80). Cumulative `ACTION USAGE`
looked dramatically different from every prior version in this lineage: `Fold 47.2% | Call 18.8% |
r33 7.3% | r66 7.6% | rPot 12.3% | All-In 6.9%` -- all-in is now the SMALLEST of the four
raise/all-in buckets, not the largest.

### Direct Q-value comparison (same cells used throughout this investigation, eq=0.55)

| stack | V23 | V24 | V24_extreme | **V25** |
|---|---|---|---|---|
| 15bb | 2.45x | 2.75x | 1.75x | **1.36x** |
| 25bb | 2.47x | 3.14x | 1.76x | **1.36x** |
| 35bb | 2.41x | 3.18x | 1.68x | **1.35x** |
| 40bb | -- | 3.18x | 1.60x | **1.35x** |

Tightest allin-vs-next-best gap ever measured in this whole investigation -- and Q-values are now
cleanly MONOTONIC by size (fold < call < r33 < r66 < rPot < allin), a smooth, sensible value curve
rather than an erratic one. All-in is still technically the argmax at these specific synthetic
cells, but by a much smaller margin than any prior version, including V24_extreme's own
extreme-parameter diagnostic.

### `model_verify --full`: 18 PASS / 2 WARN / 1 FAIL / 1 SKIP

- **`vpip_adapts_to_style`: PASS** (short delta +8.1pts, deep +6.0pts, both clear the >=5pt bar).
  This is the critical result: V24_extreme got a comparably tight Q-gap only by ALSO breaking this
  check (opponents folded to raises so readily that hero stopped adapting entry range to table
  tightness). V25 gets an equal-or-better Q-gap WITHOUT that regression -- strong evidence the
  structural (target-EV) fix is doing real work here, not just riding the same "opponents fold more"
  lever V24_extreme leaned on.
- **`deep_stack_ood_guard`: still FAILS**, but at the lowest all-in argmax confidence yet measured:
  `eq=0.55 stack=15bb -> ALL-IN @ 0.31` (V24: 0.46, V24_extreme: 0.35). Same persistent failure this
  check has shown since V19 -- not resolved, but trending the right direction.
- **`action_diversity`: `{'fold': 9, 'allin': 12}`** -- no raise bucket wins this check's own strict
  grid (unlike V24_extreme's 2-cell raise_pot win), despite the tighter underlying Q-gap; the
  check's synthetic cells apparently don't line up with where the gap has narrowed most.
  **`stack_full_sweep`** DOES show one raise_pot win in the argmax path (`[...,'allin','raise_pot',
  'allin']`) -- modest, but non-zero, real movement.
- **Win-rate checks all PASS, and strong**: `bb100_vs_standard_fields` +29.6 to +80.6 BB/100 across
  all 4 fields (tight_deep especially strong at +80.6); `beats_offformula_stress` +41.6/+56.8 BB/100.
- Two WARNs, both pre-existing/longstanding, not new: `free_check_low_fold` (present in every
  version, covered by decision.py's live mask) and `pot_type_sensitivity` (0.004, negligible --
  same persistent WARN since V23).
- `beats_frozen_predecessor` SKIP (no frozen checkpoint saved for this 50k diagnostic).

### Verdict on the diagnostic

**Best result of the entire BET-1 investigation, at REALISTIC settings, on only a fast 50k-hand
pass.** Unlike V24_extreme (which needed 5 simultaneous extreme changes and still broke
`vpip_adapts_to_style`), V25 achieves an equal-or-tighter Q-gap with the standard curriculum, the
standard bootstrap, and no parameter pushed to an extreme -- and passes the check V24_extreme
broke. Confirms the structural target-EV fix (representing future-street value) is a genuinely
different, and so far more effective, lever than the opponent-response tuning tried in V23/V24/
V24_extreme. `deep_stack_ood_guard` remains the one open failure across the whole lineage.

Per explicit authorization to run up to 100k hands: launched a longer, from-scratch confirmatory
run (100k hands, fresh weights, no `--resume_path`, per [VAL-5]) to confirm this holds up with more
training exposure before considering any deployment recommendation.

## Results (2026-07-18, `expert_main.pth`, 100k-hand confirmatory run)

`model_verify --full`: **17 PASS / 5 WARN / 1 FAIL / 0 SKIP** (vs the 50k diagnostic's 18/2/1/1) --
a genuinely mixed result relative to the 50k pass, not a clean confirmation.

**Holds up / improved:**
- `vpip_adapts_to_style` still PASSES, and its deltas are LARGER than at 50k (short +12.0pts, deep
  +9.4pts, vs 50k's +8.1/+6.0) -- the core result this whole pivot cared about (no repeat of
  V24_extreme's regression) not only held, it strengthened with more training.
- `beats_frozen_predecessor` ran for the first time (a frozen 50k-diagnostic snapshot,
  `frozen_v25_50k_diag.pth`, existed to compare against) and PASSED at +74.0 BB/100 over 4000
  hands -- a real, validated win over its own predecessor snapshot.
- Win-rate checks still broadly strong: `bb100_vs_standard_fields` +45.9/+113.2/+44.2 BB/100 across
  3 of 4 fields; `beats_offformula_stress` PASS (+26.0/+19.1 BB/100).

**Regressed / new since 50k:**
- **Direct Q-value comparison (same eq=0.55, stack 15-40bb cells): gap WIDENED to 1.73-1.78x**,
  up from the 50k diagnostic's 1.35-1.36x -- moving the WRONG direction with more training, the
  same pattern V24's own 150k run showed relative to its own earlier signal. `deep_stack_ood_guard`
  still FAILs, now at `eq=0.55 stack=40bb -> ALLIN @ 0.34` (comparable to 50k's 0.31, not
  meaningfully better or worse).
- **Two NEW WARNs**: `committed_sensitivity` dropped from 50k's PASS (0.082/0.066 -- see prior
  section) to WARN (0.025); `position_sweep` newly WARN (spread 0.001, essentially flat). Neither
  was flagged at 50k. `pot_type_sensitivity` remains WARN (0.007, consistent with every version
  since V23). `allin_exploits_opponent_foldiness` ([OPP-8], new check) WARN as expected/already
  understood (0.007 spread) -- not a new finding, this check didn't exist at 50k.
- `bb100_vs_standard_fields`'s 4th field (tight_deep) went slightly negative (-2.3 BB/100) --
  the one field that didn't hold up, though still within normal variance for a single field.

### Verdict

**Mixed, not a clean confirmation.** The specific hypothesis this version was built to test --
does the multi-street rollout fix hold `vpip_adapts_to_style` where V24_extreme couldn't -- is
CONFIRMED, and more strongly at 100k than at 50k. But the underlying Q-gap (the direct measure of
how close all-in and the next-best size are) widened with more training, echoing V24's own
negative-result pattern, and a couple of previously-clean secondary checks (`committed_sensitivity`,
`position_sweep`) drifted into WARN. This reads as: the core structural improvement is real, but
either (a) more training exposure erodes it in ways V22-V24's history has shown before, or (b) 50k
hands was too little exposure for the Q-gap number to have stabilized in the first place and 100k
is the more trustworthy read. Not enough evidence yet to say which. **Not recommending live
deployment on this data alone** -- `deep_stack_ood_guard` remains open across the whole lineage
regardless, and the Q-gap trend needs to be understood (not just the single 100k number) before
treating this as better than V24_extreme's own already-encouraging, if confounded, result.

## New infrastructure (2026-07-18, separate from the EV-fix investigation above): real-data opponents

Per user direction to explore whether opponents derived from REAL human hand-history data (not
hand-coded heuristics, not anything trained inside this simulator's own self-play loop) can fill
personality/depth gaps the current pool structurally can't -- see the extended discussion this
session and `.agents/skills/OFK/references/known-shortcomings-backlog.md` (new item, opponent
diversity). Full pipeline built and validated end to end:

1. **Data**: downloaded the complete Pluribus (10,000 hands) + WSOP (83 hands) full-information
   hand-history corpus (`uoftcprg/phh-dataset` on GitHub) into `data/full_info_hands/` -- every
   hole card is revealed for every decision (including folds), unlike any hand-history source
   previously discussed this session, because these are purpose-released research datasets, not
   real-money exports (which never reveal folded cards to any client, a security property, not a
   data gap).
2. **Extraction**: `pokerkit` (already available in the venv) replays each hand; one row per
   decision point (real equity via `core/evaluator.py`'s existing MC equity calc against the
   OTHER active players' real cards, pot/price/street/stack context, the actual action taken).
   **Bug found and fixed before trusting any output**: pokerkit yields the SAME mutated state
   object on every iteration of a hand replay (verified: 22 yielded states, 1 unique object id) --
   naively storing `prev = state` for "the state before this action" actually aliases to whatever
   the object becomes on the NEXT iteration, silently corrupting every pot/price/stack feature.
   Fixed by snapshotting plain values at each step. 74,467 decision rows from all 10,083 hands,
   0 errors.
3. **Clustering**: identity-agnostic (Pluribus's anonymized code-names are a rotating pool of ~13
   professional players across sessions, not one name = one consistent person) -- clustered
   observed BEHAVIOR (VPIP, PFR, aggression, plus equity-conditioned features only this full-info
   data can give: median equity at fold vs. continue, ground-truth bluff rate, bet-size-vs-equity
   correlation) via a Gaussian Mixture Model, k chosen by BIC (came out to 4). Only 14 of ~19 named
   players had enough decisions (>=30) to trust. **Honest finding**: the 4 clusters are only
   weakly differentiated from each other (VPIP 0.22-0.28, bluff rate 0.50-0.55 across all four) --
   Pluribus's human opponents were all elite, similarly-skilled professionals, not a diverse
   recreational population, so there wasn't much real stylistic spread in the source data to find.
   Proceeded to train all 4 anyway per explicit instruction ("no matter if they do not match our
   ideal personality specification").
4. **Per-cluster XGBoost**: one multi-class classifier per cluster, (real equity, pot_odds,
   street, stack_bb, num_active_opponents) -> action in hero's own 6-way space (fold/call/
   raise_33/raise_66/raise_pot/allin), bucketed from each real raise's actual size. 76-80% test
   accuracy per cluster -- fold/call predicted well, raise-size-bucket boundaries predicted more
   weakly (expected: real bet-sizing is a genuinely fuzzier decision than fold-vs-continue, and
   these 5 features don't capture everything a real player conditions sizing on).
5. **`TreeOpponent`** (`versions/v25/self_play/tree_opponent.py`): mirrors the existing
   `HeuristicOpponent`/`NNOpponent` interface. Samples from the model's predicted distribution
   (never argmax, to avoid a fully predictable/exploitable bot); clips inputs to the training
   data's observed range before querying (trees extrapolate badly outside it, and this simulator's
   own curriculum deliberately sweeps stack depths/prices real hands rarely reach); collapses the
   4 raise-size predictions to a single `'raise'` at the interface boundary, since the simulator
   sizes ANY opponent's raise with one fixed 0.75x-pot rule regardless of kind -- carrying a real
   predicted size through is a genuine follow-up (would need a new interface path), not attempted
   here. Wired into `opponents.py`'s `build_opponent_pool` via a new `tree_cluster: <id>` pool-
   config key.
6. **A real bug caught before trusting the smoke test**: initial integration test showed a
   degenerate ~97% VPIP for both seeded clusters. Root cause: `min(max(equity, *CLIP_RANGES[...]),
   CLIP_RANGES[...][1])` expands to `max(equity, 0.0, 1.0)`, which always returns 1.0 (1.0 is
   literally one of the arguments) -- every decision was being fed `equity=1.0` regardless of the
   bot's real hand, hence near-universal continuing. Fixed (proper two-step clip). After the fix:
   500 simulated hands, 0 crashes, VPIP 0.37-0.40 / AGG 0.44-0.57 for the two seeded seats --
   coherent, equity-sensitive, and modestly differentiated between clusters (consistent with the
   honest finding in step 3 that the source clusters themselves aren't strongly differentiated).

**Status**: infrastructure built, integrated, and validated (no crashes, sane telemetry). NOT yet
swapped into `config.yaml`'s live training pool -- that's a real experimental decision (changes
what hero trains against) requiring its own retrain to evaluate, deliberately not made
unilaterally alongside building the capability itself. Trained models live in
`versions/v25/weights/tree_opponents/` (`xgb_cluster_{0,1,2,3}.json` + manifest); reusable by any
future version via the same `tree_cluster` pool-config key.

See: `versions/v24_extreme/SPECS.md` (the diagnostic that prompted this pivot) |
`.agents/skills/OFK/references/known-shortcomings-backlog.md` [BET-1]

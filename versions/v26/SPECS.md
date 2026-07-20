# V26 SPECS

Branches from `versions/v25` (fresh weights, not resumed -- per [VAL-5]). SAME architecture,
contract, and target-EV mechanism as V25 (context_dim=44, contract_version=7, the multi-street
`_rollout_continuation_ev` fix -- see `versions/v25/SPECS.md` for that mechanism's own derivation,
calibration, and 50k/100k results, which were MIXED, not a clean win -- not re-litigated here).
V26 changes ONLY the training opponent pool.

## What changed: 2 of 5 opponent seats now play real human-fitted models, not heuristics

Per the 2026-07-18 discussion motivating this: everything in the training population up to and
including V25 is either a hand-designed formula (`opponent_bots.py`'s `FuzzyPlayerArchetype`) or a
model trained inside this same simulator's own self-play loop (the lagged-self mirror) -- so the
whole population shares whatever representational ceiling this codebase's own assumptions impose,
and the one NN opponent (lagged-self) inherits hero's own blind spots for the same reason (it grew
up against the same heuristic population hero did). A model fit directly on external, real human
decisions is the one source of behavior in this pool NOT shaped by this codebase's own design
choices.

`config.yaml`'s pool: `maniac` (weight 0.20) and `nit` (weight 0.15) swapped from their usual
heuristic archetypes to `tree_cluster: 0` and `tree_cluster: 3` respectively -- `TreeOpponent`
instances (`versions/v25/self_play/tree_opponent.py`) backed by XGBoost models fit on ~28k and
~13k real decision rows from the Pluribus/WSOP full-information hand-history corpus (see
`versions/v25/SPECS.md`'s "New infrastructure" section for the full pipeline: download -> pokerkit
replay -> identity-agnostic behavioral clustering -> per-cluster XGBoost -> Opponent-interface
integration, including two real bugs found and fixed along the way). `fish`/`tag`/`past` stay
exactly as in V25, as a stability anchor -- this experiment isolates "what happens when 2 specific
seats become real-data-driven," not "what happens to the whole pool at once."

**Known, accepted limitation (not attempted here)**: the two TreeOpponent seats still get the SAME
fixed 0.75x-pot sizing every heuristic opponent's 'raise' decision already gets in this simulator
-- carrying a real predicted bet SIZE through requires extending the `Opponent` interface itself
(a genuine follow-up, not a training-recipe change). This run tests whether the fold/call/raise
DECISION BOUNDARY these seats learned from real data (not the sizing) already changes what hero
learns.

**Honest caveat inherited from the source data** (see versions/v25/SPECS.md): the 4 behavioral
clusters found in the Pluribus corpus are only weakly differentiated from each other -- all of
Pluribus's human opponents were elite, similarly-skilled professionals, not a diverse recreational
population. This experiment is a test of "does even modest, genuinely non-heuristic diversity move
the needle," not a test of "does injecting dramatically different personalities work" -- that
would need a source with more real behavioral spread (the anonymized Bet365-style corpus discussed
earlier this session, or a future hand-history source with more stylistic variety).

## Verification (pre-training)

- `versions/v26/self_play/tree_opponent.py` inherited unchanged from V25 (same class, same clip
  bug already found and fixed there -- see that file's own docstring).
- Config-level pool wiring smoke-tested directly through `train.py`'s own YAML-loading path (not
  a hand-built pool): 200 real simulated hands via `build_opponent_pool` reading `config.yaml`
  verbatim, 0 crashes. Pool resolved correctly: `{'past': 'Past (Heuristic)', 'maniac':
  'RealPlay-0 (Tree)', 'fish': 'Calling Station (Heuristic)', 'tag': 'TAG (Heuristic)', 'nit':
  'RealPlay-3 (Tree)'}`.
- `target_hands: 100000` (matches V25's own confirmatory run scale), fresh weights, no
  `--resume_path`.

## Results (2026-07-18, `expert_main.pth`, 100k hands complete)

**Training telemetry** (final batch): hero closed at **+58.2 BB/100** (VPIP 44.4%, AGG 54.5%),
beating every seat in the pool except a narrow loss to TAG (-3.3 BB/100 on the exploitation
scoreboard). Both TreeOpponent seats got clearly outplayed once hero adapted (RealPlay-0: -26.4,
RealPlay-3: -34.0 BB/100) -- not a training-pool imbalance, hero simply learned to beat them, same
as it beats the heuristic seats other than TAG. Action usage settled at Fold 38.8% / Call 15.7% /
r33 11.3% / r66 11.2% / rPot 13.8% / All-In 9.3%.

**`model_verify --full`: 19 PASS, 3 WARN, 1 FAIL, 0 SKIP** (`tools/model_verify/results/v26__expert_main.{json,html}`,
copied to `.agents/skills/OFK/references/V26/model_verify_report.html` per the standing convention).

- **`beats_frozen_predecessor`: PASS, +42.8 BB/100** over 4000 hands vs a field including a frozen
  V25 snapshot (copied in as `versions/v26/weights/frozen_v25.pth` specifically to enable this
  check -- it SKIPped on the first run with no frozen predecessor present). A real, direct win over
  V25's own trained weights, not just a self-play telemetry number.
- **`vpip_adapts_to_style`: PASS**, short delta +10.4pts / deep delta +8.6pts -- comparable to V25's
  100k run (+12.0/+9.4pts), the core BET-1 hypothesis still holds.
- **`committed_sensitivity` and `position_sweep` are back to PASS** (0.101 TV, spread 0.378) --
  both were newly WARN in V25's own 100k confirmatory run. Can't be conclusively attributed to the
  real-data opponents (single run, no seed-controlled ablation), but it's the one measurable
  difference between the two 100k runs' check outcomes.
- **`deep_stack_ood_guard`: FAIL** -- the same known, pre-existing V14/V19/V20-era deep-stack
  OOD trash-jam (eq=0.55, stack=40bb -> ALL-IN argmax @ 0.32). Not new to V26, not fixed by it either.
- **`allin_exploits_opponent_foldiness` [OPP-8]: WARN, spread 0.011** -- confirms the backlog
  finding with real numbers from this run: P(all-in) barely differs across NIT/TAG/LAG/CALLING_STATION
  despite wildly different real `base_fold_to_pressure`. The real-data opponent seats did NOT close
  this gap -- consistent with the honest caveat below (Pluribus's humans were all similarly-skilled
  professionals, so their fitted models don't express a foldy-vs-sticky spread wide enough to teach
  hero this distinction either).
- `free_check_low_fold`, `pot_type_sensitivity`: WARN, both pre-existing/tracked, unaffected by this
  version's change.

**Live telemetry (opponent-color jam pattern, both training-time and end-of-run)**: hero's all-in
rate by opponent color tracked Blue 9-11% -> Green 16-17% -> Yellow 22-23% -> Red 19-20%,
consistently, across multiple mid-training snapshots and the final one -- a stable policy trait, not
noise. Notably this is the OPPOSITE of what a pure fold-equity exploit would predict (jam MORE
against the tightest/foldiest color, not least) -- see [OPP-8] in the backlog for the two competing
explanations (underexploitation due to no direct foldiness read vs. a legitimate range-selection
effect where reaching postflop against a tight opponent at all is already a stronger-than-average
spot for hero).

**Verdict**: a clean, real improvement over V25 (+42.8 BB/100 head-to-head, one check recovered from
WARN to PASS, no new regressions) from a training-recipe change that touched ONLY 2 of 5 opponent
seats. Whether the real-data opponents specifically (rather than just a third self-play run's normal
variance) caused the `committed_sensitivity`/`position_sweep` recovery is not established by this
single run -- would need a same-seed re-run or a second real-data-opponent version to attribute
confidently. `allin_exploits_opponent_foldiness` / OPP-8 remains open and does not appear to be
something this data source can fix on its own (see honest caveat, source data lacks behavioral
spread on foldiness specifically).

## Status

Training complete and verified 2026-07-18 (`model_verify --full`, see Results above). NOT yet
deployed live -- `core/decision.py` still serves V25 as the active model. Deployment is a decision
for the user to make (per the established pattern: V25 was deployed live explicitly on request,
not automatically on a good verify result).

See `versions/v25/SPECS.md` (the inherited EV-fix mechanism + full TreeOpponent pipeline) |
`.agents/skills/OFK/references/known-shortcomings-backlog.md`

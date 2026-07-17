# V20_preflopEq_AI — opponent-pool experiment (NN diversity vs. shove-preference)

Clone of V20_preflopEq (`versions/v20_preflopEq`). **Same architecture, same tensor schema**
(context_dim=37, contract_version=5, `PokerEVModelV4`) — this version changes ONLY the training
opponent pool. No contract/model code changes.

## How this started

A live-session review of V20_preflopEq (board `Double_Or_Nothing_1171034284`) surfaced a
qualitative observation: the model shoves all-in very readily, including in spots (e.g. AQ facing
one raise, modestly multiway) where a human would consider sizing up gradually to grow the pot
instead. Investigated with the real trained model rather than speculating:

Queried `expert_main.pth` (V20_preflopEq's 75k checkpoint) across an equity x stack grid
(heads-up, no bet to face) and printed the critic's raw Q-values (bb EV estimate) per action:

```
equity  stack |  CALL   R33   R66  RPOT ALLIN
  0.55     20 |  1.24  1.45  1.48  1.69  3.10
  0.75     40 |  2.13  2.51  2.58  3.09  5.46
  0.95     50 |  2.58  3.07  3.11  3.75  6.45
```

ALLIN is cleanly, monotonically the highest-EV action at every combo tested — roughly double the
next-biggest size. Not a bug: the model has correctly learned to exploit its OWN training
population. Traced the mechanism in `opponent_bots.py`'s `FuzzyPlayerArchetype`:

1. **The "value" branch ignores bet size entirely** — once an opponent's equity clears
   `need_for_value`, it calls/raises "regardless of price" (verbatim from the code). Shoving
   extracts the same zero-extra-fold-cost value from these hands as any smaller bet, for roughly
   double the pot.
2. **The fold-bar saturates** — `continue_bar = min(0.95, pot_odds + style_shift)`. Past a certain
   size, bigger bets barely induce more folds, so there's little downside to going bigger.

So this is a real, quantified, correctly-learned exploit of a **training-population fidelity gap**
— the heuristic bots don't punish overbets the way a real opponent (or a genuine learned policy)
would. `check_action_diversity`'s own model_verify data for V20_preflopEq already showed this
structurally: RAISE_33/RAISE_66 essentially never won argmax anywhere in the test grid at 25k,
50k, or 75k hands — only FOLD/CALL/ALLIN (occasionally RAISE_POT) ever did. This traces back to
the same "no middle gear" characteristic flagged as [P1] in `versions/v15/SPECS.md` and evidently
never fully resolved by that version's stack-range widening as once believed.

## This version's experiment

Shift the opponent pool toward real NN opponents — learned, more continuous decision boundaries —
instead of relying mainly on the heuristic bots, **without** needing the (still unbuilt) per-model
contract-selection mechanism: every opponent here is natively context_dim=37/contract_version=5,
so no cross-version scale mismatch is possible (unlike using any pre-V20_preflopEq frozen
checkpoint, which would crash on a tensor shape mismatch).

**Pool** (`config.yaml`, `style` keys `maniac`/`fish` repurposed as stat-bucket identifiers only —
see the file's own comments):

| style (bucket) | weight | actual opponent |
|---|---|---|
| `past` | 0.25 | Lagged self-play mirror of THIS run (refreshed every 5k hands) |
| `maniac` | 0.20 | `frozen_50k.pth` — V20_preflopEq's own 50k-hand checkpoint |
| `fish` | 0.15 | `frozen_25k.pth` — V20_preflopEq's own 25k-hand checkpoint |
| `tag` | 0.25 | Heuristic TAG (disciplined anchor, as requested) |
| `nit` | 0.15 | Heuristic NIT (short-stack push/fold discipline) |

`frozen_v20_preflopEq.pth` (V20_preflopEq's 75k final model) is also carried as this version's
`beats_frozen_predecessor` benchmark — same architecture as this version, so unlike
V20_preflopEq's own attempt against frozen V20 (SKIPped on a context_dim mismatch), this
comparison should actually run.

**Honest caveat, documented before training, not after**: `frozen_50k`/`frozen_25k`/`past` are all
the SAME lineage, trained under the same heuristic-dominated conditions this experiment is
questioning — they may reproduce the shove-bias rather than punish it. This tests "does self-play
diversity alone fix it." If `check_action_diversity`'s raise-bucket usage and the critic's
mid-size Q-values don't move, the more direct lever — making the heuristic bots' value-branch
price-sensitive (continuation probability decaying with bet size even above the value threshold,
not flat) — is the next thing to try, not further pool-composition iteration.

## Training plan

Fresh random-init weights (matches this project's "one variable per version" / "new-version-by-
copy, fresh weights" convention — a warm start from V20_preflopEq could mask whether the new
population actually changes the learned behavior, since gradient momentum from the old population
could persist). Target 150k hands, `checkpoint_dump_interval=35000` for sanity checks at
35k/70k/105k/140k, full `model_verify` (FAST+SLOW) at completion.

## Status

Training launched 2026-07-17. Opponent pool verified standalone (loads correctly, labels resolve
to the real underlying model per `describe_pool_entry`, 50 hands simulate with no crash) before
starting the real run. Updates logged below as checkpoints land.

**35k checkpoint (`main_hands35068.pth`) FAST model_verify: 9 PASS / 1 WARN / 0 FAIL** — notably
stronger than V20_preflopEq's own 25k/50k results at comparable training length:
- `deep_stack_ood_guard` **already PASSes** ("no marginal-equity/deep-stack/single-modest-bet cell
  jams all-in") — V20_preflopEq never fully passed this at 25k, 50k, OR its final 75k.
- `action_diversity`: `{'fold': 9, 'call': 3, 'raise_pot': 5, 'allin': 4}` — **RAISE_POT wins
  argmax in 5 of 21 grid cells.** V20_preflopEq's action_diversity NEVER showed a raise bucket
  winning argmax anywhere in its own equivalent grid at 25k, 50k, or 75k (`{'fold', 'call',
  'allin'}` only, occasionally one `raise_pot` cell). This is early (35k of 150k) but is a
  concrete, measurable first signal in exactly the direction this experiment is testing for.
- Both new-feature sensitivity checks still healthy (`hand_strength` 0.153, `equity_edge` 0.601).
- `free_check_low_fold` WARN (0.993) is the same long-standing non-gating soft spot every version
  in this line carries.

Too early to conclude the opponent-pool hypothesis is confirmed (35k hands, pre-cutover-adjacent),
but a promising early read. Continuing to 70k.

**70k checkpoint (`main_hands71426.pth`) FAST model_verify: 8 PASS / 1 WARN / 1 FAIL** — the 35k
signal partially REVERSED, reported honestly rather than cherry-picking the better number:
- `deep_stack_ood_guard` **regressed back to FAIL** (`eq=0.55, stack=30bb -> ALLIN@0.31`) — had
  passed clean at 35k.
- `action_diversity`: `{'fold': 9, 'allin': 10, 'raise_pot': 2}` — down from 5 raise_pot cells at
  35k to 2, and allin back up to 10/21 (was 4/21). Still strictly better than V20_preflopEq ever
  showed (which had ZERO raise-bucket argmax wins), but the trend from 35k did not hold.
- `hand_strength_sensitivity` dropped 0.153 -> 0.066 (still passing, >0.03 threshold) --
  `equity_edge_sensitivity` held steady (0.601 -> 0.616).
- No errors/crashes; this is normal non-monotonic mid-training variance, not a wiring problem --
  same class of fluctuation V20_preflopEq itself showed between its own 25k/50k checkpoints.

Read: the 35k result may have been a transient/exploration-phase artifact rather than a stable
shift, OR the model is still cycling before settling. Not treating this as a failure of the
hypothesis yet -- watching whether 105k/140k trend back toward diversity or consolidate around
allin-dominance once training fully clears the curriculum's later phases (Phase 4 dynamic
activity >50k, Phase 5 focus rounds >75k, both new territory this run hasn't reached before).

**105k checkpoint (`main_hands107359.pth`) FAST model_verify: 8 PASS / 1 WARN / 1 FAIL** --
consistent with 70k, not a further slide:
- `deep_stack_ood_guard` still FAILs (`eq=0.55, stack=25bb -> ALLIN@0.36`, a different specific
  cell than 70k's).
- `action_diversity`: `{'fold': 9, 'allin': 10, 'raise_33': 2}` -- still only 2 raise-bucket argmax
  cells (now raise_33 instead of raise_pot), allin holding at 10/21. Reads as the model having
  settled near this same character rather than continuing to drift either direction.
- `hand_strength_sensitivity` ticked back up (0.066 -> 0.100), `equity_edge_sensitivity` stable
  (~0.62). Both still clearly load-bearing.
- No errors/crashes.

Working read at 105k/150k: the 35k result looks like it was likely a transient exploration-phase
reading rather than a stable trend -- the model has settled into a character similar to
V20_preflopEq's own (allin-dominant, occasional raise-bucket use), modestly more diverse than
V20_preflopEq ever showed but not a dramatic shift. One checkpoint (140k) left to see if this
holds, improves, or degrades further before the final verdict at 150k.

**140k checkpoint (`main_hands143409.pth`) FAST model_verify: 8 PASS / 1 WARN / 1 FAIL** --
continues the 70k/105k trend, not a recovery:
- `deep_stack_ood_guard` still FAILs, same cell shape (`eq=0.55, stack=25bb`), slightly higher
  mass (`ALLIN@0.40`, up from 0.36 at 105k).
- `action_diversity`: `{'fold': 9, 'allin': 11, 'raise_pot': 1}` -- DOWN to only 1 raise-bucket
  argmax cell, allin now at its highest count of the whole run (11/21). The diversity gain from
  35k has fully faded by 140k.
- Both new-feature sensitivity checks remain healthy (hand_strength 0.076, equity_edge 0.621) --
  the features themselves are fine; this is specifically about sizing-bucket usage, not the new
  context features breaking.

**Working conclusion going into the 150k full verify**: the opponent-pool-diversity hypothesis
does NOT show a durable win on the sizing-diversity question by 140k -- the 35k reading was very
likely a transient/exploration-phase artifact. The model has converged to a character at least as
allin-concentrated as V20_preflopEq's own baseline. This doesn't mean the experiment was
wasted -- it's a real, negative result narrowing down the cause: simply diversifying the opponent
POOL (even with genuine NN opponents) isn't sufficient on its own. The more direct lever flagged
before training started -- making the heuristic bots' own value-branch price-sensitive, so
"value" hands don't pay off any bet size identically -- remains the untested, more targeted next
step.

## Final result -- 150k hands complete, full model_verify: 12 PASS / 1 WARN / 1 FAIL / 0 SKIP

Report: `tools/model_verify/results/v20_preflopEq_AI__expert_main.pth.json` /
`v20_preflopEq_AI_report.html`.

**The sizing-diversity hypothesis did NOT pan out** -- `action_diversity` at 150k:
`{'fold': 9, 'allin': 11, 'raise_pot': 1}`, essentially unchanged from 140k, still allin-dominant.
`deep_stack_ood_guard` still FAILs (`eq=0.55, stack=20bb`). Confirmed negative result: diversifying
the opponent POOL alone, even with genuine NN opponents, does not durably change the learned
sizing behavior. See "next step" note above -- the heuristic bots' price-insensitive value-branch
is the more direct, still-untested lever.

**But the model is a clear, measurable overall improvement over its parent** -- for the first time
in this lineage, `beats_frozen_predecessor` actually RAN (not SKIP): same architecture as
V20_preflopEq (unlike V20_preflopEq's own attempt against frozen V20), so `frozen_v20_preflopEq.pth`
loaded cleanly.
- `beats_frozen_predecessor` **PASS**: +53.5 BB/100 over 4000 hands vs a field including the
  frozen parent -- a real, validated head-to-head win, the first this lineage has actually been
  able to measure directly.
- `vpip_adapts_to_style` **PASS**, and MEANINGFULLY stronger than the parent: short +11.5pt (vs
  parent's +6.6pt), deep +9.6pt (vs parent's +7.1pt).
- `bb100_vs_standard_fields` **PASS**, positive across all 4 fields and higher than the parent's
  own baseline everywhere: loose_short +31.1 (parent +16.8), loose_deep +57.7 (parent +36.2),
  tight_short +27.0 (parent +22.8), tight_deep +65.1 (parent +61.1). Recorded as this version's own
  baseline in `baselines.json`.
- `beats_offformula_stress` **PASS**: +34.7/+65.0 BB/100 (short/deep), comparable to or slightly
  better than the parent's +31.3/+66.0 -- no overfit-to-training-formula regression.
- Both new-feature sensitivity checks remain healthy (`hand_strength` 0.104, `equity_edge` 0.606).
- `deep_stack_ood_guard` FAIL and `free_check_low_fold` WARN are the same long-standing soft spots
  every version in this line carries -- not introduced or worsened here.

**Overall verdict**: this run answered its own question with a clean negative (pool diversity
alone doesn't fix the shove-preference) while incidentally producing a genuinely stronger overall
model than V20_preflopEq across every winrate/adaptation metric tested, including the first real
validated win over its own direct predecessor this lineage has managed. Whether to deploy this
over V20_preflopEq live, and whether to chase the heuristic-bot price-sensitivity fix next, are
both open decisions for the user -- not acted on here.

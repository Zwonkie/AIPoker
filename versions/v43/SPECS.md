# V43 SPECS — corrective-prior cleanup: back to learning from correct inputs

Branches from `versions/v41` (fresh weights, not resumed — per [VAL-5]). **No contract change**:
`context_dim=54`, `contract_version=8`, identical to V29/V40/V41, so checkpoints stay compatible
and the live bridge needs no new wiring.

**Status: TRAINED (100,000 hands, 2026-07-21) and DEPLOYED LIVE 2026-07-21** — by explicit user
decision, on a MIXED scorecard, and **before `beats_frozen_predecessor` finished**. V41 (the
MILESTONE) stays registered as the one-line rollback.

Training: 2.5h, fresh weights (no `--resume_path`, per [VAL-5]), checkpoints at 21.9k/43.8k/65.7k/
87.7k. Hero +33.3 BB/100 vs the field at VPIP 48.8% / AGG 55.7% — but training-time BB/100 is
masked by the exploration anchor and must not be trusted (guardrails §0); `eval_pure_policy.py` is
the honest number.

### model_verify --full at deploy time: 19 PASS / 4 WARN / **1 FAIL** (slow checks still running)

**Better than V41 — what the cleanup bought:**
- `allin_vs_nextbest_qgap` **PASS, negative at every cell** (−0.34 → −0.59 across 15–40bb). This was
  the headline risk of the clip change; the 0.20 re-calibration held [BET-1] closed. Shipping
  clip=100 with risk left at 0.15 would have regressed it (target-space trend said +6.53).
- `opponent_style_sweep` **WARN → PASS** (0.054) — an [OPP-8] check V41 failed.
- `action_diversity` genuinely mixed: `{fold 9, raise_66 8, raise_pot 2, call 1, allin 1}` —
  `raise_66` is a real action now, not the old fold/allin collapse.
- `short_stack_polarization` 0.15 (V41 0.19), `deep_stack_ood_guard` PASS,
  `multiway_shortstack_aggression` PASS ([BET-3] still resolved), `position_sweep` spread 0.936.

**Worse than V41 — the predicted cost of removing the realization discount, and it is real:**
- `vpip_adapts_to_style` **FAIL** — short +5.4pts, deep **+4.2pts** vs the ≥5pt gate. V41 passed at
  short +5.9 / deep +7.2. Entry range widened AND opponent-adaptation weakened. This is [P4], the
  longest-standing behavioural goal in the project, moving backwards.
- `nash_bbcall_vs_jam` **47%** (V41 passed). Calls jams far wider than Nash at 5bb, frequently never
  folding (`F=0.00`). **This one should NOT be discounted** — it is the clean binary check (facing
  an all-in there is no cheap-limp confound), unlike `nash_pushfold_vs_chart`.
- `nash_pushfold_vs_chart` 65% (V41 78%) — discounted for the reasons under "Findings NOT acted on"
  below; the run again reports `971/971 commits use a sized RAISE not a literal jam`.

**The honest read**: the ablation predicted exactly this (entry rate at eq ≤ 0.35: 0.59 → 0.79 in
target space). Removing the discount did what it said it would. The open question — whether a
looser, more diverse style is *correct against this population* even though a GTO chart dislikes it
— is precisely what `beats_frozen_predecessor` and `bb100_vs_standard_fields` would answer, and
those had not finished at deploy time. **If the head-to-head goes against V43, the realization
discount was load-bearing and V41 is the rollback.**

`weights/` also holds `frozen_v41.pth` (md5-identical to V41's deployed `expert_main.pth`) so that
head-to-head is a genuine NN-vs-NN comparison.

## Premise

V40 and V41 fixed root causes in the training data itself: the betting round no longer ends on a
check (postflop and BB-option nodes exist at all for the first time), CALL is no longer exempt from
the variance penalty or the continuation credit, blinds are no longer dead, NN opponents no longer
play a degraded self, stacks are asymmetric, the min-raise rule is legal, and [OPP-7] is fixed at
the tensor boundary.

Several training-loop knobs predate those fixes and exist to *suppress behaviour the broken data was
producing*. This version removes the ones that impose an answer the model should learn from correct
inputs, and keeps the one that measurement says is still load-bearing.

Scope set by explicit user direction: remove the realization discount and the ALLIN veto, fix
`TARGET_CLIP_BB`, and reconcile the bb-normalised values that depend on it.

---

## Change 1 — Realization discount REMOVED (V12 knob)

`POLICY_TIGHTNESS_BB = 2.0` subtracted a flat bb amount from every voluntary action below eq 0.45,
in both the pre-cutover (`vectorize_hand_samples`) and post-cutover (`regret_match_policy_torch`)
actor targets. Rationale when introduced: "the all-in-equity counterfactual overvalues speculative
entries." Deleted outright — the helper, both call sites, the `equity=` parameter, the
`run_training` argument, and the config key.

**Measured before removing** (`ablate_dampeners.py`, eq × stack × field-size grid, target space):

| | entry rate, 5bb → 100bb | entry rate @ eq ≤ 0.35 |
|---|---|---|
| with discount | 0.90 → 0.82 | **0.59** |
| without | 0.90 → 0.85 | **0.79** |

It does nothing at depth but is genuinely active at its design point. **Removing it is a deliberate
decision to let correct inputs teach entry discipline rather than impose it — so EXPECT entry range
/ VPIP to widen, and check `vpip_adapts_to_style` and the entry-range checks specifically.** This is
the change most likely to regress, and V41 is the rollback.

## Change 2 — ALLIN critic-consistency veto REMOVED (V29 knob)

`critic_consistency_margin = 0.15` zeroed ALLIN's regret whenever another action's Q beat it by more
than the margin. Deleted: the veto block, the parameter, the global, the config key, and
`calibrate_critic_consistency.py` (which existed only to tune it).

**Measured**: near-inert. ALLIN's target share 0.37–0.43 with it, 0.41–0.43 without; entry rate at
eq ≤ 0.35 identical (0.59 vs 0.60). Consistent with V40's own finding that its rescope was a
provable no-op under the fold baseline. *Limitation*: that measurement exercises the pre-cutover
path; post-cutover the veto applied to the critic's Q, which the harness does not reproduce.

## Change 3 — `TARGET_CLIP_BB` 40 → 100 (review T-M5)

The clip sat at 40bb while `stack_depth_mix` reaches 100bb and the contract represents stacks to
`STACK_CEIL_BB = 100`, so a 100bb stack-off and a 40bb loss trained identical targets.

**Measured** (600 hands): `|realized go-forward return| > 40bb` on **23.4%** of decision points
(p95 = 102bb, max = 167bb); `> 100bb` on 5.9%. The clip was truncating nearly a quarter of the
realized channel, not a corner case.

Gradient-safe: the critic loss is `nn.HuberLoss(delta=2.0)`, so per-sample gradient magnitude
saturates at 2bb of error — a 100bb target does not produce a 2.5× larger gradient than a 40bb one.
The clip controls **bias**, not stability. The Q head is an unbounded MLP (`equity_base_q + head`),
so no architecture change is needed.

Residual, accepted: ~6% of returns still clip. Hero at 100bb multiway can win more than 100bb, but
the contract cannot *represent* a stack above 100bb, so clipping at the same ceiling keeps the value
scale and the input scale consistent rather than teaching the critic magnitudes its own inputs
cannot distinguish.

## Change 4 — `risk_aversion_coefficient` 0.15 → 0.20 (forced by Change 3)

**The clip and the variance penalty are not independent knobs.** The 40bb clip was acting as an
undeclared deep-stack all-in dampener: it truncated ALLIN's edge at depth, so 0.15 only ever had to
cover what the clip left over. Raising the clip re-exposes it.

ALLIN-vs-next-best gap trend across 5→100bb — the [BET-1] pathology signature, which V29/V41 finally
got fully negative on `allin_vs_nextbest_qgap`:

| config | trend |
|---|---|
| V41 (all four dampeners, clip 40) | +1.39 |
| clip 100, risk 0.15 (naive) | **+6.53** ← would have regressed [BET-1] |
| **clip 100, risk 0.20 (shipped)** | **+1.05** |
| clip 100, risk 0.25 | −0.03 |
| clip 100, risk 0.35 | −0.10 |

0.20 restores V41's damping in the new scale. 0.25+ was **not** taken: a single-cell trace (AhKh,
eq 0.75, 100bb) shows ALLIN going 13.4 → 10.4 → 7.3 → 1.3bb as the coefficient rises, i.e. by 0.35
ALLIN is *dominated* (1.3 vs raise_pot's 3.1) — that risks the opposite pathology (never jamming),
and the aggregate metric sits at its zero-crossing there, so it cannot discriminate reliably.
Verified no target is driven into the clip at any tested value. Same one-step, same-direction
discipline as the documented 0.10 → 0.15 move.

**The variance penalty is the one dampener V43 KEEPS**, because the same ablation showed its
pathology is *not* gone at the source: removing it entirely takes the trend to +8.79 (and +14.60
with nothing at all).

---

## Fail-loud, not no-op

A config still setting `policy_tightness_bb` or `critic_consistency_margin` now **raises**:

```
ValueError: config.yaml sets ['policy_tightness_bb'], which V43 REMOVED ... They are gone from
the code, not defaulted to 0 -- delete the keys, or run versions/v41 if you want them.
```

Verified by injection. This repo's recurring failure mode is the opposite — a removed thing quietly
becoming a no-op (dead `past_model` attributes, the commented-out `CRITIC_ARGMAX_MODE` line, the
stale `PHPHelp.py` version ladders that served V41 vs-random equity). A removed knob must not
silently train without the corrective its author believed was active.

## Verification so far

- 400-hand end-to-end training run completes and saves; loss decomposition prints normally.
- Removed-knob guard fires on injection; config restored afterwards.
- `versions.v43` imports rewritten (no cross-version imports); `weights/` clean.
- No stale references to the removed symbols anywhere in the slice.

## Measurement tooling added (reusable)

- `probe_nash_regression.py` — enumerates `nash_pushfold_vs_chart` disagreements across checkpoints.
- `calibrate_pushfold_dampeners.py` — dampener sweep against the Nash grid in target space.
- `ablate_dampeners.py` — the ablation behind Changes 1/2/4.
- `probe_commit_ev_decomposition.py` — decomposes commit EV into fold-equity vs showdown value.
- `probe_size_collapse.py` — measures chip-identical raise sizes in real hands.

## Findings recorded here that are NOT acted on in V43

1. **`nash_pushfold_vs_chart` is substantially a check artifact.** It scores `agg_mass > p_fold`,
   summing four aggressive heads against one fold head — and at its own probe node (pot 1.5bb,
   to_call 0.5bb) `raise_33`/`raise_66`/`raise_pot` are **all the same 1.5bb min-raise**, because
   the min-raise floor exceeds every pot fraction. Meanwhile ALLIN — the only action Nash models —
   has *negative* target EV at the failing cells, i.e. the model agrees with Nash on the actual
   push/fold question. V29's better 83% partly reflects it being too passive, the [BET-3] failure
   V40/V41 deliberately fixed. **This check should be scored on ALLIN-vs-fold before it is used as
   a training target.**
2. **Raise-size collapse is real and large**: 40.7% of hero decisions overall, **56% preflop**, have
   all three sized raises chip-identical. The review's T-M9 (stack-capped `raise_33` already an
   all-in) is only **2.4%** — a 17× smaller corner. An `allin_by_chips` flag is staged in the
   simulator, **default off** (byte-identical to V41), pending a decision between deduping
   chip-identical sizes and making the buckets min-raise-aware.
3. The four dampeners do **not** control the HU push/fold commit threshold (11 configurations, 349
   cells, all ~identical) — that threshold is set by the fold-equity model, which is review #6/H2
   (B1) territory.

See `.agents/skills/OFK/references/fable-review-resolution-log.md`.

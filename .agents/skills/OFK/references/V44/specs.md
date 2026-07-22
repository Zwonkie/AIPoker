# V44 SPECS — `equity_edge` normalized by the effective contested field

Branches from `versions/v43` (fresh weights, not resumed — per [VAL-5]).

**CONTRACT CHANGE**: `context_dim` stays 54, but `contract_version` goes **8 → 9** because
`ctx[35]` (`equity_edge`) keeps its index and width while changing **meaning**. V43 and earlier
checkpoints would *load* into this contract and behave wrongly; the version bump is what stops that
happening silently.

**Status: DEPLOYED LIVE 2026-07-22** — `active_model_name = 'Herocules (v44)'`, V43 registered as
the one-line rollback. Deployed on the 0-FAIL scorecard and the first-ever [P4] pass (below).

**Live wiring (all verified end-to-end through the real `make_decision`):**
- `core/models/v44_engine.py` declares `make_bridge()` (own V44 contract — NOT interchangeable with
  V43's despite equal width, since ctx[35] means the effective-field edge), `live_features()`,
  `is_sized`, `display_tag="V44"`, `has_aux`. No `is_vN` ladder here or in PHPHelp can misroute it.
- The live edge denominator is wired: `live_feature_providers()` now returns `effective_field_fn`
  (the version's own `effective_contested_field` + `_COLOR_TO_VPIP`), and PHPHelp sets
  `BoardState.effective_field` from the same front/after HUD split it already builds for equity —
  preflop uses the roll, postflop uses the nominal count, exactly mirroring the simulator.
  Confirmed live: preflop ctx[35]=1.458 (effective), not 3.12 (nominal). Absent → falls back to
  nominal = V43 behaviour, never worse.
- **Live dropdown pruned to the 5 newest iterations** (user request): v44, v43, v41, v40, v29.
  v28→v13 removed from the SELECTOR only — slices, weights, engine files and bridge branches all
  remain on disk, so re-registering any is a one-line add (deprecate-not-delete).

Two fixes made while deploying, both recorded in `versions/v42_liveFixes`:
- `model_verify`'s `build_ctx` derived ctx[35] from the nominal count; made version-aware for
  contract_version ≥ 9 (else V44 scored broken on an OOD feature — see above).
- `verify_v42.py`'s FOLD-mask check asserted sampling-appearance, which broke on V44's sharper
  policy (P(FOLD)=0.038 → ~0 after temp-0.5 sharpening). Re-pointed at the post-mask
  `sampled_probs` distribution, which is what masking actually controls — a hard 0 iff masked.

## Result — 21 PASS / 6 WARN / 0 FAIL (`model_verify --full`)

**The targeted goal passed.** `vpip_adapts_to_style` — [P4], the longest-standing behavioural goal
in the project — **PASS for the first time**: short +6.1pts, deep +5.9pts, against a ≥5pt gate that
V43 failed (+5.4 / **+4.2**). This version was built to make `equity_edge` usable, and the metric it
was supposed to move, moved.

| check | V44 | V43 |
|---|---|---|
| `vpip_adapts_to_style` | **PASS** +6.1 / +5.9 | FAIL +5.4 / +4.2 |
| `beats_frozen_predecessor` | **PASS +91.7 BB/100** vs frozen V43 | +74.6 vs frozen V41 |
| `nash_bbcall_vs_jam` | WARN 67% | WARN 47% |
| `bb100_vs_standard_fields` | PASS, all four fields + | PASS |
| `allin_vs_nextbest_qgap` | PASS, negative every cell ([BET-1] held) | PASS |
| `multiway_shortstack_aggression` | PASS ([BET-3] aggression held) | PASS |
| FAIL count | **0** | 1 |

**The honest cost — two features V44 attends to less than V43 did:**
- `committed_sensitivity` **WARN 0.023** (V43 PASS 0.050) — barely responds to opponent committed
  chips.
- `pot_type_sensitivity` **WARN 0.024** (V43 PASS 0.052) — barely responds to 3bet-vs-limped.

Both are "feature may be redundant" warnings, not correctness failures, but they are a real
regression: WARN count went 4 → 6 while FAIL went 1 → 0. Plausibly the sharper `equity_edge` now
carries field/commitment information those two slots used to, making them partly redundant — worth
confirming, not assuming.

## The single-hand entry sweep is only a PARTIAL win — do not overclaim

The AKs-vs-N-Yellow sweep (field SIZE axis, size 1→5) improved but did not resolve: raise region
extended (3-way flipped CALL→RAISE, P(raise) 0.771), but AKs still folds **98.5% at 5-way**. The
model still ultimately gates on absolute equity in the coupled regime; the field-size cliff moved
out one seat rather than disappearing.

`vpip_adapts_to_style` passing is NOT in tension with this: it measures the field-TIGHTNESS axis
(tight vs loose field, size fixed), which the effective-field denominator feeds directly by folding
each opponent's VPIP into the count. Size-response and tightness-response are different axes; V44
fixed the one it was aimed at. The residual size cliff is a separate item, still open.

## Verify-harness fix required to score this fairly

`tools/model_verify/scenarios.py::build_ctx` derived `ctx[35]` from the NOMINAL opponent count,
correct for contract_version ≤ 8 but wrong for V44 (trained on the effective field). Left as-is,
every non-overriding check would have fed V44 an edge feature 2–3× out of distribution and scored a
healthy model as broken. `build_ctx` is now version-aware for contract_version ≥ 9, mirroring
`effective_contested_field` exactly (machine-checked for parity across five VPIP profiles).

Frozen predecessor is `frozen_v43.pth` (md5-identical to V43's deployed `expert_main.pth`), seated
in V44's simulator — so it is fed V44's `ctx[35]` semantics, a minor OOD handicap on a feature it
barely uses. Fair as "beat your predecessor in the new contract's world"; noted for a thin margin,
which +91.7 BB/100 is not.

---

---

## The finding

`equity_edge` exists to say "this hand is strong **for this field size**". No model in this lineage
has ever used it. Measured on V43 — AKs preflop, equity computed exactly as training computes it,
`hand_strength` constant at 0.661 throughout:

| opp | equity | equity_edge | P(FOLD) | P(raise) | chosen |
|---|---|---|---|---|---|
| 1 | 0.670 | 1.34 | 0.000 | **0.826** | RAISE_POT |
| 2 | 0.610 | 1.83 | 0.006 | 0.723 | RAISE_POT |
| 3 | 0.600 | 2.40 | 0.008 | 0.674 | CALL |
| 4 | 0.570 | 2.85 | 0.120 | 0.339 | CALL |
| 5 | 0.520 | **3.12** | **0.907** | **0.005** | **FOLD** |

The edge climbs 1.34 → 3.12 exactly as designed while `P(raise)` collapses 0.826 → 0.005: **a top-5
starting hand folded 91% of the time at the precise moment its edge feature peaks.** A threshold
sweep confirms the model gates on near-constant *absolute* equity (eq\* ≈ 0.51 from 2 opponents up),
so edge\* rises linearly with field size (0.79 / 1.47 / 2.04 / 2.58 / 3.08). **If the model were
using the edge, edge\* would be flat.**

## Root cause: the two halves counted different things

`equity` is measured against the **effective contested** field. Preflop,
`compute_range_aware_equity` rolls each still-to-act opponent at their VPIP and **skips all-fold
samples** (deliberately — counting folds as wins made 72o and AA both read ~0.9). So hero's equity
"vs 5 Yellow opponents" is really equity against **1.80** expected contesting opponents, fair share
0.357.

`equity_edge = equity × (num_active + 1)` normalized by the **nominal** field, fair share 0.167.

Different denominators, diverging as the field grows — so the feature was never the clean "equity vs
fair share" ratio its docstring claimed. That is the likely reason it was never learned: a network
cannot extract a ratio whose numerator and denominator disagree about what they are counting.

## The change

`n` in the `n + 1` denominator is now the effective contested field, closed form off the same
`_COLOR_TO_VPIP` the equity roll already uses — no MC, so no added variance:

```
E[k | k>=1] = (|front| + Σ p_after) / (1 − Π(1 − p_after))
```

Front opponents (already committed this round) are `p = 1` and force the denominator to 1, since
someone is then guaranteed in and no conditioning applies. **Postflop this degenerates to the
nominal count** — there is no fold-roll postflop — so this is a **preflop-only** change and postflop
semantics are byte-identical to V43 (verified).

Effect on the feature itself (AKs): a 2.4× field-size swing becomes flat, while it still separates
hands, which is the entire point of it:

| | 1 opp | 3 opp | 5 opp | spread |
|---|---|---|---|---|
| V43 edge | 1.32 | 2.32 | 3.12 | **1.80** |
| **V44 edge** | 1.32 | 1.37 | 1.46 | **0.14** |

| hand | V44 edge band |
|---|---|
| AA | 1.70 – 2.16 |
| AKs | 1.30 – 1.49 |
| JTs | 0.88 – 1.01 |
| 94o | 0.64 – 0.67 |
| 72o | 0.59 – 0.62 |

The residual upward drift on AA is **real signal** — a monster's share genuinely does outgrow fair
share as the field widens — not noise.

## Where it is computed, and why there

At the **caller**, carried on `BoardState.effective_field` — the same pattern `equity` and
`hand_strength` already use. The contract receives seat state and cannot know the front/after
split; the simulator builds `front_colors`/`after_colors` immediately before its equity call
(`simulator.py` ~L1592) and the live path has `colors_in_pot`/`colors_still_to_act`.

`ctx[5] num_active` deliberately stays **nominal** — the model should still know how many players
are seated. Only the edge denominator changes.

`effective_field = 0.0` means "caller did not supply one" and falls back to the nominal count, i.e.
exactly V43's feature. So any construction site not yet updated degrades to the old behaviour rather
than to a silently mis-scaled `ctx[35]`.

**Two independent implementations of this feature exist** — `ContractV12.to_tensors` (inference) and
`vectorize_hand_samples` (gradient tensors). They must change together or training and serving split
silently. `verify_v44.py` asserts they agree, including on the fallback path.

## "Ignores it" is too strong — the precise claim

V43's `model_verify` passes both `equity_edge_sensitivity` (policy shift 0.613 between low and high
edge at *fixed* equity) and `equity_edge_sweep` (0.997 range across edge 0.4–3.2). The input is
wired and the network responds strongly to it **when the edge is varied artificially with equity
pinned**.

What fails is the **coupled** regime — the only one that occurs in real play, where equity and edge
move together because the field size changed. There, behaviour is governed by absolute equity and
the edge contributes nothing that survives (the table at the top of this file).

This is exactly the distinction `opponent_color_isolated_ablation` already draws for the opponent
colour inputs: *"responds at extremes but flat within realistic bounds — reads as a training-
population artifact (network CAN use this input, population never taught it to care within
realistic bounds), not dead wiring."*

And it is what mismatched denominators predict. As the field grows, `equity` says "weaker" while
`equity_edge` says "stronger"; the two disagree, and the network learns to trust the one that is
internally consistent. Fixing the denominator makes them agree instead of compete — which is the
whole bet of this version.

## Why this half of the fork

The alternative was dropping the fold-roll from `equity`. Rejected: that moves `ctx[3]`, the most
load-bearing feature in an equity-primary architecture, and destroys the conditional-on-contested
property that stops 72o and AA both reading ~0.9. This change touches only `ctx[35]`, whose signal
the model currently discards in the coupled regime — so there is little to unlearn.

## Everything else is V43, unchanged

Realization discount and ALLIN veto removed, `TARGET_CLIP_BB` 100, `risk_aversion_coefficient` 0.20.

## Verification

`versions/v44/self_play/verify_v44.py`, 18/18:
1. `effective_contested_field` matches the closed form, including front-only, mixed colours, and
   the empty case.
2. The feature is flat across field size (spread 1.80 → 0.14).
3. It still separates hands — AA / AKs / 72o bands are disjoint.
4. `ctx[35]` differs from V43 preflop, is byte-identical postflop, and falls back to V43 exactly
   when `effective_field` is unset.
5. `vectorize_hand_samples` agrees with `ContractV12.to_tensors`, on both the supplied and the
   fallback path.

Plus a 400-hand end-to-end run, and an instrumented 60-hand run confirming the field is really
populated rather than silently defaulting: **64/64 preflop decisions set** (0 falling back),
effective/nominal ratio min 0.35 / median 0.58 / max 1.00, and **postflop equal to nominal in every
case**.

`weights/tree_opponents/` was carried over from V43 deliberately: those XGBoost boosters are
training *inputs* (the V26 real-data opponent pool), not inherited model weights. Only
`expert_main.pth`/checkpoints start empty, per §6 step 2.

## What to check when it finishes

The entry curve is the point of this version. Re-run the sweep in the finding table above:
`P(raise)` for AKs should no longer collapse from 0.826 → 0.005 across 1→5 opponents, and the entry
threshold should sit at a roughly **constant edge** rather than a constant absolute equity.

Watch `vpip_adapts_to_style` and the entry-range checks — [P4] has been measured for many versions
against a feature that could not do its job, so its numbers may move for reasons unrelated to
opponent modelling.

Note also that two **live** fixes landed the same day (`front_colors` gated on committed chips;
`is_active` monotonic within a hand — see `versions/v42_liveFixes/SPECS.md`). Both reduce the field
size the live model is told about, and the entry curve is steep exactly there, so V43's live
behaviour should already have loosened without any weight change. **Do not attribute that to V44.**

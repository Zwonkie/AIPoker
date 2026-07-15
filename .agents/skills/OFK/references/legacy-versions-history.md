# Legacy Versions History (V4–V10) — durable lessons only

**Date Recorded**: 2026-07-15 (consolidated from per-version spec/sensitivity/health files during OFK cleanup)

## Context
V4–V10 are superseded architectures (older contracts, mostly 3-action, pre-regret-matching). Their
point-in-time specs, sweep tables, and health snapshots were deleted; this file keeps only the
insights and recurring bug-patterns that still inform the current line (V12→V15). For V11 see the
retained standalone files `V11/model-specifications.md` and `V11/issues-and-fixes.md` (the bridge into
the current architecture). For the current line see `V12/validated-findings.md` and `V13`–`V16` specs.

## Guidelines / durable lessons per version

- **V4** — Original 31-dim causal-transformer EV model. **LESSON (still central):** off-policy
  EV-extrapolation on unseen "garbage" hands hallucinated +EV for trash → the recurring failure that
  reappears in V7 (72o raises), V11 ("raise-everything"), and is what V15's **counterfactual
  per-action policy targets** exist to kill. Proposed mitigations of the era: CQL, data aug, policy
  masking, ε-exploration.

- **V5** — Hybrid preflop exploration split (5/50/45) + RTX-4080 CUDA/AMP/pin-memory throughput. The
  split was abandoned; the perf tips are generic PyTorch. No architecture-specific lesson survives.

- **V6** — First **6-action discretized bet-size heads** (direct ancestor of V15's action space) and
  **showdown-equity baseline subtraction** (ancestor of range-aware equity). **LESSON:** the V5/V6
  flat/collapsed Q-values root-caused to **sequence-dilution / attention-collapse** (padding present
  only at seq step-19 → ~8000× gradient dilution) — recurs in V10 mode-collapse and V11 left-padding
  fixes. Also the **"positive winrate masked by hardcoded simulator fallbacks"** trap.

- **V7** — Remediation: `key_padding_mask`, **dense all-action EV loss**, closed-loop postflop model
  play, chronological/causal packing. **LESSON:** all-action/dense loss is the fix for untaken-head
  blindness; and two **target-EV formula bugs** surfaced — (a) opponent `p_fold` wrongly tied to
  Hero's equity, (b) min-raise sizing ignores opponent pot-odds / sunk cost. These are the seed of the
  V9 river-bluff collapse and later street-aware fold-equity work.

- **V8** — NN personality league (Maniac/Nit/Sticky) + **stack-size curriculum** (short-stack focus →
  survives into V15's DoN short-stack training) + **corrected equity-independent preflop fold
  formula**. The league finally folded 72o and diverged by personality. Keep: curriculum +
  corrected-fold-formula lesson; league detail superseded.

- **V9** — Opponent profiling window widened 10–20 → 50 hands (not load-bearing now). **LESSON
  (canonical bug-pattern): terminal-state value inflation** — a flat `p_fold=0.70` applied to *all*
  streets over-inflated **river** fold-equity → model shoved air. Fix = **street-aware fold equity**
  (informs current MC targets / DoN street handling).

- **V10** — De-rigidified opponent bots + mandated **collapse-detection telemetry** (bluff matrix,
  showdown vs non-showdown, fold-equity-by-street, action entropy) — a reusable diagnostic checklist.
  Replaced analytical target-EV with `_calculate_mc_target_evs` (true MC GTO). **LESSON:** its mode
  collapse (entropy→0, ~71% unconditional-call VPIP) root-caused to **PID-controller EV penalties
  distorting the true Q-targets** → the principle *"don't distort true MC targets to force behavior,"*
  which became V12's framing of the target clip as **variance control, not a bias hack**.

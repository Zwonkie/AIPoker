# V20_preflopEq — range-aware equity calibration + field-size-aware features

**Date Recorded**: 2026-07-17
**Related Files**: [versions/v20_preflopEq/SPECS.md](file:///c:/REPO/Antigravity/AIPoker/versions/v20_preflopEq/SPECS.md) (full detail), `versions/v20_preflopEq/core/contract.py`, `versions/v20_preflopEq/self_play/simulator.py`

## Context

Clone of V20. Two real calibration bugs fixed in the shared training/live equity function, two
new engineered features added, plus a serious pre-existing bug found and fixed while implementing.
`context_dim` 35→37, `contract_version` 4→5.

## Guidelines / Summary

- **Finding 1** (live-only): unknown HUD-color opponents were silently dropped from the live
  equity call instead of mapped to 'Yellow' — see [[known-shortcomings-backlog]] RESOLVED-4.
- **Finding 2** (shared train+live): hero's range-aware equity gave every active opponent the
  identical flat VPIP-fold-roll regardless of the simulator's own correct action-order ground
  truth. Fixed with a front (already acted, guaranteed in) vs after (still to act, normal roll)
  split — quantified as a 31-point equity swing in one real multiway spot (0.496 flat → 0.185
  corrected). Live wiring uses a position-only approximation (no real per-seat action-state) — see
  [[known-shortcomings-backlog]] OPP-4.
- **Bug found while implementing**: `train.py::vectorize_hand_samples` never received V20's own
  `/100,/250` rescale — see [[known-shortcomings-backlog]] RESOLVED-3.
- **Two new features**: `equity_edge` (equity's edge over field-size fair share) and
  `hand_strength` (field-independent card quality, `preflop_equities.csv`-backed preflop, cheap MC
  postflop). Both confirmed load-bearing via `model_verify`'s new sensitivity checks.
- **model_verify --full @ 75k**: 11 PASS/1 WARN/1 FAIL/1 SKIP. `vpip_adapts_to_style` PASS
  (short +6.6pt, deep +7.1pt). `bb100_vs_standard_fields` PASS, positive all 4 fields.
  `beats_frozen_predecessor` SKIP (context_dim 35 vs V20's own 35... no, mismatch is width 37 vs
  V20's 35 — see [[known-shortcomings-backlog]] VAL-2). `deep_stack_ood_guard` FAIL,
  `free_check_low_fold` WARN — same long-standing soft spots, see backlog.
- Also found+fixed a real bug in `tools/model_verify/scenarios.py` itself (wrong money-feature
  scale fed to V20's FAST checks since it shipped) — see [[known-shortcomings-backlog]]
  RESOLVED-2.
- Deployed live 2026-07-17, then SUPERSEDED by V20_preflopEq_AI the same day (opponent-pool
  experiment, see `V20_preflopEq_AI/specs.md`). V20_preflopEq's own checkpoints
  (`main_hands25089.pth`, `main_hands50092.pth`, `expert_main.pth`) are reused as frozen
  NN-opponent inputs for that next version.

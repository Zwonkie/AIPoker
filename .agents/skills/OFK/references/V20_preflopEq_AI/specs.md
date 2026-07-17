# V20_preflopEq_AI — opponent-pool experiment (NN diversity vs. shove-preference)

**Date Recorded**: 2026-07-17
**Related Files**: [versions/v20_preflopEq_AI/SPECS.md](file:///c:/REPO/Antigravity/AIPoker/versions/v20_preflopEq_AI/SPECS.md) (full detail), `versions/v20_preflopEq_AI/self_play/config.yaml` (opponent pool), `core/models/v20_preflopEq_AI_engine.py`

## Context

Clone of V20_preflopEq — IDENTICAL architecture/tensor schema (context_dim=37,
contract_version=5), training-RECIPE-only experiment. Motivated by a live-session finding that the
model shoves all-in very readily (see [[known-shortcomings-backlog]] BET-1) — traced to
`opponent_bots.py`'s heuristic bots being price-insensitive once a hand clears their value
threshold. This version tested whether shifting the opponent pool toward real NN opponents (a
lagged self-play mirror + V20_preflopEq's own 25k/50k checkpoints, replacing most of the heuristic
weight) would change that.

## Guidelines / Summary

- **Pool**: `past` (lagged self-play, 0.25), `maniac`-slot carrying `frozen_50k.pth` (0.20),
  `fish`-slot carrying `frozen_25k.pth` (0.15), `tag` heuristic (0.25), `nit` heuristic (0.15).
  Trained 150k hands fresh (not warm-started — isolates the pool as the one variable).
- **Result: the hypothesis did NOT pan out.** `action_diversity` showed a promising signal at 35k
  (5/21 raise-bucket argmax cells) but fully faded by 140k/150k (down to 1/21, allin at its
  highest share of the whole run). `deep_stack_ood_guard` still fails. **Real negative result,
  not a wasted run** — narrows the fix to the opponent RESPONSE FUNCTION (see backlog BET-1), not
  pool composition. Don't re-attempt "just add more NN diversity" without changing the
  opponent_bots.py value-branch price-sensitivity first.
- **But the model is a clear overall improvement over its parent.** For the first time in this
  lineage, `beats_frozen_predecessor` actually RAN (not SKIP) — same architecture as V20_preflopEq
  (`frozen_v20_preflopEq.pth` loads cleanly), and PASSED at +53.5 BB/100 vs a field including the
  frozen parent. Beats V20_preflopEq's own `bb100_vs_standard_fields` baseline in all 4 fields
  (e.g. tight_deep +65.1 vs +61.1, loose_short +31.1 vs +16.8) and shows stronger
  `vpip_adapts_to_style` deltas (short +11.5pt vs +6.6pt, deep +9.6pt vs +7.1pt) — see
  [[known-shortcomings-backlog]] RESOLVED-1.
- **model_verify --full @ 150k**: 12 PASS/1 WARN/1 FAIL/0 SKIP. Full report:
  `tools/model_verify/results/v20_preflopEq_AI_report.html`.
- Deployed live for user testing 2026-07-17 (`Herocules (v20_preflopEq_AI)`, dropdown default).
  V20_preflopEq and V20 both remain fully intact in `core/decision.py`'s registry as rollbacks.
- **Next step under discussion**: go directly at `opponent_bots.py`'s value-branch price
  insensitivity (make continuation probability decay with bet size even above the value
  threshold) rather than iterating further on pool composition.

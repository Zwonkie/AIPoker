# V41 — MILESTONE (kept reference / fallback)

**Tagged milestone 2026-07-21.** V41 is the version that **resolved [BET-3]** — the multiway
passivity that drove a real live complaint ("v29 is really bad — too tight, almost not aggressive")
— while posting the cleanest validation scorecard any version has produced. Keep it as a known-good
reference to roll back to.

- Manifest: `versions/v41/core/manifest.py` has `milestone=True` (schema: `shared/manifest.py`).
- **Do NOT delete `versions/v41/weights/expert_main.pth`** — it is the fallback checkpoint.
- Deployed live 2026-07-21 as `Herocules (v41)` (`core/models/v41_engine.py`,
  `core/decision.py`'s `active_model_name`).
- Contract is **unchanged since V29** (`context_dim=54`, `contract_version=8`), so the whole
  V29 → V40 → V41 chain is checkpoint-compatible and rolling back needs no bridge work.

## Why it earns the tag

**[BET-3] resolved.** `multiway_shortstack_aggression` PASSES outright. 3-way aggression at
eq 0.65 went from **~0.01 (V29) → 0.81 (V41)**, flat from heads-up. V29 collapsed all 6 short-stack
cells; V40 fixed 3; V41 carried the rest.

The root cause was never the network. A single check ended the betting round in the simulator, so
check-behind, check-raise, "checked to me in position", delayed c-bet and BB-option nodes had
**never appeared in a single training sample** (0 of 849 postflop checks in an instrumented
1000-hand run were followed by anyone acting). This is the most valuable lesson in the version:
*a persistent behavioural gap in the model is worth suspecting the data-generating process for
before the architecture.*

**`model_verify --full`: 22 PASS / 5 WARN / 0 FAIL / 0 SKIP.** Cleanest of any version (V29 was
21/2/0/1) and the first ever with zero skips. Report:
`.agents/skills/OFK/references/V41/model_verify_report.html`.

**First real predecessor comparison since the V18 refactor.** `beats_frozen_predecessor` = +64.3
BB/100 over 4000 hands with frozen V40 seated as an actual `NNOpponent`. Every such result between
V18 and V40 was silently measuring "beats a TAG field" (`sim.past_model` / `sim.disable_past_self`
had been dead attributes since V18 — Fable review finding #4). Caveat per finding #5: SE(BB/100)
≈ ±13–19 at this sample size, so the margin is real but the point estimate is not precise.

Provenance: V41 is the second of two versions implementing the 2026-07-20 Fable full-stack review
of V29 — V40 carried findings #1/#2/#3 (the [BET-3] package), V41 carries #7/#8/#9/#10/#11 (the
simulation-realism package). Full per-finding status:
`.agents/skills/OFK/references/fable-review-resolution-log.md`.

## Known limitations carried forward — do NOT read this tag as "solved"

- **`nash_pushfold_vs_chart` regressed 83% → 78%** and V41 did **not** fix it. Introduced by V40,
  with the error direction *flipped*: V29 folded where Nash shoves; V40/V41 shove where Nash folds,
  with weak suited trash at 5bb (94s, 93s, 92s, 83s). This is the clearest open defect in the
  milestone and wants its own investigation via the V30/[VAL-1] tooling.
- **Multiway is improved, not solved.** Training's postflop average is still **1.96 active
  players**, and the eq-0.55 multiway cells still soften (0.81 → ~0.68).
- **Every opponent raise is still exactly 0.75 pot** (review finding #6) — the last open member of
  the [BET-3] bundle. Hero has never faced an open-jam, overbet or min-raise. This is the natural
  next version, since it is what the [OPP-2] raise features and every fold-vs-raise response are
  ultimately calibrated against.
- **[OPP-8] untouched**: `opponent_style_sweep` (0.004) and `allin_exploits_opponent_foldiness`
  (0.000) — the model's aggression barely differs by opponent archetype.
- `free_check_low_fold` and `pot_type_sensitivity` remain standing WARNs.

Precedent for this tag: [../v13/MILESTONE.md](../v13/MILESTONE.md) (V13, the first live-viable
foundation). A new version should be a NEW folder copied from here; this folder stays frozen as the
reference.

**To also mark it in git (optional, recommended):**
```
git tag -a v41-milestone -m "V41: resolved [BET-3] multiway passivity; 22/5/0/0 model_verify"
```

See: [SPECS.md](SPECS.md) | [../v40/SPECS.md](../v40/SPECS.md) (the [BET-3] package it builds on) |
[../v29/SPECS.md](../v29/SPECS.md) (contract, [OPP-2], critic-consistency filter)

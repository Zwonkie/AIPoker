# V48 — MILESTONE (kept reference / fallback)

**Tagged milestone 2026-07-24.** V48 is the **table-geometry foundation** — the version that made
the simulator deal, position and price hands like a real short-handed table, and it is the base
every subsequent seat-aware version (including V50) is copied from. The best model to date; keep it
as a known-good reference to roll back to.

- Manifest: `versions/v48/core/manifest.py` has `milestone=True` (schema: `shared/manifest.py`).
- **Do NOT delete `versions/v48/weights/expert_main.pth`** — it is the fallback checkpoint AND
  V50's frozen head-to-head predecessor (`versions/v50/weights/frozen_v48.pth`).
- Deployed live 2026-07-23 as `Herocules (v48)` (`core/models/v48_engine.py`,
  `core/decision.py`'s `active_model_name`); now the one-line rollback behind `Herocules (v50)`.
- Contract is **unchanged since V44** (`context_dim=54`, `contract_version=9`), so the whole
  V44 → V47 → V48 → V50 chain is checkpoint-compatible and rolling back needs no bridge work.

## Why it earns the tag

**The table geometry became real.** V48 is a V47 clone with a four-part realism package:

- **Change 0 — generalized chip-identity collapse.** Raise buckets that resolve to the same slider
  chips are collapsed (serve mirror gated on the engine's `collapse_aliased_buckets = True`),
  superseding V47's all-in-only flag — a proper train/serve invariant pair.
- **Change 1 — true N-handed dealing.** 3–6 seats with ring-relative positions, replacing the
  fixed-geometry deal. This is the change the whole 3–6 seat line depends on.
- **Change 1b — realistic opponents.** Raise repertoires and the pool mixture were **fitted from
  the 99-player bet365 hand-history corpus**, not hand-tuned.
- **Change 2 — measured seat × depth joint curriculum** (the DoN life-cycle), the direct ancestor
  of V50's wider raw-empirical mix.

**`model_verify --full`: 20 PASS / 9 WARN / 1 FAIL.** Headline wins:

- **`opponent_style_sweep` 0.027 → 0.105 (PASS)** — undoes V47's flattening; the model's aggression
  once again differs by opponent archetype ([OPP-8], a standing weakness since the line began).
- **`nash3_btn_jam` 79% (PASS)** — the **first** version measured on a 3-max Nash axis at all.
- **`vpip_adapts_to_style` held** (+10.2 short / +7.7 deep) — the [P4] style-adaptation win from
  V44 survived the geometry rewrite.
- `table_size` spread 0.015 → 0.027 (partial — see limitations).

Report: `.agents/skills/OFK/references/V48/model_verify_report.html`.

## Known limitations carried forward — do NOT read this tag as "solved"

- **`deep_stack_ood_guard` FAIL** (known at deploy). `eq0.55@15bb` picks ALL-IN argmax 0.34 — a
  regression vs V47's 0-FAIL card. **[STACK-1] reopened.** Root cause identified as a **seat-count
  coverage hole** (the measured curriculum starved heads-up/3-handed), not depth — which is exactly
  what **V50** was built to widen. This is the clearest open defect in the milestone.
- **Head-to-head vs frozen V47 was PARITY, not a decisive win.** Solo 8k-hand/axis tie-breaker:
  6-handed −4.1 ±29.8 (parity), DoN mix +13.7 ±31.2 (positive lean, CI includes 0). Same verdict
  class V47 itself deployed on — deployed on the geometry improvements and 0 net regression, not a
  proven BB/100 edge. (LESSON logged: never run two model_verify batteries concurrently — the
  concurrent-load runs gave contradictory ±50 swings before the solo tie-breaker settled it.)
- **`nash_pushfold` regression** (83% → 78%, inherited via V40/V41→…) is still carried, not fixed.

Precedent for this tag: [../v41/MILESTONE.md](../v41/MILESTONE.md) (resolved [BET-3]) and
[../v13/MILESTONE.md](../v13/MILESTONE.md) (first live-viable foundation). A new version should be a
NEW folder copied from here (V50 already is); this folder stays frozen as the reference.

**To also mark it in git (optional, recommended):**
```
git tag -a v48-milestone -m "V48: table-geometry foundation; true N-handed dealing; 20/9/0/1 model_verify"
```

See: [SPECS.md](SPECS.md) | [../v47/SPECS.md](../v47/SPECS.md) (the clone base) |
[../v50/SPECS.md](../v50/SPECS.md) (the curriculum retrain that widens V48's seat coverage)

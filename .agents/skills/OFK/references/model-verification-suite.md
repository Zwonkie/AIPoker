# Model Verification Suite (tools/model_verify)

**Date Recorded**: 2026-07-15
**Related Files**:
*   [scenarios.py](file:///c:/REPO/Antigravity/AIPoker/tools/model_verify/scenarios.py)
*   [checks.py](file:///c:/REPO/Antigravity/AIPoker/tools/model_verify/checks.py)
*   [run.py](file:///c:/REPO/Antigravity/AIPoker/tools/model_verify/run.py)
*   [render_report.py](file:///c:/REPO/Antigravity/AIPoker/tools/model_verify/render_report.py)
*   [shared/registry.py](file:///c:/REPO/Antigravity/AIPoker/shared/registry.py)

## Context

Before this tool existed, model verification was a set of one-off scripts written to a
session-scratchpad each time a question came up (`playtest_style.py`, `eval_shortstack.py`,
`eval_v15.py`, ad hoc spot tests). The scratchpad doesn't survive between sessions, so the exact
scripts that caught the P4 VPIP-flatness bug and the V14 deep-stack-jam incident were **gone** by
the next session — nothing accumulated, and every version got re-verified from scratch with
whatever the agent happened to reinvent that day. `tools/model_verify` replaces that with a
persistent, committed, GROWING curriculum: every diagnosed issue becomes a permanent regression
check, appended once and never re-invented. Supersedes the old
[Model Testing Suite](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/model-testing-suite.md)
doc (V4-V6 era, predates the 6-action contract, manual/ad hoc, not committed as code) — that file
is kept for historical scenario-design ideas only.

## What it covers

Two speed tiers, both version-agnostic — loads ANY version via `shared/registry.py`
(`get_manifest`/`load_model`), no per-version code needed:

- **FAST checks** (`checks.py:FAST_CHECKS`, run every invocation, milliseconds each): single
  synthetic-scenario forward passes built directly from the 35-feature context layout
  (`scenarios.py:build_ctx`), bypassing the need to construct a full `BoardState` graph.
  - `equity_ablation_monotonic` — P(fold) falls / P(aggressive) rises as equity rises.
  - `free_check_low_fold` — raw policy shouldn't want to fold a free option (WARN, not FAIL — it's
    a real, quantified characteristic already covered by `core/decision.py`'s free-check mask, not
    a deploy blocker: V15 carries up to 0.35 raw mass here and ships fine).
  - `air_folds_mostly` / `nuts_aggressive_mostly` — spot-test baselines from the old suite,
    ported forward.
  - `deep_stack_ood_guard` — **regression test for the V14 P0 incident**: reproduces the exact
    board conditions (43% equity, 20bb, single modest bet) that caused hero to jam K9o live.
    Sweeps the marginal-equity x 15-40bb x single-modest-bet neighborhood; FAILs if any cell has
    ALL-IN as argmax.
  - `short_stack_polarization` — tracks [P3] (preflop flattening) without gating on it.
  - `action_diversity` — guards the V11 raise-/call-everything collapse. Tests for ONE action
    monopolizing >85% of an equity x stack grid, NOT "few distinct argmax winners" — the latter
    false-flagged V15's known-accepted "no middle gear" bimodal sizing ([P1],
    `versions/v15/SPECS.md`) as a collapse. Get this distinction right if extending it.
  - `no_nan_or_crash` — regression guard for the NaN-inference crash fixed by removing
    `key_padding_mask` (commit 55a1bc9).
- **SLOW checks** (`checks.py:SLOW_CHECKS`, `--full` only, real simulated hands via the version's
  own `self_play.simulator`, minutes each):
  - `vpip_adapts_to_style` — the exact [P4] regression gate (hero VPIP must move >=5pts with
    opponent tightness at both short and deep stacks).
  - `bb100_vs_standard_fields` — winrate vs loose/tight fields, diffed against
    `tools/model_verify/baselines.json` (WARN on a >15 BB/100 regression from the last accepted
    number, not just a raw report).
  - `beats_frozen_predecessor` — every version must beat a frozen snapshot of its immediate
    predecessor (the `frozen_v{N-1}.pth` pattern used since V15); auto-discovers
    `versions/<id>/weights/frozen_*.pth`.

## Running it

```
.venv/Scripts/python.exe -m tools.model_verify.run --version v16
.venv/Scripts/python.exe -m tools.model_verify.run --version v16 --full
.venv/Scripts/python.exe -m tools.model_verify.run --version v16 --full --update-baseline
```

Prints a PASS/WARN/FAIL/SKIP table, exits non-zero on any FAIL (usable as a gate later). Every run
also writes the full raw per-scenario data to
`tools/model_verify/results/<version>__<weights>.json` — feed that into
`tools/model_verify/render_report.py <json> --out <html>` to get an interactive HTML report
(heatmaps/line charts/hover tooltips over the raw sweep data, dataviz-skill palette, dark/light
themed) instead of just the printed summary. NOT auto-wired into `train.py` (deliberate,
2026-07-15 decision) — run it as a manual follow-up step after a training run for now.

## How to extend the curriculum

Append a new function + a tuple entry to `FAST_CHECKS` or `SLOW_CHECKS` in `checks.py`. That's the
entire extension surface — `run.py` and `render_report.py` need zero changes. Every check returns a
`CheckResult(status, detail, data=...)`; populate `data` with the raw per-scenario records (not just
the aggregate) so it shows up in the JSON dump and can be charted later — this is what let the
V15 deep-stack finding below get visualized instead of just reported as one FAIL line.

## Durable lessons from calibrating it (2026-07-15)

A brand-new check suite needs calibrating against a KNOWN-GOOD baseline before it's trusted — two
real bugs were caught this way, running the fresh tool against the already-validated, already-live
V15 checkpoint and treating any disagreement as "my check is probably wrong" first, not "the deployed
model is suddenly broken":

1. **Substring-match trap**: `'all' in 'call'.lower()` is `True` in Python — a naive
   `needle in action_name` search for the all-in action silently matched CALL first (since it
   iterates in the action tuple's order and CALL comes before ALLIN). This corrupted both the
   "aggressive mass" sum and the deep-stack OOD guard's all-in detection until fixed. **Fix**: match
   on the WHOLE action name (`k.lower() == canonical`), never substring — see `checks.py:_find`.
2. **Over-strict diversity threshold**: "at least N distinct actions must win argmax somewhere"
   flagged V15 (deployed, known-good) as a collapse, because its raise buckets legitimately never
   peak above fold/call/allin in a specific price context — that's the pre-existing, accepted
   "no middle gear" characteristic, not a new bug. **Fix**: test for single-action monopolization
   of the whole grid (>85% share) instead of counting distinct winners.

## A real finding this tool surfaced (not a suite bug)

After the above fixes, `deep_stack_ood_guard` genuinely FAILS on the live V15 checkpoint: at the
exact original-incident conditions (43% equity, 20bb, single modest bet), V15's argmax is STILL
ALL-IN (0.37 probability), climbing smoothly from a 0.24-probability all-in argmax even at a
clearly-losing 35% equity. V15's aggregate BB/100 evals looked like the deep-stack OOD was fixed —
this decision-level probe suggests that conclusion may have been an artifact of aggregate stats
(e.g. exploiting weak opponents) rather than the underlying decision actually being corrected. Full
detail and the V16 implication in `versions/v16/SPECS.md` ("[P0-recheck]").

**Second real finding (2026-07-15, `v16_foldregret`):** this tool caught a genuine deploy-blocking
regression that the training dashboard alone would have missed. A single-variable experiment (fold-
relative regret-matching baseline, see `versions/v16_foldregret/SPECS.md`) fixed its target problem
cleanly (air/draws Fold% up sharply, held stable, without collapsing profitable equity tiers) — the
training telemetry alone looked like an unambiguous win. `model_verify --full`'s
`vpip_adapts_to_style` check, run against both the new checkpoint AND a freshly-regenerated V16
baseline for direct comparison, showed the deep-stack style-adaptation delta dropped from V16's own
+8.4pts (PASS) to +2.0pts (FAIL) — a real side effect invisible in the training dashboard's own
metrics (which don't segment by opponent style at all). This is exactly the failure mode the suite
exists to catch: a fix that visibly helps the thing it targeted while quietly breaking something the
training loop never measures. Lesson reinforced: always run the FULL suite (not just the fast
checks) against BOTH the new checkpoint and a same-day baseline re-run before treating a training-
dashboard improvement as a deploy-ready result.

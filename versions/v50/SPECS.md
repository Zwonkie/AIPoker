# V50 — wider seat×depth curriculum (250k hands)

**Base**: clone of `versions/v48` (fresh weights, no `--resume_path`, per [VAL-5]).
**Contract**: UNCHANGED — `context_dim=54`, `contract_version=9` (identical to V48/V47/V44).
**Status**: scaffolded 2026-07-23, NOT TRAINED. V48 remains the live model and the rollback.

## Scope — curriculum only

V50 changes **one thing** versus V48: `table_stack_joint_mix` in `self_play/config.yaml`, plus
`target_hands` 100k → 250k. No simulator, contract, model, target, or veto edits — the slice is
byte-identical to V48 except the mix and hand count. The live bridge, `live_features()`, and the
engine carry over unchanged; a `v50_engine.py` is a declaration-only clone at deploy time.

## Why — V48's coverage holes were on the seat axis

V48 introduced seat-count *conditioning* (Change 1 ring geometry) but its measured joint mix
starved the low-seat-count end (HU 0%, 3-handed 1%), so the model extrapolates a "fewer players →
jam harder" trend into regimes it never trained — the reopened `deep_stack_ood_guard` failure
([STACK-1], V48). The depth axis was already fine (43% of V48 mass sat at 30–100bb). So V50 rebuilds
the mix from the **full** hero corpus rather than a 4k snapshot, and restricts it to the real DoN
range with even coverage across it.

## The curriculum — raw empirical, cut to the DoN-real range

Built from **4,492 hero-dealt hands** across the whole hand store (`live2/historydb`, all
sessions), binned Seats × hero-BB-depth, then cut per owner direction (2026-07-23):

- **Seats: keep 3–6.** Drop `< 2 opponents` (heads-up + the seat-1 `"?"`-tournament artifact);
  clamp at 6-max (drop the 7–10-handed rebuy/MTT hands). Floor = **hero + 2 opponents**.
- **Depth: keep 2–100bb.** Drop `< 2bb`; fold the `100+` tail into `60–100` (the `STACK_CEIL_BB=100`
  clamp represents it there anyway).

4,010 hands survive; each surviving cell is its raw fraction of that total — **not** a parametric
fit. The seat→depth correlation is inherent to the data (4-handed bubble short, 6-handed early
game deep). Marginals: seats **3:2.9% · 4:22.2% · 5:26.1% · 6:48.8%**; depth **2-5:6.8 · 5-8:8.3 ·
8-14:14.9 · 14-30:29.9 · 30-60:25.5 · 60-100:14.6**.

**Known minor:** the raw 3-handed row is Turbo/Twister-sourced (mode 14–30, some 30–60/60–100 mass)
rather than a short DoN endgame — but at 2.9% of hands (<1% of total in its deep cells) the effect
is negligible. Left raw by owner call; a short-endgame override for seat-3 is a one-line change.

## 250k hands

Raised from V48's 100k (owner-authorized): a wider distribution needs more hands to keep each
region sampled. **Fresh** run (no resume, [VAL-5]). Checkpoint every 20k; run `model_verify --full`
at **100k / 150k / 200k / 250k** and pick the best — V24/V25 showed more exposure can *widen* the
Q-gap, so 250k is not assumed to be the winner.

## `deep_stack_ood_guard` — OOD by construction, not a gate to chase

The check probes a **heads-up** (`num_active_opp=1`), 15–40bb, single-modest-bet spot. V50 trains
**zero** heads-up (floor = 3 seats), so this gate is out-of-distribution by construction and is
**expected to stay soft/FAIL**. This is a deliberate trade: DoN never reaches heads-up, and the
real short-stack money is the **4-handed bubble**, now densely trained (22%). Treat `deep_stack_ood`
as informational for a DoN-scoped model, not a deploy blocker. If a live HU/3-handed spot ever
matters, handle it with a serve-side short-handed jam guard rather than distorting the curriculum.
(Training 3-handed at ~2.9% gives the model a nearer low-seat anchor than V48's 4-handed floor, so
the extrapolation to HU is one step shorter than before.)

## Acceptance gates

- **Hold V48's wins**: geometry (`nash3_btn_jam` ~79%, `table_size_sweep` load-bearing),
  `opponent_style` (~0.105), `allin_vs_nextbest_qgap` negative worst-cells, action diversity.
- **Improve / watch**: 4-handed short-stack behavior; `nash_bbcall`/`nash3_bb_call` (finding-A,
  BB too tight calling jams @5–6bb) — the denser short bands should help.
- **Head-to-head**: mirrored-deal paired vs `frozen_v48.pth` (staged) — gate on the 6-handed pair.
- **Expected soft**: `deep_stack_ood_guard` (see above).

## Build log

- 2026-07-23: cloned from `versions/v48`; weights cleared (V48 `expert_main.pth` staged as
  `weights/frozen_v48.pth` = head-to-head predecessor; `tree_opponents/` kept); all
  `versions.v48` imports rewritten to `versions.v50`; manifest `version_id=v50`, contract
  unchanged; config `version: v50`, `target_hands: 250000`, new `table_stack_joint_mix`.

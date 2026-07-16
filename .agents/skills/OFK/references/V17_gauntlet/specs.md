# V17_gauntlet — widened frozen-opponent pool (overnight run)

## CORRECTION (2026-07-16, found while planning the V18 opponent-architecture refactor)

**Frozen V16 was NEVER actually used in the `tag` seat during this run.** The opponent-assignment
code (`simulator.py`, the `else: # 'tag'` branch) had `opp_model = self.tag_model` immediately
followed by a leftover, unintentional `opp_model = None` on the next line -- a stray line left
behind from the pre-fix code that silently nullified the wiring. The `tag` seat fell back to the
scripted TAG heuristic bot for the entire 200k-hand run, exactly as it did before the "fix" below
was written. **This also means the "temperature-sensitivity" explanation given below for why
`Tag Bot` showed 12.6% VPIP (much tighter than V16's own documented 58-66%) was wrong** -- the real
reason is simpler: it wasn't V16 playing at all.

Net effect on this run: it actually trained against **4 real components, not 5** -- fish, maniac
(heuristic), frozen-V15 (`nit`), lagged-self (`past`), and TAG heuristic (not frozen V16). The
`nit`/`past` wiring and the forcing-bypass fix were both genuinely correct and did work as
documented; only the `tag` slot was silently broken. This does NOT invalidate the trained checkpoint
or its `model_verify` results below -- those are real measurements of the actual weights -- but the
"3 skilled opponents" framing throughout this doc is overstated; it was 2. Not retroactively fixed
in this checkpoint (would require retraining); the bug is fixed going forward in `versions/v18`'s
opponent-architecture refactor, which structurally prevents this class of silent-nullification bug
(no more elif chains with room for a stray leftover line). See `versions/v18/SPECS.md`.

**Purpose (2026-07-16):** clone V17 (deployed live, see `versions/v17/SPECS.md`), same actor-critic
training recipe, ONE variable: the opponent pool. Pulls the `versions/v18/SPECS.md` backlog item
"widen the frozen-opponent pool" forward. Every version through V17 trained against scripted
heuristics + a single frozen predecessor; this run adds THREE genuinely skilled frozen/lagged
opponents so the hero faces real learned play, not just heuristic archetypes, in 60% of its
opponent seats.

## Opponent pool

`pool: ["fish", "maniac", "nit", "tag", "past"]`, `weights: [0.25, 0.15, 0.20, 0.20, 0.20]`

- `nit` seat -> **frozen V15** (`frozen_v15.pth`)
- `tag` seat -> **frozen V16** (`frozen_v16.pth`) -- NEW model-loading wiring, this slot was
  previously hardcoded to always use the pure heuristic bot (no model option existed for it)
- `past` seat -> **TRUE lagged self-play mirror** of v17_gauntlet's own training history
  (`freeze_past_self: false`) -- NOT a static frozen file, saves a fresh snapshot every 5k hands
- `fish` / `maniac` stay pure scripted heuristics -- continued non-NN diversity

**Prerequisite fix required and implemented:** `maniac`/`nit`/`fish` style slots have
ACTION-FORCING logic in `_opponent_decide` (`simulator.py`) that probabilistically overrides the
seat's actual decision toward a target archetype's stats (e.g. nit: 80% chance to force-fold
whenever realized VPIP>15%, regardless of what the model actually wanted). Appropriate for a
scripted heuristic hitting its target stats; would badly distort a genuine trained network's
judgment. **Forcing is now bypassed whenever a real model is loaded for that seat**
(`opponent.get('model') is not None`) -- scripted heuristic seats are completely unaffected,
forcing still applies to them exactly as before. This is the actual engineering change this run
needed beyond "point a config at some weight files."

`frozen_v15.pth` / `frozen_v16.pth` / `frozen_v17.pth` all copied into
`versions/v17_gauntlet/weights/` (the V17 copy is unused by training -- 'past' uses a true lagged
mirror, not a static file -- but kept as the benchmark file for post-training `model_verify`, see
Caveats below).

## Curriculum

**Deliberately UNCHANGED from V17's validated recipe** (`stack_depth_mix`
`[[5,14,.55],[14,30,.30],[30,50,.15]]`, `disable_extreme_stacks: true`, `disable_focus_rounds: true`,
phase settings identical). The opponent pool is already the one variable this run tests; changing
the curriculum at the same time would confound whether any result is the new field or the new
stack distribution. Reviewed and kept, not changed.

## Other changes

- `equity_sims` reverted 5000->2000 (see `versions/v18/SPECS.md` "MC equity_sims budget" --
  measured 2.46x wall-clock cost for a ~0.3-percentage-point noise reduction, small next to the
  critic's own much more powerful denoising via regression across thousands of hands).
- `target_hands: 200000` (up from V17's 100k test budget) -- full production run.
- `live_players: 6` (6-max, unchanged).

## Pre-launch checks (all passed)

- `overfit_sanity`: noisy on synthetic critic check across repeated runs (same known unseeded-RNG
  variance as every prior version in this line — 2 of 3 runs clean), real targets learnable.
- 200-hand smoke run: config header confirms `Nit seat: FROZEN frozen_v15.pth (unforced)`,
  `Tag seat: FROZEN frozen_v16.pth (unforced)`, `Past-Self seat: enabled (lagged mirror)`, pool/
  weights correct, no load warnings, dashboard prints cleanly (including the "FACING A BET ONLY" /
  Free-column telemetry fix from V17, which carried over in the copy).

## Caveat for the eventual `model_verify` run

`tools/model_verify/run.py`'s `_find_frozen_predecessor` picks the FIRST alphabetically-sorted
`frozen_*.pth` in the weights dir as "the" predecessor for `beats_frozen_predecessor`. This weights
dir has THREE (`frozen_v15.pth`, `frozen_v16.pth`, `frozen_v17.pth`) -- alphabetical sort would pick
`frozen_v15.pth`, not the true immediate parent `frozen_v17.pth`. Not fixed tonight (doesn't block
training). Whoever runs `model_verify --full` on this checkpoint next should either pass
`--weights` pointing explicitly at the right comparison, temporarily move the other two frozen
files aside, or fix `_find_frozen_predecessor` to prefer the highest version number.

## Launch

Fresh (no warm-start), matching every version in this line:
`python -m versions.v17_gauntlet.self_play.train --personality main --num_hands 200000`

## Training COMPLETED (2026-07-16)

200,001 hands, 2h37m, clean exit, zero NaN/crash/traceback across the whole run. Weights saved to
`versions/v17_gauntlet/weights/expert_main.pth`.

**Terminal-tick caveat (same class of artifact seen before in this line):** the very last two
dashboard ticks show Val Loss swinging wildly (0.0046 -> 23.16) — a tiny trailing batch (7 hands,
`batch_hands = min(sim_batch_size, num_hands - hands_done)` shrinks near the target) producing a
noisy single-batch loss read. Cosmetic only, doesn't affect the saved weights. **VPIP, unlike V17
round 1/2's terminal tick, is NOT an artifact here** — checked the full trailing trend (last 15
ticks) and it's smooth and stable at 40-41.5%, no wild swing.

**Final health read:** Action Entropy declined smoothly throughout to a notably low 0.168 (V17 solo
finished at 0.633) — the policy converged more decisively/less-mixed than V17 did alone, plausibly
because a harder opponent pool (real skilled NNs, not just heuristics) rewards more committed play.
Hero VPIP converged to ~40-41% (between V17 solo's ~36-40% and V15's looser 51-66%) — plausible
given the pool mixes V15's genuinely loose character back in via the 'nit' seat. Marginal/Strong/
Nuts equity tiers all clearly profitable (net chips +420k/+560k/+290k) -- no sign of a tightening
or looseness collapse. Hero +46.4 BB/100 cumulative (raw training self-play, not the deployed-temp
eval number -- not directly comparable to `model_verify`'s numbers).

**Labeling quirk, not a bug:** the frozen V15 (`nit` seat) shows 65.5% VPIP in this dashboard --
genuinely consistent with V15's own documented loose-aggressive character against a loose field,
just misleadingly labeled "Nit" (a leftover from the stat-bucket's original heuristic-archetype
naming, unrelated to what's actually loaded there now).

~~Frozen V16 (`tag` seat) shows a notably tighter 12.6% VPIP here than its own solo `model_verify`
read (58-66%) -- likely because opponent NNs are queried at whatever `policy_temperature` the
simulator defaults to during training rollout (not the sharpened temp=0.5 deploy setting V16's own
eval used), which is a known, already-documented temperature-sensitivity of this whole codebase,
not new or alarming.~~ **WRONG -- see the CORRECTION at the top of this doc.** 12.6% VPIP is the
TAG heuristic bot's real number; frozen V16 was never queried in this run at all.

**Tooling fix made while evaluating this checkpoint:** `tools/model_verify/run.py`'s
`_find_frozen_predecessor` picked the alphabetically-FIRST `frozen_*.pth` file, which for this
version's THREE frozen files (v15/v16/v17) grabbed the oldest (v15) instead of the true immediate
parent (v17). Fixed to pick the highest parsed version number instead -- a general fix, not
specific to this run, since "oldest predecessor" is almost never the right default comparison.

**`model_verify --full`: 10 PASS, 1 WARN, 1 FAIL** (`results/v17_gauntlet__expert_main.pth.json`):
- `deep_stack_ood_guard` FAIL (eq=0.55, stack=40bb -> ALL-IN argmax @ 0.39) -- same pre-existing
  carried defect every version in this line fails (V15/V16/foldregret/V17). Not new.
- `vpip_adapts_to_style` PASS, and notably stronger than V17 alone: short +10.3pt (V17: +9.7pt),
  deep **+12.3pt** (V17: +5.8pt -- more than doubled).
- `bb100_vs_standard_fields`: loose_short +32.5 (V17 +28.9), loose_deep +69.8 (V17 +90.3 -- came
  down but still strongly positive, reads as a more balanced policy not a regression),
  tight_short +26.8 (V17 +18.4), tight_deep +35.4 (V17 +32.6). Three of four fields improved.
- `beats_frozen_predecessor`: +84.3 BB/100 vs frozen-V17 (the CORRECT immediate parent, thanks to
  the picker fix above) over 4000 hands -- convincingly beats its own predecessor.
- `air_folds_mostly` still a clean 1.00 -- the core V17 fix holds with the new opponent pool.

**Verdict:** genuine improvement over V17, not just a wider/harder training field for its own sake
-- more balanced field performance and meaningfully better deep-stack style-adaptation, no new
regressions. Reported to user with a deploy recommendation; not deployed yet pending their
go-ahead (same policy as every prior deploy decision in this line).

# V15 — Specs, live observations & build plan

Deferred refinements + the evidence-backed V15 retrain plan. V15 clones V14 (same 6-action
contract + equity-primary architecture); only the sim-table setup + training budget change.

## Size-scaled opponent bluffing (deferred from V14 P1b)

V14 P1b made the bots' CONTINUE/FOLD decision size-aware (fold more to bigger bets, anchored to
pot-odds/MDF, modulated by style — see `versions/v14/self_play/opponent_bots.py`). It intentionally
left the bots' **bluff-RAISE** frequency flat (`bluff_freq · agg`), unchanged from v13.

The refinement (defer until/unless bot bluffing looks too rigid): scale bluff-raise frequency by the
size of the bet faced —

```python
bluff_room = 1.0 - pot_odds          # facing a SMALL bet -> cheap fold equity -> bluff-raise more
if (random.random() < self.current_bluff_freq * bluff_room
        and random.random() < self.current_agg_freq * 1.5):
    return 'raise'
```

Rationale: a bluff-raise is cheaper and folds out more when the bet you're raising is small; it's
worse when facing a big bet. Optional polish; not load-bearing for V15's goals.

---

# Live-deployment observations (V14, 2026-07-14)

Two full DoN boards were recorded and reviewed turn-by-turn from `history/<board_id>/turns.jsonl`
(decoding the actual model-input tensors, not the raw vision read). Findings ranked by cost.

## [P0] Deep-stack OOD — CONFIRMED costly (top priority; addressed by V15)

V14 was trained ONLY at short/DoN depths (`config.yaml fixed_stack_bb: [5, 14]`), validated as a
strong winner there (+15..+23 BB/100), and DEPLOYED live across ALL depths (user decision
2026-07-14: ship + monitor). Monitoring caught the failure:

- **Board `1170810410`, turn 13: hero jammed K9o for a full 20bb all-in into a single limper**
  (43% equity, heads-up). ALL-IN was the model's ARGMAX (0.42) — the model genuinely mis-valuing
  the spot, NOT sampler noise. Next recorded turn (t14) the stack is 1.6bb: the jam almost certainly
  cratered the stack.
- Root cause: at 20bb the model applies short-stack push/fold logic (all it ever saw) where it is
  badly wrong — you never stack off 20bb K9o vs a limp.

Fixed by the V15 build plan below (widen the training stack range so one model covers all depths).

## [P1] Bimodal deep-stack sizing — "no middle gear"

Above ~15bb the raise sizing is degenerate: the pot-fraction buckets floor to a min-raise, so deep
the hero only ever **min-raises (meaninglessly small) OR jams** — nothing in between. Live evidence
(board `1170810410`): t2 min-raised 2bb at 40bb; t5/t6 tiny bets at 20–24bb; t13 the 20bb jam. This
is the short-only training expressed through the pot-fraction action space at depth. SUBSUMED by the
V15 widen-stack-range retrain; the min-raise floor logic itself (`_raise_size_for_fraction`) is
correct for short stacks and should NOT change — the fix is training coverage, not the formula.

---

# V15 BUILD PLAN — DECIDED (2026-07-14)

User decisions after reviewing V14 live play. Foundation = clone V14 (same 6-action contract/arch);
change only the sim-table setup + training budget below.

**STATUS: TRAINED + VALIDATED — PASSES (2026-07-15). Deploy pending user review (v14 stays live).**
200k hands / 2h46m, fresh init, weights `versions/v15/weights/expert_main.pth`. Pre-launch checks all
passed (overfit_sanity 0.83bb/KL0.0021; stack mix 55/29/16%; frozen-V14 plays; all 6 actions used).

DEPLOYED-POLICY EVAL (temp=0.5):
| Field                 | SHORT 5-14bb | DEEP 30-50bb |
|-----------------------|-------------:|-------------:|
| vs Loose (fish/tag/nit) | +15.0        | +27.6        |
| vs Tight (nit/tag)      | +16.0        | **-4.2**     |
| vs FROZEN-V14 (bench)   | +17.9        | +53.4        |
| Full DoN mix vs loose   | +13.3        |              |
| Full DoN mix vs frozen  | +35.6        |              |

**DEEP-STACK OOD FIXED** (v14 jammed trash 20bb deep; v15 wins +27.6 deep vs loose, holds short
+15/+16, BEATS frozen-V14 at every depth +18/+53/+36). inspect_policy_vs_target clean (monotonic
equity response, no trash jams, all 6 sizes used). SOFT SPOT (noted, NOT a blocker): ~breakeven deep
vs a TIGHT field (-4.2, within ~1SE of 0), from a looser style than v14 (VPIP ~55% vs ~45%) — likely
learned to punish the 20% frozen-V14 opponent (spews deep). Wins big vs loose/station fields (what the
live tables actually are). DID NOT retrain: re-tuning `policy_tightness_bb` to fix deep-vs-tight risks
regressing the loose-field wins (over-tightening collapsed v14's loose winrate) — a tradeoff for the
user to own.

REMAINING TO DEPLOY (not done autonomously — user decision, v14 stays active meanwhile): wire a
`core/models/v15_engine.py` (copy v14's, load v15 weights) + register in `core/decision.py`; the 6-action
contract is identical so `_v14_size_to_slider`/executor path is reused as-is. Optionally combine with
backlog [P2] stack-scaled live temperature.

## 1. Stack curriculum — DoN-shaped depth band (replaces flat `[5,14]`)

Sample effective depth PER HAND from a fixed DoN-shaped MIXTURE (short-weighted with a real deep
tail), so every batch contains both short and deep spots — more robust than a training-time-gated
curriculum. Keeps v14's short-stack density (where it's already strong) while adding the 15–50bb
coverage it never saw.

```yaml
# versions/v15/self_play/config.yaml  (curriculum)
# each entry [lo, hi, weight]; per hand pick a band by weight, then uniform depth within it
stack_depth_mix:
  - [5, 14, 0.55]     # late-game short (v14's zone — keep dense)
  - [14, 30, 0.30]    # mid-game
  - [30, 50, 0.15]    # early-game deep tail (the missing OOD coverage; e.g. the t13 20bb spot)
```

Implemented via a `_get_starting_stack` extension: accept a `stack_depth_mix` list (weighted band
pick → uniform within). Scalar / `[lo,hi]` forms still work; `fixed_stack_bb` unset when a mix is used.

## 2. Opponent field — stations + FROZEN V14 (no maniac)

Keep the station-heavy heuristic base (best postflop/value data) and ADD **frozen V14** as a static
expert seat so the hero learns sizing/shove dynamics vs a competent, adaptive opponent (not just
heuristics) — and gives a clean benchmark (V15 must beat frozen-V14).

- `versions/v14/weights/expert_main.pth` → `versions/v15/weights/frozen_v14.pth` (STATIC, never
  overwritten during training — NOT a lagged self-snapshot).
- Reuse the existing `past` seat plumbing (`simulator.py` seat 4, style `'past'`, `past_model`) PINNED:
  load `frozen_v14.pth` as `past_model` and DISABLE the 5k-hand snapshot write in `train.py` via a
  `freeze_past_self` flag.
- Frozen-V14 shares the 6-action contract → runs through `_query_model_decide` unchanged.
- Field:
  ```yaml
  opponents:
    pool:    ["fish", "tag", "nit", "past"]   # 'past' = frozen_v14
    weights: [0.40, 0.20, 0.20, 0.20]
    live_players: 6
    disable_past_self: false     # ENABLE the seat...
    freeze_past_self: true       # ...but pin frozen_v14 (no lagged snapshotting)
    disable_focus_rounds: true
  ```
- CAVEAT: frozen-V14 is short-stack-specialised, so it plays the deep (30–50bb) band weakly (same
  OOD). Fine — hero learns to exploit a 20bb-jammer — but don't read deep-stack winrate vs it as truth.

## 3. Hands — 200k (up from 100k)

`target_hands: 200000` (~2 hrs). Wider depth range + stronger opponent + bigger state space.
`mid_flight_diagnostics_interval: 10000`.

## Training recipe decision (2026-07-14)

**Train FRESH (no warm-start from v14).** Warm-starting would entrench v14's bad deep-stack behaviour
(the 20bb trash-jam) which the run would then have to un-learn, plus bootstrap-interaction
subtleties. Fresh + the proven v14 recipe (bootstrap warmup on, exploration on) + wider stack mix +
200k budget is lower-risk and learns the full depth range cleanly. Fallback if fresh underperforms:
warm-start via `--resume_path` from v14 weights with `disable_bootstrap: true`.

## Validation gates before deploying V15
- `overfit_sanity` PASS (pipeline still wired after the sampler/opponent changes).
- `eval`-style pure-policy eval at BOTH short (5–14bb) AND deep (30–50bb) fields — must win at short
  (hold v14's +15..+23) AND not bleed deep.
- `inspect_policy_vs_target` — sizing spreads sanely at depth; no 20–40bb trash jams.
- Beats frozen-V14 head-to-head.
- THEN deploy (replace v14 as active in `core/decision.py`; keep v14 + v13 as fallbacks).

## BACKLOG — MOVED TO `versions/v16/SPECS.md` (2026-07-15)

V15 is TRAINED, VALIDATED, and DEPLOYED LIVE (2026-07-15; active model `Herocules (v15 DoN)` in
`core/decision.py`, `core/models/v15_engine.py`; v14/v13 fallbacks). All deferred items below —
plus the newly-found [P4] VPIP-conformance gap and the size-scaled bluffing polish — are consolidated
into the **V16 roadmap** (`versions/v16/SPECS.md`). Kept here for history:

- **[P2] Live sampler spew — stack-scaled temperature.** `core/decision.py`-only change (no retrain):
  make `LIVE_POLICY_TEMPERATURE` stack-scaled — near-argmax (~0.2 / hard argmax) at `<= ~8bb` where
  short-stack strategy is pure, easing to ~0.5 deeper. Kills the dominated-action sampling (folded
  50% eq HU, spew-raised 2–14% air). Independent of any training; can ship to live anytime.
- **[P3] Preflop flattening — polarized preflop targets.** Training-side: more POLARIZED preflop
  targets at short depth (push/fold prior, or down-weight the CALL bucket preflop when effective
  stack is short). Bundle into a FUTURE retrain (not the first V15 run).
- **[P4] Opponent-aware PREFLOP entry range (VPIP conformance).** DISCOVERED via style play test
  (2026-07-15, `scratchpad/playtest_style.py`): hero VPIP does NOT adapt to opponent style — FIXED
  entry range (v15 ~51%, v14 ~40%) whether the field is tight or loose, at BOTH short and deep
  (35-50bb) stacks. Adaptation shows only in AGGRESSION (correctly more vs tight / less vs loose)
  + postflop, never in entry. CONSEQUENCE: both are hard-swing style specialists — vs LOOSE deep
  v15 +111 / v14 +98 BB/100, but vs TIGHT deep v15 **-42.6** / v14 -25.0. Great for the live
  station-heavy population, bad vs tight deep reg tables. ROOT: range-aware equity is being absorbed
  into aggression, not into a tighter preflop entry range. FIX (training-side): make the preflop
  entry decision opponent-aware — e.g. steepen how the actor's fold-vs-enter threshold responds to
  the range-aware equity delta preflop, or add an opponent-tightness feature/target so a tighter
  field lowers VPIP. Bigger than [P3]. Only matters if playing tight/reg tables; deprioritize while
  the live field is loose.

## (add future parking-lot items here)

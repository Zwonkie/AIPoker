# V41 SPECS — simulation-realism package from the Fable V29 review

Branches from `versions/v40` (fresh weights, not resumed — per [VAL-5]). Inherits V40's [BET-3]
package unchanged (betting round no longer ends on a check; CALL variance-penalised and given
continuation credit; ALLIN veto rescoped) — see `versions/v40/SPECS.md`, not re-litigated here.

**No contract change.** `context_dim=54`, `contract_version=8`, identical to V29/V40. Every change
below is simulation or query-encoding, so V29/V40/V41 checkpoints stay contract-compatible and the
live bridge needs no new wiring.

Scope set by explicit user direction (2026-07-21): review findings **#7 (dead blinds)**, **#8 (NN
opponents play a degraded self)**, **#9 (all stacks identical / min-raise / reopened action)**,
**#11 ([OPP-7] defeated at the tensor boundary)**, plus the fixes those depend on. Per-finding
status: `.agents/skills/OFK/references/fable-review-resolution-log.md`.

---

## Change 1 [review #11 + #10]: the opponent-seat query encoding

Two defects in `_query_model_decide`, both silent, both invisible to the verification that
originally signed the V27 fix off (it checked the `board_state` dict, not what survived encoding).

**(a) [OPP-7] was defeated at the tensor boundary.** V27 correctly remapped the 5 opponent slots to
"every seat except the acting one", but keyed each slot by the **absolute** seat number
(`seat_{seat_id}`) while `ContractV12.to_tensors` only ever reads `seat_1..seat_5`. For any
non-hero actor `other_seats` contains 0, so the real hero was written to a `seat_0` key the encoder
never reads — hero stayed invisible to every non-hero NN query, exactly what V27 set out to fix —
**and** the surviving slots were misaligned (for `actor_seat=4` the code wrote seat_0/1/2/3/5, so
the encoder's 4th slot looked up the missing `seat_4` and fell back to an inactive default).

Fix: key by **slot index** (`seat_{idx+1}`), which is what the encoder addresses; the real seat
number is retained in `name` for debugging, and `opponents_profiles` lookups now explicitly use the
absolute seat key. For `actor_seat == 0` the two indices coincide, so hero's own query is
byte-identical to V40's.

Measured (40 hands, NN opponent seated): **V40 dropped the hero on 128 of 128 NN-opponent queries;
V41 drops zero, and the hero occupies a readable slot in every query where it is still live.**

**(b) The rollout encoder was a third, drifted encoder** (review #10). `is_active` was
`idx < num_opponents` — which marks the first N *slots*, not the seats that are actually live — and
every opponent's `stack` was a `hero_stack` placeholder. Meanwhile hero's own **gradient** record
(`add_decision`) has always used the real mask and the real per-seat stacks, so the rollout policy
was generating trajectories from features that disagreed with what it was trained and served on.
Both now come from ground truth, threaded through `table_state` as `folded` / `stacks` alongside
the existing `committed` / `raised_this_hand` arrays. The active *count* is unchanged (the call
site already computes `num_opponents` as exactly the number of unfolded non-actor seats), so
`ctx[5]` is identical — what changes is *which* slots carry the live seats' features, i.e. the
per-seat alignment that [OPP-2], [OPP-7] and [V22]'s `committed` all depend on.

(b) is a hard prerequisite for Change 3: once stacks differ per seat, a `hero_stack` placeholder
stops being an approximation and becomes a lie.

## Change 2 [review #8]: NN opponents no longer play a degraded self

- **Range-aware equity was gated on `current_actor == 0`.** Every NN-backed opponent — above all
  the lagged-self mirror, which *is* this run's own network — was fed plain vs-random equity
  despite being trained with `range_aware_equity: true`. A direct train/serve inconsistency for the
  opponent pool, and one that quietly flattered every NN head-to-head. Any NN-backed actor now gets
  the range-aware number it trained on, with the front/after split generalised from a hardcoded
  seats-1-5 block to "every live seat except the actor" (identical to before when the actor is
  hero). Heuristic and Tree opponents are deliberately left on vs-random: their thresholds and
  fitted features are calibrated against that number, so converting them would be the same class of
  mismatch in the other direction.
- **`call_amount` was corrupted.** Both `NNOpponent` query sites passed `pot_odds * pot_size`,
  which given the caller's own `pot_odds = to_call / (pot + to_call)` equals
  `to_call * pot / (pot + to_call)` — **not** `to_call`. A pot-sized bet reached the network as
  half its real size, with the error growing with bet size. Fixed by inverting the caller's
  definition exactly (`t = pot_odds * p / (1 - pot_odds)`, clamped near the singularity) in the new
  `_call_amount_from_pot_odds` helper.

## Change 3 [review #9]: asymmetric stacks, and the two rule bugs that exposes

**Asymmetric starting stacks.** All six stacks were identical for an entire run, so the model never
once saw a covered opponent or a short stack it had already out-chipped — live tables always have
both. Hero's depth still comes straight from the curriculum (`stack_depth_mix` / `fixed_stack_bb`),
so every stack sweep, `deep_stack_ood_guard`, and model_verify's fixed-depth fields keep measuring
what they always measured; each *opponent* seat is now scaled by an independent log-uniform
multiplier in **0.35×–2.0×** (so "half hero" and "double hero" are equally likely), floored at 1bb.
The band is deliberately moderate: wide enough for genuinely covered/covering spots, narrow enough
that a 5bb hero doesn't face a 100bb opponent and blow past the money-feature scaling the V20
rescale calibrated.

Measured (250 hands): opponent stack max/min spread **1.05 → 1.64**; hero decisions facing a
covering opponent **0 → 199**.

That symmetry was the only thing keeping two rule bugs unreachable, so both are fixed here:

- **Min-raise floor was `to_call + 1bb`, not `to_call + last increment`.** After a 3bb open the
  legal "min 3-bet" was 4bb instead of 5bb — a systematic illegal under-raise at every node with
  prior aggression. `min_raise_inc` is now tracked per street (seeded with the big blind, updated
  to each full raise's increment) and threaded into `_raise_size_for_fraction`, the opponent raise
  branch, **and** `_mc_target_evs_sized`, so the counterfactual targets price only sizes hero can
  legally make. Unit-verified: unraised pot → min raise to 2bb; after a 3bb open → min 3-bet to
  5bb (V40 produced the illegal 4bb); stack-capped short all-in still caps correctly.
- **Short all-ins reopened action.** Any raise, including an all-in for less than a full increment,
  reset `acted_this_round` for the whole table, handing out re-raise rights real NLH keeps closed.
  Only a **full** raise re-opens now. Players who already acted still face and must answer the
  extra chips (the loop keeps walking while `all_matched` is false) — they simply do not get the
  round re-opened to them. *Residual deviation, deliberately not fixed:* such a player could still
  choose a raise from the action space; enforcing call-or-fold would need a per-actor action mask.

V40's `highest_bet = max(highest_bet, street_committed[actor])` guard — added there defensively —
is genuinely load-bearing as of this version, since a stack-capped raise below `to_call` is now
reachable.

## Change 4 [review #7]: dead blinds

Pre-folding ran **before** blinds were posted, and blinds were posted unconditionally, so a
pre-folded seat could pay a blind and then never act. With `target_hands: 100000` the curriculum
pre-fold is active for 60% of the run and folds E[2] of 5 opponent seats — hero was learning that
attacking blinds prints money because the blind literally cannot fight back, inflating steal EV and
distorting preflop pot odds across the majority of late-run training data.

Fix: `sb_seat`/`bb_seat` are resolved at the top of `simulate_hand`, before any pre-folding, and
excluded from the pre-fold candidate pool (both in the curriculum path and the `live_players`
diagnostic path). The duplicate blind-seat assignment further down is deleted so the two can never
drift. **Honest cost**: with 1–2 of the 5 opponent seats protected, the deepest pre-folds are
capped, shifting the starting-field distribution slightly toward *larger* fields — which moves
training toward the multiway conditions [BET-3] is about, not away from them.

Measured (400 hands, past the 40k pre-fold threshold): hands reaching a flop with at least one dead
blind **47.6% → 0.0%** (111 dead blind seats → 0).

---

## Status

**TRAINED (100,000 hands, 2026-07-21), DEPLOYED LIVE 2026-07-21, and tagged MILESTONE.**
`core/decision.py`'s `active_model_name` is `'Herocules (v41)'`, replacing V40 (which was live for a
few hours as an interim play-test model). V25–V40 remain registered as rollback options.

**MILESTONE** (`milestone=True` in `core/manifest.py`, plus [MILESTONE.md](MILESTONE.md)) — the
second version ever tagged, after V13. **Do NOT delete `weights/expert_main.pth`.** Read
MILESTONE.md for the limitations this tag does NOT cover, chiefly the `nash_pushfold_vs_chart`
regression V41 inherited from V40 and did not fix.

### model_verify --full: 22 PASS / 5 WARN / 0 FAIL / 0 SKIP

The cleanest scorecard of any version — V29's was 21/2/0/1, and this is the first run with **zero
skips**. Report saved to `.agents/skills/OFK/references/V41/model_verify_report.html` per the
standing convention.

**[BET-3] is resolved** — the live symptom that started this entire line of work.
`multiway_shortstack_aggression` now PASSES outright, where V29 collapsed all 6 short-stack cells
and V40 fixed 3 of 6:

| cell | V29 | V41 (HU → 3-way) |
|---|---|---|
| 5bb / eq 0.65 | ~0.01 | 0.81 → **0.81** |
| 6bb / eq 0.65 | ~0.01 | 0.81 → **0.81** |
| 8bb / eq 0.65 | ~0.01 | 0.81 → **0.80** |
| 5–8bb / eq 0.55 | — | 0.81 → 0.67–0.69 |

Aggression at eq 0.65 is now flat from heads-up to 3-way; the eq-0.55 cells still soften multiway,
which is defensible (marginal equity against more opponents) rather than the old collapse.

**`beats_frozen_predecessor` ran for real for the first time since the V18 refactor**: +64.3 BB/100
over 4000 hands with frozen V40 seated as an actual `NNOpponent` (`frozen_v40.pth`, md5-identical to
V40's deployed weights). Note review finding #5's caveat — SE(BB/100) ≈ ±13–19 at this sample size,
so the margin is real but the point estimate is not precise. Other slow checks:
`bb100_vs_standard_fields` +28.0/+83.4/+31.5/+105.4 across the four fields, `vpip_adapts_to_style`
PASS (short +5.9pts, deep +7.2pts), `beats_offformula_stress` PASS.

Also newly passing vs V40: `deep_stack_ood_guard`, `short_stack_polarization` (0.19 — V40 had
regressed this to 0.22), `allin_vs_nextbest_qgap` (negative at every cell).

**The 5 WARNs, all pre-existing and none introduced by V41:**
- `nash_pushfold_vs_chart` 78% — unchanged from V40, and the one regression V40 introduced that V41
  did NOT fix. Still shoves weak suited trash at 5bb where Nash folds (94s/93s/92s/83s). Worth its
  own look; see the V30/VAL-1 tooling.
- `free_check_low_fold` — standing, covered by decision.py's free-check mask.
- `opponent_style_sweep` (0.004), `allin_exploits_opponent_foldiness` (0.000) — **[OPP-8]**,
  untouched by this version.
- `pot_type_sensitivity` (0.024) — pot_type may be redundant with call_amount/pot_size/committed.

**Live-serving smoke test** (through `core/decision.py`'s real `make_decision`): all four streets
emit executable `RAISE_SLIDER_x` actions with real chip amounts, and the multiway [BET-3] spot
returns a full-pot raise 3-way at both eq 0.65 and 0.90. This check exists because the V40 deploy
silently emitted `RAISE_POT` with `size=0.0` — see the resolution log's #16/H4.

Every change above has a measured before/after, not just a code diff. Not addressed here and still
open: #6 (every opponent raise is exactly 0.75 pot) — the last member of the reviewer's [BET-3]
bundle, and the natural next version, since it is what the OPP-2 raise features and the
fold-vs-raise responses are ultimately calibrated against.

See: `.agents/skills/OFK/references/fable-review-resolution-log.md` |
`versions/v40/SPECS.md` ([BET-3] package) | `versions/v29/SPECS.md` (contract, [OPP-2])

# V16 — Roadmap / consolidated backlog

All open suggestions from the V14→V15 line, gathered here (2026-07-15) after **V15 was deployed
live**. V15 = the active model (`core/models/v15_engine.py`, `Herocules (v15 DoN)`); v14/v13 are
fallbacks. V15 fixed the deep-stack OOD and is a strong loose-aggressive winner vs loose/station
fields (the live population) — see `versions/v15/SPECS.md`. The items below are what V15 did NOT
address; none is required for the current live setup, which faces mostly loose fields.

Nothing here is built. Pick items per the guardrail: pull one forward only when live play shows the
concrete leak it fixes — don't build speculatively. Ordered by leverage for the current situation.

---

## [P4] Opponent-aware PREFLOP entry range (VPIP conformance) — biggest structural gap

**Evidence (style play test 2026-07-15, `scratchpad/playtest_style.py`, temp 0.5, 4 opponents):**
hero VPIP does NOT adapt to opponent style — a FIXED entry range regardless of field, at BOTH short
and deep (35-50bb) stacks (v15 ~51%, v14 ~40%; tight-vs-loose delta only ±1-2pts, sometimes the
wrong way). Adaptation shows only in AGGRESSION (correctly more vs tight / less vs loose) + postflop,
never in which hands it enters with.

**Consequence:** both models are hard-swing STYLE SPECIALISTS —
| depth 35-50bb | vs LOOSE | vs TIGHT |
|---------------|---------:|---------:|
| v15           | **+111** | **-42.6** |
| v14           | +98      | -25.0    |

Great for the live station-heavy/loose population (crushes it); bad vs tight, deep reg tables.

**Root cause:** range-aware equity lowers the hero's computed equity vs tight opponents, but that
signal is absorbed into aggression/postflop, NOT into a tighter preflop entry range.

**Fix (training-side):** make the preflop ENTER-vs-FOLD decision opponent-aware. Options:
- steepen how the actor's preflop fold threshold responds to the range-aware equity delta (so a
  tighter field, which lowers equity, actually folds more hands preflop);
- and/or add an explicit opponent-tightness feature/target that lowers VPIP vs tight fields;
- validate by re-running `playtest_style.py` — success = VPIP drops meaningfully vs tight, rises vs
  loose, at deep stacks (where it currently doesn't move).

**Priority:** only bites at tight/reg tables. Deprioritize while the live field is loose; promote if
you start sitting tougher tables.

**STATUS (2026-07-15): IN PROGRESS as the V16 retrain.** Root cause re-diagnosed precisely (see
`versions/v16` implementation): the preflop CALL/FOLD target inside `_mc_target_evs_sized` used
oracle equity vs opponents' literal dealt cards, which is style-independent at the entry decision
(no selection has occurred yet) — that's the real asymmetry vs RAISE, which already gets a real
style signal via `p_all_fold`. Fix = swap the preflop-only CALL/FOLD equity basis to the
already-computed range-aware equity (single substitution, no new tuned constants, no contract
bump). Full detail + rationale in the approved plan / `versions/v16/self_play/simulator.py`.

## [P2] Live sampler spew — stack-scaled temperature (LIVE-SERVE, no retrain) — DONE (2026-07-15)

`core/decision.py`-only. `LIVE_POLICY_TEMPERATURE` is a flat 0.5; at short stacks (≤~8bb) where the
strategy is near-PURE push/fold, sampling still occasionally picks a dominated action (live evidence:
folded 50% eq HU; spew-raised 2-14% air — boards `1170808251`/`1170810410`). Make the temperature
STACK-SCALED: near-argmax (~0.2 / hard argmax) at ≤~8bb, easing to ~0.5 deeper where mixing has
value. Cheapest win here (no retrain, ~5-10 lines); can ship independently of any V16 training.

**IMPLEMENTED:** `_stack_scaled_temperature(board_state)` in `core/decision.py` — 0.2 at
`stack_bb <= SHORT_STACK_BB(8)`, linear ease to `LIVE_POLICY_TEMPERATURE(0.5)` by
`DEEP_STACK_BB(20)`, flat 0.5 beyond. Wired into both the sized-model (V14/V15) and legacy
actor-policy (v13) sampling branches, replacing the flat `LIVE_POLICY_TEMPERATURE` constant in the
`sharp = {...}` calls; reason string now logs the actual `temp` used. NOT smoke-tested in the live
GUI yet (logic-verified only: 3bb->0.2, 10bb->0.25, 14bb->0.35, 20bb+->0.5).

**KNOWN GAP:** eval scripts (`scratchpad/eval_shortstack.py`, `eval_v15.py`) still set a FLAT
`sim.policy_temperature=0.5` for train/serve parity checks — they no longer bit-for-bit match live
at ≤20bb (live now samples sharper there). Conservative direction (eval overstates short-stack spew
risk vs what ships), so not urgent, but true parity would need `simulator.policy_temperature` made
stack-aware too. Not done — flagged for later.

## [P3] Preflop flattening — polarized preflop targets (training-side)

Persistent CALL mass in the preflop policy at short stacks where the spot is shove-or-fold
(v13/v14/v15 all show it). Fix: more POLARIZED preflop targets at short depth — push/fold prior, or
down-weight the CALL bucket preflop when the effective stack is short. Overlaps with [P4] (both are
preflop-entry quality); consider tackling them together in one retrain.

**STATUS (2026-07-15): DEFERRED, not pre-solved in the V16 retrain.** Decided NOT to add a
dedicated short-stack CALL penalty constant alongside the P4 fix — once RAISE and CALL/FOLD share a
consistent, style-aware equity basis, regret-matching across all K actions may polarize toward
push/fold on its own. Validate CALL frequency at short stacks after the V16 training run; only
design a targeted follow-up if it hasn't moved.

## Size-scaled opponent BLUFFING (deferred since V14 P1b)

Opponent bots' DEFENSE is size-aware (fold more to bigger bets — `opponent_bots.decide_postflop`),
but their bluff-RAISE frequency is flat. Refinement: scale bluff-raise frequency by the size of the
bet faced (`bluff_room = 1 - pot_odds`; raise-bluff more into small bets). Realism polish for how
opponents apply pressure back; only matters if bot bluffing looks too rigid. Low priority.

## [P5] Bet-size perception — size-aware history + scaled to-call feature (input contract)

**Current state:** the model sees the CURRENT price it faces via `pot_odds` (ctx[4], well-resolved)
and `to_call` (ctx[9] = call_amount/BB/400 — but /400 is scaled for 400bb-deep pots, so a 1-5bb
raise lands at ~0.003-0.013, near-zero/poorly resolved). The action-HISTORY tokens
(`contract.VOCAB`, `'r'=6`) are SIZE-BLIND — a min-raise and a pot-overbet are the same token, so
the model can't see the *pattern* of prior sizing.

**Change:** (a) give the raise/bet history tokens a size bucket (e.g. distinct tokens or a parallel
size feature per action step: small / medium / pot / overbet / all-in), so the model can read a
sizing sequence; (b) add a properly-scaled absolute bet-size feature (e.g. to_call/BB clamped to a
sane short-stack range, not /400). Input-contract change → contract_version bump + retrain. Modest
value on its own; bundle with [P6].

## [P6] Opponent-action attribution — WHO raised, and how many (input contract)

**Current state (the real gap):** the input has NO opponent-action encoding at all. The `act` tensor
is HERO-only (`contract.to_tensors` fills it from `hero_actions`); live it's empty. The model's
entire read of opponents each step is the aggregate snapshot: which seats are still active
(`active_mask`), each active seat's HUD vpip/agg COLOR + stack, the averaged `opp_vpip_norm`/
`opp_agg_norm`, and the total `to_call`/`pot_odds`. So it CANNOT tell: who the aggressor is, whether
one opponent raised or two raised (a raise + a cold-call vs a 3-bet), or that "the maniac 3-bet and
the nit flatted." It infers opponent strength only from WHO REMAINS ACTIVE + their static HUD color
(via range-aware equity), never from the action taken.

**Change:** encode per-opponent actions this hand — at minimum an aggressor flag + the aggressor's
HUD tendency (so a nit-raise reads differently from a maniac-raise beyond just "still active"), ideally
a compact per-seat action+size sequence. This is the highest-information opponent feature the model
currently lacks. Input-contract change (bump + retrain); bundle with [P5] into one contract revision.
Priority: meaningful, but the current range-aware-equity proxy (equity vs active opponents' color
ranges) covers the first-order effect; promote if postflop/3-bet-pot play looks range-blind live.

## Also worth carrying into any V16 retrain
- **Widen the frozen-opponent pool.** V15 used a single frozen-V14 in the `past` seat. A V16 could
  pin BOTH frozen-V14 AND frozen-V15 (or a small gauntlet) so the hero faces multiple competent
  styles, not one. New benchmark: V16 must beat frozen-V15.
- **Deep-stack critic variance.** V15's critic Q-loss ran high (~9.6) because 30-50bb pots have large
  EV magnitudes near `target_clip_bb=40`. Actor was fine, so not urgent — but if a V16 goes deeper,
  revisit the clip / consider normalizing Q targets by pot or stack.

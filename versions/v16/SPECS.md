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

## [P0-recheck] Deep-stack OOD trash-jam — RECONFIRMED PRESENT in V15 (2026-07-15)

`tools/model_verify`'s `deep_stack_ood_guard` check (see below) reproduces the exact original V14
incident conditions (43% equity, 20bb stack, facing a single small bet) against the LIVE V15
checkpoint: **ALL-IN is still the model's argmax there (0.37 probability)**, climbing smoothly from
a 0.24-probability all-in argmax even at a clearly-losing 35% equity. V15's aggregate BB/100 numbers
looked like the OOD was fixed (beats frozen-V14 at every depth), but that was likely because the
aggregate winrate improvement came from elsewhere (better postflop, exploiting a weak/short-stack-
specialized frozen-V14 opponent) rather than this specific decision actually being corrected.

**Implication for V16:** the P4 fix in this retrain is scoped to the PREFLOP CALL/FOLD equity basis
only (deliberately, per the plan discussion) and does not touch this postflop/deep-stack pathway —
so this trash-jam tendency will very likely carry into V16 unchanged. Not a V16 regression if seen
again; a pre-existing, now-precisely-characterized V15 carryover. Track separately; candidate for a
dedicated deep-stack decision-quality pass in a future version if it's still present after V16.

## [P7] Opponent-pool NN personalities — close the "Yellow"/LAG training gap (2026-07-15)

**Finding:** the current opponent training pool (`config.yaml` `pool: ["fish","tag","nit","past"]`)
never includes an individual opponent whose VPIP lands in the "Yellow" HUD band (0.26-0.35):
NIT=0.11 (Blue), TAG=0.22 (Green), fish=0.45 (Red), frozen-past≈0.50+ (Red). `LAG` (VPIP=0.32,
`opponent_bots.py`) is the one existing archetype that WOULD be Yellow, but it's excluded from the
pool. Checked empirically whether this is a functional hole: swept the opponent-VPIP input feature
continuously through the untrained 0.26-0.35 gap on the live V15 model — response was smooth and
monotonic with no discontinuity (likely because the AVERAGED `opp_vpip_norm` feature crosses that
range often anyway, from mixed Blue+Red tables). So NOT currently an urgent correctness bug, but the
underlying training-population gap is real and worth closing properly rather than relying on
interpolation.

**Bigger idea raised in discussion:** rather than only adding LAG as a scripted heuristic bot, train
DEDICATED NN opponent personalities using the current architecture, so opponents are both
STYLE-CONSTRAINED and genuinely skilled (not simplistic threshold heuristics). The mechanism to do
this already exists, dormant, from the V8-era codebase:

- `simulator.py:_hero_decide` has a `hero_personality` mode (`train.py --personality
  {main,maniac,nit,sticky}`) that PROBABILISTICALLY FORCES the network's chosen action toward a
  target style during self-play data generation when the hero's own running VPIP/AGG drifts off
  target (e.g. `nit`: 80% chance to override to `fold` preflop if VPIP > 15%), combined with a
  style-matched heuristic-anchor bootstrap early in training. This lets the network learn genuinely
  good judgment (sizing, bluff selection, board reads) WITHIN a style that's exogenously steered,
  rather than a hard mask — the same technique already used to train the `main` personality itself.
- `_opponent_decide`'s "querying personality NNs" path exists to LOAD a personality checkpoint as an
  opponent seat's brain, but is currently disabled (`train.py` comment: "personality NNs disabled
  (they use fuzzy heuristic bots)") — likely because those old checkpoints predate the current
  6-action sized contract, not because the mechanism is broken.

**Proposed path (not started):**
1. Add a `lag` branch to the action-forcing logic in `_hero_decide` (mirror nit/maniac's pattern,
   target ~28-32% VPIP, moderate aggression), anchored to the existing (currently unused) `LAG`
   heuristic archetype.
2. Run a dedicated `--personality lag` training pass under V16's CURRENT contract → `expert_lag.pth`
   (native compatibility with the sized opponent-decision path, unlike the old disabled checkpoints).
3. Re-enable the opponent-seat personality-NN loading path; add `lag` to the pool (e.g.
   `["fish","tag","nit","lag","past"]`).
4. Validate, don't assume: action-forcing is probabilistic, not guaranteed — measure `expert_lag.pth`'s
   REALIZED VPIP via `tools/model_verify` (or a dedicated eval) before trusting it in the pool.

Same technique could eventually upgrade nit/tag/maniac from scripted heuristics to skilled NN
opponents too — LAG is just the immediate, concrete gap. Backlog item, not scheduled.

## [BUG] Opponent AGG stat undercounted for any sized-model (NN) opponent seat — FIXED (2026-07-15)

**Symptom (spotted live on the training dashboard):** "Past Self" (the frozen-V15 seat) showed
RAISE=49225 (a large, plausible count) but AGG=1.6% (implausibly near-zero) — a direct contradiction
between two counters that should broadly agree.

**Root cause:** `simulator.py`'s opponent AGG tracking (`if decision == 'raise': agg_acts += 1`,
was line ~1165) used an EXACT string match. That's correct for the legacy heuristic bots (plain
`'raise'`), but "Past Self" is the frozen V15 model itself, which returns sized-bucket strings
(`'raise_0'..'raise_3'`) per the 6-action contract — `== 'raise'` never matches those, so
`agg_acts` almost never incremented for it, while the SEPARATE `raises` counter (used for the
dashboard's RAISE column) uses a catch-all `else` branch that counts correctly. Two counters, one
bug — the mismatch between them is what exposed it.

**Impact — real, not cosmetic:** `opponents_profiles[seat]['agg']` (fed into the network's input
context via `map_agg_to_midpoint`, both the per-seat HUD feature and the averaged `global_agg`) is
sourced from this same broken counter. Every hand "Past Self" was active (20% of the pool), the
network was taught a contradictory read: "enters very wide (VPIP correctly ~64%) but plays
passively (AGG wrongly ~2%)" — not what the frozen model actually does. **Pre-existing since V15**
first wired a real NN into an opponent seat (V15 shipped/validated fine despite it — bounded impact,
not catastrophic). Does NOT affect the P4 fix in this retrain (range-aware equity color
classification uses only `vpip`, never `agg`).

**Fix:** `decision == 'raise'` → `decision.startswith('raise')` at all 4 call sites in
`simulator.py` (opponent AGG counter; opponent-personality action-forcing for `fish`/`maniac`
styles, both preflop and postflop; and the analogous HERO-side `sticky`-personality forcing branch,
which has the same latent bug for any FUTURE `--personality` training run under the sized contract)
— matching the pattern hero's own AGG tracking already used correctly (`decision.startswith('raise')`
at line ~1114).

**Not retroactive to the currently-running V16 job**: Python doesn't hot-reload source changes into
an already-running process/worker pool, so this fix takes effect on the NEXT training launch, not
the in-progress one. Decided NOT to kill and restart the current run over this — it's a pre-existing,
bounded, already-shipped-fine issue unrelated to what's under test this retrain; restarting would
lose the hands already trained for no benefit to the thing actually being validated.

**Follow-up worth doing:** if `expert_lag.pth` ([P7] above) or any other sized-model opponent seat
gets built, re-verify its AGG stat looks plausible post-training (a quick sanity a future
`model_verify` extension could check, since this class of bug is specifically "a new NN-driven
opponent seat + an old string-matching assumption").

## Tooling: `tools/model_verify` — the standing model-verification suite (2026-07-15)

Replaces the old practice of writing one-off eval scripts to a session-scratchpad (which didn't
survive between sessions — the exact scripts that caught the P4 VPIP-flatness bug and the V14
deep-stack incident were lost after those sessions ended). See
`.agents/skills/OFK/references/model-verification-suite.md` for the full writeup: what it covers, how
to run it, how to extend the curriculum, and durable lessons from calibrating it (a substring-match
bug that silently miscounted CALL as ALL-IN, and an over-strict diversity threshold that flagged
V15's known-accepted bimodal sizing as a false collapse). Run it after every training run going
forward: `python -m tools.model_verify.run --version v16 --full`.

## [BUG-derived experiment] `v16_foldregret` — fold-relative regret baseline — trained, NOT deployed (2026-07-15)

**Motivation:** live dashboard showed `<20%`/`20-40%` equity buckets net-losing chips while still
continuing/jamming ~half the time. Traced to `regret_match_policy`'s mean-centered baseline: a
bluff-raise's fold-equity inflates the shared mean enough that independently-bad actions (e.g.
calling with air) still show positive regret relative to it. `versions/v16_foldregret/` isolates
ONE change — regret measured against FOLD's value (always 0) instead of the mean — same config as
this main V16 line otherwise, trained fresh, 100k hands. Full trace: `versions/v16_foldregret/SPECS.md`.

**Result:** the air/draws fix worked exactly as intended (Fold% up sharply and stable to
completion — Pure Air 66% fold/16% all-in, Draws 84% fold/6% all-in — without collapsing the
profitable Marginal/Strong/Nuts tiers). But `tools/model_verify --full` caught a real regression:
`vpip_adapts_to_style` deep-stack delta dropped from this line's own +8.4pts (PASS) to +2.0pts
(FAIL) — the fold-relative baseline appears to zero out the small style-conditioned edge [P4]'s
range-aware-equity fix relies on at deep stacks, alongside the bad actions it was targeting.
`deep_stack_ood_guard` also FAILs on foldregret, but V16 fails the identical check too (pre-existing
carried gap from V15, not a new regression).

**Decision:** kept as a validated experiment result, NOT promoted over this line. Follow-up ideas
(blended baseline, street/depth-scaled bite, larger sample to confirm the deep delta isn't partly
noise) are logged in `versions/v16_foldregret/SPECS.md`'s Outcome section for whoever picks this
back up.

## New: live "thinking" narrative (2026-07-15, `core/decision.py` + `PHPHelp.py`)

User asked for the model's 3 aux heads (`bluff`/`strength`/`equity`) to drive a human-readable
"what is it thinking" line in the live HUD. Investigated first: those heads train against
`opp_bluff_prob`/`opp_strength` (an OPPONENT read, not hero self-reflection) and every active
config ships `aux_loss_weight: 0.0` — zero gradient, so their live outputs are untrained noise.
Built the equivalent from real, already-trained signal instead: `_narrate_thinking(action,
board_state, evs)` in `core/decision.py` bands the model's own equity input against the chosen
action (fold/aggressive/call) into a line like `"Thinking: weak hand (14% equity) -- bluffing,
betting on fold equity rather than hand strength."` Surfaced via `ev_dict['thinking']`, a new
`self.thinking_lbl` under the action-reason label in the PHPHelp HUD, and the append_log/turn-
history trail. Version-agnostic (reads `board_state.equity` + the chosen action string, not any
model-specific head) — applies to whichever model is active, no per-version wiring needed.

## Periodic restore-point checkpoints — every 25k hands (2026-07-15)

**Gap:** `train.py` already saved rolling checkpoints (`expert_main.pth` every
`mid_flight_diagnostics_interval`=10k hands, `temp_active_model_main.pth` every batch) but these get
OVERWRITTEN each time — there was no history of intermediate models to fall back to or evaluate if
a later stage of training regressed or diverged.

**Fix:** new `checkpoint_dump_interval` config (`self_play/config.yaml`, default 25000). Every
~25k hands (fires on the first batch that crosses the boundary, so actual points land near
26k/52k/78k/... for a 2000-hand batch size, not exactly on the round number), `run_training` saves a
DISTINCT, never-overwritten checkpoint to `weights/checkpoints/{personality}_hands{hands_done}.pth`
— filename postfixed with the actual cumulative hand count reached, not the configured interval
value. Implemented editing the file directly while the current V16 job was already running: safe,
since Python doesn't hot-reload source into an already-running process — this takes effect on the
NEXT launch (a restart, or v17 copied from this folder), same caveat as the AGG-bug fix above.

**Verified (not live-trained yet):** `py_compile` passes; cadence/filename logic simulated standalone
against a mock 210k-hand loop — produced 8 dump points at the expected ~26k-hand-spaced boundaries.

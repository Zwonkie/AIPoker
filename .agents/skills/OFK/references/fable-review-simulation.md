# Fable Review — Simulation Layer (V29)

**Date Recorded**: 2026-07-20
**Related Files**: [simulator.py](file:///c:/REPO/Antigravity/AIPoker/versions/v29/self_play/simulator.py), [opponents.py](file:///c:/REPO/Antigravity/AIPoker/versions/v29/self_play/opponents.py), [opponent_bots.py](file:///c:/REPO/Antigravity/AIPoker/versions/v29/self_play/opponent_bots.py), [config.yaml](file:///c:/REPO/Antigravity/AIPoker/versions/v29/self_play/config.yaml)

## Context

Simulation-area report from the 2026-07-20 four-way V29 audit (see
`fable-review-consolidated.md`). Scope: poker-rule fidelity, equity computation, opponent pool
realism, counterfactual target generation, statistical hygiene. The H1 betting-engine finding was
**empirically confirmed** with a 1000-hand instrumented run (849 postflop checks, zero times anyone
acted after a check; zero BB-option decisions preflop).

## HIGH severity

**H1. A single check ends the postflop betting round; the BB never gets its preflop option.**
`simulator.py:1675` (and the companion break at `:1411`) — the round terminates on `all_matched and last_raiser == -1`, and with `highest_bet = 0.0` postflop, `all_matched` is true from the street's first instant. Only the seat the loop starts on (button+1) ever gets a free action; if it checks — or is already folded — the street ends for everyone. Empirically verified: over 1000 instrumented hands, 0 of 849 postflop checks were followed by any other player acting; 0 preflop decisions ever occurred at price 0 (limped-pot BB option denied). The `acted_this_round` array (`:1394`) exists but is never consulted by the termination logic. Failure scenario: the model has literally zero training data for "checked to me in position", check-behind, check-raise, delayed c-bet, or BB limped-pot option nodes — all of which occur constantly live. This is a plausible root cause of the tracked [BET-3] multiway passivity (model collapses to call/fold): it has never once seen a betting node that follows a check.

**H2. Every opponent — heuristic, lagged-self NN, and TreeOpponent — raises exactly 0.75-pot; NN/tree size choices are discarded.**
`simulator.py:1627-1629` — the opponent raise branch hardcodes `min(pot*0.75, stack)` regardless of what the agent returned. A frozen/lagged NN returning `raise_3` (all-in) executes a 0.75-pot raise; TreeOpponent's 6-class size prediction is collapsed to 'raise' (admitted in `tree_opponent.py:22-29`). Hero trains 100k hands never facing an open-jam, overbet, min-raise, or small probe from anyone. Failure scenario: live opponents shove and min-raise routinely; the model's per-seat raise features (OPP-2) and fold-vs-raise responses are calibrated to a world with exactly one opponent bet size.

**H3. Counterfactual CALL target is single-street-terminal while RAISE targets get a continuation bonus.**
`simulator.py:1018` (`evs = [0.0, true_equity*(pot+to_call) - to_call]`) vs `:1037-1043` where only non-all-in raises receive `_rollout_continuation_ev`'s implied-odds/fold-equity correction. Calling with a draw has real future-street value; the target says it doesn't, while a raise at the same node gets that value credited. Additionally the raise EV never models being re-raised: `_ev_target_fold_decision` (`:766`) returns only fold/continue, and continue is scored as an exact call of the full amount (`:1032-1035`) — hero's raise can never be 3-bet off its equity, biasing small-raise EV upward vs aggressive opponents. Failure scenario: systematic pro-aggression, anti-call tilt baked into every training target — the actor is regret-matched against these numbers, so the bias is structural, not noise. (Note: the V28/V29 variance penalty is the opposing asymmetry — applied to raises only, never CALL — see the training report; the two do not cancel, they just distort in different regimes.)

**H4. Dead blinds: pre-folded seats post blinds and never defend, in ~90% of hands past 40k.**
Pre-folding happens at `simulator.py:1232-1242` before blinds are posted unconditionally at `:1331-1335`. With `target_hands: 100000` (`config.yaml`), 60% of the run is in the pre-fold phase; the BB seat is a pre-folded corpse in ~40% of those hands (E[folds]=2 of 5 seats). Failure scenario: hero learns that attacking blinds prints money because the blind literally cannot defend — inflating steal EV and distorting preflop pot odds in the majority of late-run training data.

## MED severity

**M1. NN opponents are fed a corrupted `call_amount`.**
`opponents.py:176,189` pass `pot_odds * pot_size` as call_amount; since `pot_odds = to_call/(pot+to_call)` (`simulator.py:1607`), the passed value is `to_call·pot/(pot+to_call)` — a pot-sized bet shows up as half its real size. Combined with **M2**, the lagged-self mirror and any frozen checkpoint play a degraded, off-distribution version of themselves — which also quietly inflates every "beats frozen VX head-to-head" result.

**M2. NN opponents get vs-random equity though they were trained on range-aware equity.**
`simulator.py:1415-1436` — range-aware equity is gated on `current_actor == 0`; every other seat (including the lagged-self, trained with `range_aware_equity: true`) gets the plain vs-random number. A direct train/serve inconsistency for the opponent pool.

**M3. Range-aware equity's effective sample count collapses vs tight fields.**
`compute_range_aware_equity` (`simulator.py:278-307`): preflop VPIP fold-rolls *skip* samples; vs one Blue (VPIP 0.10) yet-to-act opponent, ~15 of 150 sims are counted → stderr ≈0.13 on the model's *primary* input feature. Also the underlying preflop ranking is built from only 80 sims/combo (`:211`, stderr ≈0.056 — substantial mid-range misordering) and frozen forever in `preflop_ranking.json`.

**M4. Min-raise rules are wrong and short all-ins reopen action.**
`_raise_size_for_fraction` (`simulator.py:756-764`) and `:1628` floor raises at `to_call + 1bb`, not `to_call + last raise increment` — systematic illegal under-raises (open 3bb, "min 3-bet" to 4bb). An all-in under-raise unconditionally resets `acted_this_round` for everyone (`:1586-1588`, `:1644-1647`), reopening action that real NLH keeps closed. Self-consistent in training, but a genuine rule deviation the live bridge must paper over.

**M5. All stacks are always identical — and that symmetry is the only thing masking a chip-corruption bug.**
`simulator.py:1202` (`stacks = [starting_stack_chips]*6`) means effective stacks are symmetric for the entire run; the model never sees covered/short opponents (live tables always have them). Worse: `highest_bet = street_committed[actor]` after a raise (`:1577`, `:1638`) would *lower* the current bet (yielding negative `to_call` and money flowing backwards through the call branch at `:1616-1622`) whenever an actor's stack < to_call — provably unreachable only because equal starting stacks guarantee `to_call ≤ stack`. Any future stack-asymmetry feature detonates this latently.

**M6. Heuristic opponents' behavior is card-blind in ways no human's is.**
Bots decide purely on (own equity vs random, pot odds) — no position, board texture, hero range, or history (`opponent_bots.py:146-256`). Stat-forcing overrides are holding-agnostic: `_force_preflop_nit` (`opponents.py:46-49`) folds 80% of decisions once realized VPIP drifts >0.15 — including AA. Opponent actions get decoupled from their oracle cards, which is exactly what the counterfactual targets condition on, so hero's fold-equity credit is earned against fold patterns no live player exhibits.

**M7. Postflop ranges never narrow.**
Range-aware sampling always draws opponents from their full preflop VPIP band (`simulator.py:267-297`) — a preflop 3-bettor or a two-street caller is still sampled from their whole range. The front/after fix (V20_preflopEq) only handles the fold-roll, not action-conditioned range narrowing → hero equity biased high vs shown aggression. (Live-side sibling of [OPP-9].)

**M8. Target-noise budget is thin and there is zero variance reduction.**
Per-size fold-outs use 10 Bernoulli draws per opponent (`simulator.py:1029-1034`, granularity 0.1, multiplied across seats); continuation rollout is 4 trials × 150-sim equity (`:51-52`). No duplicate/mirrored deals, no antithetic dealing, no AIVAT-style baseline anywhere. 100k hands × ~2-4 decisions is enough for aggregate stats but state-conditional target noise in a 44-dim context is large, and the biases above (H3, M6) don't average out.

## LOW severity

- **L1.** Multiway `ev_if_called` assumes exactly one full caller (`simulator.py:1035`) while `p_all_fold` is a product over all opponents; fold-model opponents' equity is vs 1 random hand (`:1011`) regardless of field size.
- **L2.** Preflop targets mix probability spaces: fold-weighted conditional range equity multiplied against the actual literal pot (`:1010,1018`), and the call target assumes call closes action (no squeeze behind).
- **L3.** Every opponent slot's stack feature is `hero_stack` (`simulator.py:645`) — acknowledged placeholder, wrong once commitments diverge.
- **L4.** Hero's own action-history token is size-blind — every raise bucket appends `6` (`:1583`) (known backlog [OPP-3]).
- **L5.** `phase_1_stacks: [100.0]` in config is dead code: `stack_depth_mix` takes precedence over all curriculum from hand 0 (`simulator.py:429-432`).
- **L6.** True heads-up (button=SB posts and acts first) never exists as a starting configuration — HU only arises mid-hand inside the 6-seat scaffold — yet HU push/fold (VAL-1 Nash axis) is a headline eval.
- **L7.** Per-personality VPIP/AGG stats are cumulative over the whole run, never windowed — colors converge and reflect bootstrap-era behavior long after it's gone.

## What's actually solid

The side-pot slicing algorithm (`simulator.py:1697-1730`) is correct, including multiway all-ins, ties, and uncalled-bet refund via the leftover path; chip conservation checks out. Seat order (UTG first preflop, SB first postflop), blind posting capped at stack, and the button-relative position fix (V19) are right. The counterfactual per-size target architecture itself — oracle-hand equity, decoupled fold model (V24), closed-form variance penalty whose mean provably matches the EV blend (`_outcome_variance`, `:928`) — is a thoughtfully engineered design, as is the shared `compute_range_aware_equity` for train/serve consistency and the fail-loud model-query error surfacing. The V27 per-actor seat remapping and V29 per-seat raise attribution are correctly threaded end to end (but see the contract report: the remap's `seat_0` keys are unreadable by `to_tensors`, defeating the hero-visibility goal downstream). The problems are concentrated in the betting-round state machine (H1 — one localized fix using the existing `acted_this_round` array would repair most of it) and in what opponents are allowed to do (H2), not in the accounting or the target math's plumbing.

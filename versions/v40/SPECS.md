# V40 SPECS — the [BET-3] package from the Fable V29 review

Branches from `versions/v29` (fresh weights, not resumed — per [VAL-5]). SAME contract
(`context_dim=54`, `contract_version=8`, [OPP-2] per-seat raise attribution), SAME opponent pool,
deep-stack curriculum, risk-adjusted target, critic-consistency filter and
`risk_aversion_coefficient=0.15` as V29 — see `versions/v29/SPECS.md`; none of that is
re-litigated here.

**No contract change.** `context_dim`/`contract_version` are byte-identical to V29's, so V29 and
V40 checkpoints are contract-compatible, `beats_frozen_predecessor` can load a frozen V29 into
V40's architecture (unlike V28→V29, which SKIPped for exactly that reason), and the live bridge
needs no new wiring when/if V40 is deployed.

Scope was set by explicit user direction (2026-07-20): fix the first tier of
`.agents/skills/OFK/references/fable-review-consolidated.md` — "Explains live behavior already
logged in the backlog", i.e. findings #1, #2, #3 — in a new version cloned from V29, in narrow
scope, without chasing adjacent findings. Per-finding status is tracked side-by-side in
`.agents/skills/OFK/references/fable-review-resolution-log.md`.

---

## Change 1 [review #1, BET-3]: a single check no longer ends the betting round

**Defect.** `simulate_hand`'s betting loop terminated on
`all_matched and (last_raiser == -1 or current_actor == last_raiser)`. Postflop `highest_bet`
starts at `0.0`, so `all_matched` is true from the street's very first instant, and the
`last_raiser == -1` disjunct made "nobody has raised yet" sufficient on its own. The round
therefore closed as soon as the OPENING seat (button+1) acted — or immediately, if that seat was
already folded. A second guard (`to_call == 0.0 and not first_round and last_raiser == -1 → break`)
closed the same door from the other side, firing on the street's second live seat.

The reviewer confirmed this empirically over 1000 instrumented hands: **0 of 849 postflop checks
were followed by any other player acting**, and the BB never once received its limped-pot preflop
option. Check-behind, check-raise, "checked to me in position", delayed c-bet and BB-option nodes
had literally never appeared in any training sample — a strong root-cause candidate for **[BET-3]**
(V29 live: collapses to call/fold with 3+ opponents, won't raise at 90% equity).

**Fix.** The simulator already maintained `acted_this_round` — for the V20_preflopEq front/after
range-aware-equity split, including its reset-on-raise, which is exactly the "a raise re-opens
action" rule — and simply never consulted it for termination. A round now ends when every
still-live seat WITH CHIPS has both acted this round and matched the current bet. Seats with
`stacks == 0` are all-in and excluded. This subsumes the old `current_actor == last_raiser`
disjunct (action returns to the raiser precisely when everyone else has responded). The early
`break` is now additionally gated on `acted_this_round[current_actor]`, demoting it from de-facto
terminator to safety net.

**Hardening required by the fix** (review M5): both raise branches now do
`highest_bet = max(highest_bet, street_committed[actor])` instead of a bare assignment. A
stack-capped "raise" below `to_call` used to LOWER the bet level, producing negative `to_call` for
everyone else and an `all_matched` that can never become true — a hung hand. Previously unreachable
only because all six stacks start equal; longer betting rounds make it reachable, so it is guarded
now.

**Verified** (instrumented probe over 750 hands per version, all seats counted, three seeds):

| metric (per hand, all seats) | V29 | V40 |
|---|---|---|
| postflop actions | 3.13 | **5.16** (+65%) |
| preflop actions | 6.70 | 7.23 |
| preflop decisions at price 0 (BB option) | **0** | **259 / 750 hands** |

## Change 2 [review #2, BET-3]: CALL is no longer exempt from the risk penalty or the continuation credit

**Defect.** `_mc_target_evs_sized` gave every SIZED action two corrections that CALL never got:

1. the [V25] `_rollout_continuation_ev` implied-odds / continued-fold-equity credit (every non-
   all-in raise, preflop/flop/turn). Calling with a draw has real future-street value; the CALL
   target said it had none, while a raise at the IDENTICAL node had that value credited — a
   structural pro-aggression/anti-call tilt in every training target, and the actor is
   regret-matched against these numbers, so the bias is structural rather than noise.
2. the [V28/V29] `risk_aversion_coefficient * sqrt(Var[X])` penalty. CALL was the only action left
   as a raw risk-free point estimate, so every raise was docked variance RELATIVE to calling. That
   penalty scales with pot size — i.e. it bit hardest in exactly the multiway/high-equity spots
   where V29 refuses to raise live.

**Fix.** Both are now applied to CALL through the same shared helpers. The variance reuses the same
closed-form `_outcome_variance`, instantiated for CALL's own 2-point outcome mixture
(`p_fold=0`, base pot `= pot + to_call`, `"raise_size" = to_call` → win ⇒ `+pot`, lose ⇒
`-to_call`), which is exactly the mixture whose mean is the existing CALL EV — the same algebraic
consistency V28 established for the sized actions. The continuation credit is skipped when the call
is itself all-in (nothing left to play) and on the river (no next street).

**One deliberate exception**: the variance penalty is NOT applied when `to_call == 0`. A free check
risks no chips, and FOLD (the actor's regret baseline) is a flat `0.0` carrying no penalty of its
own, so penalizing a free check would tilt the target toward folding for free — the exact corner
`free_check_low_fold` already tracks as a standing WARN.

**Verified numerically** at a flop node (pot 20, to_call 6, hero AKs on Qh Jd 2c vs 77/T9, three-way,
`risk_aversion=0.15`): the CALL target moved by −3.57 (variance penalty ≈ −1.5, continuation delta
≈ −2.0, the latter within the 4-trial rollout's own noise); and at the same node with `to_call = 0`
the check target stayed strictly above fold (+7.87 vs 0.00), confirming the free-check carve-out.

**Explicitly NOT included** from finding #2's "Related:" clause (user scope decision): the multiway
`ev_if_called` single-caller-geometry fix, and "raise EV never models being re-raised". Both remain
open — see the resolution log.

## Change 3 [review #3, STACK-3]: ALLIN critic-consistency veto rescoped

`regret_match_policy_torch`'s ALLIN veto compared ALLIN against `values[..., :-1]` — every
non-ALLIN action, FOLD included. It is now `values[..., 1:-1]` (non-fold alternatives only), so it
can never fire merely because FOLD outranks ALLIN. This is the review's own suggested fix for #3.

**Documented honestly, in-code and here: under the `baseline_mode='fold'` that BOTH call sites use,
this is a provable no-op.** It only changes `best_non_allin` when FOLD is the argmax, and in exactly
that case `regrets[..., -1]` is already clamped to `0` by the fold-relative
`(values - values[..., 0:1]).clamp(min=0.0)` before the veto runs. It is kept because it is
correct-by-construction if a `'mean'` baseline is ever used here. A consequence worth recording:
the review's stated feedback loop for #3 ("veto → policy never jams → ALLIN Q trains only on the
risk-penalized counterfactual → Q stays low → veto keeps firing") **does not hold as written** — the
ALLIN Q-head is trained from the counterfactual target at every visited state regardless of what the
policy sampled, which is the whole point of the counterfactual-target architecture.

The genuinely open half of finding #3 — margin `0.15` possibly below critic noise, and four
risk-dampeners (variance penalty, realization discount, this veto, `TARGET_CLIP_BB=40`) stacked with
no joint calibration — is a **calibration** question and was deliberately not changed blind. Note
V40 changes two of those four dampeners' targets (Change 2 adds the variance penalty to CALL), so a
joint re-calibration pass is a reasonable follow-up before or after the first V40 run.

---

## Status

**TRAINED (100,003 hands, 2026-07-20, 2h44m) and DEPLOYED LIVE 2026-07-21 as an INTERIM model**
by explicit user request, for play-testing while V41 finishes training.
`core/decision.py`'s `active_model_name` is `'Herocules (v40)'`; V25-V29 remain registered as
rollback options (one-line change, or `set_active_model` at runtime).

Deployment honesty: `model_verify --full` was only PARTIALLY completed for this version. All FAST
checks ran (18 PASS / 5 WARN / 0 FAIL) plus `vpip_adapts_to_style` PASS and
`beats_offformula_stress` PASS; `bb100_vs_standard_fields` and `beats_frozen_predecessor` were cut
short to free CPU for V41's training run. The headline result that motivated deploying:
`multiway_shortstack_aggression` improved from **6/6 collapsed cells (V29) to 3/6**, with 3-way
aggression at eq 0.65 rising 0.01 -> ~0.5 -- i.e. the live [BET-3] symptom. Regressions to watch on
the table: `nash_pushfold_vs_chart` 83% PASS -> 78% WARN with the error direction FLIPPED (V29 folded
where Nash shoves; V40 shoves where Nash folds, at the very bottom of the range), and
`short_stack_polarization` 0.14 -> 0.22.

Smoke tests run:
- 750 hands/version instrumented betting-loop probe (table above) — no hangs, no exceptions.
- Direct numeric probe of `_mc_target_evs_sized` including the `to_call == 0` corner.
- **Full real pipeline**, `python -m versions.v40.self_play.train --num_hands 2200`: multiprocessing
  workers, the real config-driven pool (two TreeOpponents, lagged-self NN, Calling Station + TAG
  heuristics), vectorization, training loop and checkpoint save all end-to-end clean.
  **Sim speed 19 hands/sec** — the extra CALL continuation rollout (4 rollouts per decision instead
  of 3) keeps it well above the ~10 hands/sec floor where equity cost becomes a concern.

Next: a real training run (`target_hands: 100000`, fresh weights, no `--resume_path`) followed by
`model_verify --full`. Because the contract is unchanged from V29, `beats_frozen_predecessor` can
run for real this time by copying V29's `expert_main.pth` in as `frozen_v29.pth` — but note review
finding #4: that check does not currently seat the frozen model at all (`sim.past_model` /
`sim.disable_past_self` are dead attributes since the V18 refactor). Fix that first or the
head-to-head is again "beats a TAG field".

See: `.agents/skills/OFK/references/fable-review-resolution-log.md` (per-finding status) |
`versions/v29/SPECS.md` (contract, [OPP-2], critic-consistency filter, risk coefficient) |
`.agents/skills/OFK/references/known-shortcomings-backlog.md` ([BET-3], [STACK-3])

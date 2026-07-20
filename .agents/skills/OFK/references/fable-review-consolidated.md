# Fable Review — Consolidated Critique (V29 full-stack audit)

**Date Recorded**: 2026-07-20
**Related Files**: [simulator.py](file:///c:/REPO/Antigravity/AIPoker/versions/v29/self_play/simulator.py), [train.py](file:///c:/REPO/Antigravity/AIPoker/versions/v29/self_play/train.py), [contract.py](file:///c:/REPO/Antigravity/AIPoker/versions/v29/core/contract.py), [decision.py](file:///c:/REPO/Antigravity/AIPoker/core/decision.py), [table_state.py](file:///c:/REPO/Antigravity/AIPoker/core/table_state.py), [PHPHelp.py](file:///c:/REPO/Antigravity/AIPoker/PHPHelp.py), [checks.py](file:///c:/REPO/Antigravity/AIPoker/tools/model_verify/checks.py)

## Context

Full-stack critical review of the V29 ("Herocules") pipeline, run 2026-07-20 while V29 was the
active live model. Four parallel reviewers covered (1) data contract / feature encoding,
(2) simulation & opponents, (3) training & targets, (4) the live serving path; the highest-impact
claims were then independently spot-checked against source. Detailed per-area reports:
`fable-review-contract-encoding.md`, `fable-review-simulation.md`, `fable-review-training.md`,
`fable-review-live-serving.md` (same directory).

**Verification status of the top claims**:
- CONFIRMED by direct read: CALL exempt from variance penalty + continuation credit
  (simulator.py:1018-1047); rollout `is_active` slot-index bug + `stack=hero_stack` placeholder
  (simulator.py:593-645); OPP-7 remap emits `seat_0` keys the contract never reads
  (contract.py:194-243 reads only seat_1..seat_5); `beats_frozen_predecessor` sets dead attributes
  (`past_model`/`disable_past_self` appear nowhere in the V18+ simulator — checks.py:964-965 vs
  train.py:1531's comment that the V18 refactor replaced them).
- CONFIRMED empirically by the reviewer (1000-hand instrumented run): betting-round check bug —
  0/849 postflop checks were followed by any other player acting; BB never got its preflop option.
- All other findings: reviewed with file:line anchors but not independently re-executed.

## Overall verdict

Engineering discipline (append-only contract indices, shared money-scale helpers, fail-loud
checkpoints, calibrate-before-retrain) is genuinely good. But there are real, load-bearing defects
in every layer, and the recurring pattern is that **the validation layer is weaker than the
engineering layer** — several "verified" claims verified the wrong object.

## Top findings (ranked)

### Explains live behavior already logged in the backlog

1. **Betting round ends on any check — zero training data for post-check nodes** (simulator.py:1675,
   companion break ~L1411). Round terminates on `all_matched and last_raiser == -1`; postflop
   `highest_bet` starts 0 so `all_matched` is instantly true. Check-behind, check-raise,
   "checked to me", and BB limped-pot option nodes have literally never been trained on.
   `acted_this_round` exists but is never consulted by termination. Strong root-cause candidate
   for **[BET-3]** multiway passivity.
2. **Risk penalty (V28/V29) applies to raises only — CALL is exempt — and CALL gets no
   multi-street continuation credit either** (simulator.py:1018 vs 1037-1047). The penalty scales
   with pot size, i.e. exactly the multiway/high-equity spots where V29 refuses to raise. Second
   code-level mechanism for **[BET-3]**. Related: raise EV never models being re-raised;
   multiway `ev_if_called` assumes exactly one caller while `p_all_fold` multiplies over all seats.
3. **Critic-consistency veto margin (0.15bb) is below critic noise and self-confirms an ALLIN
   blackout** (train.py:274-278): veto → policy never samples jams → ALLIN Q-head trains only on
   the risk-penalized counterfactual → Q stays low → veto keeps firing. This is the concrete
   mechanism hypothesized in **[STACK-3]** and matches [VAL-1]'s 851/851 no-literal-jam finding.
   Four risk-dampeners now stack with no joint calibration: variance penalty (0.15, bump never
   recalibrated), realization discount, ALLIN veto, TARGET_CLIP_BB=40 (which also aliases 100bb
   losses to 40bb while the curriculum reaches 100bb).

### Validation integrity

4. **`beats_frozen_predecessor` never seats the frozen predecessor** (checks.py:964-965 sets
   `sim.disable_past_self`/`sim.past_model` — attributes the V18+ simulator does not have; the
   'past' seat falls back to a TAG heuristic). The V26→V25, V27→V26, V28→V27 head-to-head chain
   was actually "beats a TAG field". `_run_field`'s nit/fish fields are likewise stat-forced TAG
   bots, not the real archetypes. Same class as the v17_gauntlet dead-attribute bug.
5. **No confidence intervals anywhere**: at 3-4k hands SE(BB/100) ≈ ±13-19, so `bb100 > 0` and the
   −15 regression gate are ~1-SE decisions. `gameplay_eval.py` still runs at temp 1.0 vs serve 0.5
   (the documented trap, fixed in checks.py but not there). Both V29 knobs were calibrated against
   `deep_stack_ood_guard`'s own grid, then success partly declared on that same check (Goodhart).

### Simulation realism

6. **Every opponent raise is exactly 0.75 pot** (simulator.py:1627-1629) — heuristic, lagged-self
   NN, and TreeOpponent size choices are all discarded. Hero has never faced an open-jam, overbet,
   or min-raise in 100k hands; undercuts what the OPP-2 features can learn.
7. **Dead blinds**: pre-folding happens before blinds are posted (simulator.py:1232-1242 vs
   1331-1335) — in most late-run hands blind seats are corpses that paid and cannot defend;
   steal EV systematically inflated.
8. **NN opponents play a degraded self**: vs-random equity instead of the range-aware equity they
   trained on (gated `current_actor == 0`, simulator.py:1415-1436) and corrupted `call_amount` =
   `pot_odds * pot` (opponents.py:176,189). Also flatters NN head-to-heads.
9. **All six stacks always identical** (simulator.py:1202): never a covered/short opponent; the
   symmetry is the only thing masking a latent negative-`to_call` chip bug (L1577/L1638). Min-raise
   floor is `to_call + 1bb` (not last-increment); short all-ins reopen action incorrectly.

### Contract / encoding drift (the V20 disease, still active in the unshared parts)

10. **Rollout queries use a third, drifted encoder**: `is_active = (idx < num_opponents)` marks
    slots not real fold state, and every opponent stack is a `hero_stack` placeholder
    (simulator.py:597, 645) — while gradient tensors and live use real masks/stacks. The policy
    generates trajectories reading features that disagree with what it is trained and served on.
11. **[OPP-7]'s V27 fix is defeated at the tensor boundary**: the remap emits `seat_0` keys for
    non-hero actors but `ContractV12.to_tensors` reads only `seat_1..seat_5` — the real hero is
    still invisible to every non-hero NN query. The fix's verification checked the board_state
    dict, not what survives encoding. Backlog status "RESOLVED" should be revisited.
12. **`contract_version` is never validated — only `context_dim` width** (shared/manifest.py:82-88).
    V29 is protected only by accidentally-unique width; the next same-width contract change
    (e.g. a rescale) re-arms silent cross-scale loading.

### Live serving

13. **Call-button OCR miss silently becomes "free check" and force-masks FOLD**: PHPHelp.py:1275
    initializes `call_amount = 0.0` and never produces the `None` sentinel decision.py:534-538
    treats as the safe path — on a miss the model is forbidden from folding a real bet.
14. **Live serves an all-PAD action-history sequence** (found independently by two reviewers):
    training and all model_verify rollouts populate hero's action tokens; the live call
    (PHPHelp.py:1323-1331) never passes `action_history_raw`. No eval reproduces this live input.
    `table_state.action_history` is maintained but never consumed.
15. **Missing/corrupt weights degrade to random-weight play**: v29_engine.py:62-68 swallows the
    load failure; decision.py:442-444 never checks `.loaded`.
16. **Version dispatch fragility, worse than documented**: three hand-synchronized `is_vN` ladders
    (decision.py + two more in PHPHelp.py for equity/hand-strength imports), unknown-name fallback
    to V20, and an unmatched future version falls through to bridge_v9 → exception → caught →
    silently folds every hand while play continues.

Second tier (live): per-seat raise attribution inverts when two stack drops land in one ~1s frame
(seat-order iteration, table_state.py:236-255) — noise on exactly the new V29 features; raw-frame
pot under-reads trigger false hand resets (table_state.py:74-77) and `max()` latching (L113);
`committed` excludes posted blinds (0.5-1bb/hand train/serve skew); unknown HUD defaults to
Blue/super-nit live vs Yellow/average in training; empty seats break position arithmetic in
3-5-handed DoN endgames; the ≤8bb temp-0.2 ramp is serve-only (no eval applies it); decimal-stake
tables mix cents-scale stacks with float call amounts; 2-card partial board reads encode as river.

## What's genuinely solid

Side-pot/chip accounting correct incl. multiway all-ins; append-only contract indexing + shared
money-scale helpers killed the V20 drift class where used; self-describing width-checked
checkpoint loading; the counterfactual-target architecture (oracle equity, decoupled fold model,
variance formula whose mean provably matches the EV blend) is thoughtfully built; fold-when-free
mask and temp-0.5 sharpening exactly mirrored train↔serve; unusually honest in-code documentation
of partial fixes and calibration provenance.

## Suggested triage order

1. Live-safety, cheap: #13 (call_amount sentinel), #15 (check `.loaded`).
2. Evidence-chain, cheap: #4 (make beats_frozen_predecessor actually seat the frozen model via
   `build_opponent_pool`).
3. Model-quality retrain levers (bundle in one version): #1 (betting-round fix using the existing
   `acted_this_round`), #2 (variance-penalize CALL uniformly / give CALL its continuation credit),
   #6, #7. This is the [BET-3] package.
4. Then #3 (rescope the ALLIN veto so it can never fire when the only better action is FOLD —
   matches [STACK-3]'s own suggestion), #11, #12.

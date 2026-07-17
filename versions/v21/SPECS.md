# V21 — CLEANUP BUILD (structural, not a new-behavior experiment)

Clone of v20_preflopEq_AI. Same tensor contract (`context_dim=37`, `contract_version=5` —
unchanged), same architecture, same opponent pool composition. The explicit goal of this version
is "does the core train/sim loop still produce a model at least as good after removing dead
weight and consolidating duplicated mechanisms" — not a new hypothesis test. See the parent
session's review of the core train/sim loop for the full inventory this was drawn from.

## Motivation

A structural-soundness pass on the core training/simulation loop (requested directly, independent
of any specific model-behavior bug): identify methods/tunings that are load-bearing vs. accumulated
complexity (dead branches, two systems doing one job, knobs nobody remembers exist), before adding
new features on top. See `.agents/skills/OFK/references/known-shortcomings-backlog.md` for the
separate model-BEHAVIOR backlog (BET-1, OPP-2/3, etc.) — several items below connect to it.

## CONFIRMED

1. **Delete the critic-side preflop tightness prior** (`TIGHTNESS_PENALTY_BB`, `ENTRY_EQUITY_MARGIN`,
   the `if action_taken >= 1 and dp['street'] == 0` block in `vectorize_hand_samples`) and the now-fully-inert
   `disable_target_shaping` flag end to end (config key, `run_training` param, CLI wiring).
   Superseded by range-aware equity (dp['equity'] is already VPIP-conditioned once
   `range_aware_equity: true`, which every recipe since V15/16 runs) — validated dormant
   (`disable_target_shaping: true`) across V17/V17_gauntlet/V19/V20/V20_preflopEq/V20_preflopEq_AI.

2. **`model_share: 0.95`, `eps` stays `0.05`** — sums to 1.0, cleanly removes the steady-state 15%
   heuristic-anchor floor past the bootstrap cutover. `eps` NOT raised to 10%: its job is state-visitation
   coverage, not target quality (counterfactual EV targets are computed model-free regardless of
   what's visited), and no diagnosed gap motivates more noise in the ground-truth realized-return head.
   Exception-path fallback (model query failure -> heuristic) is untouched.

3. **Phase-4 `num_to_fold` reweight**: `[0.40,0.25,0.20,0.10,0.05]` -> `[0.10,0.25,0.30,0.25,0.10]`
   for `[0,1,2,3,4]` folded (i.e. `[6,5,4,3,2]`-handed). Biases hard toward 3-5-handed starting
   conditions while keeping thin full-ring/heads-up exposure (live play is 6-max). Confirmed
   already CORRECTLY excludes pre-folded seats from the model entirely (not a "fold that still
   lingers") — `active[s]=False; folded[s]=True` set at hand-init before dealing, and every
   downstream consumer (`active_opponents_mask`, `active_opps_count`, both `front_colors` AND
   `after_colors` in the range-aware equity call) filters on `not folded[s]`. Real in-hand folds
   (hero at simulator.py:1144, opponents at :1209) do the same immediately. The only thing ever
   "still considered" is a genuinely-active opponent who hasn't acted yet this street
   (`after_colors`'s VPIP fold-roll) -- that's Finding 2's deliberate uncertainty modeling, not a bug.

4. **Bootstrap/cutover timing, shifted 10k earlier**:
   - `bootstrap_alpha`: flat 1.0 for hands < 5,000 (was 10,000); linear decay 5,000-20,000 (was
     10,000-30,000); 0 after 20,000 (was 30,000).
   - `ACTOR_CRITIC_CUTOVER_HANDS`: 30,000 -> 20,000 (stays tied to the bootstrap-decay-end milestone
     by construction, not a separately-tuned number).
   - Phase-4 dynamic active players (`current_hand > 50000`): -> `40000`.
   - Phase-5 focus rounds (`hands_done >= 75000`): -> `65000`. Currently `disable_focus_rounds: true`
     in config so this doesn't fire this run either way, but shifted for consistency if re-enabled later.
   - `print_dashboard`'s hardcoded "Phase 1..5" labels: currently DISCONNECTED from the real active
     config (e.g. still says "Phase 3: Extreme Stacks" even with `disable_extreme_stacks: true` +
     `stack_depth_mix` actually driving stacks) -- fix while touching these thresholds so the
     dashboard stops actively lying about what's running.
   - `run_intermediate_sensitivity_check`'s 10k/25k print-checkpoints: shift to 5k/20k to match the
     new bootstrap milestones (cosmetic/diagnostic only, not a training-behavior change).

5. **Actor-target pipeline consolidation** (was inventory items #7-#12 — "two full target-construction
   systems glued together at a hardcoded hand count"). Plan:
   - Keep the two-regime CURRICULUM itself (pre-cutover: dataset-precomputed mean-baseline regret
     over the model-free MC EVs; post-cutover: live fold-baseline regret over the critic's own
     detached Q) -- this is a deliberate, validated design (the critic isn't trustworthy enough for
     a live fold-baseline regret computation before it's had a full bootstrap-decay's worth of
     training), not accidental duplication, and shouldn't be collapsed into one.
   - What SHOULD change: the realization-discount math (`POLICY_TIGHTNESS_BB`/`POLICY_TIGHTNESS_PIVOT`)
     is currently implemented TWICE independently — once inline in `vectorize_hand_samples` (Python,
     per-hand, dataset time) and again inline in `regret_match_policy_torch` (tensor, live, tagged
     "reproduces the SAME... realization discount"). Factor into ONE shared helper both call, so the
     two regimes can't silently drift apart if the constant/pivot is ever retuned.
   - Stop computing `Y_pol`/`pol_t` once `hands_done >= ACTOR_CRITIC_CUTOVER_HANDS` for the REST of
     the run — currently computed and shipped to the GPU every batch for the full run even though
     the post-cutover loss ignores it entirely (~86.7% of a 150k run: (150k-20k)/150k). Thread
     `hands_done` (available in `run_training`'s loop) into `vectorize_hand_samples` and skip the
     regret-matching computation once past cutover; leave `pol_t` as a placeholder zero tensor.
   - Side note (no action needed now, just documented): `regret_match_policy_torch`'s `baseline_mode='mean'`
     branch is currently never invoked anywhere (training always calls it with `'fold'`) — dead code
     kept for the V17-round-1 mechanism's reference/diagnostic value, not wired into any active path.

6. **Live-serve sampling temperature**: remove the stack-scaled sharpening divergence
   (`_stack_scaled_temperature`, `LIVE_POLICY_TEMPERATURE=0.5`, `SHORT_STACK_TEMPERATURE=0.2`,
   `SHORT_STACK_BB`/`DEEP_STACK_BB`) in `core/decision.py` entirely rather than neutering it with
   matching constants — delete the function, sample live at the SAME temperature training/eval
   actually uses (1.0, i.e. the raw regret-matching distribution, unsharpened). Also update
   `eval_pure_policy.py`/`gameplay_eval.py`'s `policy_temperature` override (currently set to 0.5
   specifically "to MATCH the deployed serve config") to 1.0 for the same reason, so eval stays
   consistent with whatever live actually does.
   - Rationale: this is the one item in the whole review that's a genuine TRAIN/SERVE BEHAVIORAL GAP,
     not a training-side redundancy — the live model has never actually been tested at the temperature
     it was trained/validated at. It's also the one hand-tuned "by feel" lever in the list, and has
     already caused a live incident (the 0.4 retune, reverted 2026-07-16, dropped rough VPIP 23%->8%
     one-sidedly on one example before being caught).
   - Known risk: `_stack_scaled_temperature` was originally added because raw-policy sampling produced
     observed live spew (folded 50% eq HU, spew-raised 2-14% air) under an EARLIER model generation.
     It's plausible the current regret-matching + fold-baseline + counterfactual-EV critic chain no
     longer produces that spew at all (regret-matching should already assign ~0 mass to clearly-bad
     actions once well-calibrated) -- testing at temp=1.0 live is exactly how to find out whether the
     original justification still holds, rather than assuming it does. Validate via `model_verify --full`
     AT temp=1.0 before deploying, same discipline the existing code comment already calls for.

6b. **Live-serve temperature fix is DEFERRED to V21's deploy step, not part of this implementation
   pass.** `core/decision.py` is shared (not version-namespaced) and today's live model
   (`v20_preflopEq_AI`) is still being served through `_stack_scaled_temperature` — editing it now
   would silently change PRODUCTION sampling behavior before it's been trained/validated at the new
   temperature, which is exactly the undisciplined-retune risk item 6 exists to avoid. V21 isn't
   wired into `core/decision.py`/`core/models/` at all yet (no engine class, no registry entry) —
   `model_verify` doesn't need one (it loads checkpoints generically via `shared.manifest`/
   `shared.registry`, not the per-version live-serving engine files), so training + verification can
   proceed without touching live serving. The temperature change lands when V21 is actually wired up
   for live testing, after `model_verify --full` validates it at temp=1.0.

## OPEN — discuss before deciding

7. **Aux heads (`opp_bluff`/`opp_strength`/`self_equity`, currently `aux_loss_weight=0.0`)**: fully
   inert today (zero gradient contribution; also NOT used by the live "Thinking:" narrative, which
   deliberately avoids them as untrained — see `[[live-thinking-narrative]]` memory). Original intent
   was multi-task regularization of the shared trunk (forcing the transformer to explicitly encode
   "what does this board/history imply about opponent strength" even though only Q/policy heads are
   consumed) -- a legitimate technique, just never validated in THIS lineage since the weight has
   sat at 0 since V14. Two real possible future uses: (a) representation-learning regularization if
   turned back on, (b) IF it's ever validated well-calibrated, a legitimate opponent-read signal for
   the live HUD (reopening a feature explicitly declined earlier for lack of training signal).
   Recommendation: don't fold into the main V21 150k run blind. Run a small, ISOLATED ablation
   (e.g. two ~50k-hand candidates, `aux_loss_weight=0.0` vs `0.05-0.1`, otherwise identical) and check
   `model_verify` for any movement before deciding to keep training it on or delete the heads outright.

## FOLLOWUP — explicitly NOT in V21's scope (own version)

8. **Boardstate/OCR information parity** — does the simulator record / does live OCR expose
   everything the model could use, and are the two symmetric? Concretely checked "do we record
   who/how much each opponent entered the pot with":
   - **Training (`simulator.py`)**: NO. Per-opponent context is stack (current), position,
     VPIP/AGG-color (an aggregate historical tendency across many hands, not this-hand-specific),
     and active mask. There's no per-opponent record of "entered for $X" or a per-opponent action
     sequence this hand -- `committed[]` is tracked internally for pot math but never surfaced into
     `decision_points`/`ctx`. This is exactly backlog `OPP-2` (no per-opponent action attribution)
     and `OPP-3` (size-blind action history).
   - **Live (`core/table_state.py`)**: the RAW signal already exists and is computed every tick —
     `_generate_timeline_actions` diffs each player's (including each opponent seat's) stack
     drop (`diff = last_stack - current_stack`) to detect bets/calls/raises — but it's immediately
     collapsed into one scalar (`self.current_street_bet_level = max(..., diff)`, "what's the
     current bet to call") instead of being kept per-opponent or written into a structured record.
     `self.action_history` is initialized and serialized in `to_dict()` but — in this file at least —
     never actually appended to; looks like unwired/vestigial scaffolding, not a working feature.
   - **Net finding**: this is good news for feasibility — live OCR does NOT need new vision work to
     support per-opponent entry-sizing; the per-tick stack-drop diff already IS that signal, it's
     just being thrown away on both sides (sim never records it, live collapses it to a scalar).
     Building this is a data-modeling change (new context feature(s) + wiring on both train and
     live paths + retraining to prove value), not a vision-capability gap.
   - Deliberately kept OUT of V21: this is a new-feature/new-context-width change, not a structural
     cleanup, and would confound "did the cleanup regress anything" with "did the new feature help."
     Own version once V21 is validated (working name: V22, or whatever's next in sequence).

9. **Deeper stack-depth curriculum + `STACK_CEIL_BB` raise** (raised 2026-07-17, while picking
   `stack_depth_mix` bands for the 100k run): widening the curriculum's sampled range past 50bb
   (proposed: `[5-14bb:0.35, 14-30bb:0.35, 30-60bb:0.20, 10-100bb:0.10]` or a disjoint variant)
   only teaches the model anything about deep stacks if it can actually PERCEIVE the difference --
   checked `contract.py` and found `scaled_stack_bb` hard-clips at `STACK_CEIL_BB=50.0` for BOTH
   hero's own stack and every opponent's stack: a 60bb and a 95bb stack produce the identical
   context value today (both saturate at `50/100=0.5`). Sampling deeper hands without raising this
   ceiling would train on more hands the model can't distinguish from a 50bb stack -- looks like
   progress, doesn't actually fix anything.
   - Fixing it means raising `STACK_CEIL_BB` (+ rescaling `STACK_SCALE` to match) in `contract.py` --
     a semantic change to an EXISTING feature, not just adding a new one, so it needs a
     `contract_version` bump per this codebase's self-describing-contract discipline (even though
     train and serve already share this exact function, so there's no cross-version scale-mismatch
     risk the way V20's original rescale had).
   - Deliberately kept OUT of V21 (decided 2026-07-17): would break V21's own stated goal of
     isolating the loop cleanup as the ONLY variable. Own version once V21 is validated -- bundle
     the `STACK_CEIL_BB` raise + a real deep-stack-aware `stack_depth_mix` together then, and
     re-derive the band shape (this doc's proposed numbers were provisional, not decided).

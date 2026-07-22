# Fable Review — Resolution Log (side-by-side)

**Date Recorded**: 2026-07-20
**Related Files**: [fable-review-consolidated.md](fable-review-consolidated.md), [SPECS.md](file:///c:/REPO/Antigravity/AIPoker/versions/v40/SPECS.md), [simulator.py](file:///c:/REPO/Antigravity/AIPoker/versions/v40/self_play/simulator.py), [train.py](file:///c:/REPO/Antigravity/AIPoker/versions/v40/self_play/train.py)

## Context

Companion tracker for the 2026-07-20 Fable full-stack V29 audit. The `fable-review-*.md` documents
are the reviewer's own words and are **never edited** — this file is the only place resolution
status is recorded. One row per numbered finding in `fable-review-consolidated.md`'s ranked list,
plus the second-tier live items.

Work so far, all under explicit user direction, all with **no contract change** (`context_dim=54`,
`contract_version=8` throughout, so every checkpoint in this chain stays compatible and the live
bridge needs no new wiring):

| pass | date | where | scope |
|---|---|---|---|
| 1 | 2026-07-20 | `versions/v40` (clone of v29) | Tier 1 — "explains live behavior already logged in the backlog" (#1, #2, #3) |
| 2 | 2026-07-20 | `tools/model_verify/checks.py` | validation integrity (#4) |
| 3 | 2026-07-21 | `versions/v41` (clone of v40) | simulation realism + encoding drift (#7, #8, #9, #10, #11) |
| 4 | 2026-07-21 | `core/decision.py`, `core/models/v41_engine.py` | live-serving safety (#15/H1, #14/H3, #16/H4) |
| 5 | 2026-07-21 | `PHPHelp.py`, `core/decision.py`, `core/table_state.py`, `core/models/v4[01]_engine.py` | **V42_liveFixes** — live money/OCR integrity (#13, M4, M6) + train↔serve encoding (#8-CE, #6-CE, #12-CE, #10-CE), plus the #16/H4 remainder that was actively degrading V41 |
| 6 | 2026-07-21 | `versions/v43` (clone of v41, NOT TRAINED) | **V43** — corrective-prior cleanup: realization discount + ALLIN veto removed, `TARGET_CLIP_BB` 40→100, variance penalty re-scaled 0.15→0.20. Resolves #3's calibration half and T-M5; overturns the premise of the `nash_pushfold_vs_chart` regression |

Passes 1 and 2 were deliberately narrow — no chasing of adjacent findings. Passes 3 and 4 were each
scoped by the user to a named list plus whatever those depend on.

**Pass 4 is the only one that changes LIVE behaviour**, and it does so for whichever model is
active — including V29, which was serving when it landed. All three are train↔serve corrections or
failure-mode fixes; none touch the tensor schema. The behavioural one to be aware of is #14: live
decisions now carry hero's own action history, which the model was trained and validated with but
had never been served. That is a genuine change in what the network sees mid-hand, in the direction
of train≡serve.

Status vocabulary: **FIXED** (code changed + verified) · **PARTIAL** (part of the finding
addressed, remainder named) · **OPEN** (untouched) · **NOT-A-DEFECT** (investigated, claim does not
hold) · **DEFERRED** (deliberate scope decision, who/why recorded).

---

## Tier 1 — "Explains live behavior already logged in the backlog" (pass 1, 2026-07-20)

| # | Finding | Status | Where |
|---|---|---|---|
| 1 | Betting round ends on any check — zero training data for post-check nodes | **FIXED** | `versions/v40/self_play/simulator.py` |
| 2 | Risk penalty applies to raises only (CALL exempt); CALL gets no continuation credit | **FIXED** (main claim) / **DEFERRED** (the "Related:" clause) | `versions/v40/self_play/simulator.py` |
| 3 | Critic-consistency veto self-confirms an ALLIN blackout | **PARTIAL** — rescope applied, mechanism **NOT-A-DEFECT**, calibration half **OPEN** | `versions/v40/self_play/train.py` |

### #1 — FIXED

Termination now requires every still-live seat WITH CHIPS to have both acted this round
(`acted_this_round`, already maintained, never consulted) and matched the current bet; the companion
early `break` is demoted to a safety net gated on `acted_this_round[current_actor]`. Also hardened
alongside, because longer rounds make it reachable: both raise branches use
`highest_bet = max(highest_bet, street_committed[actor])` so a stack-capped "raise" can no longer
LOWER the bet level and hang the hand (this is the latent bug the review's own **M5** describes —
recorded here as a byproduct fix, not as M5 being resolved).

Verified, 750 hands/version, three seeds, all seats instrumented: postflop actions/hand
**3.13 → 5.16**; preflop decisions at price 0 (the BB option the reviewer measured as literally
never occurring) **0 → 259**. Full derivation and the reviewer's own 0/849 measurement:
`versions/v40/SPECS.md` Change 1.

### #2 — FIXED (main claim) / DEFERRED (the "Related:" clause)

CALL now receives BOTH corrections every sized raise already had: the [V25]
`_rollout_continuation_ev` credit (skipped when the call is itself all-in, and on the river), and
the [V28/V29] closed-form variance penalty via the same `_outcome_variance` helper instantiated for
CALL's own 2-point mixture. **One deliberate exception**: no variance penalty when `to_call == 0` —
a free check risks no chips while FOLD, the regret baseline, is a flat `0.0` with no penalty, so
penalizing a free check would tilt the target toward folding for free (the `free_check_low_fold`
corner). Verified numerically including that corner — `versions/v40/SPECS.md` Change 2.

**DEFERRED, user scope decision (2026-07-20)**: the finding's "Related:" clause — (a) raise EV never
models being re-raised, (b) multiway `ev_if_called` assumes exactly one caller while `p_all_fold`
multiplies over all seats. A candidate fix for (b) exists, written in an earlier session and left
behind in the untracked `versions/v30/` scratch clone (call-probability-weighted HU called-equity,
consistent with the single-caller pot geometry). It was explicitly excluded from V40 to keep the
variable count small. **If `versions/v30/` is ever cleaned up, that fix is the only thing in it
worth keeping** — and note its clone was broken anyway (its `train.py`/`simulator.py` still imported
`versions.v29.*`, so training it would silently have run V29's code).

### #3 — PARTIAL: rescope applied; the stated mechanism does NOT hold; calibration half open

Applied the review's own suggested fix: the veto's dominance comparison runs over `values[..., 1:-1]`
(non-fold alternatives) instead of `values[..., :-1]`, so it can never fire merely because FOLD
outranks ALLIN.

**Investigation result, recorded so it is not re-derived**: under `baseline_mode='fold'` — which
BOTH call sites use — this rescope is a **provable no-op**. It changes `best_non_allin` only when
FOLD is the argmax, and in exactly that case ALLIN's regret is already clamped to `0` by the
fold-relative `(values - values[..., 0:1]).clamp(min=0.0)` that runs before the veto. Consequently
the review's stated feedback loop ("veto → policy never samples jams → ALLIN Q-head trains only on
the risk-penalized counterfactual → Q stays low → veto keeps firing") **does not hold as written**:
the ALLIN Q-head is trained from the counterfactual target at every visited state regardless of
which action the policy sampled — that is the entire point of the counterfactual-target
architecture. The rescope is kept as correct-by-construction should a `'mean'` baseline ever be used
here.

**STILL OPEN** (calibration, deliberately not changed blind): margin `0.15` may sit below critic
noise; four risk-dampeners — variance penalty, realization discount, this veto, `TARGET_CLIP_BB=40`
(which also aliases 100bb losses to 40bb while `stack_depth_mix` reaches 100bb) — stack with no
joint calibration. V40's Change 2 alters one of the four (the variance penalty now covers CALL), so
a joint re-calibration pass is a sensible follow-up. `versions/v40/self_play/calibrate_critic_consistency.py`
is the existing harness.

---

## Tier 2 — validation integrity (pass 2, 2026-07-20)

| # | Finding | Status | Where |
|---|---|---|---|
| 4 | `beats_frozen_predecessor` never seats the frozen predecessor; `_run_field`'s nit/fish fields are stat-forced TAG bots | **FIXED** | `tools/model_verify/checks.py` |

### #4 — FIXED

New shared helper `_seat_opponent_pool(rc, sim, pool_config, model_loader=None)` in
`tools/model_verify/checks.py` calls the version's own `opponents.build_opponent_pool` exactly the
way `train.py`'s `simulate_worker` does (same `heuristic_bots` mapping, same injected
`_query_model_decide` / `_note_query_error`). This tool never called that builder, so since the V18
refactor `sim.opponent_pool` stayed `{}` and EVERY seat fell through simulator.py's lookup-miss
fallback — `HeuristicOpponent(style, self.tag_heuristic, forced=True)`.

Both halves fixed:
- `_run_field` now seats the REAL archetype bots. Verified: the `nit` seat is `Nit (Heuristic)` and
  the `fish` seat is `Calling Station (Heuristic)`, neither backed by `sim.tag_heuristic` any more.
- `check_beats_frozen_predecessor` drops the two dead attribute writes
  (`sim.disable_past_self` / `sim.past_model`) and seats the frozen checkpoint as a real
  `NNOpponent` on the `past` seat, plus a **fail-loud guard**: if that seat is not `kind == 'NN'`
  the check returns SKIP rather than reporting a head-to-head against a heuristic. This bug class
  has now recurred twice (v17_gauntlet's `tag_model`, then this), always by degrading quietly — the
  guard is the point, not just the fix. The result string and JSON now name what actually played
  (`past_seat`).

Verified end-to-end against V29 (hero vs a frozen copy of itself): seat reported as
`EXPERT_MAIN (NN)`, and the degraded-loader case correctly resolves to `Heuristic` so the guard
would fire.

**Consequence to remember**: every "beats frozen V(N-1)" result from V18 through V29 measured
"beats a TAG field". Do not treat that chain as evidence. `tools/model_verify/baselines.json` only
holds entries for `v20_preflopEq` / `v20_preflopEq_AI`, so the changed field composition creates no
spurious regression WARNs for any current version — but any baseline recorded before 2026-07-20 was
measured against TAG-bot fields and is not comparable.

**For V40 specifically**: `versions/v40/weights/frozen_v29.pth` is staged (a copy of V29's
`expert_main.pth`) and verified to load into V40's architecture — V40's contract is unchanged from
V29, so this is the first genuine version-over-version head-to-head available since the V18
refactor. Note only the exact filename `expert_v7_selfplay.pth` triggers train.py's warm-start hook
(review L5), so a `frozen_*.pth` in the weights dir cannot contaminate a fresh run.

---

## Tier 3 — simulation realism + encoding drift (pass 3, 2026-07-21, `versions/v41`)

`versions/v41` clones V40 (BET-3 package inherited unchanged), **no contract change**
(`context_dim=54`, `contract_version=8`). Scope set by explicit user direction. Every row below has
a measured before/after, not just a code diff — full derivations in `versions/v41/SPECS.md`.

| # | Finding | Status | Measured effect (V40 → V41) |
|---|---|---|---|
| 7 | Dead blinds — pre-folding happens before blinds are posted | **FIXED** | hands reaching a flop with ≥1 dead blind **47.6% → 0.0%** (111 dead blind seats → 0, over 400 hands) |
| 8 | NN opponents play a degraded self (vs-random equity; corrupted `call_amount`) | **FIXED** | range-aware equity now serves any NN actor; `call_amount` was `to_call·pot/(pot+to_call)`, a pot-sized bet arriving at half size — now inverted exactly |
| 9 | All six stacks identical; min-raise floor; short all-ins reopen action | **FIXED** (3 of 3) | hero decisions facing a covering opponent **0 → 199** per 250 hands; opp stack spread 1.05 → 1.64; min 3-bet after a 3bb open **4bb (illegal) → 5bb** |
| 10 | Rollout queries use a third, drifted encoder (`is_active` slot-index, `stack=hero_stack`) | **FIXED** | both now read ground truth via `folded`/`stacks` threaded through `table_state` |
| 11 | [OPP-7]'s V27 fix defeated at the tensor boundary (`seat_0` keys unread) | **FIXED** | hero dropped on **128/128** NN-opponent queries in V40, **0** in V41 |

### #11 / #10 — the same block, fixed together

V27's remap ("the 5 slots are every seat except the actor") was right, but it keyed each slot by the
**absolute** seat number while `ContractV12.to_tensors` reads only `seat_1..seat_5`. For a non-hero
actor that meant the real hero was written to `seat_0` — a key the encoder never reads — *and* the
surviving slots were misaligned (actor_seat=4 wrote seat_0/1/2/3/5, so the encoder's 4th slot found
no `seat_4` and fell back to an inactive default). Now keyed by **slot index**; the absolute seat
number survives in `name`, and `opponents_profiles` lookups use the absolute key explicitly. For
`actor_seat == 0` both indices coincide, so hero's own query is byte-identical.

**Backlog action**: [OPP-7]'s "RESOLVED" status is misleading and should read *resolved in the
board_state dict from V27, resolved at the tensor boundary only from V41*. The general lesson —
V27 verified the dict, not what survived encoding — is the same failure mode as #4 (verifying the
wrong object), which is why both were caught by the same reviewer pass.

### #9 — what asymmetry unlocked

Equal stacks were the only thing keeping two rule bugs unreachable, so fixing the symmetry required
fixing both: the min-raise floor (`to_call + last increment`, not always `+1bb`) threaded through
`_raise_size_for_fraction`, the opponent raise branch **and** `_mc_target_evs_sized` so the
counterfactual targets only price legally-makeable sizes; and short all-ins no longer re-opening
betting. V40's `highest_bet = max(...)` guard, added there defensively, is genuinely load-bearing
now. *Residual deviation, deliberately not fixed and worth knowing*: a player facing an incomplete
all-in can still choose a raise from the action space; enforcing call-or-fold needs a per-actor
action mask.

### #7 — the honest cost

Protecting 1–2 of the 5 opponent seats as blinds caps the deepest pre-folds, shifting the starting
field slightly toward **larger** fields. That is a real change to the training distribution — and it
moves toward the multiway conditions [BET-3] is about, not away from them.

---

## Tier 4 — live-serving safety (pass 4, 2026-07-21, `core/decision.py`)

Requested by name (the live report's H1/H3/H4). These touch the LIVE path, so they change behaviour
for whichever model is active — including V29, which was serving at the time. All three are
train↔serve corrections or failure-mode fixes, none change the tensor schema.

| # | Finding | Status |
|---|---|---|
| 15 (H1) | Missing/corrupt weights degrade to random-weight play | **FIXED** |
| 14 (H3) | Live serves an all-PAD action-token sequence | **FIXED** |
| 16 (H4) | Version dispatch: substring ladders, V20 fallback, crash-fold default | **PARTIAL** — both silent-failure paths closed |

### #15 (H1) — FIXED

`make_decision` now refuses to act when the active engine's `.loaded` is False, returning FOLD with
the load error in the reason instead of serving a freshly-initialised network at a real table.
Engines without the flag are treated as loaded (legacy). The engine constructor still swallows the
exception deliberately — the registry builds *every* engine at init and one missing rollback
checkpoint must not take the app down — so `.loaded` is the contract, and the guard has to live at
the point of use. Added alongside: a startup health line per engine (`weights=OK/FAILED`,
`tensors=own bridge/version ladder`, active model marked), because the old single WARNING scrolled
past unnoticed among many.

### #14 (H3) — FIXED

Training and every model_verify rollout feed hero's own past-action tokens (7=fold, 3=call,
6=raise) into `bridge.to_tensors`; live never passed `action_history_raw`, so `to_tensors` filled
all 20 slots with PAD. The transformer was trained and validated with its own line populated and
served with it blank — anything it learned to condition on its own past actions (barrelling after
raising, giving up after checking) was silently unavailable, and **no eval reproduced the live
input**.

Fixed inside `decision.py` rather than in PHPHelp deliberately: `decision.py` already owns
`hand_history_buffer` and its reset rules, so appending exactly one token per state **in the same
call that appends the state** makes misalignment structurally impossible. (`PHPHelp.py` does
maintain a `table_state.action_history`, but it is a list of chars appended *after* the decision,
outside the buffer's lifecycle — wiring that in would have re-created the alignment bug the fix
exists to prevent.) An explicit `action_history_raw` from a caller still wins, for replay harnesses.

Verified over a 4-decision hand: tokens fed were `[…0,0,0]` → `[…6,0]` → `[…6,6,0]` → `[…6,6,6,0]`,
with the current step correctly left as PAD (the transformer shifts by one internally), and
`len(hero_action_buffer) == len(hand_history_buffer)` after each call.

Raises are size-blind (every bucket emits 6) — that is the pre-existing [OPP-3] limitation carried
over from the simulator, not something introduced here.

### #16 (H4) — PARTIAL: both silent failures closed, the ladders remain

Two failure modes fixed, both of which failed *quietly*:
- **Unknown model name silently swapped in V20** — nine versions stale, a different contract, and
  indistinguishable at the HUD from the model you asked for. Now it keeps the currently-active
  (known-good) model and logs an error naming the registry contents.
- **A registry entry the ladder didn't recognise fell through to `bridge_v9`**, threw, was caught,
  and **folded every hand** behind a generic "Fatal decision engine crash" while play continued.
  The v9 branch is now explicit, and the final `else` raises an error that names the actual fix.

Structural improvement: engines may now declare `make_bridge()`, and `decision.py` resolves it once
at startup into `_engine_bridges`, short-circuiting the ladder entirely. `V41ModelEngine` does this,
so V41 needs no ladder entry and cannot be misrouted by substring collision (a future `'v41b'`
matching `'v41'`). This is the direction the OFK guardrails point — manifest-driven dispatch rather
than more `is_vN` branching — applied incrementally: existing engines keep the ladder untouched.

**Demonstrated for real, same day**: deploying V40 an hour after writing this fix, I added its
engine and registry entry, confirmed it loaded and served -- and it emitted a bare `RAISE_POT` with
`size=0.0` instead of an executable `RAISE_SLIDER_x`, because I had not added `is_v40_model` to the
`is_sized_model` ladder. The bridge was fine (V40 declares `make_bridge()`); it was one of the
OTHER ladder consumers that silently degraded. Live, that is a raise the executor cannot size. This
is precisely the finding, reproduced by the person who had just read it -- which is the argument for
finishing the job: `make_bridge()` only removed ONE of the ladder's consumers. `is_sized_model`, the
aux gate and the tag ternary all still need the same treatment (an engine should declare "I am a
sized model", not be recognised by substring).

**Demonstrated for real a SECOND time, same day** — and this one reached the user. The V40
deployment was reported as live and testable, but the model never appeared in the dashboard's
**Decision Model dropdown**: `PHPHelp.py` carried a *fourth* hand-maintained copy of the registry
(values AND default), with a comment explicitly warning "bump both here too" that I did not follow.
The user saw a dropdown offering v29..v13 and displaying `Herocules (v29)` while `decision.py` was
actually serving v40 — **the HUD naming a different model than the one acting**, which is exactly
the class of silent mislabelling this finding is about. Fixed by DERIVING both the values and the
default from `decision_engine.models` / `active_model_name` (the engine is constructed at
`PHPHelp.py:157`, well before the dropdown at :268), filtering entries whose `.loaded` is False
since #15's guard would only make them fold. Deploying a version now needs no dropdown change at
all. Verified headlessly: 16 registered, all loaded, default `Herocules (v40)`, present in list.

**Still open**: the ladder itself (13 flags, the nested-ternary tag, the aux gate) and the two
*other* ladders in `PHPHelp.py` (per-version `compute_range_aware_equity` and
`preflop_hand_strength` imports), which can still silently drop a new version to vs-random equity —
a train/serve mismatch that does not announce itself. Converting those to the same
engine-declares-its-own-behaviour pattern is the natural follow-up.

**Lesson worth generalising** (twice in one day, both by the author of the fix): "deploy a version"
is not one edit, it is N hand-synchronized edits across files that do not reference each other, and
every miss degrades *quietly* — a raise with no size, a dropdown that omits the model, a HUD label
naming the wrong net. Neither miss threw. Any remaining hand-maintained per-version list should be
treated as a latent silent-failure site, not a style complaint.

---

## Outcome — V41 deployed live 2026-07-21, [BET-3] resolved

The review's first two tiers landed as V40 (findings #1/#2/#3) and V41 (#7/#8/#9/#10/#11), plus the
live-serving tier (#14/#15/#16) and the tooling fix (#4). V41's `model_verify --full`:
**22 PASS / 5 WARN / 0 FAIL / 0 SKIP** — the cleanest of any version and the first with zero skips.
Report at `V41/model_verify_report.html`.

- **[BET-3] resolved.** 3-way aggression at eq 0.65: **~0.01 (V29) → 0.81 (V41)**, flat from
  heads-up. The reviewer's root-cause call on finding #1 was correct and was the single highest-value
  item in the document: the model wasn't passive, it had **never seen a postflop node** where anyone
  acted after a check.
- **Finding #4 paid off immediately.** `beats_frozen_predecessor` ran as a genuine head-to-head for
  the first time since the V18 refactor (+64.3 BB/100 vs a field including frozen V40 as a real
  `NNOpponent`). Every such result before this fix was measuring something else.
- **What the review did NOT fix, and V41 still carries**: `nash_pushfold_vs_chart` 83% → 78%,
  introduced by V40 with the error direction FLIPPED (V29 folded where Nash shoves; V40/V41 shove
  where Nash folds, with weak suited trash at 5bb). Not addressed by any finding here — it wants its
  own investigation via the V30/VAL-1 tooling. [OPP-8]'s two flat-response WARNs are likewise
  untouched by this review.

Finding **#6** (every opponent raise is exactly 0.75 pot) is the last open member of the [BET-3]
bundle and the reviewer's own nomination for the next version.

---

## Tier 5 — V42_liveFixes: live money/OCR + train↔serve encoding (pass 5, 2026-07-21)

Live layer only — `PHPHelp.py`, `core/decision.py`, `core/table_state.py`, two engines. **No
contract, simulator, training or weights change**, so it applies to whichever model is active and
V41 keeps serving unchanged. Full derivations and measured before/after:
`versions/v42_liveFixes/SPECS.md`.

**No training clone, no retrain, and the reason matters**: every finding in this pass turned out to
live in the shared live layer, and every fix moves the live input distribution TOWARD what V41 was
trained on (range-aware equity instead of vs-random, a real `hand_strength` instead of a constant,
training's own unknown-HUD default, real button-relative positions). A `versions/v42/` clone would
have been a byte-identical copy of V41's slice needing a 100k retrain to become servable, to
reproduce a model identical by construction. The one item here that WOULD justify a clone + retrain
is named under "still open" below.

### The finding that was not in the review, and was live

**V41 — the milestone model, deployed that morning — was being served vs-random equity and a
constant `hand_strength = 0.5`.** `PHPHelp.py`'s two remaining per-version substring ladders (the
ones #16/H4's "still open" paragraph names explicitly) stopped at `'v29'`; V40 and V41 were both
deployed without being added. So `ctx[3]` — the feature this equity-primary architecture is built
around — came from an estimator the model was never trained on, `equity_edge` (ctx[35]) drifted
with it, and `hand_strength` (ctx[36]) was the neutral default for AA and 72o alike. Nothing threw;
the only outward symptom was a HUD panel showing "-".

Fixed the same way `make_bridge()` fixed the tensor ladder: the engine declares `live_features()`
(V40 and V41 do), `core/decision.py::live_feature_providers()` resolves it, and the legacy name
mapping lives there as the single copy. A model matching neither resolves to `source='unresolved'`
and is logged as an ERROR rather than degrading quietly. Verified: all 17 registered models resolve
to their own package, byte-identical to the old ladder for V13–V29.

**This is the third time in two days that "deploy a version" silently missed a hand-maintained
per-version list** (after `is_sized_model` and the dropdown registry). The lesson in pass 4 —
"any remaining hand-maintained per-version list is a latent silent-failure site" — was right, and
this one was live on the primary input feature for the whole deployment.

| # | Finding | Status | Note |
|---|---|---|---|
| 13 | Call-button OCR miss becomes "free check" and force-masks FOLD | **FIXED** | The `None` sentinel was unreachable from live; `make_decision` now takes `call_amount_known` and masks FOLD only on a *positively identified* free check. The fabricated `2.0` and `100.0` chip constants (which knew nothing about the blind level) are replaced by real evidence: this street's observed bet level, or hero's stack when there is no call button. |
| — (M4) | `check_call_available` accepted and ignored | **FIXED** | CALL is now masked when the button is absent, in both policy paths and the degenerate fallbacks. Verified over 400 draws/case. |
| — (M6) | Decimal-stake money units | **FIXED** | Vision digit-strips stacks (`"1.50"`→150) and blinds are ×100, so the pipeline is in cents — but the call parser used `float("0.20")=0.2` against bb=20, making a €0.20 bet 0.01bb. `_parse_button_money` mirrors vision's digit-strip exactly. |
| 8 (CE) | Live unknown-HUD default is a super-nit; training's is average | **FIXED** | `Blue` (0.10/0.18, the tightest band expressible) → `Yellow/Green` (0.30/0.46), matching training AND the equity path, which already used `or 'Yellow'`. The same opponent was being described two different ways to two consumers in one decision. |
| 6 (CE) | Live equity estimator differs from the trained-on feature | **FIXED** | The front/after split now driven by the version's own `live_features()['front_colors']`; the 250-vs-150 sims difference is unchanged and documented. |
| 12 (CE) | Short-handed tables / position arithmetic assumes gap-free seating | **FIXED** | Positions counted over the OCCUPIED ring. No contract change needed for opponents either: the contract reads slot `j` as `(j+1+hero_position)%6`, so writing an opponent whose true position is `p` into slot `(p-hero_position)%6` makes the encoder emit exactly `p` (injective mod 6, never collides with hero's 0). **Identical to the old behaviour at a full table** — verified for all 6 button positions. 4-seated measured: hero BB encoded as UTG (3) → correct (2); opponents 4,5,1 → 3,0,1. |
| 10 (CE) | Partial 1–2 card board reads alias to River | **FIXED** | The contract sends any `board_len` outside {0,3,4} to `street_level=3.0`, so a mid-deal frame encoded as a river state with three PAD cards. The decision loop now waits for a complete board. |
| — (live L3) | `HEROCULES_CRITIC_ARGMAX` silently swaps live selection to an eval-unvalidated critic-argmax mode | **FIXED** | Found by the user spotting the amber "Q-CRITIC MODE" HUD header. `PHPHelp.py`'s `CRITIC_ARGMAX_MODE = False` was a **dead variable**: the line that pushes it into the env var `core/decision.py` reads at import was commented out (`##`), so the flag's own documented promise — "Authoritative: this wins over any `HEROCULES_CRITIC_ARGMAX` left in the shell" — was false, and a leftover env var from a testing session put live play on the critic's argmax-Q instead of the sampled actor policy. Re-armed, and `_report_engine_health` now prints a loud startup line when the mode is ON (the review's actual complaint was that nothing surfaced it). Verified both ways: with `HEROCULES_CRITIC_ARGMAX=1` in the shell the flag still resolves False; importing `decision.py` alone with the var set still resolves True, so the diagnostic mode is not broken, just no longer reachable by accident. |

**Still open from this area, and the one thing that would justify a V43 clone + retrain**: a live
short-handed table leaves the empty contract slots at the absent-seat default (mask 0, stack 0),
while training's inactive seats are always *folded players with real stacks and colours*. Positions
are now correct; that residual only closes by teaching the simulator to seat 3–6 players. Also
open: on an unreadable call button with no observed bet this street the price estimate is 0 but
FOLD stays available, so a weak hand can fold a genuinely free check — the deliberate direction of
the asymmetry, but a one-frame retry would remove both errors.

---

## Tier 6 — V43: corrective-prior cleanup (pass 6, 2026-07-21, `versions/v43`, NOT TRAINED)

Clone of V41, **no contract change** (`context_dim=54`, `contract_version=8`). Scope set by explicit
user direction: strip the corrective priors that were compensating for defects V40/V41 have since
fixed at the source, fix `TARGET_CLIP_BB`, and reconcile what depends on it. Full derivations:
`versions/v43/SPECS.md`. **V41 remains live; V43 has no trained weights.**

| # | Finding | Status | Note |
|---|---|---|---|
| 3 (calibration half) | Four risk-dampeners stacked with no joint calibration | **RESOLVED** | Not by calibrating all four — by ABLATING them. Realization discount and ALLIN veto REMOVED; variance penalty KEPT (measurement says its pathology is not gone at the source); `TARGET_CLIP_BB` re-set and the penalty re-scaled to match. |
| T-M5 | `TARGET_CLIP_BB=40` vs a 100bb curriculum | **FIXED** | 40 → 100, matching `STACK_CEIL_BB`. Measured first: the clip truncated **23.4%** of realized go-forward returns (p95 102bb, max 167bb), not a corner. Gradient-safe — the critic loss is `HuberLoss(delta=2.0)`, so gradient magnitude saturates at 2bb of error regardless. |
| T-M9 | Short-stack action aliasing | **STAGED, default OFF** | An `allin_by_chips` flag exists in the simulator but is inert (byte-identical to V41) — because measurement showed T-M9 is only **2.4%** of decisions while a *different*, unreported aliasing (min-raise floor swallowing every pot fraction) is **40.7% overall / 56% preflop**. Fixing the small one first would have been backwards. |

### The measurements, and what they overturned

Three hypotheses died on contact with their own numbers — the V20 discipline, applied before any
retrain rather than after:

1. **The fixed-bb realization discount does NOT cause the inverted commit-vs-stack slope.** 349 Nash
   cells × 11 dampener configurations: turning the discount OFF ENTIRELY moves commit mass by 0.02.
   None of the four dampeners controls that threshold. **The locked V12 fix was never touched on a
   hypothesis.**
2. **`nash_pushfold_vs_chart` is substantially a CHECK ARTIFACT.** It scores `agg_mass > p_fold` —
   four aggressive heads against one fold head — and at its own probe node `raise_33`/`raise_66`/
   `raise_pot` are **all the same 1.5bb min-raise** (the min-raise floor exceeds every pot fraction
   at a 1.5bb pot). ALLIN, the only action Nash models, has NEGATIVE target EV at the failing cells:
   on the question Nash actually asks, the model AGREES. V29's better 83% partly reflects it being
   too passive — the [BET-3] failure V40/V41 deliberately fixed. **Score this check on
   ALLIN-vs-fold before treating it as a training target.** Same "validated the wrong object"
   pattern as #4 and #11, now inside the review's own recommended metric.
3. **The 40bb clip was an undeclared deep-stack all-in dampener.** Raising it to 100 with
   `risk_aversion_coefficient` left at 0.15 takes the ALLIN-vs-next-best gap trend from V41's +1.39
   to **+6.53** — i.e. the naive clip fix alone would have regressed [BET-1], the check V29/V41
   finally got fully negative. 0.20 restores it (+1.05). **The clip and the variance penalty are not
   independent knobs**; 0.25+ was rejected because a single-cell trace shows ALLIN becoming
   *dominated* (1.3 vs raise_pot's 3.1 at 100bb), risking the opposite pathology.

### Removed knobs fail loud

A config still setting `policy_tightness_bb` or `critic_consistency_margin` now raises. Verified by
injection. The repeated lesson from this whole review — dead `past_model` attrs, the commented-out
`CRITIC_ARGMAX_MODE` line, the stale `PHPHelp.py` ladders that served V41 vs-random equity — is that
this codebase's characteristic failure is a removed thing quietly becoming a no-op.

**Expected regression to watch when V43 trains**: entry range / VPIP should WIDEN (entry rate at
eq ≤ 0.35 measured 0.59 → 0.79 without the discount). That is the intended consequence of letting
correct inputs teach entry discipline instead of imposing it — but it is the most likely thing to
go wrong, and V41 is the rollback.

---

## Tier 7 — V42_liveFixes round 2: the front/after equity split (pass 7, 2026-07-21)

Not from the review. From a **flagged live hand**:
`history/Turbo_1171580052/flagged/turn_2_20260721_201440` — V43 folded **QQ preflop at 100%
confidence** facing 1bb with 75bb behind, and the bot clicked it.

**Not a model fault, and not a V43 regression.** Replayed on the identical input tensor V41 folds
it too (0.996), so rolling back would not have helped. The model was fed **equity 0.38**; fed the
equity the training simulator produces for that hand it raises at *every* opponent count 1–6
(0.72→0.59 equity, P(FOLD) = 0.000 throughout).

**Cause**: `front_colors` means "guaranteed to showdown, no fold-roll" — the strongest claim the
equity model can make about an opponent — and `_classify_opponents_by_action_order` awarded it on
**seat position alone**. Training builds `front` from real `acted_this_round`/`folded`/all-in
state; live has no fold detection, so folded seats became locked-in showdown opponents. The
flagged pot held 30 chips = the blinds and nothing else, and three opponents were marked committed.
QQ: **0.38 → FOLD** vs **0.62 → RAISE_POT** once `front` is correctly empty.

Fixed by making committed chips the criterion (preflop ignores position entirely, which also
catches a 3-bet from *behind* hero that position could never see; posted blinds are involuntary and
don't count; postflop keeps the positional read, intersected with chips-in-hand). Details and the
full before/after in `versions/v42_liveFixes/SPECS.md`; regression test
`versions/v42_liveFixes/verify_front_colors.py` (7/7).

**Same hand, second finding — the diagnostic was lying.** `summary.txt` rendered the real
75BB/1.5BB/1.0BB node as 300BB/6.0BB/4.0BB and raised a **false** `MODEL-INPUT vs RAW-OCR MISMATCH
-> BRIDGE issue` banner against a correct bridge, costing triage time on a fabricated lead.
`_decode_model_input` held *another* substring ladder (`is_v20_family`) that stopped at `v29` —
the **third** time this one decoder has been wrong for the live model. Now sourced from the version
contract's own `STACK_SCALE`/`POT_SCALE`/`CALL_SCALE` via `core/decision.py::context_scales()`.
Third symptom of the same bug: every turn recorded under V40/V41/V43 logged `to_call` 4× too large.

**The generalisable lesson, now demonstrated a third time in two days**: a hand-maintained list of
version names in a consumer file is not a mechanism, it is a countdown. Each instance
(`is_sized_model`, the equity/hand_strength ladders, `is_v20_family`) failed *silently* and only
surfaced through a live hand. Prefer asking the authoritative source — the engine, or the contract
module itself.

**Third finding, same round — `is_active` was not monotonic within a hand.** `TableState.update()`
assigned `is_active = raw_active` unconditionally, so a folded seat could go False → True on one
bright frame (deal animation, timer overlay, chip graphic over the name plate). A folded player
cannot re-enter; the section header already claimed "Monotonic Decay" and the inline comment
already claimed "they stay folded". Cost of one phantom seat, AKs preflop: **4 opponents → CALL
(fold 0.12), 5 opponents → FOLD (fold 0.91)** — `num_active` feeds equity AND
`equity_edge = equity × (num_active + 1)`. Fixed with a per-hand `folded_this_hand` latch;
`verify_fold_monotonic.py` 15/15.

**A false lead worth recording, because the reasoning error generalises.** The turn record listed
5 active opponents while the stored screenshot plainly shows a player sitting out — which looked
like proof that vision cannot detect folds, and nearly bought a card-back detector plus a retuned
brightness threshold. Running the real `read_board_state` on that frame disproves it (`seat_1:
is_active=False, state='Folded'` → `num_active_players = 4`). The record and the screenshot are
**different frames**: `save_diagnostics` stores `last_raw_img` at F12-press time, seconds after the
decision. **When a record and a screenshot disagree, confirm they describe the same instant before
blaming the detector.**

**Watch next**: this depressed equity on *multiway* hands specifically, and it has been live for
every version since V20_preflopEq. Any conclusion that a model "plays too tight" drawn from live
observation during that window is suspect, since `model_verify` measures in the simulator where
this bug does not exist. The [P4]/`vpip_adapts_to_style` thread should be re-read with that in
mind.

---

## Tier 2+ — not addressed

All **OPEN**. Listed so nothing silently drops off; ranking is the reviewer's, not a work order.

| # | Finding | Note |
|---|---|---|
| 4 | `beats_frozen_predecessor` never seats the frozen predecessor (dead `past_model`/`disable_past_self` attrs) | **FIXED 2026-07-20** — see below. |
| 5 | No confidence intervals anywhere; `gameplay_eval.py` runs temp 1.0 vs serve 0.5; Goodhart on `deep_stack_ood_guard`'s own grid | Now the binding limit on #4's gate: the head-to-head is real, but at 4k hands SE(BB/100) ≈ ±13–19. |
| 6 | Every opponent raise is exactly 0.75 pot | **The last open member of the [BET-3] bundle** and the natural next version: it is what the OPP-2 raise features and every fold-vs-raise response are ultimately calibrated against. Hero has never faced an open-jam, overbet or min-raise. |
| 7 | Dead blinds | **FIXED in V41** — see Tier 3. |
| 8 | NN opponents play a degraded self | **FIXED in V41** — see Tier 3. |
| 9 | All six stacks identical; min-raise floor; short all-ins reopen action | **FIXED in V41** — see Tier 3. |
| 10 | Rollout queries use a third, drifted encoder | **FIXED in V41** — see Tier 3. |
| 11 | [OPP-7]'s V27 fix defeated at the tensor boundary | **FIXED in V41** — see Tier 3. Backlog status "RESOLVED" still needs the correction noted there. |
| 12 | `contract_version` never validated — only `context_dim` width | |
| 13 | Call-button OCR miss silently becomes "free check" and force-masks FOLD | **FIXED 2026-07-21 (V42_liveFixes)** — see Tier 5. |
| 14 | Live serves an all-PAD action-history sequence | **FIXED 2026-07-21** — see Tier 4. |
| 15 | Missing/corrupt weights degrade to random-weight play | **FIXED 2026-07-21** (during the V41 deployment) — `core/decision.py`'s `make_decision` now refuses to act when the active engine's `.loaded` is False, returning FOLD with the load error in the reason string instead of serving random weights. Engines without the flag are treated as loaded (legacy). Verified by pointing the engine at a missing checkpoint: `FOLD — Model 'Herocules (v41)' failed to load (FileNotFoundError...) - refusing to act`. The engine constructor still swallows the exception on purpose — the registry builds every engine at init and one missing rollback checkpoint must not take the app down — so `.loaded` is the contract. |
| 16 | Version dispatch fragility | **PARTIAL 2026-07-21** — the two silent-failure paths are closed; the ladders themselves remain. See Tier 4. |
| — | Second-tier live list (raise attribution inversion, pot under-reads, `committed` excludes blinds, HUD default colour, empty-seat position arithmetic, serve-only temp ramp, decimal stakes, 2-card board reads) | **PARTIAL 2026-07-21 (V42_liveFixes)** — HUD default colour, empty-seat position arithmetic, decimal stakes and 2-card board reads are FIXED (Tier 5). Still open: raise-attribution inversion, pot under-reads/latching, `committed` excluding blinds, the serve-only temp ramp. See `fable-review-consolidated.md`'s closing paragraph. |

## Guidelines

- Never edit any `fable-review-*.md`. Record outcomes here.
- Update a row's status in place; do not append duplicate rows for the same finding.
- When a finding is fixed, name the version and the verification, not just "done".
- When an investigation shows a finding does not hold, say so explicitly (**NOT-A-DEFECT**) and keep
  the reasoning — that is the expensive part, and re-deriving it costs more than the fix did.

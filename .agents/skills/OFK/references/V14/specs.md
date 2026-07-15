# V14 — Specs & Observations

**Status:** PLANNING. **Spine = a discretized BET-SIZE action space** (let the model *learn* sizing).
V14 builds from the [[V13 milestone]] (`versions/v13/`), which plays reasonably well live and stays
frozen as the reference/fallback. Inherits all locked fixes in `versions/v13/VALIDATED_FINDINGS.md`.

---

## Root cause this version exists to fix

V13's action space is `{FOLD, CALL, RAISE}` and **every raise is a hardcoded 0.75×pot** (capped at
stack) — see `simulator.py` and `core/decision.py`. Because sizing was never a *choice*, the model
**cannot learn it**. Bet sizing is one of the highest-information decisions in poker (small = thin
value / cheap bluff, big = polarized, overbet/shove = commitment), and V13 collapses all of it into
one action. The live symptoms we flagged are downstream of this:

- **Short-stack failure** (JJ folded at 14.9 BB, `RAISE Q = −0.78`): at ~15 BB a 0.75-pot raise is a
  ~1.5 BB min-raise, not a shove — the model literally **cannot express an open-jam** in one action.
- **Preflop "flattening"**: with only one raise size, the actor can't distinguish "raise small to
  steal" from "jam"; combined with count-dominated equity it plays a generic range.

The strong poker AIs (Pluribus, Libratus, DeepStack) all use a small **discretized set of bet sizes**
as distinct actions. That is the fix.

---

## P1 (THE SPINE) — Discretized bet-size action space

**Action space:** `{FOLD, CALL, RAISE_33, RAISE_66, RAISE_POT, ALLIN}` — i.e. **3 pot-fraction raise
sizes + all-in** (small bucket set chosen deliberately; solvers show sharply diminishing returns past
2–3 sizes per node, and each extra bucket costs its own data + MC-target compute). Fractions are
tunable; ~⅓ / ~⅔ / ~1×pot is a sane start.

- **Model heads** widen `[3] → [6]` (both `q_vals` and `policy_logits`). Small architectural change.
- **Contract bump:** the `act` token vocabulary grows (new raise tokens) → new `contract_version`,
  fresh weights (fail-loud manifest already enforces this).
- **Sim betting loop:** apply the CHOSEN size instead of the hardcoded 0.75-pot. Preflop, map
  pot-fraction buckets to sane BB-equivalents with the existing min-raise floor (a "small" preflop
  raise ≈ a standard open; ALLIN covers the short-stack jam). Postflop = literal pot-fractions.
- **Regret-matching policy target** now mixes over 6 actions (the mixed strategy the actor samples).

**This SUBSUMES short-stack (old P1):** once `ALLIN` is a real action with its own EV target, the
model can *learn* that jamming JJ at 15 BB is +EV and pick it — no hardcoded "short→shove" patch. Still
train on a **short/medium tournament stack distribution** (DoN ≈ 5–14 BB), NOT the fat-tail 10–300 BB
extreme curriculum (kept disabled for good reason in v12_validated).

### P1a (REQUIRED sub-part) — Per-size counterfactual EV targets

The learning signal for sizing IS the per-size EV — an EV for **each** raise size, via MC, not one
blended raise. Without it the extra actions are noise.

**IMPLEMENTED + verified (2026-07-14, unwired):** `simulator._mc_target_evs_sized(...)` returns
`[ev_fold, ev_call, ev_raise(frac_0), …]` for `raise_pot_fractions` in config (`[0.33,0.66,1.0,null]`;
null = all-in), with `_raise_size_for_fraction` for the sizing (min-raise floored, stack capped).
Each size's opponent fold-out is sampled from the **same** `bot.decide_*` used in play, so the target
matches the data by construction — meaning the **shared-curve TODO is resolved** (the bot's own
size-aware function IS the shared curve; no separate helper needed). Verified it produces the correct
opponent-adaptive signal: bluff→shove vs nits / give-up vs stations; value→jam vs stations / size-down
vs nits; all-in stack-bounded. **Not yet wired** into the model/train (see P1 flip below) so v14 stays
trainable on 3 actions until the atomic switch.

**Counterfactual = cheap exploration of overbet/shove.** Because the target evaluates *every* size at
*every* decision regardless of what was played, the model learns the EV of overbetting/shoving without
having to stumble into it. State-VISITATION still needs: short-stack curriculum, ε-random extended to
sample the new actions (incl. all-in), non-degenerate size-logit init, and sampled (not argmax)
rollouts. P1b makes the big-size targets correct (bots fold more to bigger bets).

### P1b (REQUIRED sub-part / HARD prerequisite) — Size-SENSITIVE opponents

**This make-or-breaks P1.** The fuzzy bots were **bet-size-BLIND** — `decide_postflop` collapsed
`pot_odds` to a boolean (`facing_bet > 0`), so a 3 BB bet and an all-in were identical input. If we
add sizes but keep size-blind opponents, a bigger bet folds out the *same* hands while risking more
→ the model learns to **min-bet/min-bluff everything** (degenerate).

**DEFENSIVE size-response — IMPLEMENTED (2026-07-14, `versions/v14/self_play/opponent_bots.py`).**
`decide_postflop` now anchors the continue/fold bar to the price: `continue_bar = pot_odds +
(fold_to_pressure − 0.5)·STYLE_SHIFT_SCALE` (STYLE_SHIFT_SCALE=0.30, tunable). Because
`pot_odds == bet/(pot+bet)` rises with bet size, the bar rises with size → bots fold more to bigger
bets (defend freq ≈ MDF = 1 − pot_odds). Style shifts it off the price: **nit over-folds, station
under-folds** → hero will learn to bluff bigger vs nits, value-bet bigger vs stations. This replaced
the old flat `fold_to_pressure` "sticky float". Verified monotonic size-response (fold% rises with
bet size for every archetype). Bluff-RAISE frequency left flat here; **size-scaled bluffing deferred
to `versions/v15/SPECS.md`** (polish, not load-bearing).

**Still TODO for P1b:**
- **Shared fold curve:** factor `continue_bar(...)` into ONE helper used by BOTH `decide_postflop`
  AND P1a's `_calculate_mc_target_evs` (`P(opponent folds | size)`), so the per-size EV targets match
  how opponents actually behave in the generated data (internal train consistency).
- **Bot size SELECTION (secondary):** have bots *choose* varied sizes (polarized: big/all-in with
  nuts or bluffs, small/medium with marginal made hands) so the hero faces a range of sizings. Less
  critical than the response above. The self-play mirror (`disable_past_self=False`) develops this
  organically once the action space lands.

### P1 — THE ATOMIC FLIP — TRAINING PIPELINE DONE + VERIFIED (2026-07-14)

Steps 1–4 below are IMPLEMENTED and the whole 6-action pipeline is verified by `overfit_sanity`
(critic synth |Q−target| 0.80bb PASS, actor synth KL 0.0013 PASS, real per-size targets learnable
KL 0.016). Done: manifest (contract_version 3, 6-action space); model heads `[3]→[6]` (`num_actions=6`);
sim (`_mc_target_evs_sized` wired as the target, betting loop applies `_raise_size_for_fraction` per
chosen bucket, `_query_model_decide` + ε-random sample over 6, fold-when-free mask kept, records
6-way action + 6-EV target); train.py `vectorize` widened to K (inferred from target width). Warm-start
weights removed (heads reshaped) → **v14 trains FRESH**.

**REMAINING before/after the first real run:**
- **eval/inspect tools 3→6** (`inspect_policy_vs_target`, `eval_pure_policy`, `inspect_ev_targets`):
  widen the per-action columns; the per-size policy spread is the key readout (validation plan below).
  Untestable until a checkpoint exists — do alongside the first training run.
- **LIVE serve (decision.py + PHPHelp)**: map the 6-way output to the executor's sizing controls
  (`RAISE_POT_x` / all-in) via the shared `_raise_size_for_fraction`; widen the sampled-policy +
  temperature + fold-mask path from 3 to 6. Only needed to DEPLOY v14 (v13 stays live meanwhile).
- Then: short-stack curriculum config + the actual training run.

Original order (for reference):
1. **manifest:** `contract_version 2→3`, `action_space=(fold,call,raise_33,raise_66,raise_pot,allin)`.
   context_dim stays 35 (input schema unchanged). Weights train FRESH (head shapes change → no warm start).
2. **model.py:** widen head outputs `[…,3] → [,6]` for `equity_base_q`, `equity_base_pi`, `head`,
   `head_policy` (K=6). SP_IDX base unchanged.
3. **simulator:** swap `_calculate_mc_target_evs` → `_mc_target_evs_sized` (already built); betting
   loop applies `_raise_size_for_fraction` for the chosen raise action; `_select_action`/`_query_model_decide`
   sample over 6 + keep the fold-when-free mask (zero fold idx 0); **ε-random samples all 6 incl. all-in**;
   record/decision_points store the 6-way action idx + 6-EV target. Action HISTORY (`act` input) can stay
   coarse ('r' for any raise) to avoid a vocab change, or add size tokens later.
4. **train.py:** regret-matching policy target + critic loss over K=6; keep target_clip_bb=40.
5. **eval/inspect tools:** 3→6 columns (the per-size policy is the key readout — see validation plan).
6. **LIVE (decision.py + PHPHelp):** map the 6-way output to the executor's sizing controls
   (`RAISE_POT_x` buttons / all-in); the sampled-policy + temperature + fold-mask path already exists,
   widen to 6. Shared `_raise_size_for_fraction` so live sizing == training sizing.

### P1c — Live sizing consistency (train/serve — the recurring trap)

Live already mis-sizes *today*: v13 emits a plain `'RAISE'` → the executor clicks the bare raise
button = client default (≈ min-raise), while `bet_size` (computed as 3×BB by default, since
`layer_sizing_var` defaults False) is **never even passed to the executor**. With V14's action space,
the decision path emits the **sized action** and the executor maps it to its existing sizing controls
(`POT_50/70/100/125` buttons + slider + all-in). The size rule MUST be identical in sim and live —
same principle that bit us on equity, fold-when-free, and sampling.

## P2 — Preflop flattening (partly resolved by P1, finish the rest)

Equity is dominated by opponent COUNT/colors, not hand strength (v13 §6), and the actor folds hands
its own critic rates +EV (JJ: **VERIFIED not perception** — decoded the actual input tensor: preflop,
equity 0.61, to_call 1.0 BB, pot_odds 0.32 all correct; critic said CALL +0.54, actor folded). A
learned action space with a real jam option should relieve part of this (the model can finally express
the correct aggressive action). Still do the **strength/count decouple** (compute strength
heads-up-vs-aggregate-continuing-range so nit vs station diverges; represent multiway separately) and
consider an **actor↔critic consistency loss** so the policy can't fold what the critic values.

## P3 — AGG axis (learned aggression)

Still VPIP-only in v13. Feed **per-opponent AGG** (not the global ctx average) and re-enable the
bluff/strength aux heads; validate bluff frequency dropping vs stations / rising vs nits. Naturally
pairs with P1b (size-aware, aggression-aware opponents) and P1 (sizing IS how aggression expresses).

## Observed-GOOD — KEEP, do not regress in V14

- **Postflop made-hand valuation works.** Live A8o flopped top-pair aces → critic Q +2.0…+2.66 BB and
  the model value-raised a station correctly. Equity-primary architecture is doing its job postflop.
- Range-aware equity adapts across colors; the **fold-when-free mask + sharpened sampling (temp 0.5)**
  live-serve fixes match training. (See [[live-action-selection]].)

## Tooling to add alongside V14

- **Opponent-action + outcome logging** in the recorder: per-turn villain actions/bet sizes and the
  hand RESULT (won/lost, showdown, villain cards). Today `turns.jsonl` logs hero decisions only; we can
  only infer "hero faced a bet" from `pot_odds`/`to_call` in the tensor. Needed to score sized battles
  and to *validate P1b* (are opponents actually folding more to bigger bets?). (See
  [[live-turn-history-and-shortcuts]].)
- Two-layer history (`board_state` / `evaluation` / `action`) is in place; a replay engine can consume
  it (still deliberately not built).

## Validation plan (how we'll know P1 worked)

- `inspect_policy_vs_target`: policy should put mass on DIFFERENT sizes by spot (small for thin
  value/steals, big/overbet polarized, ALLIN when short) — not collapse to one size.
- Size-response check: hold the hand fixed, vary opponent color/stack → the chosen size should shift
  (bigger vs stations, jam when short). If every spot picks the smallest size → P1b (size-aware bots)
  is broken.
- `eval_pure_policy` at **short-stack fields (5–15 BB)**: JJ-type hands should now JAM, not fold.
- Judge at ≥70k mature, ≥8000-hand / multi-seed evals (30k single-seed was noisy for v12/v13).

# V42_liveFixes — live-serving correctness package (Fable review bundles A + C)

**Date**: 2026-07-21
**Serves**: `Herocules (v41)` — unchanged weights, unchanged contract (`context_dim=54`,
`contract_version=8`), unchanged simulator.

## Why there is no `versions/v42/` training slice, and no retrain

The task was scoped as "clone from V41, do bundles A and C, retrain if needed". Every finding in
those two bundles turned out to live in the **shared live layer** (`PHPHelp.py`,
`core/decision.py`, `core/table_state.py`, `core/models/*_engine.py`) — not in a version's
`core/contract.py`, `self_play/simulator.py`, `train.py` or `config.yaml`. Nothing under
`versions/v41/` was read differently, let alone written.

A clone would therefore have been a byte-identical copy of V41's slice, and per the guardrails'
own checklist (§6 step 2) it would start with **empty weights** — so making it servable would mean
retraining 100k hands to reproduce a model that is statistically identical to V41 by construction.
That is real cost for zero information.

More importantly, **every fix here moves the live input distribution TOWARD what V41 was trained
on**, never away from it: range-aware equity instead of vs-random, a real `hand_strength` instead
of a constant, the training-side unknown-HUD default, real button-relative positions. V41's
existing weights are strictly *more* appropriate after this package than before it. Retraining
would be undoing nothing and learning nothing.

The one item in this area that genuinely *would* need a simulator change plus a retrain is called
out under "Not fixed" below, and is left as a V43 proposal rather than done speculatively.

---

## The finding this package started from, which was not in the review

**V41 — the live model — was being served vs-random equity and a constant `hand_strength = 0.5`.**

`PHPHelp.py` carried two hand-maintained per-version substring ladders, one choosing the version's
`compute_range_aware_equity` (L1072) and one its `preflop_hand_strength` (L1177). Both stopped at
`'v29'`. V40 and V41 were deployed on 2026-07-21 without being added to either, so for as long as
each was the active model:

- `ctx[3]` (equity) — the single feature this architecture is built around (equity-primary base,
  see `model.py`'s `SP_IDX`) — came from the plain vs-random evaluator, an estimator the model was
  never trained on, and `equity_edge` (ctx[35]) is derived from it, so that drifted too;
- `hand_strength` (ctx[36]) was the neutral 0.5 default for every hand, AA and 72o alike;
- the HUD's "HAND WIN% / EQ EDGE" panel silently showed "-", the only outward symptom.

Nothing threw. The tensors were shape-valid. This is precisely review finding #16/H4's "still
open" note — *"the two other ladders in `PHPHelp.py` … can still silently drop a new version to
vs-random equity — a train/serve mismatch that does not announce itself"* — realised on the
primary feature, on the milestone model, in production.

**Fix**: the same pattern `make_bridge()` used for the tensor ladder. An engine declares
`live_features()` (V40 and V41 do), and `core/decision.py::live_feature_providers()` resolves it;
the legacy name mapping now lives there as `_LEGACY_LIVE_FEATURES`, in **one** place, and a model
matching neither path resolves to `source='unresolved'` which the live log reports as an ERROR
instead of degrading quietly. Verified: all 17 registered models resolve to their own package's
implementations, byte-identical to the old ladder for V13–V29.

---

## Bundle A — live money / OCR integrity

### A1 [#13, live-serving H2] An unreadable Check/Call button no longer means "free check"

`call_amount` was initialised to `0.0` and every failure path left it there, so a Check/Call OCR
miss while facing a real bet reached `core/decision.py` as `free_check=True` → `probs['FOLD']=0.0`:
**the model was forbidden from folding a bet it could not see.** decision.py's `call_amount is
None` parse-miss sentinel was unreachable from live — dead code from the live path's perspective,
exactly as the review said.

Now every branch reports whether the price was actually *read*:

| button OCR | before | after |
|---|---|---|
| call word + digits | parsed amount | parsed amount (`known`) |
| call word, no digits | fabricated `2.0` chips | this street's observed bet level, `known=False` |
| check word | 0.0 | 0.0 (`known`) — positively identified |
| nothing matched | **0.0 → free check, FOLD masked** | this street's observed bet level, `known=False` |
| no call button at all | fabricated `100.0` chips | hero's stack, `known=False`, **CALL masked** |

`make_decision` gained `call_amount_known` and masks FOLD only on a *positively identified* free
check. The estimate still drives the tensor (`pot_odds`, `scaled_call`) — both channels stay safe,
because feeding 0 while unmasking FOLD would still have told the model checking was free.

The estimate is `table_state.current_street_bet_level`: the largest single-player contribution
actually observed this street (the same tracker [OPP-2]'s raise classifier uses), seeded to the big
blind preflop and 0 on a street where nobody has bet — which is itself the correct answer. It
replaces two absolute chip constants that knew nothing about the blind level (2.0 chips is 0.1bb at
bb=20 and 20bb at bb=0.1).

### A2 [live-serving M6] Decimal-stake money units

`core/vision.py` reads every stack and the pot by stripping non-digits (`"1.50"` → `150`) and the
window-title parser multiplies decimal blinds by 100, so on a €0.10/€0.20 table the whole pipeline
is denominated in cents — except this parser, which used `float("0.20") = 0.2` against a big blind
of 20. **A €0.20 bet arrived as 0.01bb, i.e. effectively free.** `_parse_button_money` now
digit-strips the matched *number* (not the whole string — `clean_stack_string`'s misread table maps
`A→4`, so running it over `"KALD 0.20"` yields 4020), reproducing vision's semantics exactly.
Verified: `"KALD 0.20"→20`, `"CALL 1.50"→150`, `"KALD 40"→40`, `"CALL 0,20"→20`, `"KALD"→None`.

### A3 [live-serving M4] `check_call_available` is no longer accepted and ignored

`make_decision` took the parameter and never read it, so with no Check/Call button on screen the
sampler could still return CALL and the executor would click where no button exists. CALL is now
masked exactly as the raise buckets are when the raise button is gone, in both the sized and the
legacy 3-way path, including the degenerate fallbacks. Verified over 400 samples per case: CALL is
never chosen without a call button, and with neither button the only action is FOLD.

---

## Bundle C — train↔serve encoding mismatches

### C1 [#8-CE] Unknown HUD colour: super-nit → average

`to_board_state` defaulted an unclassified opponent to `Blue` = VPIP 0.10 / AGG 0.18, **the
tightest band the contract can express**. Training's absent-profile default is the opposite
judgement — 0.30/0.46, i.e. Yellow/Green (`train.py`'s `map_*_to_midpoint` defaults, and
`ContractV12`'s own absent-seat default). So a player whose badge had not been OCR'd yet (new
player sits, badge obscured, colour crop misread) was read as the nittiest possible villain, and
hero over-bluffed and over-folded to their aggression. Live now matches training, and also matches
the equity path, which already used `or 'Yellow'` ([V20_preflopEq] Finding 1) — the same opponent
was previously described two different ways to two different consumers in the same decision.

### C2 [#6-CE] Range-aware equity with the front/after split

This is the ladder finding above: the front (already committed, no VPIP fold-roll) / after (still
to act, normal roll) split was implemented for V20_preflopEq…V29 but gated behind the stale ladder,
so V40/V41 got neither the split nor range-aware equity at all. Now driven by the version's own
`live_features()['front_colors']`. The documented residual is unchanged and accepted: live runs 250
sims vs training's 150 (same estimator, less noise — see the call site's own note).

### C3 [#12-CE] Short-handed position arithmetic — the DoN bubble

`hero_position` was `(0 - dealer_idx) % 6` and the contract derives every opponent's position from
it as `(slot + 1 + hero_position) % 6`. Both assume six gap-free seats — true in training, which
always deals 6, and false at exactly the moment that matters most: the 3–5 handed DoN endgame.
With seat_5 busted and the button on seat_4, the blinds skip the empty seat, so **hero can be the
big blind while being encoded as UTG**, with every opponent shifted alongside.

Positions are now counted over the **occupied ring**, which is what the blinds actually follow.
No contract change was needed for the opponent side either: since the contract reads slot `j` as
position `(j + 1 + hero_position) % 6`, writing an opponent whose true position is `p` into slot
`(p - hero_position) % 6` makes the encoder emit exactly `p`. Those slot indices are distinct for
distinct `p` (mod 6 is injective on 0..5) and never collide with hero's 0. Per-seat features
(stack, `committed`, [OPP-2] raise flags) travel with the opponent because they are read from the
opponent record, not from the key.

**For a full 6-handed table this is arithmetically identical to the old behaviour** — slot
`(p - hp) % 6` reduces to the physical seat number — so the common case is unchanged, verified for
all six button positions.

Measured on a 4-seated table (hero, seat_1, seat_3, seat_4; button on seat_3):

| | before | after |
|---|---|---|
| hero_position | 3 (UTG) | **2 (BB)** — hero is genuinely 2 seats after the button |
| opponent positions decoded by the contract | 4, 5, 1 | **3, 0, 1** — matches ground truth |

### C0 [live-serving L3] The Q-critic test flag was a dead variable

Found because the live HUD was showing the amber **"ACTION DISTRIBUTION — Q-CRITIC MODE"** header
after the testing session was over. `PHPHelp.py` had:

```python
CRITIC_ARGMAX_MODE = False   # <-- flip to True to test Q-critic mode
##os.environ['HEROCULES_CRITIC_ARGMAX'] = '1' if CRITIC_ARGMAX_MODE else '0'
```

The second line was commented out, so `CRITIC_ARGMAX_MODE` was **assigned and read by nothing**.
`core/decision.py` resolves `USE_CRITIC_ARGMAX_ACTION` from the environment at import, so the
flag's own comment four lines above — *"Authoritative: this wins over any
`HEROCULES_CRITIC_ARGMAX` left in the shell"* — was untrue in exactly the direction that matters:
a variable left set in a shell from a testing session silently put **live play on the critic's
argmax-Q instead of the sampled actor policy**, a selector no `model_verify` check has ever
evaluated. Setting the flag to False could not turn it off.

Re-armed, plus a loud startup line from `_report_engine_health` when the mode is on — the review's
L3 complaint was precisely that *nothing at startup surfaces that it's on*. Verified both ways:
with `HEROCULES_CRITIC_ARGMAX=1` exported, the flag still resolves False; importing `decision.py`
alone with the var set still resolves True, so the diagnostic mode still works, it just can't be
entered by accident.

### C4 [#10-CE] Partial board reads no longer alias to River

`ContractV12` buckets `board_len` as 0/3/4 and sends **everything else** to `street_level = 3.0`,
so a 1- or 2-card mid-deal frame encoded as a *river* state carrying three PAD cards — a
combination with zero training support, on the same tick a decision can be requested. The decision
loop now waits for a complete board instead of deciding on that fiction (community cards are
monotonic in `TableState`, so it resolves as soon as the flop is fully read).

---

## Round 2 (2026-07-21, from a flagged live hand) — the front/after equity split

`history/Turbo_1171580052/flagged/turn_2_20260721_201440`: **V43 folded QQ preflop, 100%
confidence, facing 1bb with 75bb behind, and the bot clicked it.** Not a model fault — replayed on
the identical tensor, V41 folds it too (0.996), so this predates V43 and rollback would not have
helped. The model was simply fed **equity 0.38**. Given the equity the training simulator produces
for that hand, V43 folds QQ at **no** opponent count:

| opponents | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| equity | 0.72 | 0.70 | 0.67 | 0.66 | 0.63 | 0.59 |
| P(FOLD) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| chosen | RAISE_POT | RAISE_POT | RAISE_POT | RAISE_POT | RAISE_POT | RAISE_POT |

**Root cause.** `front_colors` tells `compute_range_aware_equity` "these opponents are guaranteed
to showdown, do not roll them for a fold" — the strongest claim the equity model can make about an
opponent. `_classify_opponents_by_action_order` awarded it on **seat position alone**: everyone
sitting before hero was assumed to have acted *and* stayed. Training never does this — the
simulator builds `front` from real `acted_this_round[s]` / `folded[s]` / all-in state
(`simulator.py` L1592). Live has no per-seat fold detection, so folded seats were silently
promoted to locked-in showdown opponents. In the flagged hand the pot held **30 chips = the blinds
and nothing else** (`pot_type=0`, every `opp_raised_*` flag 0) and three opponents were classified
as committed.

| | QQ equity | action |
|---|---|---|
| 3 opponents wrongly locked in | **0.38** | FOLD (0.996) |
| front correctly empty | **0.62** | **RAISE_POT** (fold 0.000) |

**The fix**: chips in the pot are the criterion; seat order was only ever a proxy for them.
- Preflop, position is not consulted at all — committed chips *are* this street's chips. This also
  fixes a case the positional read could never have handled: a **3-bet from behind hero** reopens
  the action, so the raiser is committed *and* positionally "after".
- A **posted blind is involuntary** and can still fold behind it, so a blind counts only once it
  puts in more than it was forced to.
- Postflop, `committed` spans earlier streets and can't answer "acted *this* street", so the
  positional read stands there (it is sound postflop) — intersected with chips-in-hand, which
  filters folded and phantom seats.
- Anything unconfirmed falls through to `after` rather than being dropped, so a limper we couldn't
  confirm is under-weighted, not erased — the same conservative direction as [V20_preflopEq]
  Finding 1.

Supporting change in `core/table_state.py`: `committed_chips()` (extracted so the classifier and
`to_board_state()`'s per-seat `committed` feature cannot drift) and `blind_seat_keys()`.

### Same round: the diagnostic was lying, twice over

`summary.txt` for that turn rendered the real 75BB/1.5BB/1.0BB node as **300BB/6.0BB/4.0BB** and
raised `(!) MODEL-INPUT vs RAW-OCR MISMATCH -> BRIDGE issue` — against a bridge that was working
correctly. `_decode_model_input` carried *another* hand-maintained substring ladder
(`is_v20_family`) that stopped at `v29`, so V40/V41/V43 decoded with the pre-V20 `/400`,`/1000`
constants. This is the **third** time this one decoder has been wrong for the live model (its own
docstring records the V20_preflopEq_AI case), and it cost real triage time on a false lead while
the actual bug sat one layer away.

Fixed at the source rather than by extending the list: `core/decision.py::context_scales()` reads
`STACK_SCALE`/`POT_SCALE`/`CALL_SCALE` from the version contract's own module constants — the
literal values the encoder divided by, which cannot drift from the encoder by construction. The
`_LEGACY_LIVE_FEATURES` scales column is a fallback consulted only for pre-V20_preflopEq contracts
that bake the divisor inline. Engine resolution is shared with `live_feature_providers()` via
`_resolve_live_spec()`, so the two can never disagree about which version a live model is.

**Third symptom, same cause**: `_build_turn_record` sources `to_call`/`pot_odds` from that decoder,
so *every* turn recorded under V40/V41/V43 logged a `to_call` 4× too large — the flagged hand shows
`80.0`/`4.0BB` for a real 20-chip/1.0BB price. Fixed by the same change.

### Same round: `is_active` was not monotonic within a hand

Chasing the field-size question produced a **false lead worth recording**, because the reasoning
error is more instructive than the fix. The flagged turn's record listed all 5 opponents active
while the stored screenshot clearly shows one player sitting out, which looked like proof that
vision could not detect folds. Running the real `PokerVision.read_board_state` on that exact frame
disproves it:

```
RAW vision   seat_1: is_active=False  state='Folded'  stack=0
AFTER update num_active_players = 4        <- correct
```

The record's stacks (`1500/1500/1480/1480/1480`) and the frame's (`0/1560/1480/1470/1460`) are
**different frames**: `save_diagnostics` stores `last_raw_img` at F12-press time, three seconds
after the decision. A decision was being compared against a later picture. **When a record and a
screenshot disagree, check they describe the same instant before blaming the detector.**

The real defect it did expose: `TableState.update()` assigned `tracked_opp['is_active'] =
raw_active` unconditionally in *both* branches, so `is_active` was free to go False → True inside a
hand. A folded player cannot re-enter, so one bright frame (deal animation, timer overlay, a chip
graphic over the name plate) silently put a folded seat back in the pot for the rest of the hand.
The section header already claimed "Monotonic Decay" and the inline comment already claimed "they
stay folded" — `pot_size` and stacks were monotonic, `is_active` never was.

Cost of a single phantom seat, AKs preflop with training-computed equity: **4 opponents → CALL
(fold 0.12); 5 opponents → FOLD (fold 0.91)**. One flicker turns a clear continue into a 91% fold
of a top-5 hand, because `num_active` feeds equity *and* `equity_edge = equity × (num_active + 1)`.

Fixed with a per-hand `folded_this_hand` latch, cleared in `reset()`. Safe in the other direction
because the underlying signal is bimodal rather than marginal: across all 16 stored flagged frames
the name-plate brightness clusters at **58–105** (out) and **242–254** (in) with nothing between,
and vision's threshold (160.0) sits in the middle of that gap. A false fold needs a ~140-point
excursion; the resurrect path needed one frame.

## Verification

`versions/v42_liveFixes/verify_fold_monotonic.py`, 15/15 passing — flicker-resurrect, cumulative
folds, folded-on-first-frame, per-hand reset, and all-in seats (stack 0) staying in.

`versions/v42_liveFixes/verify_front_colors.py`, 7/7 passing — the flagged hand (front must be
empty), posted-blinds-only, a real raiser, a 3-bet from behind hero, postflop positional read,
postflop phantom active seat, and no-button fallback.

`versions/v42_liveFixes/verify_v42.py`, all passing:

1. **No regression at a full table** — hero_position and slot assignment identical to the legacy
   formula for all 6 button positions.
2. **Short-handed** — hero_position and all three contract-decoded opponent positions correct;
   per-seat features follow the opponent through the remap.
3. **HUD default** — unknown colour encodes Yellow/Green.
4. **Contract** — still 54 features from a short-handed table; `ctx[0] = 0.4` (position 2/5).
5. **Sentinel + masking** — known free check masks FOLD; unknown price does not; no call button ⇒
   CALL never sampled (400 draws/case); neither button ⇒ FOLD only.
6. **`_parse_button_money`** — decimal, comma-decimal, integer, and no-digit cases.

Plus a live-serving smoke test through the real `make_decision` with V41's actual weights: all four
streets emit executable `RAISE_SLIDER_x` actions with real chip sizes, and the 4-handed board
encodes `ctx[0] = 0.4`.

---

## Not fixed (deliberate, named)

- **Absent vs folded seats.** A live 4-handed table leaves two contract slots at the absent-seat
  default (mask 0, stack 0); training's inactive seats are always *folded players with real stacks
  and real HUD colours*. Positions are now right, but that residual gap can only be closed by
  teaching the simulator to seat 3–6 players — a real simulator change, and the one thing in this
  area that **would** justify a V43 clone and a 100k retrain. Not done speculatively.
- **Free check misread as a priced spot.** If the Check/Call button is unreadable *and* no bet has
  been observed on this street, the price estimate is 0 but `known=False`, so FOLD stays available
  and a weak hand can fold a genuinely free check. That is the deliberate direction of the
  asymmetry (folding a free check costs a little and is visible; calling an unseen bet with FOLD
  forbidden costs a lot), but it is a real, if rare, EV leak. A one-frame retry before deciding
  would remove both errors and is the obvious refinement.
- Everything in the review's bundles B (live state-tracking noise), D ([BET-3] finish / #6), E
  (confidence intervals), F (#12 `contract_version` validation, the remaining `is_vN` ladders), and
  live M3 (the serve-only short-stack temperature ramp that no eval applies).

See `.agents/skills/OFK/references/fable-review-resolution-log.md` for per-finding status.

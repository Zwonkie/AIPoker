# Fable Review — Live Serving Path (V29 "Herocules")

**Date Recorded**: 2026-07-20
**Related Files**: [decision.py](file:///c:/REPO/Antigravity/AIPoker/core/decision.py), [table_state.py](file:///c:/REPO/Antigravity/AIPoker/core/table_state.py), [vision.py](file:///c:/REPO/Antigravity/AIPoker/core/vision.py), [PHPHelp.py](file:///c:/REPO/Antigravity/AIPoker/PHPHelp.py), [v29_engine.py](file:///c:/REPO/Antigravity/AIPoker/core/models/v29_engine.py)

## Context

Live-serving-area report from the 2026-07-20 four-way V29 audit (see
`fable-review-consolidated.md`). Scope: version dispatch, vision/state-tracking correctness,
serve-time policy transforms, failure modes, latency.

## HIGH

**H1. A missing/corrupt weights file degrades to random-weight play with only a startup console warning.**
`core/models/v29_engine.py:62-68` — on load failure the engine sets `self.loaded = False`, prints a warning, and keeps a freshly-initialized network. `core/decision.py:442-444` (`make_decision`) only checks that the *name* resolves in the registry; it never checks `active_model.loaded`. Failure scenario: `versions/v29/weights/expert_main.pth` renamed/corrupted → app starts, HUD looks normal, every decision comes from untrained weights at a real table. The registry comment at `decision.py:175-180` explicitly names this danger, but the only guard is "prune bad entries by hand."

**H2. A call-amount OCR miss silently converts "facing a bet" into "free check," force-masking FOLD.**
`decision.py:534-538` treats `call_amount is None` as the parse-miss sentinel (safe default: don't mask fold). But `PHPHelp.py:1275` initializes `call_amount = 0.0` and never produces `None`: if the Check/Call button OCR (`PHPHelp.py:1283-1297`) fails to match `["KALD","CALL","KLD","KND"]` while a real bet is pending, `call_amount` stays `0.0` → `free_check=True` → `probs['FOLD']=0.0` (`decision.py:549-550`) → the model *must* call or raise a bet it can't fold. The `'CHECK'` safeguard at `PHPHelp.py:1346-1354` never fires because sized models emit `'CALL'`, never `'CHECK'`. The None-sentinel safety in decision.py is dead code from live's perspective.

**H3. Live serves with an all-PAD action-token sequence; training and rollout eval feed real action histories.**
Training queries pass `hero_actions_history` (tokens f=7/c=3/r=6) into `bridge.to_tensors` (`versions/v29/self_play/simulator.py:662, 1538-1583, 1613-1643`), and model_verify's BB/100 rollouts go through that same simulator. Live, `PHPHelp.py:1323-1331` calls `make_decision` without `action_history_raw`, so `decision.py:491` → `contract.py:308-312` fills `act_ints=[0]*20` (PAD). The transformer was trained and validated with its own past-action tokens populated; live it never sees them. Any behavior the model learned to condition on its own line (e.g. barreling after raising) is silently unavailable — and no eval configuration reproduces this exact live input. (`table_state.action_history` is even maintained at `PHPHelp.py:1357-1362` but never consumed.)

**H4. Version dispatch is three hand-synchronized substring ladders with a crash-fold default; active model is a hardcoded string.**
The actual mechanism: hardcoded `self.active_model_name = 'Herocules (v29)'` (`decision.py:374`); `set_active_model` falls back to **V20** — nine versions stale — on any unknown name (`decision.py:397-402`). Branch enumeration: 13 `is_vN` flags at `decision.py:461-485` (each `getattr(engine,'is_vN')` OR name-substring fallback with documented ordering traps for `v20`/`v20_preflopeq`/`v20_preflopeq_ai` and `v17`/`v17_gauntlet`), bridge selection at `487-520`, a 12-deep nested ternary tag ladder at `577`, aux gate at `682`. Two *more* independent ladders live in `PHPHelp.py:1079-1117` (per-version `compute_range_aware_equity` import) and `PHPHelp.py:1184-1199` (`preflop_hand_strength` import). Concrete failure: a future `'Herocules (v30)'` added to the registry but not the ladders matches no flag → falls through to `bridge_v9.to_tensors(board_state,...)` (`decision.py:520`) → exception → caught at `522-523` → **returns FOLD every hand** ("Fatal decision engine crash" in the reason, but play continues); meanwhile PHPHelp's ladder silently drops to vs-random equity (train/serve mismatch). A name like `'v29b'` would substring-match `'v29'` and misalign silently if its contract differs.

## MED

**M1. Single-frame pot OCR glitches can wipe mid-hand state (false hand reset), and inflated pot reads latch.**
`table_state.py:74-77` — `detect_hand_reset` compares the *raw, unfiltered* frame pot against stabilized pot: one under-read (dropped digit: 1500→150) mid-hand resets the whole `TableState`, destroying `hand_start_stacks`/`raise_count`/`raised_this_*` — exactly the inputs V29's 10 new features and `pot_type`/`committed` depend on. Conversely `table_state.py:113` (`pot_size = max(pot_size, median_pot)`) latches any 2-of-3-frame over-read for the rest of the hand and then *causes* a false reset when the true pot returns.

**M2. Per-seat raise attribution inverts when two stack drops land in one ~1s frame; missed frames merge streets.**
`table_state.py:236-255` iterates `current_stacks` in *seat insertion order*, not action order, updating `street_bet_before = diff` as it goes. If seat_5 bets 10 and seat_2 calls 10 within one polling tick (loop cadence `PHPHelp.py:991/1408` is ~1s plus multi-second OCR time), seat_2 is processed first, its diff (10) exceeds the old level → the *caller* gets `raised_this_hand=True`/`raise_count+=1` and the real raiser is classified as a call. Street transitions also reset the bet level (`206-211`) before diffing, so a call from the *previous* street seen late is counted as a new-street raise. These directly corrupt ctx[43] (`pot_type`) and ctx[44:54] — V29's headline new features.

**M3. The short-stack temperature ramp (0.2 @ ≤8bb) is a serve-only transform no eval applies.**
`decision.py:37-39, 65-74` ramps 0.2→0.5 between 8-20bb. `tools/model_verify/checks.py:878, 973` sets `sim.policy_temperature = 0.5` flat, and the simulator has no stack-scaled temp at all (`simulator.py:678`). All short-stack eval numbers (e.g. the 5-14bb field evals) were produced at 0.5; live plays near-argmax there. The file's own history (`decision.py:22-29`) documents how an unvalidated sharpening change cratered VPIP 23%→8% — the same class of risk, still live.

**M4. `check_call_available` is accepted and ignored by `make_decision`.**
`decision.py:439` takes the parameter; the body never reads it (only `bet_raise_available`, L551-552). If the check/call button is absent (`PHPHelp.py:1259-1264`, all-in situations), the model can still sample `CALL` and the executor clicks where no button is.

**M5. The hero-stack OCR fallback is dead for the model input.**
`PHPHelp.py:1237-1242` falls back to `last_valid_hero_stack` — into a local variable that is never used again. `to_board_state` (`PHPHelp.py:1312`) reads `table_state.hero_stack`, which is 0 if OCR never succeeded this hand (timer overlay): the model then sees stack=0bb (contract `scaled_stack_bb(0,·)`), and `_stack_scaled_temperature` drops to 0.2 near-argmax on a fiction.

**M6. Money-unit inconsistency on decimal stakes: stacks/pot in "concatenated digit" cents, call_amount as a decimal float.**
`vision.py:228-232, 251-255` strip to digits ("1.50" → 150) while `PHPHelp.py:1287-1291` parses the call button as `float("0.20") = 0.2`, and blinds from the title are ×100 (`PHPHelp.py:898-899`). On a €0.10/€0.20 table a €0.20 bet becomes call_amount=0.2 vs bb=20 → 0.01bb, effectively "free." Related: matched "KALD" with no digits fabricates `call_amount = 2.0` chips (`PHPHelp.py:1292-1294`), and `check_call_available=False` forces `call_amount = 100.0` regardless of blind level (`PHPHelp.py:1300`).

**M7. `committed` features systematically exclude posted blinds.**
`table_state.py:219-224` seeds `hand_start_stacks` from the first *observed* (post-blind-posting) frame; training's `committed` includes blind posts (`simulator.py` seeds `street_committed` with sb/bb, ~L1391-1398). Hero/opponents in the blinds under-report `committed`/`hero_committed` by 0.5-1bb every hand — a small but systematic train/serve skew on features V29 actively reads.

## LOW

- **L1.** False all-in latch: any OCR text containing `IN`/`ALL` (whitelist permits A,L,I,N — `vision.py:412, 474`) marks state `All-In`, stack 0, which the monotonic-min tracker (`table_state.py:124-128, 167-172`) makes permanent for the hand; hero's full stack is then "committed" and diffed as a raise.
- **L2.** Dead/stale code: v9 river guardrail keyed on `'Herocules (v9 Main)'` which is not in the registry (`decision.py:624-628`); `bridge_v11`/`bridge_v9` branches unreachable from the current registry (`517-520`); the entire math-engine override block (`637-668`) is bypassed for every registered model; `is_v11_model` computed twice (`481`, `636`); `v29_engine.py:9-19` docstring still claims the OPP-2 features are "NOT yet functional in LIVE serving" — false since the 2026-07-20 table_state work, and misleading for future readers.
- **L3.** `HEROCULES_CRITIC_ARGMAX` env var (`decision.py:48-49`) silently swaps live selection to an eval-unvalidated critic-argmax mode; nothing at startup surfaces that it's on except per-decision reason text.
- **L4.** NaN/degenerate policy output is swallowed: NaN probs → `max(0.0, nan)`→0.0 for all actions → empty candidate set → silent `'CALL' if free_check else 'FOLD'` (`decision.py:558-559, 594-595`) with no anomaly log.
- **L5.** Latency/robustness: each frame runs ~13 sequential Tesseract subprocess calls (no timeout) plus, on hero's turn, range-aware equity at 250 sims, a 200-sim `hand_strength` MC, and a 2000-sim vs-random fallback (`simulations_var` default, `PHPHelp.py:172`) — all blocking on the single worker thread with no decision deadline/watchdog against the site's turn clock. `compute_range_aware_equity` is bounded (25 rejection retries/opponent, fixed sims) — no unbounded loop.
- **L6.** Straddles would be counted as raises (diff > bb+ε) and antes as calls; when the dealer button isn't detected, `dealer_idx` defaults to 0 → hero_position=Button (`table_state.py:32-33`), feeding a wrong position feature until the template match lands.

## What's actually solid

The core train≡serve invariants hold up well under scrutiny: equity is computed by the *same* `compute_range_aware_equity` function training used (only sims 150→250, same estimator); the live `probs**(1/temp)` sharpening is mathematically identical to training/eval's `softmax(logits/temp)`; the fold-when-free mask exactly mirrors `simulator._select_action`; model_verify explicitly pins `policy_temperature=0.5` to the serve value (a previously-fixed trap); V29 correctly got its own 54-dim bridge with append-only feature indices; the contract's 100bb clamp matches the actual 5-100bb training mixture; the bb-seeded preflop bet level and raise-vs-bet-level classification in `table_state.py` genuinely match the simulator's `highest_bet` semantics; and the per-decision `model_input` replay payload is a strong foundation for post-hoc debugging.

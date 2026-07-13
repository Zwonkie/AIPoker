# V11 EV Scaling Mismatch (Maniac Collapse)

**Date Recorded**: 2026-07-12
**Related Files**: 
- [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/train_selfplay.py)
- [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/six_max_simulator.py)

## Context
During the evaluation of `expert_v11_main.pth` using `evaluate-model-health`, the model demonstrated a catastrophic "Maniac Collapse". It would exclusively output `RAISE` in all scenarios, even with pure air preflop or postflop. However, the telemetry dashboard reported a conservative 33% VPIP and 0% Aggression. This discrepancy was because the training simulator's `bootstrap` logic enforced an 80% manual fold rate and `math_engine` guardrails suppressed the model's maniac tendencies during evaluation.

## Root Cause
The Q-learning target values generated during `train_selfplay.py` vectorization were mixed in two different scales:
1. The untaken actions' target EVs were returned from `_calculate_mc_target_evs` in **RAW CHIPS** (e.g. +13.5 chips).
2. The taken action's target EV was overridden by the actual profit `mc_return` in **BIG BLINDS** (e.g. +1.35 BBs).

Since 1 Big Blind = 10 chips, the model observed that the untaken actions (which often included Raise) were intrinsically ~10x more valuable than the action it actually took. Over thousands of hands, the Q-values for `Raise` hyper-inflated, leading the model to always select it. 

## Resolution
Modified `train_selfplay.py` line 116 to scale the `target_evs` from raw chips to Big Blinds before overriding the taken action:
```python
# These come back in RAW CHIPS. We must scale them to BIG BLINDS!
t_evs = [ev / bb for ev in list(dp.get('target_evs', [0.0, 0.0, 0.0]))]
```
This forces all target EVs (Fold, Call, Raise) to exist in the same mathematical space (BBs), ensuring the neural network gradients are structurally sound.

**Next Steps:** Retrain V11 from scratch, monitoring the raw EV outputs using `evaluate-model-health` directly after early epochs to ensure no runaway values.

---

# V11 Vectorization Misalignment (Monster-Under-The-Bed Collapse)

**Date Recorded**: 2026-07-13
**Related Files**: 
- [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/train_selfplay.py)
- [contract_v11.py](file:///c:/REPO/Antigravity/AIPoker/core/bridge/v11/contract_v11.py)

## Context
During retraining, the model began exhibiting the "Monster-Under-The-Bed" syndrome (folding the absolute nuts). A forensic review revealed that the Target EV equations were 100% correct. However, there was a catastrophic mismatch between how sequence tensors were padded during training vs. inference.

## Root Cause
- **Training Side (`train_selfplay.py`)**: Data was being populated sequentially from index 0 (Right-Padding). Thus, valid data occupied indices `0, 1, 2`, while `3..19` were `[0.0]`.
- **Inference Side (`contract_v11.py`)**: The data contract was placing the single current state at `context_seq[-1]` (index 19), leaving `0..18` as `[0.0]`.
- **The Bug**: During live play, the model extracted the prediction from `q_vals[-1]`. However, during training, position embedding 19 was almost never populated (only for extremely long 20-action hands), meaning it consisted of untrained random noise. Additionally, the transformer was robbed of its state history because inference left indices `0..18` blank.

## Resolution / Guidelines
**Mandatory Fix**: Unify the padding direction. Left-padding is the standard for autoregressive transformers when extracting the final prediction. `train_selfplay.py` must offset decision points by `max_seq_len - len(dps)` to match the inference expectation, and the `ContractV11` must maintain a rolling history buffer rather than only populating index `-1`.

*Update 2026-07-13:* The fix was successfully verified during live training. The "Monster-Under-The-Bed" syndrome (folding >80% equity hands) was eliminated. Intermediate sensitivity checks showed the model accurately outputting positive EV for RAISING with the Nuts (`Ah As`) where it previously FOLDED.

---

# V11 Variable Shadowing in `vectorize_hand_samples()` (Silent Sequence Corruption)

**Date Recorded**: 2026-07-13
**Related Files**: 
- [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/train_selfplay.py)

## Context
During a Deep Check audit, a critical variable shadowing bug was discovered in `vectorize_hand_samples()` that silently corrupted all multi-step hand training sequences.

## Root Cause
The outer loop (`for i, dp in enumerate(dps)`) set `idx = start_idx + i` to track the current decision point's position in the sequence. Two inner loops (`for idx in range(5)` on lines 97 and 121) iterated over opponent seats but **reused the same `idx` variable**, overwriting it. After the inner loops completed, `idx` was always `4` regardless of the actual decision point index.

This caused `context_seq[idx]`, `action_seq[idx]`, `target_evs_seq[idx]`, `loss_mask[idx]`, and all auxiliary labels to always write to position `[4]`. For any hand with multiple decision points (e.g., Preflop→Flop→Turn→River), only the **last** decision point survived, and it was always placed at index 4 instead of its true sequential position.

**Impact**: The transformer effectively trained on single-step corrupted samples, destroying its ability to learn sequential patterns across streets.

## Resolution
Renamed both inner loop variables from `idx` to `j`:
```python
# Line 97: was `for idx in range(5):`
for j in range(5):
    if dp['active_opponents_mask'][j] == 1.0:
        seat_key = f"seat_{j+1}"
        ...

# Line 121: was `for idx in range(5):`
for j in range(5):
    seat_key = f"seat_{j+1}"
    ...
```

**Status**: ✅ Fixed. Requires full retraining of all V11 models.

---

# V11 Architecture ↔ Weight Shape Mismatch (All V11 Models Broken)

**Date Recorded**: 2026-07-13
**Related Files**: 
- [poker_transformer_v11.py](file:///c:/REPO/Antigravity/AIPoker/core/models/v11/poker_transformer_v11.py)
- [contract_v11.py](file:///c:/REPO/Antigravity/AIPoker/core/bridge/v11/contract_v11.py)

## Context
All V11 weight files fail to load with: `size mismatch for state_proj.0.weight: copying a param with shape torch.Size([256, 159]) from checkpoint, the shape in current model is torch.Size([256, 163])`.

## Root Cause
The V11 inference contract (`contract_v11.py`) was expanded from 31 to 35 context features (adding per-opponent position encoding, changing scaling of call_amount). This changed the `state_proj` input dimension from `64+64+31=159` to `64+64+35=163`. The model architecture in `poker_transformer_v11.py` was updated to `context_dim=35` to match, but the existing weight checkpoints were trained on the old 31-feature contract.

| Component | Old (trained) | New (code) |
|:---|:---:|:---:|
| Base features | 11 | 10 |
| Per-opponent features | 4×5=20 | 5×5=25 |
| **Total context_dim** | **31** | **35** |
| **state_proj input** | **159** | **163** |

## Resolution
**Decision: Option B** — Keep the 35-feature contract. All existing V11 weights are incompatible and must be retrained from scratch after fixing the variable shadowing bug above.

**Status**: ⚠️ Requires full retraining. No code change needed (architecture is correct at 35 features).

---

# V11 Per-Seat Stat Attribution Collapse (Personality Averaging)

**Date Recorded**: 2026-07-13
**Related Files**:
- [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/six_max_simulator.py)

## Context
The training dashboard reported every opponent seat (Maniac, Nit, Sticky, Past Self, TAG) with near-identical VPIP (~25%) and AGG (~43%), down to matching raise/fold counts, and the numbers never diverged no matter how long training ran. A Maniac and a Nit looked statistically identical.

## Root Cause
The opponent *style* occupying each seat was reshuffled every hand (`random.shuffle(base_styles)`), but cumulative stats (`self.seat_histories`) were keyed by **table seat index** while the HUD labels (`seat_labels`) were **hard-coded per personality**. Because each seat saw a roughly uniform mix of all archetypes over time, every seat's cumulative VPIP/AGG converged to the population *average* of all personalities. It had "converged" — to a meaningless blended mean with no relationship to the row's label.

## Resolution
Re-keyed all cumulative tracking (VPIP/AGG/profit/exploitation) by **personality slot** instead of table seat via a new `STYLE_SLOT` map (`0 Hero | 1 Maniac | 2 Nit | 3 Sticky/fish | 4 Past | 5 TAG`). Each hand builds a `seat_slot` map and every stat increment routes through it, so a personality's stats accumulate against that personality regardless of which chair it occupies. The worker return contract (a dict indexed 0-5) is unchanged — the index now means "personality slot," which is exactly what the fixed HUD labels already assumed, so no trainer/parser changes were required.

**Status**: ✅ Fixed. Verified: VPIP spread across personalities went from ~1pt (collapsed) to ~50pts (Maniac loose-aggressive, Nit tight-passive, Sticky loose-passive).

---

# V11 Model Never Queried During Self-Play (Silent Heuristic Fallback)

**Date Recorded**: 2026-07-13
**Related Files**:
- [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/six_max_simulator.py)

## Context
While fixing the attribution bug above, discovered that the neural network was **never actually generating self-play decisions** — every action came from the heuristic bootstrap chart. This is the deeper reason earlier runs showed no policy feedback loop / no convergence.

## Root Cause
`_query_model_decide` built the model's opponent-HUD input by reading `self.seat_histories[idx+1]['vpip']` / `['agg']` — keys that never existed in the history buckets (which hold `vpip_ops`/`vpip_acts`, not `vpip`). This raised `KeyError` on **every** call. Both callers (`_hero_decide`, `_opponent_decide`) wrap the query in `try/except: pass`, so the exception was swallowed and the code silently fell back to the heuristic chart every time.

## Resolution
Threaded the per-hand `opponents_profiles` dict (which already holds correct per-seat VPIP/AGG floats) into `_query_model_decide` via `table_state`, and read HUD colors from it instead of the non-existent keys. The NN is now genuinely queried. Verified `_query_model_decide` returns a valid decision without exception.

**Status**: ✅ Fixed.

---

# V11 Opponent Action-Forcing Gated Off for Heuristic Bots

**Date Recorded**: 2026-07-13
**Related Files**:
- [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/six_max_simulator.py)

## Context
Personality archetypes (Maniac/Nit/Sticky) were not being pinned to their target VPIP/AGG ranges. During the fresh V11 retrain, opponent NN models are disabled (`m/n/s/p_path = None`), so all opponents run pure heuristics.

## Root Cause
The archetype action-forcing in `_opponent_decide` was gated on `if model is not None:`. With opponent models disabled, the forcing never fired, so the heuristics ran unshaped.

## Resolution
Dropped the `model is not None` gate on both the preflop (kept the `roll >= bootstrap_alpha` guard) and postflop forcing blocks, so archetype shaping applies to heuristic opponents too. Verified: Maniac VPIP jumped to ~61%, Nit pinned to ~11%, VPIP spread widened from ~24pts to ~50pts.

---

# V11 Loose-Collapse Ratchet (Policy Divergence to Loose-Passive/Loose-Aggressive)

**Date Recorded**: 2026-07-13
**Related Files**:
- [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/train_selfplay.py)
- [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/six_max_simulator.py)

## Context
Once the bootstrap crutch decayed (α: 1.0→0 over 10k-30k hands) and the Hero's own policy took over, the Hero's VPIP ratcheted monotonically upward (30% → 34% → 50%) while BB/100 fell (+10 → −11 → −29). It collapsed into the loose-passive then loose-aggressive (Maniac-like) losing quadrant and kept bleeding — a genuine objective/policy-divergence leak, not a training trough.

## Root Cause
Three compounding roots in the Q-target design:
1. **No baseline / taken-action-only masking**: the loss trained only the action actually taken (`action_mask`), so the model never saw a counterfactual and passivity/looseness was self-reinforcing (an action never taken never gets corrected).
2. **Fat-tail ratchet**: the taken-action target was the raw realized go-forward return clipped to a huge ±100 BB. Entering has a positive-tailed distribution (rarely stack a fish for 100BB+), so "enter" Q-values stayed buoyantly ≥ 0 with nothing pulling them down; fold's target of exactly 0 rarely won the argmax.
3. **Half-fish pool**: 2 of 5 opponents were forced spew-fish (Maniac, Sticky), so realized-MC rewarded over-loosening to exploit dead money — the Hero overfit to the fish and got stacked by the disciplined seats.

## Resolution (three fixes)
1. **Counterfactual fold baseline** — replaced taken-action masking with a per-action weight tensor (`target_w_seq`): always train the Fold head toward its exact go-forward value (**0**) plus the action actually taken toward its MC return. The model now always has an anchored "is entering better than folding (0)?" comparison.
2. **Anti-ratchet target shaping** — (a) preflop tightness prior: entering with equity below `1.15 × 1/(active_opps+1)` has its target lowered by up to `TIGHTNESS_PENALTY_BB` (4 BB), pushing weak entries under the fold baseline; (b) tightened the target clip from ±100 → **±40 BB** (`TARGET_CLIP_BB`) to damp the fat tail. Constants live at the top of `train_selfplay.py`.
3. **Pool rebalance** — opponent seats now sample from a disciplined-majority pool (`OPPONENT_POOL_STYLES`/`WEIGHTS`: tag 30 / past 25 / nit 20 / maniac 15 / fish 10), cutting spew-fish from ~40% → ~25% of seats.

**Status**: ✅ Implemented + smoke-tested. A fresh 6k-hand micro-run already showed disciplined play (weak <20%-equity hands folded ~95%, nuts raised/all-in ~88%) instead of loose-collapse. **Watch after ~30k hands** (post-bootstrap): Hero VPIP should hold ~20-25% (no ramp to 50%) and BB/100 stay non-negative.

---

# V11 Past-Self League Opponent Enabled

**Date Recorded**: 2026-07-13
**Related Files**:
- [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/train_selfplay.py)
- [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/six_max_simulator.py)

## Context
The "Past Self" seat was intended to be a frozen lagged snapshot of the training Hero, but the snapshot was dead code — it saved to `v11_past_checkpoint.pth` every 5,000 hands yet the workers never loaded it (`p_path = None`), so "Past Self" secretly played the TAG heuristic (identical stats to the "TAG Bot" seat gave it away).

## Resolution
Wired `p_path = past_path if os.path.exists(past_path) else None` so workers load the lagged checkpoint once the first 5k-hand snapshot of the current run exists. Added a fresh-run cleanup (`os.remove(past_path)` when `initial_hands_done == 0`) so Past Self only ever plays a snapshot of the *current* run, never a stale leftover. Until the first snapshot exists it cleanly falls back to the TAG heuristic. Also split the `past` vs `tag` styles in the simulator so the two HUD rows are genuinely distinct (`past` → past_model, `tag` → static TAG heuristic).

**Status**: ✅ Verified end-to-end: snapshot saved at the 5k crossing and loaded back as an opponent with no error.

---

# V11 Action-EV Extrapolation (Raise-Everything Hallucination)

**Date Recorded**: 2026-07-13
**Related Files**:
- [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/train_selfplay.py)
- [six_max_simulator.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/six_max_simulator.py)
- [health_report_expert_v11_main_200k.md](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/resources/models/v11/health_report_expert_v11_main_200k.md)

## Context
The 200k model (post loose-collapse fixes) was run through the model testing suite and **failed CRITICALLY**: it output positive `Raise` EV — and therefore `argmax` = RAISE — for *every* scenario, including a 0-equity river pure-air bluff and 72o. `Fold` was correctly ~0 (the fold-baseline fix), which only made it worse: since Raise EV never dipped below 0, the model never folded. Verified as a genuine model defect (identical across all inference conventions: pad direction, mask, read index), not a harness artifact.

## Root Cause
Both the fold-baseline design and the earlier taken-action masking trained the `Call`/`Raise` Q-heads **only on states where those actions were actually taken**. Because the model rarely voluntarily raised trash in-game, the raise-head **never received a negative signal for raising air** and extrapolated a positive EV to every unseen state — the classic off-policy data-gap hallucination the OFK suite (Scenario A/F) is designed to catch. The model learned relative equity *ordering* but not the absolute break-even for aggression.

## Resolution
Train **all three heads** using the simulator's true-equity Monte-Carlo counterfactual (`_calculate_mc_target_evs`, which returns `ev_call = equity·pot − cost`, `ev_raise` with a heuristic fold model — both correctly **negative for air**):
- Fold → `0.0` (weight 1.0), taken action → realized `mc_return` (weight 1.0), **untaken actions → MC counterfactual EV (weight `COUNTERFACTUAL_WEIGHT` = 0.5)**.
- Retained the ±40 BB clip and the preflop tightness prior.

**Status**: ✅ Implemented; smoke-tested on 6k hands — raw `Raise` EV collapsed from ~2.0–3.8 (raise-everything) to ~0.1–0.45, confirming the counterfactual signal closes the hole. Full fresh retrain launched; **re-run the model testing suite on completion to confirm** air→fold / nuts→raise discrimination.

---

# V11 Windows File-Lock Crash on Weight Save (Continuation Died at 199k/200k)

**Date Recorded**: 2026-07-13
**Related Files**:
- [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v11/train_selfplay.py)
- Pairs with [windows-pytorch-multiprocessing-deadlock.md](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/windows-pytorch-multiprocessing-deadlock.md)

## Context
A resumed 100k→200k continuation run crashed at 199,355/200,000 hands with `RuntimeError: File ...temp_active_model_main.pth cannot be opened` inside `torch.save`. The final checkpoint save never ran; recovery relied on the intact `v11_past_checkpoint.pth` (195k snapshot).

## Root Cause
`temp_active_model_{personality}.pth` is rewritten by the main process at the top of every batch while worker processes read it. On Windows, opening the file for write while a worker holds a read handle (or AV/indexer touches it) transiently fails — a non-deterministic file-lock race, unrelated to the model.

## Resolution
Added `robust_torch_save(state_dict, path)` and routed all four save sites (active-model, past-self, mid-flight diagnostic, final) through it. It writes to a unique temp file then **atomic `os.replace`** (workers never see a half-written file) with **6 retries + backoff**; a total failure warns instead of crashing (workers reuse the last complete file). Also added a `--num_hands` CLI flag so continuation is first-class: `--resume_path <ckpt> --hands_done 100000 --num_hands 200000` continues a run for another 100k (resume loads weights, appends the CSV, preserves past-self).

**Status**: ✅ Verified: the finishing tail run saved cleanly through the hardened path; final 200k checkpoint loaded strict/163-dim.


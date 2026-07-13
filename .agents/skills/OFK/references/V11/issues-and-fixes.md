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


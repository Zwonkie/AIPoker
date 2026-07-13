# V12 Issues and Fixes

**Date Recorded**: 2026-07-13
**Related Files**:
- [simulator.py](file:///c:/REPO/Antigravity/AIPoker/versions/v12/self_play/simulator.py)
- [contract.py](file:///c:/REPO/Antigravity/AIPoker/versions/v12/core/contract.py)
- [engine.py](file:///c:/REPO/Antigravity/AIPoker/core/models/engine.py)

## Context: Context Alignment Mismatch (P0)
The legacy `ContractV8V9` extracted opponent HUD stats differently than the offline `vectorize_hand_samples` in `train.py`.
- `train.py` defaults an empty opponent seat to VPIP `0.30` and AGG `0.40`. 
- The old contract defaulted empty seats to `"Blue"` (`0.10`/`0.18`).
This resulted in context vectors being out of alignment at inference time.

## Resolution
`ContractV12` was updated to correctly default empty seats to `"Yellow"` and `"Green"` to exactly mirror the training vectorizer's inputs. A `test_v12_alignment.py` was created to assert exact alignment.

---

## Context: PyTorch Transformer Mask Bug & NaN Crash (P0.5)
While aligning tensors in V12, a major PyTorch issue was hit: passing `key_padding_mask` during inference caused padded tokens to output `NaN`, which infected downstream unpadded tokens through the causal multi-head attention.
Worse, the `key_padding_mask` was NEVER passed during training in `train.py`. This meant the model was being trained to treat the sequence padding as valid zero-states, and passing a mask at inference time was actually pushing it out-of-distribution.

## Resolution
Removed `key_padding_mask` entirely from `engine.py`, `simulator.py`, and `ContractV12`. Because the network is trained with zeros for padded sequence steps without a mask, it inherently learns to attend to them properly. This completely resolves the `NaN` crash and ensures the inference attention distribution exactly matches training.

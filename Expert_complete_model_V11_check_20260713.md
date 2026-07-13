# Expert Complete Model Check: V11
**Date**: 2026-07-13
**Model Name**: expert_v11_main / expert_v11_nit

## 1. Executive Summary
A comprehensive helicopter review of the entire V11 training pipeline and data processing architecture has been conducted under the Zero-Trust Engineering Mandate. 

The review successfully isolated the root cause of the "Monster-Under-The-Bed" syndrome (the model folding the absolute nuts). The issue is **not** caused by flawed target EV math or incorrect heuristic actions. Instead, it is caused by a catastrophic **Vectorization Misalignment** between the training data loaders and the inference data contract.

## 2. Forensic Findings

### A. Vectorization Misalignment (The Fatal Flaw)
There is a massive discrepancy between how sequence tensors are populated during training vs. live play:
- **Training Phase (`train_selfplay.py`)**: Hands are vectorized using **Right-Padding**. Decision points are populated sequentially starting from index `0`. Thus, for a 3-action hand, valid data occupies indices `0, 1, 2`, while `3..19` remain `[0.0]`. The positional embeddings at early indices learn actual poker states.
- **Inference Phase (`contract_v11.py`)**: The inference contract places the single current state at `context_seq[-1]` (index 19) and leaves indices `0..18` entirely blank (**Left-Padding**).

**Why this breaks the model:**
During live play, the model is queried at `q_vals[-1]`. Since `pos_emb[19]` was almost never populated during training, it consists of untrained, random noise. Furthermore, by leaving indices `0..18` blank during inference, the transformer is robbed of its state history (e.g., Flop context). The model evaluates out-of-distribution garbage, panics, and defaults to low-variance actions (Folding).

### B. Reward System & Target Generation
The Monte Carlo Target EV calculations in `six_max_simulator.py` were audited and mathematically verified.
- The simulator correctly evaluates `Fold EV` as `0.0`.
- It accurately computes the exact net profit for Calling and Raising.
- It correctly models `opp_equity` against random hands, causing heuristic opponents to call raises when Hero holds the nuts, which correctly produces a massive `Raise EV`.
- The scaling in `train_selfplay.py` properly aligns target EVs and actual MC returns to Big Blinds.

### C. Model Architecture
The autoregressive causal mask bug from V10 (`~torch.eye`) has been successfully removed. V11 uses a standard causal mask (`torch.triu(..., diagonal=1)`), which properly allows historical sequence attention without padding dilution.

## 3. Mandatory Remediation Steps
To fix the V11 pipeline, the padding direction must be unified. Autoregressive transformers extracting final predictions require **Left-Padding**.
1. **Update `train_selfplay.py`**: Offset the decision points by `max_seq_len - len(dps)` to match the inference structure.
2. **Update `contract_v11.py`**: Refactor the inference contract to maintain a rolling buffer of historical `BoardState` context vectors, rather than only populating index `-1` in an empty array.
3. **Retrain**: Flush the old weights and restart the training league with the aligned tensors.

---
*Report generated via standard checklist operating procedure.*

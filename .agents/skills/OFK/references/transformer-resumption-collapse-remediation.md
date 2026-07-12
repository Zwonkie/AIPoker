# Transformer Resumption Collapse & LR Remediation

**Date Recorded**: 2026-07-11
**Related Files**: [train_selfplay.py](file:///c:/REPO/Antigravity/AIPoker/tools/self_play/v8/train_selfplay.py)

## Context
When resuming training of the `PokerEVModelV4` Decision Transformer from a weights-only checkpoint (such as `expert_v8_selfplay_50k.pth`), the training state and optimizer momentum values (moments) are initialized from scratch. Resuming training at the default learning rate of `1e-3` caused immediate representation/value collapse. The network outputs converged to near-zero flat predictions across all hand types, leading to critical failure behaviors like folding pocket Aces preflop to minimize Huber loss against the high volume of folding sequences.

**League Training Symptom**: In self-play league simulations (V8/V9) where `bootstrap_alpha` decays heuristics in favor of NN inference, a collapsed model causes catastrophic VPIP drift. As `bootstrap_alpha` drops, opponent bots querying the corrupted NN will drift towards 50% VPIP (random action noise), which completely destroys the validity of the training environment data.

## Resolution / Guidelines
1. **Reduce Learning Rate on Resumption**: Always scale down the learning rate by a factor of 10 (e.g., from `1e-3` to `1e-4`) when resuming training from a weights-only checkpoint without loaded optimizer momentum state.
2. **LR Config Parameter**: Added the `--lr` argument to the V8 training script command-line interface to allow configurable optimizer learning rates.
3. **Verification**: Run `run_active_sanity_check.py` immediately after the first batch finishes training to verify that the EV predictions for pocket Aces (`Ah As`) preflop remain well-separated (+7.0 BB to +9.5 BB) and that the model recommends raising.

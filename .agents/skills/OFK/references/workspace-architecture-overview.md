# Workspace Architecture Overview

**Date Recorded**: 2026-07-12
**Related Files**: 
- [PHPHelp.py](file:///c:/REPO/Antigravity/AIPoker/PHPHelp.py)
- [board_state.py](file:///c:/REPO/Antigravity/AIPoker/core/board_state.py)
- [contract_v8_v9.py](file:///c:/REPO/Antigravity/AIPoker/core/bridge/contract_v8_v9.py)
- [engine.py](file:///c:/REPO/Antigravity/AIPoker/core/models/engine.py)

## Context
The repository evolved from a monolithic structure where UI, OCR vision, data normalization, and ML tensor math were tightly coupled in `PHPHelp.py`, `simulator.py`, and `ml_bridge.py`. A massive refactoring enforced a strict 4-layer architecture.

## Resolution / Guidelines

The workspace is strictly partitioned into the following sequence:

1. **Dashboards (`PHPHelp.py`)**: 
   The top-level GUI and vision/OCR entry point. It captures screen data, reads table components, and filters noise via `TableState`. Its only job is to emit a clean, domain-pure `BoardState`.
   
2. **BoardState (`core/board_state.py`)**: 
   The pure domain object (dataclass). It tracks the physical properties of a poker hand (stacks, pot, community cards, opponent HUD stats) completely independently of machine learning concepts (e.g., tensors, padding masks).
   
3. **Model Bridge / Data Contract (`core/bridge/`)**: 
   The translator. It takes a generic `BoardState` and converts it into the exact tensor schema required by a specific model version. 
   *(e.g., `ContractV8V9` translates state to the 31-feature model for `PokerEVModelV4`)*.

4. **Model Engine (`core/models/`)**: 
   The PyTorch inference wrapper. It receives the standardized tensors from the Bridge, runs a forward pass through the Neural Network (`engine.py`), and returns universal Fold/Call/Raise Q-values.

## Sandbox Separation

To decouple stateless GUI parsing from stateful ML sequences, follow the sandbox paradigm:

1. **Live Vision Sandbox (`PHPHelp.py`)**: Must remain strictly stateless. It extracts the table frame-by-frame and emits a single current `BoardState` on each turn. It does not track hand history.
2. **Training Simulator Sandbox (`tools/self_play/`)**: Evaluates the network without OCR. Because it plays sequentially, the simulator is responsible for natively maintaining `model_state_history` lists for all players and passing them to the bridge.
3. **Live Bridge Sandbox (`core/decision.py`)**: Intercepts the single `BoardState` from the Live Vision Sandbox. It must act as a stateful memory buffer (`hand_history_buffer`) across a hand, accumulating states until the hand ends or the street resets, before passing the fully reconstructed history array to the model contract.

### Best Work Practices for this Workspace:
- **Simulator Neutrality**: RL self-play simulators (`tools/self_play/`) **must** construct and mutate `BoardState` objects natively. Do not duplicate model bridge logic in simulators. Let the simulator act exactly like `PHPHelp.py` does in live play.
- **Model Isolation**: When developing a new model version (e.g., `V10`), **create a new bridge contract** (`contract_v10.py`). Do not clutter old contracts with feature toggles or legacy support flags.
- **UI Decoupling**: Core decision logic (`core/decision.py`) only receives a `BoardState`. It must never import `customtkinter`, `mss`, or PyTorch models directly.

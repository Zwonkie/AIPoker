# Model Versioning & Cloning Standards

> [!IMPORTANT]
> **Expanded & superseded by [versioned-architecture-guardrails.md](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/versioned-architecture-guardrails.md)** (2026-07-13), which is now the CANONICAL, agent-binding guide. It adds the enforcement machinery this doc lacked: a version manifest/registry as the single dispatch point, self-describing weights with fail-loud loading, live/training contract unification, a new-version checklist, and Golden Rules. Read that first; the SOP below is a subset kept for continuity.

**Date Recorded**: 2026-07-12
**Related Files**: 
*   [core/models/poker_transformer.py](file:///c:/REPO/Antigravity/AIPoker/core/models/poker_transformer.py)
*   [core/decision.py](file:///c:/REPO/Antigravity/AIPoker/core/decision.py)

## Context
As the project evolved from V1 to V10, critical neural network architectures (like `poker_transformer.py`) and data contracts were modified in-place. This broke backwards compatibility, causing older weights (e.g. V8) to fail or behave incorrectly when loaded into the newer architecture. 

To maintain a strict sandbox and ensure that any past or future model version can be accurately re-evaluated, we now strictly isolate versions.

## Standard Operating Procedure (Version Cloning)
When beginning a new major model version (e.g. `V11`), the following folder structures and files **MUST** be cloned into a version-specific namespace before any modifications are made:

1. **Neural Network Architecture**
   - **Source:** `core/models/poker_transformer.py` (or the previous version's file).
   - **Target:** `core/models/v11/poker_transformer_v11.py`
   - *Reason:* Neural network `state_dict` weights are tightly coupled to the class definitions. Changing a layer dimension or adding an auxiliary head breaks past weights.

2. **Data Contracts**
   - **Source:** `core/bridge/contract_v8_v9.py`
   - **Target:** `core/bridge/v11/contract_v11.py`
   - *Reason:* If the sequence vectorization or context array changes, older models will receive misaligned inputs and hallucinate.

3. **Training & Simulation Loop**
   - **Source:** `tools/self_play/v10/`
   - **Target:** `tools/self_play/v11/`
   - *Reason:* The simulator and telemetry logic dictates how the model learns. Training hyper-parameters, curricula, and opponents should be sandboxed.

### Engine Integration
When running in production or executing mid-flight diagnostics, the `ModelEngine` inside `core/decision.py` must be explicitly instructed to import the correct `poker_transformer_vX.py` and `contract_vX.py` based on the weight file being loaded.

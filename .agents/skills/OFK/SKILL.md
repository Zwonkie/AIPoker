---
name: ofk-memory-manager
description: |
  Manages the project's Open Knowledge Format (OFK) memory. Use this skill 
  whenever creating, updating, reading, or indexing persistent memory files 
  and knowledge items in the workspace.
---
# Open Knowledge Format (OFK) Memory Management Guidelines
This skill guides the agent on how to interact with and maintain the workspace's persistent memory layer. The goal is to prevent repetitive context-gathering and ensure resolved bugs or design decisions are permanently recorded.
## 1. Directory Structure
The memory base is organized as follows:
- `SKILL.md`: The entry point, index, and rules for memory management.
- `references/`: A directory containing individual markdown knowledge files categorized by topic (e.g., `references/api-integration.md`, `references/resolved-bugs.md`).
## 2. When to Create or Update Memory
At the end of a task or chat session, the agent should evaluate if a new memory should be written or updated. You **MUST** record information if:
- **Critical Bugs Resolved**: You spent significant time debugging a complex issue or an obscure error. Document the root cause and the fix.
- **Architectural Decisions**: A design choice was made (e.g., *"Why we used library X instead of Y"*).
- **Environment Setup**: Complex setup steps or local dependencies are configured.
- **APIs and Schemas**: Crucial request/response shapes or DB schemas are identified.

## 2.5 Bot models (NN) and version management
Anything related to bot models (NN) and version management (retrain, issues, fixes, and so on) should have its own OFK ressource folder under that specific bot version it relates to. E.g. if you are working on the V8 bot model, you should create a folder named `V8` under the `references/` directory and store all related memory files in it. This should also apply to subversions of the bot model, such as `V8.1`, `V8.2`, etc. information can be catogorized into the following: "model_specifications" (NN model architectures, training parameters, etc.), "issues and fixes" (known issues, bugs, suggestion on how to improve it, for the next version) 

## 3. Formatting Memory Files (OKF Standard)
All memory files in the `references/` directory must follow these rules:
- **File Format**: Standard markdown with a `.md` extension.
- **File Name**: Descriptive, lowercase, and hyphenated (e.g., `database-setup.md`).
- **Structure**:
  ```markdown
  # [Descriptive Title]
  
  **Date Recorded**: YYYY-MM-DD
  **Related Files**: [main.go](file:///path/to/main.go)
  
  ## Context
  Provide 2-3 sentences explaining the background or the problem.
  
  ## Resolution / Guidelines
  Provide the specific steps, code patterns, or commands required to handle this topic.
  ```

## 4. Memory Directory Index
*   ⭐ [Versioned Architecture Guardrails (CANONICAL)](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/versioned-architecture-guardrails.md): **Read first before touching any model, contract, simulator, weight, or live-play code.** Binding rules for isolating each model version — manifest/registry dispatch, self-describing weights + fail-loud loading, live/training contract unification, the "start a new version" checklist, and Golden Rules that prevent old/new mixing.
*   [Workspace Architecture Overview](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/workspace-architecture-overview.md): High-level description of the 4-layer architecture (Dashboards -> BoardState -> DataContract -> ModelEngine) and best practices for extending simulators and models.
*   [Model Testing Suite](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/model-testing-suite.md): Comprehensive suite of critical edge-case scenarios (Pure Air Bluffs, Nutted Traps) for evaluating model health.
*   [Decision Transformer Data Contract](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/decision-transformer-data-contract.md): Data schema guidelines for state context variables.
*   [Transformer Resumption Collapse Remediation](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/transformer-resumption-collapse-remediation.md): Documents representation collapse on model training resumption and its LR mitigation.
*   [Windows PyTorch Multiprocessing Deadlock](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/windows-pytorch-multiprocessing-deadlock.md): Resolves PyTorch 0% CPU/GPU hang by keeping worker pools alive instead of respawning them.
*   [V4 Transformer Architecture](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V4/model-specifications.md): Design notes for Pluribus V4 model sequence inputs.
*   [V5 Training Optimizations](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V5/training-optimizations.md): Details hybrid preflop splits and PyTorch CUDA/AMP training speedups.
*   [V6 Model Specifications & Improvements](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V6/model-specifications.md): Outlines showdown variance reduction, structured action tokens, MARL pool, and PER.
*   [V5 vs V6 Sensitivity Report](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V6/v5-v6-sensitivity-report.md): Side-by-side validation results and attention collapse diagnosis.
*   [V7 Model Specifications & Architectural Remediations](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V7/model-specifications.md): Outlines key padding masks, all-action target EV losses, and postflop model play.
*   [V7 Sensitivity Report](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V7/sensitivity-report.md): Side-by-side validation results comparing V5, V6, and V7.
*   [V8 Model Specifications & League Training](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V8/model-specifications.md): Outlines diversity-based league training, pre-training strategies, and 10-20 hand dynamic profiling.
*   [V8 Sensitivity Report](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V8/sensitivity-report.md): Side-by-side preflop sensitivity analysis and exploitative strategy convergence results for all V8 personalities and Main Hero.
*   [V9 Model Specifications & Improvements](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V9/model-specifications.md): Outlines expanded 50-hand opponent profiling windows for stable strategy estimations.
*   [V9 River Bluffing Collapse](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V9/issues-and-fixes.md): Documents the target EV formula flaw causing V9 to over-bluff on the River and outlines short/long-term fixes.
*   [V9 Model Health Report](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V9/model_health_report.md): Final diagnostics grade (Critical Fail) exposing the Pure Air Hallucination.
*   [V10 Model Specifications & Training Improvements](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V10/model-specifications.md): Details fixes for rigid opponent exploits (e.g. River over-folding) and mandates the Bluff Matrix / action entropy training telemetry.
*   [V10 Implementation Plan](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V10/implementation-plan.md): Outlines the Monte Carlo GTO evaluation and opponent bot dynamics introduced to fix the Pure Air Hallucination.
*   [Monitor Training Session Skill](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/monitor-training-session/SKILL.md): Standard operating procedure for actively evaluating live training telemetry, detecting rigidity, and stopping early collapses.
*   [V11 Model Specifications & Training Improvements](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V11/model-specifications.md): Details the roadmap for V11, including Weighted Personality Focus Rounds, Interpretable Auxiliary Heads, and Fuzzy Heuristic Opponents.
*   [V11 Issues & Fixes](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V11/issues-and-fixes.md): Running log of V11 defects and remediations — per-personality stat attribution, the swallowed-KeyError that disabled the NN, the loose-collapse ratchet, the raise-everything hallucination + counterfactual-target fix, past-self wiring, and the Windows file-lock save crash + robust_torch_save.
*   [V12 Design Proposal](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V12/design-proposal.md): Root-cause-driven roadmap for V12 — unify the inference contract, replace the MC-return objective with regret-matching/actor-critic, fail-loud checkpoint versioning, gameplay-based evaluation, and bet-sizing / NN-league upgrades.
*   [Model Versioning & Cloning Standards](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/versioning-standards.md): SOP for isolating neural network architectures, data contracts, and training loops into version-specific namespaces to prevent breaking backwards compatibility.
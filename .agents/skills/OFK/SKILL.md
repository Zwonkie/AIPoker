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
*(Cleaned 2026-07-15: merged the V4–V10 per-version specs/reports into `legacy-versions-history.md`; deleted point-in-time health reports for dead models, the self-superseded versioning-standards, the stale 31-dim data-contract doc, and the realized V12 design-proposal. Current model line = V15 live.)*

### Architecture & process
*   ⭐ [Versioned Architecture Guardrails (CANONICAL)](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/versioned-architecture-guardrails.md): **Read first before touching any model, contract, simulator, weight, or live-play code.** Binding Golden Rules for isolating each version — manifest/registry dispatch, self-describing fail-loud weights, live/training contract unification, the "start a new version" checklist. §0 tracks the current foundation (V15).
*   [Workspace Architecture Overview](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/workspace-architecture-overview.md): The 4-layer partition (Dashboards → BoardState → Contract → ModelEngine) + the three-sandbox rule. (Examples reference the older ContractV8V9; the layering itself is current.)
*   [Pipeline Flow (Sim/Train vs Live)](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/pipeline-flow.md): Two maintained Mermaid diagrams of every condition/logic/data tweak in the sim/training pipeline and the live path, colour-coded by scope (global/board/player-state), with ID'd boxes + a train≡serve invariants checklist. Update on any pipeline tweak.
*   [Simulation Architecture (deep dive)](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/simulation_architecture.md): Prose companion to the pipeline diagram — opponent pool/weights, 5-phase curriculum, bootstrap decay, HandRecord, and the per-head counterfactual+realized Q-target math. (V10/V11-era internals; the 6-action/ContractV12 specifics live in the V13–V15 specs.)

### Live play & infra
*   [Decision Pipeline Tracing & GUI Overrides](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/decision-pipeline-tracing-and-gui-overrides.md): Live-play mechanics — decision_path trace, the ⚡ Auto-Live quick-start + window matching, cents-basis blind parsing from the window title, and hero-position tracking. (Default-model/action-space framing predates the 6-action space.)
*   [HUD Thresholds & Aggression](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/hud-thresholds-and-aggression.md): Canonical VPIP/AFq HUD colour bands (Blue/Green/Yellow/Red) + the colour→midpoint vectorization map. Consumed by both live HUD reads and sim opponent modelling.
*   [Model Testing Suite](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/model-testing-suite.md): The standard sensitivity/sanity scenarios (72o-shove extrapolation, air bluffs, nutted traps, equity monotonicity) for evaluating model health. (Re-express "Raise EV" against the 6-action space when using.)
*   [Windows PyTorch Multiprocessing Deadlock](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/windows-pytorch-multiprocessing-deadlock.md): Version-agnostic infra fix — keep the worker Pool alive (reuse via starmap) + ship weights via a temp .pth, to avoid the spawn+CUDA re-init hang.
*   [Monitor Training Session Skill](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/monitor-training-session/SKILL.md): SOP for evaluating live training telemetry, detecting rigidity, and stopping early collapses.

### History
*   [Legacy Versions History (V4–V10)](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/legacy-versions-history.md): One paragraph per superseded version — durable lessons only (off-policy EV hallucination, attention/padding collapse, all-action loss, terminal-state value inflation, stack-size curriculum, collapse-detection telemetry). Tables/point-in-time reports dropped.

### Current model line (V11 bridge → V15 live)
*   [V11 Model Specifications](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V11/model-specifications.md): The bridge into the current architecture — 35-feature contract, counterfactual policy targets, causal masking, left-padding, aux heads, fuzzy heuristics. (Retained standalone — not folded into the legacy history.)
*   [V11 Issues & Fixes](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V11/issues-and-fixes.md): The richest catalogue of still-live bug patterns — unit-scale (chips vs BB) mismatch, train/infer padding parity, `idx` shadowing, arch↔weight shape mismatch, silent-heuristic fallback, loose-collapse ratchet, off-policy raise-everything, atomic weight saves.
*   [V12 Validated Findings](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V12/validated-findings.md): The frozen v12_validated baseline — locked fixes (target clip 40, counterfactual policy target, equity-primary arch, realization discount, postflop-data field) + the safe-change workflow.
*   [V12 Issues & Fixes](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V12/issues-and-fixes.md): Two durable V12 gotchas — the `key_padding_mask` NaN/OOD trap and the train↔ContractV12 context-alignment mismatch.
*   [V13 (specs / milestone / validated-findings)](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V13/specs.md): Equity-primary + range-aware equity (opponent adaptation); first live-viable MILESTONE foundation. Folder also has `milestone.md`, `validated-findings.md`.
*   [V14 Specs](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V14/specs.md): Discretized 6-action bet-size space {fold,call,raise_33/66/pot,allin}; short-stack winner; later found to have deep-stack OOD.
*   [V15 Specs](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V15/specs.md): **CURRENTLY LIVE.** Same 6-action contract retrained on a DoN-shaped stack mixture (5–50bb) + a frozen-V14 opponent; fixes the deep-stack OOD; loose-aggressive winner vs loose fields.
*   [V16 Roadmap / Backlog](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V16/specs.md): Open items — P2 stack-scaled temp (done), P4 opponent-aware preflop VPIP, P3 preflop polarization, P5 bet-size perception, P6 opponent-action attribution, size-scaled bluffing.
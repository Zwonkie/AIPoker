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
*   [Decision Pipeline Tracing & GUI Overrides](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/decision-pipeline-tracing-and-gui-overrides.md): Live-play mechanics — decision_path trace, the ⚡ Auto-Live quick-start + window matching, cents-basis blind parsing from the window title, hero-position tracking, and the live "Thinking" narrative (§7, 2026-07-15 — equity+action-banded human-readable read, deliberately NOT sourced from the untrained/opponent-read aux heads). (Default-model/action-space framing predates the 6-action space.)
*   [HUD Thresholds & Aggression](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/hud-thresholds-and-aggression.md): Canonical VPIP/AFq HUD colour bands (Blue/Green/Yellow/Red) + the colour→midpoint vectorization map. Consumed by both live HUD reads and sim opponent modelling.
*   ⭐ [Model Verification Suite](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/model-verification-suite.md): **The current standing tool — `tools/model_verify`.** Version-agnostic fast scenario checks (deep-stack OOD guard, equity-ablation monotonicity, action-collapse guard, etc.) + slow simulated-hand checks (`--full`: VPIP-vs-style, BB/100-vs-baseline, beats-frozen-predecessor), a growing append-only curriculum, and an HTML raw-data report generator. Run after every training run. Documents 2 calibration bugs (substring-match trap, over-strict diversity threshold) and a live finding (V15's deep-stack OOD trash-jam is still present under decision-level probing despite passing aggregate evals).
*   [Model Testing Suite (superseded)](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/model-testing-suite.md): V4-V6 era manual/ad hoc sensitivity scenarios (72o-shove extrapolation, air bluffs, nutted traps, equity monotonicity), predates the 6-action contract and isn't committed as runnable code. Superseded by Model Verification Suite above; kept only for historical scenario-design ideas.
*   [Windows PyTorch Multiprocessing Deadlock](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/windows-pytorch-multiprocessing-deadlock.md): Version-agnostic infra fix — keep the worker Pool alive (reuse via starmap) + ship weights via a temp .pth, to avoid the spawn+CUDA re-init hang.
*   [Monitor Training Session Skill](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/monitor-training-session/SKILL.md): SOP for evaluating live training telemetry, detecting rigidity, and stopping early collapses. Dashboard tool itself lives standalone at `tools/training_monitor/` (not skill-owned).

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
*   [V16 Roadmap / Backlog](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V16/specs.md): **P2 done** (stack-scaled live temp). **P4 trained** (preflop CALL/FOLD target swapped from oracle equity to range-aware equity — single substitution, no new tuned constants, no contract bump); V16 itself is trained but NOT yet deployed live (V15 still active/default). **P3 deferred** (later CONFIRMED resolved via model_verify — see V17). Also: P5/P6 (input-contract gaps), size-scaled bluffing, a RECONFIRMED V15 deep-stack-OOD finding (P0-recheck, V16 inherits it), P7 (opponent-pool "Yellow"/LAG training gap), and the live "Thinking" narrative feature (equity+action-banded HUD line, `core/decision.py`/`PHPHelp.py`). **`v16_foldregret` sub-experiment trained + evaluated, NOT deployed** (2026-07-15): fold-relative regret baseline fixed the air/draws free-riding continuation problem cleanly, but flipped the model's style (tight_deep fixed, loose_deep collapsed vs the actual loose-heavy live population) — kept as a reference result, root-cause analysis superseded it (see V17).
*   ⭐ [V17 Roadmap (planning only, no code yet)](file:///c:/REPO/Antigravity/AIPoker/.agents/skills/OFK/references/V17/specs.md): Synthesized from reading the full V16 line's results. Carries forward P2/P4/AGG-fix/checkpoints/thinking-narrative/model_verify-as-gate; explicitly does NOT carry forward foldregret's fold-relative baseline. Flagship item: **[P-actor-critic]** — route the actor's regret-matching target through the critic's OWN learned (denoised, accumulated) Q-values instead of a fresh noisy single-hand simulator sample, replacing a rejected "blend two tuned baselines" patch with a structural fix (reuses the existing `bootstrap_alpha` schedule pattern rather than a new tuned constant). Also: [P0] deep-stack OOD (3 versions running unaddressed — top priority), [P3] confirmed resolved, [P5]/[P6] promoted over further target-formula tuning, a model_verify weighted-composite-score tooling gap.
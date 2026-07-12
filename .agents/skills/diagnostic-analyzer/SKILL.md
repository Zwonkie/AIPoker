---
name: diagnostic-analyzer
description: |
  Analyzes the latest poker bot diagnostics. Triggers on user prompts containing keywords like:
  - "latest diagn."
  - "diagonsticsd"
  - "diag."
  - "check dia*" (check diagnostics)
---

# Diagnostic Analyzer Skill

This skill outlines how the agent should automatically inspect, analyze, and report on the latest session diagnostics to detect OCR errors, model EV inconsistencies, or pipeline bugs.

---

## 1. Trigger Verification
When a user asks to inspect diagnostics (using terms like `diag.`, `diagonsticsd`, `latest diagn.`, `check dia*`), this skill is activated.

## 2. Analysis Workflow

### Step 1: Locate the Latest Diagnostic Directory
Scan the `diagnostics/` folder in the project root. Sort directories by timestamp (naming pattern is typically `turn_YYYYMMDD_HHMMSS`) and pick the most recent one.

### Step 2: Read Log and Telemetry Data
Read and parse the files in the target diagnostic directory:
1.  **`logs.txt`**: Read the chronological steps leading up to the decision (OCR scans, window matching, state changes).
2.  **`telemetry.json`**: Parse the serialized table state, the active opponents mask, the raw EVs/Q-values calculated by the neural network, and the triggered decision path (heuristic charts, bluff engine, math guardrails, active model name).

### Step 3: Inspect Visual Assets
Identify any raw screenshots or cropped debug images in the folder (e.g. `screenshot.png`, `card_crops/`, `seat_crops/`) to verify if the OCR template matching was correct.

### Step 4: Generate Structured Report
Generate a summary report containing:
- **Observations**: Hand info, street, Hero cards, board cards, stakes, active players, pot size, stacks, GTO position.
- **Decision Trace / Reasoning**: Active model, raw EV values (Fold, Call, Raise), and the decision path taken (which guardrails triggered or bypassed).
- **Divergence / Issue Analysis**: Highlight any bugs (e.g. OCR misread, stack size mismatch, incorrect blind scale, active player mask error).
- **Proposed Fixes**: Detailed bullet points suggesting how to resolve the identified discrepancy.

### Step 5: Stop & Request Feedback
**DO NOT make any changes or implementations.** 
Present the diagnostic report to the user and explicitly ask:
> "Should I proceed with implementing these improvements?"

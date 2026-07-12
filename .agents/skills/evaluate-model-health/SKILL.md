---
name: evaluate-model-health
description: |
  Automatically executes the model testing suite to objectively evaluate a newly trained model for mathematical holes, bluff collapses, and exploitability.
  Trigger when the user asks to "test the model", "run diagnostics on V10", or "evaluate model health".
---

# Evaluate Model Health Skill

This skill defines the exact workflow for running the objective Model Testing Suite on any trained neural network, and comparing its output against the required criteria to issue a Pass/Fail grade.

## 1. Trigger Verification
When a user asks to evaluate a model (e.g., "test V10", "evaluate Pluribus (v9 Main)", "check the model for holes"), this skill activates.
Always ask the user which exact model name they want to test if it's not provided (e.g., `Pluribus (v9 Main)`).

## 2. Execution Workflow

### Step 1: Run the Diagnostic Script
Execute the following terminal command, passing the requested model name as the argument:
```bash
.venv\Scripts\python.exe scripts\math\run_model_diagnostics.py "Model Name Here"
```

### Step 2: Parse the Raw EVs
The script outputs the raw Fold, Call, and Raise EVs across 8 critical scenarios. You must read these values into your context.

### Step 3: Grade Against OFK Criteria
Cross-reference the EV outputs with the rigorous expectations defined in `OFK/references/model-testing-suite.md`. 
Particularly check:
- **River Pure Air (Bluff Collapse Check)**: Did EV(Raise) or EV(Call) exceed EV(Fold) when equity is 0.0? If yes -> **CRITICAL FAIL**.
- **River The Nuts**: Did EV(Raise) dominate EV(Call)? If EV(Call) > EV(Raise) -> **FAIL** (Missing value).
- **Preflop 72o vs Shove**: Is EV(Fold) optimal? (Note: Update the script if 72o is needed, otherwise check AA).
- **Multi-Way TPTK**: Did the model overvalue TPTK in a 4-way pot?

### Step 4: Generate the Artifact Report
Do not just dump the terminal output to the user. Create a beautifully formatted markdown artifact named `model_health_report.md`.
It must include:
1. **Target Model**
2. **Overall Grade**: PASS, WARNING, or CRITICAL FAIL
3. **Scenario Breakdown**: A table summarizing each scenario, the model's action, and whether it Passed or Failed the OFK criteria.
4. **Holes Discovered**: Detailed bullet points explaining exactly *why* the model failed a specific scenario (e.g., "The model hallucinated 0.63 EV for calling a bet with 3-high on the River").

### Step 5: Stop & Present
Present the artifact to the user and ask if they would like to review the architecture to fix any identified holes.

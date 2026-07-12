---
name: monitor-training-session
description: |
  Monitors active live training sessions. Trigger this skill every 5 minutes (e.g. via schedule) during training to parse the logs, extract the boardstate, and run automated health checks against rigidity and collapse.
---

# Monitor Training Session Skill

This skill defines the standard operating procedure (SOP) for checking on an active poker neural network training loop.

## 1. Trigger Verification
When a training session is actively running in the background, use the schedule tool (e.g., `CronExpression="*/5 * * * *"`) to trigger this skill every 5 minutes.

## 2. Execution Workflow

### Step 1: Retrieve the Active Log
Find the active background training task. Run a command like `Get-Content <logfile> -Tail 50` to pull the latest output block from the training dashboard.

### Step 2: Parse Telemetry
Run the parsing script to extract the dashboard output and generate the `telemetry.json` data:
`.venv\Scripts\python.exe .agents\skills\monitor-training-session\scripts\parse_training_log.py <logfile>`
Carefully write the parsed dashboard output in the chat and extract the following essential metrics into your context:
1. **Progress**: Hands Simulated and ETA.
2. **Loss Metrics**: Train Loss and Val Loss.
3. **Action Entropy**: The entropy of the model's policy distribution.
4. **Equity Matrix**: The full matrix showing Fold, Call, Raise, RR, All-In, Avg End Street, and Net Chips across 5 equity brackets (<20%, 20-40%, 40-60%, 60-80%, >80%).
5. **Overall Boardstate**: Extract the full 6-seat table showing `BB/100`, `VPIP`, `AGG`, and the counts for `[R:  F:  AI: ]` for Hero and all opponents.

### Step 3: Automated Health & Rigidity Checks
Compare the extracted telemetry against the following critical thresholds. If any threshold is breached, you MUST flag it prominently using a `> [!WARNING]` or `> [!CRITICAL]` alert.
- **Rigidity Check**: If `Action Entropy < 0.10`, the model is becoming deterministic and rigid.
- **Bluff Collapse Check**: If the `<20% (Pure Air)` row in the Equity Matrix shows a `Raise %` > 5.0% or `RR %` > 0.0%, the model is hallucinating fold equity and attempting pure-air bluffs into calling stations.
- **Exploitability Check**: If Hero's `BB/100` drops below 0 against static opponent bots.
- **Loss Divergence**: If Val Loss is exploding while Train Loss drops (overfitting).

### Step 4: Present the Status Overview
Create a clean, well-formatted markdown response for the user containing:
1. The **Progress** and **ETA**.
2. The **Action Entropy** and the fully formatted **Equity Matrix**.
3. The **Overall Boardstate** formatted clearly (using a markdown table or bulleted list).
4. Any **Alerts/Warnings** triggered by the health checks.
5. A brief 1-2 sentence analysis of how the model is behaving (e.g. "Hero is heavily exploiting the Sticky bot" or "The model is becoming rigid").
6. **Remind the user** that they can view a live UI version of this data by running `.\.agents\skills\monitor-training-session\scripts\start_dashboard.ps1`.

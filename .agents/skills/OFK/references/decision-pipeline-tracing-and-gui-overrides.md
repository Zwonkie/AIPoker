# Decision Pipeline Tracing and GUI Overrides

**Date Recorded**: 2026-07-11
**Related Files**: [decision.py](file:///c:/REPO/Antigravity/AIPoker/core/decision.py), [PHPHelp.py](file:///c:/REPO/Antigravity/AIPoker/PHPHelp.py)

## Context
The poker play engine features three override layers around the raw neural network (preflop range chart, postflop bluffing, and math-engine call boundaries). Previously, these overrides were opaque, making it difficult for the user to understand which layer determined the recommended action. 

Additionally, starting the bot for live capture was a repetitive manual process: selecting "Automatic Play", switching to "Live Capture", looking up the correct Bet365 window title (which always contains exactly 4 pipes `|`), and clicking start.

## Resolution / Guidelines
We implemented real-time decision path tracing and a side-by-side Auto-Live shortcut button to solve these usability challenges:

### 1. Decision Path Tracing
Inside `PokerDecisionEngine.make_decision`, we construct a `decision_path` trace dictionary documenting the status and details of each of the 4 logical stages:
- **`preflop_chart`**: Bypassed (postflop), Triggered (active chart override), or Disabled.
- **`active_model`**: Active (standard neural network output) or Overridden (by preflop chart).
- **`bluff_engine`**: Bypassed (preflop), Passed (checks not met), or Triggered (active draw/stab bluff).
- **`math_engine`**: Bypassed, Passed, or Triggered (forced fold due to equity < pot odds + buffer).

To prevent breaking existing scripts that unpack a 4-tuple from `predict_action`, the path is injected into the existing `ev_dict` payload:
```python
if ev_dict is None:
    ev_dict = {}
ev_dict['decision_path'] = decision_path
```

### 2. Window Title Blind Parsing
When capturing a live window, the engine automatically extracts the Small Blind (SB) and Big Blind (BB) from the window title (e.g. `50/100` or `€0.10/€0.20`).
To prevent decimal matching bugs (such as parsing `0.10/0.20` as `10/0`), the regex parser executes a two-tier extraction:
1. **Decimal Stakes**: Matches patterns like `€0.10/€0.20` and converts the parsed float values to cents (e.g. `sb = 10`, `bb = 20`).
2. **Integer Stakes**: Fallback for tournament or tournament-like stakes (e.g. `50/100` or `10/20`), keeping them as raw integers.

### 3. GUI Visualization
The `PHPHelp` dashboard width was expanded to `1240x820` to host a new **Decision Flow Pipeline** column:
- Displays four vertical boxes (one per stage) with connecting flow arrows.
- Color-coded status dots (`●`):
  - Green (`🟢`): Triggered override.
  - Blue (`🔵`): Stage evaluated the state and approved the action.
  - Red (`🔴`): Mathematical safety guardrail overrode the action (e.g. forced FOLD).
  - Gray (`⚪`): Bypassed/disabled.

### 3. Auto-Live Quick Start & Robust Window Matching
We added a smaller `⚡ LIVE` shortcut button side-by-side with the `START BOT` button. Clicking this:
1. Keeps the current selected Decision Model.
2. Automates selection of "Automatic Play" mode and "Live Capture" source.
3. Automatically scans all active window titles using four fallback strategies:
   * **Strategy 1**: Matches window titles with exactly 4 pipes (`|`), representing `x | x | x | x | x`.
   * **Strategy 2**: Matches titles containing `"bet365"` (case-insensitive).
   * **Strategy 3**: Matches titles containing poker terms (e.g. `"hold'em"`, `"omaha"`, `"no limit"`) + at least one pipe (`|`).
   * **Strategy 4**: Matches any title containing `"hold'em"`, `"omaha"`, or `"poker"`.
4. Selects the matched window and triggers the bot to start.
5. If no window is found, it dumps a list of all visible window titles to the GUI logs to provide immediate diagnostic feedback.

### 4. Default Heuristic Settings
All three guardrails (Pre-flop Chart, Math Engine, and Bluff Engine) now have dedicated checkboxes in the sidebar and are **disabled (unchecked) by default**. This ensures the bot defaults to running raw model decisions while allowing you to turn safety overrides on and off dynamically.

### 5. Pluribus (v4 Self-Play) Decision Transformer Registration
The Pluribus v4 Decision Transformer (`expert_v4_selfplay.pth`) has been registered in the `PHPHelp` dashboard selection dropdown and default-selected. 
*   Because the V4 model has been trained end-to-end on the complete game tree (including pre-flop ranges), it **bypasses the heuristic pre-flop range chart by default** (just like the V3 model) to allow you to run and evaluate true model-generated actions.

### 6. Dynamic Hero GTO Position Tracking
Previously, the `hero_position` feature passed to the ML models was not dynamically computed and always defaulted to `0` (Button). We implemented a robust hybrid tracking system:

1.  **OCR-Based Dealer Detection (Primary Real-time Source)**:
    *   **Template Matching**: Scans the screenshot for a cropped yellow dealer token (`card_templates/dealer_button.png`) using `cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)`.
    *   **Euclidean Anchor Mapping**: Calculates the Euclidean distance from the button's matched center coordinate $(x_D, y_D)$ to each of the 6 seat centers:
        $$d_i = \sqrt{(x_D - x_i)^2 + (y_D - y_i)^2}$$
    *   The seat key with the minimum distance is set as the active dealer.
2.  **XML Extraction (Transaction History Fallback)**:
    *   If the dealer button is not visually detected (e.g. during animations), the system parses the Bet365 XML game file (looking up the `dealer="1"` attribute in `<player>` elements) as a baseline fallback seed.
3.  **GTO Position Computation**:
    We calculate the clockwise seat offset of Hero relative to the dealer seat index $D$ (where Hero is $0$, and `seat_i` is $i$):
    $$\text{hero\_position} = (0 - D) \pmod 6$$
    This maps positions exactly to the GTO index expected by the Pluribus models:
    *   `0`: Button (BU)
    *   `1`: Small Blind (SB)
    *   `2`: Big Blind (BB)
    *   `3`: Under the Gun (UTG)
    *   `4`: Middle Position (MP)
    *   `5`: Cut Off (CO)

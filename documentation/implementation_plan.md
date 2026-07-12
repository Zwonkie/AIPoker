# Implementation Plan: Dashboard Cleanup

To clean up the GUI dashboard and provide a more intuitive display of the game state and ML reasoning, we will restructure `PHPHelp.py`.

## Proposed UI Changes

### 1. Sidebar Cleanup
We will permanently remove the deprecated controls from the left sidebar:
- `Opponents Range (1-5)` slider and labels.
- `Active Modules` checkboxes (Preflop Engine, Math Engine, Bluffing Engine, Sizing Shortcuts).

### 2. Restructuring the Visual Board
We will remove the isolated `CARDS DETECTED` panel from the left column of the telemetry display. 
Instead, we will embed the cards directly into the 6-seat table grid to make it look like a physical poker table:
- **Community Cards** will be added inside the center yellow `POT` block.
- **Hero Hole Cards** will be added inside the bottom center `Hero` seat block.
- The 6-seat grid will stretch to the left to occupy the space vacated by the `CARDS DETECTED` panel, giving the grid more breathing room.

### 3. Adding EV Breakdown
In the rightmost panel (under `RECOMMENDED ACTION` and the explanation), we will add a new **EV BREAKDOWN** section showing the exact expected values calculated by the neural network for each branch:
- EV(Fold) = X.XX
- EV(Call) = Y.YY
- EV(Raise) = Z.ZZ

To pass this data to the UI without breaking other decision models, we will update `MoE_PyTorch_Engine.predict_action` to return a 4-element tuple `(action, reason, bet_size, ev_dict)`, which `PHPHelp.py` will catch and display.

## Open Questions

> [!NOTE]  
> Are there any specific colors you'd like for the EV breakdown text? (e.g., green for positive EV, red for negative EV). If not, I will default to a standard gray/white to keep it clean.

## Verification Plan
1. Apply the UI structural removals and additions to `PHPHelp.py`.
2. Update `moe_pytorch_engine.py` to forward EV values.
3. Restart `PHPHelp.py` and ensure the GUI loads cleanly with the new layout and no crashes.

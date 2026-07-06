import os
import sys
import numpy as np
import xgboost as xgb

# Add workspace path to system path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Paths to model binaries
BINARY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "models", "binaries")
pro_path = os.path.join(BINARY_DIR, "xgboost_poker.json")
human_path = os.path.join(BINARY_DIR, "xgboost_human.json")

# Load models
pro_model = xgb.XGBClassifier()
pro_model.load_model(pro_path)

human_model = xgb.XGBClassifier()
human_model.load_model(human_path)

action_map = {0: 'FOLD', 1: 'CHECK', 2: 'CALL', 3: 'BET', 4: 'RAISE'}

def get_probabilities(model, features, temp=1.0):
    probs = model.predict_proba(features)[0]
    if temp != 1.0:
        probs = np.power(probs, 1.0 / temp)
    probs = probs / np.sum(probs)
    return probs

# Test scenarios (is_preflop, num_opponents, equity, pot_odds, stack_pot_ratio, br_avail, cc_avail)
scenarios = [
    # 1. Post-flop, 2 opponents, monster hand (high equity), betting/raising available
    {
        "name": "Post-flop Monster (High Equity)",
        "features": [0.0, 2.0, 0.85, 0.0, 10.0, 1.0, 1.0]
    },
    # 2. Post-flop, 2 opponents, weak draw/trash (low equity), facing no bet
    {
        "name": "Post-flop Weak Hand, No Bet faced",
        "features": [0.0, 2.0, 0.35, 0.0, 10.0, 1.0, 1.0]
    },
    # 3. Post-flop, 2 opponents, middle pair (medium equity), facing half-pot bet (pot odds 0.25)
    {
        "name": "Post-flop Medium Hand, Facing Half-Pot Bet",
        "features": [0.0, 2.0, 0.55, 0.25, 10.0, 1.0, 1.0]
    },
    # 4. Post-flop, 1 opponent (heads up), low equity bluff spot, facing check
    {
        "name": "Heads Up Bluff Opportunity (Low Equity)",
        "features": [0.0, 1.0, 0.25, 0.0, 15.0, 1.0, 1.0]
    },
    # 5. Pre-flop, 3 opponents, premium hand (AAs/KKs - high equity)
    {
        "name": "Pre-flop Premium Hand",
        "features": [1.0, 3.0, 0.80, 0.33, 25.0, 1.0, 1.0]
    },
    # 6. Pre-flop, 3 opponents, marginal hand (suited connectors - medium equity)
    {
        "name": "Pre-flop Marginal Hand",
        "features": [1.0, 3.0, 0.52, 0.33, 25.0, 1.0, 1.0]
    },
    # 6a. Pre-flop, 3 opponents, weak hand (weak king/queen - lower equity)
    {
        "name": "Pre-flop Weak Hand",
        "features": [1.0, 3.0, 0.38, 0.33, 25.0, 1.0, 1.0]
    },
    # 6b. Pre-flop, 3 opponents, trash hand (e.g. 72o - very low equity)
    {
        "name": "Pre-flop Trash Hand",
        "features": [1.0, 3.0, 0.12, 0.33, 25.0, 1.0, 1.0]
    },
    # 7. Post-flop, Short Stacked (SPR = 1.0), premium hand, facing shove
    {
        "name": "Post-flop Short-Stacked Commitment (High Equity)",
        "features": [0.0, 2.0, 0.75, 0.33, 1.0, 1.0, 1.0]
    },
    # 8. Post-flop, Deep Stacked (SPR = 50.0), weak hand facing large bet
    {
        "name": "Post-flop Deep-Stacked Facing Large Bet",
        "features": [0.0, 2.0, 0.20, 0.40, 50.0, 1.0, 1.0]
    }
]

print("Starting sensitivity analysis...\n")

md_lines = []
md_lines.append("# Poker Model Sensitivity & Distribution Analysis")
md_lines.append("\nThis report shows how the **Pro Model** (`xgboost_poker.json`) and **Human Model** (`xgboost_human.json`) respond to different game parameters (Equity, Pot Odds, Stack-to-Pot Ratio) under both raw distributions ($T=1.0$) and temperature-softened distributions ($T=2.5$).")

for s in scenarios:
    name = s["name"]
    feats = np.array([s["features"]])
    
    # Extract features for printing
    is_pre = "Yes" if s["features"][0] == 1.0 else "No"
    opps = int(s["features"][1])
    eq = s["features"][2]
    odds = s["features"][3]
    spr = s["features"][4]
    
    md_lines.append(f"\n## Scenario: {name}")
    md_lines.append(f"**Inputs:** Pre-flop: `{is_pre}` | Opponents: `{opps}` | Equity: `{eq:.1%}` | Pot Odds: `{odds:.1%}` | SPR: `{spr:.1f}`")
    
    # Get raw and scaled probabilities
    pro_raw = get_probabilities(pro_model, feats, temp=1.0)
    pro_soft = get_probabilities(pro_model, feats, temp=2.5)
    
    human_raw = get_probabilities(human_model, feats, temp=1.0)
    human_soft = get_probabilities(human_model, feats, temp=2.5)
    
    md_lines.append("\n| Model & Config | FOLD | CHECK | CALL | BET | RAISE |")
    md_lines.append("| --- | --- | --- | --- | --- | --- |")
    
    md_lines.append(f"| **Pro Model (Raw, T=1.0)** | {pro_raw[0]:.1%} | {pro_raw[1]:.1%} | {pro_raw[2]:.1%} | {pro_raw[3]:.1%} | {pro_raw[4]:.1%} |")
    md_lines.append(f"| **Pro Model (Softened, T=2.5)** | {pro_soft[0]:.1%} | {pro_soft[1]:.1%} | {pro_soft[2]:.1%} | {pro_soft[3]:.1%} | {pro_soft[4]:.1%} |")
    
    # For human model, remember that fold (0) and bet (3) are mucked/dummy so we display them but they will be 0/dummy
    md_lines.append(f"| **Human Model (Raw, T=1.0)** | {human_raw[0]:.1%} | {human_raw[1]:.1%} | {human_raw[2]:.1%} | {human_raw[3]:.1%} | {human_raw[4]:.1%} |")
    md_lines.append(f"| **Human Model (Softened, T=2.5)** | {human_soft[0]:.1%} | {human_soft[1]:.1%} | {human_soft[2]:.1%} | {human_soft[3]:.1%} | {human_soft[4]:.1%} |")

# Save report
report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sensitivity_analysis_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"Analysis complete! Saved report to: {report_path}")

import os
import torch
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.moe_pytorch_engine import MoE_PyTorch_Engine
from core.ml_bridge import MLBridge

def run_analysis():
    engine_limit = MoE_PyTorch_Engine(game_type="limit")
    engine_nlh = MoE_PyTorch_Engine(game_type="nlh")
    
    # We will test scenarios:
    # 1. Preflop, Turn, River
    # 2. Low, Med, High Equity hands
    # 3. Facing All-in
    
    scenarios = [
        {
            "name": "Preflop - High Equity (AA) - Facing All-in",
            "state": {
                "hero_cards": ["Ah", "As"],
                "community_cards": [],
                "pot_size": 200.0,
                "hero_stack": 1000.0,
                "action_history": ["r", "r"]
            },
            "call_amount": 1000.0
        },
        {
            "name": "Preflop - Low Equity (72o) - Facing All-in",
            "state": {
                "hero_cards": ["7h", "2c"],
                "community_cards": [],
                "pot_size": 200.0,
                "hero_stack": 1000.0,
                "action_history": ["r", "r"]
            },
            "call_amount": 1000.0
        },
        {
            "name": "Turn - Medium Equity (Draw) - Facing Pot Bet",
            "state": {
                "hero_cards": ["Jh", "Th"],
                "community_cards": ["2h", "5h", "Kc", "7d"],
                "pot_size": 100.0,
                "hero_stack": 1000.0,
                "action_history": ["c", "c", "b"]
            },
            "call_amount": 100.0
        },
        {
            "name": "Turn - High Equity (Set) - Facing Small Bet",
            "state": {
                "hero_cards": ["7h", "7c"],
                "community_cards": ["2h", "5h", "Kc", "7d"],
                "pot_size": 100.0,
                "hero_stack": 1000.0,
                "action_history": ["c", "c", "b"]
            },
            "call_amount": 20.0
        },
        {
            "name": "River - Low Equity (Missed Draw) - Facing All-in",
            "state": {
                "hero_cards": ["Jh", "Th"],
                "community_cards": ["2h", "5h", "Kc", "7d", "2s"],
                "pot_size": 200.0,
                "hero_stack": 1000.0,
                "action_history": ["c", "c", "c", "c", "b"]
            },
            "call_amount": 1000.0
        },
        {
            "name": "River - High Equity (Nut Flush) - Facing Small Bet",
            "state": {
                "hero_cards": ["Ah", "Jh"],
                "community_cards": ["2h", "5h", "Kc", "7h", "2s"],
                "pot_size": 200.0,
                "hero_stack": 1000.0,
                "action_history": ["c", "c", "c", "c", "b"]
            },
            "call_amount": 20.0
        },
        {
            "name": "River - Medium Equity (Top Pair) - Facing All-in",
            "state": {
                "hero_cards": ["Kh", "Qc"],
                "community_cards": ["2h", "5h", "Kc", "7d", "2s"],
                "pot_size": 300.0,
                "hero_stack": 1000.0,
                "action_history": ["c", "c", "c", "c", "b"]
            },
            "call_amount": 1000.0
        }
    ]
    
    os.makedirs("documentation", exist_ok=True)
    out_path = "documentation/nlh_sensitivity_analysis.md"
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# MoE Sensitivity Analysis\n\n")
        f.write("This report isolates the pure (100%) evaluation of each Tier's PyTorch Neural Network, bypassing the blending Gating Network. Values shown are raw EV predictions.\n\n")
        
        for sc in scenarios:
            f.write(f"## {sc['name']}\n")
            f.write(f"- **Hero Cards:** `{sc['state']['hero_cards']}`\n")
            f.write(f"- **Community Cards:** `{sc['state']['community_cards']}`\n")
            f.write(f"- **Pot Size:** `{sc['state']['pot_size']}`\n")
            f.write(f"- **Facing Bet:** `{sc['call_amount']}`\n\n")
            
            f.write("| Action | Tier 1 (Limit) EV | Tier 2 (Limit) EV | Tier 3 (Limit) EV | NLH Expert EV |\n")
            f.write("|---|---|---|---|---|\n")
            
            for action_str, char in [("FOLD", "f"), ("CALL", "c"), ("RAISE", "r")]:
                test_state = sc['state'].copy()
                test_state['action_history'] = sc['state'].get('action_history', []) + [char]
                
                h, b, c, a = MLBridge.state_to_tensors(test_state)
                with torch.no_grad():
                    ev1 = engine_limit.expert_t1(h, b, c, a).item()
                    ev2 = engine_limit.expert_t2(h, b, c, a).item()
                    ev3 = engine_limit.expert_t3(h, b, c, a).item()
                    ev_nlh = engine_nlh.expert_nlh(h, b, c, a).item()
                    
                f.write(f"| **{action_str}** | {ev1:.2f} | {ev2:.2f} | {ev3:.2f} | {ev_nlh:.2f} |\n")
            
            f.write("\n")
            
    print(f"Analysis successfully written to {out_path}")

if __name__ == '__main__':
    run_analysis()

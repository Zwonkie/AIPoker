import os
import torch
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.moe_pytorch_engine import MoE_PyTorch_Engine
from core.ml_bridge import MLBridge

def generate_report():
    engine = MoE_PyTorch_Engine(game_type="nlh")
    
    out_path = "documentation/nlh_param_analysis.md"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Single Parameter Sensitivity Analysis (NLH)\n\n")
        f.write("This report analyzes how the No-Limit Hold'em model scales EV when isolating single parameters.\n\n")
        
        # 1. Action Sequence (Proximity of Bet)
        f.write("## 1. Proximity of the Raise\n")
        f.write("Testing how the model evaluates a CALL/RAISE when a bet comes from far away vs right in front of Hero. Hero has QQ Preflop.\n\n")
        
        base_state = {
            "hero_cards": ["Qh", "Qs"],
            "community_cards": [],
            "pot_size": 50.0,
            "hero_stack": 1000.0,
            "position": 5, 
        }
        
        # 'r' is raise, 'f' is fold, 'c' is call
        histories = [
            ("Raise far away (3 folds after)", ["r", "f", "f", "f"]),
            ("Raise middle (1 fold before, 2 after)", ["f", "r", "f", "f"]),
            ("Raise right in front (3 folds before)", ["f", "f", "f", "r"]),
            ("Lots of action (raise, call, reraise)", ["r", "c", "r"])
        ]
        
        f.write("| Scenario | Sequence | FOLD EV | CALL EV | RAISE EV |\n")
        f.write("|---|---|---|---|---|\n")
        
        for name, hist in histories:
            test_state = base_state.copy()
            evs = []
            for action in ['f', 'c', 'r']:
                test_state['action_history'] = hist + [action]
                h, b, c, a = MLBridge.state_to_tensors(test_state)
                with torch.no_grad():
                    ev = engine.expert_nlh(h, b, c, a).item()
                evs.append(ev)
            f.write(f"| {name} | `{hist}` | {evs[0]:.3f} | {evs[1]:.3f} | {evs[2]:.3f} |\n")
            
        f.write("\n")
        
        # 2. Equity Scaling
        f.write("## 2. Preflop Hand Strength (Equity) Scaling\n")
        f.write("Testing various hands facing a single raise in front of them (`['f', 'r']`).\n\n")
        
        hands = [
            ("72o (Trash)", ["7h", "2c"]),
            ("T5o (Weak)", ["Th", "5c"]),
            ("JTo (Drawing)", ["Jh", "Tc"]),
            ("88 (Mid Pair)", ["8h", "8s"]),
            ("AKs (Premium)", ["As", "Ks"]),
            ("AA (Nuts)", ["Ah", "As"])
        ]
        
        f.write("| Hand | FOLD EV | CALL EV | RAISE EV |\n")
        f.write("|---|---|---|---|\n")
        
        for h_name, h_cards in hands:
            test_state = base_state.copy()
            test_state["hero_cards"] = h_cards
            evs = []
            for action in ['f', 'c', 'r']:
                test_state['action_history'] = ["f", "r", action]
                h, b, c, a = MLBridge.state_to_tensors(test_state)
                with torch.no_grad():
                    ev = engine.expert_nlh(h, b, c, a).item()
                evs.append(ev)
            f.write(f"| {h_name} (`{h_cards}`) | {evs[0]:.3f} | {evs[1]:.3f} | {evs[2]:.3f} |\n")
            
        f.write("\n")
        
        # 3. Pot Size Scaling
        f.write("## 3. Pot Size Scaling\n")
        f.write("Hero has AKs facing a raise `['r']`. Varying the pot size.\n\n")
        
        test_state = base_state.copy()
        test_state["hero_cards"] = ["As", "Ks"]
        
        pots = [10.0, 50.0, 200.0, 500.0, 1000.0]
        
        f.write("| Pot Size | FOLD EV | CALL EV | RAISE EV |\n")
        f.write("|---|---|---|---|\n")
        
        for p in pots:
            test_state["pot_size"] = p
            evs = []
            for action in ['f', 'c', 'r']:
                test_state['action_history'] = ["r", action]
                h, b, c, a = MLBridge.state_to_tensors(test_state)
                with torch.no_grad():
                    ev = engine.expert_nlh(h, b, c, a).item()
                evs.append(ev)
            f.write(f"| {p} | {evs[0]:.3f} | {evs[1]:.3f} | {evs[2]:.3f} |\n")
            
    print(f"Report written to {out_path}")

if __name__ == '__main__':
    generate_report()

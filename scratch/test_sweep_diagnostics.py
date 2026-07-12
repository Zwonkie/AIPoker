import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.models.poker_transformer import PokerEVModelV4
from core.ml_bridge import MLBridge

def main():
    w_path = r"core/weights/expert_v4_selfplay.pth"
    model = PokerEVModelV4()
    state_dict = torch.load(w_path, map_location='cpu')
    model.load_state_dict(state_dict)
    
    # AA
    h1, b1, c1, a1 = MLBridge.state_to_tensors_v4({
        'hero_cards': ['As', 'Ah'],
        'community_cards': [],
        'pot_size': 15.0,
        'call_amount': 10.0,
        'hero_stack': 1000.0,
        'hero_position': 0,
        'opponents': {
            'seat_1': {'is_active': True, 'stack': 1000.0, 'vpip_color': 'Blue', 'agg_color': 'Blue'}
        },
        'big_blind': 10.0,
        'equity': 0.85
    })
    
    # 72o
    h2, b2, c2, a2 = MLBridge.state_to_tensors_v4({
        'hero_cards': ['7h', '2d'],
        'community_cards': [],
        'pot_size': 15.0,
        'call_amount': 10.0,
        'hero_stack': 1000.0,
        'hero_position': 0,
        'opponents': {
            'seat_1': {'is_active': True, 'stack': 1000.0, 'vpip_color': 'Blue', 'agg_color': 'Blue'}
        },
        'big_blind': 10.0,
        'equity': 0.30
    })
    
    print("=== TENSOR CHECK ===")
    print(f"h1 (hole AA): {h1.tolist()}")
    print(f"h2 (hole 72o): {h2.tolist()}")
    print(f"b1 shape: {b1.shape}, values at step 19: {b1[0, -1].tolist()}")
    print(f"c1 shape: {c1.shape}, values at step 19:\n{c1[0, -1].tolist()}")
    print(f"c2 shape: {c2.shape}, values at step 19:\n{c2[0, -1].tolist()}")
    print(f"a1 shape: {a1.shape}, values: {a1.tolist()}")
    
    model.eval()
    with torch.no_grad():
        out1 = model(h1, b1, c1, a1).squeeze(0)[-1]
        out2 = model(h2, b2, c2, a2).squeeze(0)[-1]
        
    print("\nModel outputs at step 19:")
    print(f"AA:  {out1.tolist()}")
    print(f"72o: {out2.tolist()}")

if __name__ == '__main__':
    main()

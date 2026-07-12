import torch
import sys
import os
sys.path.append(r"c:\REPO\Antigravity\AIPoker")

from core.models.poker_transformer import PokerEVModelV4
from core.ml_bridge import MLBridge

def test_outputs():
    model = PokerEVModelV4()
    w_path = r"c:\REPO\Antigravity\AIPoker\core\weights\expert_v4_selfplay.pth"
    model.load_state_dict(torch.load(w_path, map_location='cpu'))
    model.eval()
    
    # 1. State 1 (AA)
    ts1 = {
        'hero_cards': ['As', 'Ah'],
        'community_cards': [],
        'pot_size': 15.0,
        'call_amount': 10.0,
        'hero_stack': 1000.0,
        'hero_position': 0,
        'opponents': {},
        'big_blind': 10.0
    }
    h1, b1, c1, a1 = MLBridge.state_to_tensors_v4(ts1)
    
    # 2. State 2 (72o)
    ts2 = {
        'hero_cards': ['7h', '2d'],
        'community_cards': [],
        'pot_size': 15.0,
        'call_amount': 10.0,
        'hero_stack': 1000.0,
        'hero_position': 0,
        'opponents': {},
        'big_blind': 10.0
    }
    h2, b2, c2, a2 = MLBridge.state_to_tensors_v4(ts2)
    
    # 3. State 3 (Postflop AA)
    ts3 = {
        'hero_cards': ['As', 'Ah'],
        'community_cards': ['Ks', 'Qh', '2c'],
        'pot_size': 100.0,
        'call_amount': 50.0,
        'hero_stack': 900.0,
        'hero_position': 0,
        'opponents': {},
        'big_blind': 10.0
    }
    h3, b3, c3, a3 = MLBridge.state_to_tensors_v4(ts3)
    
    with torch.no_grad():
        out1 = model(h1, b1, c1, a1).squeeze(0)[-1]
        out2 = model(h2, b2, c2, a2).squeeze(0)[-1]
        out3 = model(h3, b3, c3, a3).squeeze(0)[-1]
        
    print("AA Preflop:", out1.tolist())
    print("72o Preflop:", out2.tolist())
    print("AA Flop:", out3.tolist())

if __name__ == '__main__':
    test_outputs()

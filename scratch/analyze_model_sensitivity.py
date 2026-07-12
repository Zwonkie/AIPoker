import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.models.pluribus_engine import PokerEVModel

def analyze():
    device = torch.device('cpu')
    model = PokerEVModel().to(device)
    model.load_state_dict(torch.load('core/weights/expert_nlh_combined.pth', map_location='cpu'))
    model.eval()
    
    hole = torch.tensor([[48, 42]], dtype=torch.long)
    board = torch.tensor([[37, 12, 40, 52, 52]], dtype=torch.long)
    actions = torch.tensor([[0]*20], dtype=torch.long)
    
    print("Testing with Equity = 0.15:")
    context = torch.tensor([[0.0, 30.4 / 500.0, 3.2 / 500.0, 0.15]], dtype=torch.float32)
    with torch.no_grad():
        preds = model(hole, board, context, actions).squeeze(0)
    print(f"EV(Fold): {preds[0].item():.4f}")
    print(f"EV(Call): {preds[1].item():.4f}")
    print(f"EV(Raise): {preds[2].item():.4f}")

if __name__ == '__main__':
    analyze()

import torch
import torch.nn as nn
import sys
import os
sys.path.append(r"c:\REPO\Antigravity\AIPoker")

from core.models.poker_transformer import PokerEVModelV4

def test_gradients():
    model = PokerEVModelV4()
    model.train()
    
    # Create random inputs
    batch_size = 4
    seq_len = 20
    
    hole = torch.randint(0, 52, (batch_size, 2))
    board = torch.randint(0, 52, (batch_size, seq_len, 5))
    context = torch.randn(batch_size, seq_len, 31)
    actions = torch.randint(0, 10, (batch_size, seq_len))
    
    target_evs = torch.randn(batch_size, seq_len, 1)
    sa = torch.randint(0, 3, (batch_size, seq_len)) # actions taken (0, 1, 2)
    mask = torch.ones(batch_size, seq_len)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    
    preds = model(hole, board, context, actions) # [batch, seq_len, 3]
    pred_ev = preds.gather(2, sa.unsqueeze(-1))
    
    criterion = nn.MSELoss(reduction='none')
    loss = criterion(pred_ev, target_evs)
    masked_loss = loss * mask.unsqueeze(-1)
    final_loss = masked_loss.sum() / mask.sum()
    
    final_loss.backward()
    
    print("Checking gradients for parameters:")
    has_gradients = True
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  {name:50} | GRADIENT IS NONE!")
            has_gradients = False
        else:
            grad_norm = param.grad.norm().item()
            print(f"  {name:50} | Grad Norm: {grad_norm:.6f}")
            
    if has_gradients:
        print("\nAll parameters have valid gradients!")
    else:
        print("\nWarning: Some parameters are missing gradients!")

if __name__ == '__main__':
    test_gradients()

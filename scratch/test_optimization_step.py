import torch
import torch.nn as nn
import sys
import os
sys.path.append(r"c:\REPO\Antigravity\AIPoker")

from core.models.poker_transformer import PokerEVModelV4

def test_opt():
    model = PokerEVModelV4()
    model.train()
    
    batch_size = 4
    seq_len = 20
    
    hole = torch.randint(0, 52, (batch_size, 2))
    board = torch.randint(0, 52, (batch_size, seq_len, 5))
    context = torch.randn(batch_size, seq_len, 31)
    actions = torch.randint(0, 10, (batch_size, seq_len))
    
    target_evs = torch.randn(batch_size, seq_len, 1)
    sa = torch.randint(0, 3, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len)
    
    # Let's print output before step
    with torch.no_grad():
        out_before = model(hole, board, context, actions).clone()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1) # huge learning rate to see changes clearly
    criterion = nn.MSELoss(reduction='none')
    
    for _ in range(5):
        optimizer.zero_grad()
        preds = model(hole, board, context, actions)
        pred_ev = preds.gather(2, sa.unsqueeze(-1))
        loss = criterion(pred_ev, target_evs)
        masked_loss = loss * mask.unsqueeze(-1)
        final_loss = masked_loss.sum() / mask.sum()
        final_loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        out_after = model(hole, board, context, actions)
        
    diff = torch.abs(out_after - out_before).mean().item()
    print(f"Mean absolute difference in outputs after 5 steps: {diff:.6f}")

if __name__ == '__main__':
    test_opt()

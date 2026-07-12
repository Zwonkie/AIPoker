import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.self_play.train_selfplay import create_opponent_pool, simulate_worker, PokerEVModelV4
from scratch.train_static_poc import vectorize_hand_samples

def main():
    opponent_pool = create_opponent_pool()
    num_hands = 100
    equity_sims = 200
    
    print("Simulating 100 hands...")
    sim = simulate_worker(opponent_pool, 10.0, 10, 400, equity_sims, num_hands)
    
    X_hole, X_board, X_ctx, X_act, X_sa, Y_ev, X_mask = [], [], [], [], [], [], []
    for rec in sim:
        samples = vectorize_hand_samples(rec)
        for h, b, c, a, sa, ev, mask in samples:
            X_hole.append(h)
            X_board.append(b)
            X_ctx.append(c)
            X_act.append(a)
            X_sa.append(sa)
            Y_ev.append(ev)
            X_mask.append(mask)
            
    X_hole = torch.tensor(X_hole, dtype=torch.long)
    X_board = torch.tensor(X_board, dtype=torch.long)
    X_ctx = torch.tensor(X_ctx, dtype=torch.float32)
    X_act = torch.tensor(X_act, dtype=torch.long)
    X_sa = torch.tensor(X_sa, dtype=torch.long)
    Y_ev = torch.tensor(Y_ev, dtype=torch.float32)
    X_mask = torch.tensor(X_mask, dtype=torch.float32)
    
    print(f"Dataset size: {X_hole.shape[0]} samples.")
    
    model = PokerEVModelV4()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction='none')
    
    print("Training for 1000 epochs to check overfitting...")
    model.train()
    for epoch in range(1, 1001):
        optimizer.zero_grad()
        preds = model(X_hole, X_board, X_ctx, X_act)
        pred_ev = preds.gather(2, X_sa.unsqueeze(-1))
        loss = criterion(pred_ev, Y_ev)
        masked_loss = loss * X_mask.unsqueeze(-1)
        final_loss = masked_loss.sum() / X_mask.sum().clamp(min=1.0)
        final_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {final_loss.item():.6f}")
            
    # Now evaluate on a high-equity hand vs a low-equity hand in the training set
    model.eval()
    with torch.no_grad():
        preds = model(X_hole, X_board, X_ctx, X_act)
        
    print("\nChecking first 10 samples in dataset:")
    for i in range(min(10, X_hole.shape[0])):
        equity = X_ctx[i, -1, 3].item()
        action = X_sa[i].item()
        target = Y_ev[i, -1].item()
        pred = preds[i, -1, action].item()
        print(f"Sample {i} | Equity: {equity:.3f} | Action: {action} | Target EV: {target:.2f} | Predicted EV: {pred:.2f}")

if __name__ == '__main__':
    main()

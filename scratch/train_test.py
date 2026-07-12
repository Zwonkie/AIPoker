import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.models.pluribus_engine import PokerEVModel, PluribusEngine
from tools.run_standard_sensitivity import run_analysis

def train_test():
    dataset_name = 'nlh_combined'
    epochs = 40
    batch_size = 256
    lr = 1e-3
    
    print(f"Loading data for {dataset_name}...")
    dataset_dict = torch.load(f"tools/data/vectorized/{dataset_name}_tensors.pt")
    
    hole = dataset_dict['hole']
    board = dataset_dict['board']
    context = dataset_dict['context']
    actions = dataset_dict['actions']
    stage_action = dataset_dict['stage_action']
    ev = dataset_dict['ev']
    
    dataset = TensorDataset(hole, board, context, actions, stage_action, ev)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PokerEVModel().to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for b_hole, b_board, b_ctx, b_act, b_stage_act, b_ev in train_loader:
            b_hole, b_board, b_ctx, b_act, b_stage_act, b_ev = (
                b_hole.to(device), b_board.to(device), b_ctx.to(device), 
                b_act.to(device), b_stage_act.to(device), b_ev.to(device)
            )
            
            optimizer.zero_grad()
            preds = model(b_hole, b_board, b_ctx, b_act)
            pred_ev = preds.gather(1, b_stage_act.unsqueeze(1))
            loss = criterion(pred_ev, b_ev)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_hole.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_hole, b_board, b_ctx, b_act, b_stage_act, b_ev in val_loader:
                b_hole, b_board, b_ctx, b_act, b_stage_act, b_ev = (
                    b_hole.to(device), b_board.to(device), b_ctx.to(device), 
                    b_act.to(device), b_stage_act.to(device), b_ev.to(device)
                )
                preds = model(b_hole, b_board, b_ctx, b_act)
                pred_ev = preds.gather(1, b_stage_act.unsqueeze(1))
                loss = criterion(pred_ev, b_ev)
                val_loss += loss.item() * b_hole.size(0)
        val_loss /= len(val_loader.dataset)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
    print("Training complete! Saving weights...")
    os.makedirs('core/weights', exist_ok=True)
    save_path = 'core/weights/expert_nlh_combined.pth'
    torch.save(model.state_dict(), save_path)
    
    print("\nRunning Sensitivity Analysis on new weights:")
    run_analysis()

if __name__ == '__main__':
    train_test()

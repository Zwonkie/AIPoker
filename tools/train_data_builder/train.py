import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.models.pluribus_engine import PokerEVModel, PokerEVModelV2, PokerPolicyModel

def train(dataset_name='nlh_combined', epochs=5, batch_size=256, lr=1e-3, model_type='policy'):
    print(f"Loading data for {dataset_name}...")
    try:
        dataset_dict = torch.load(f"tools/data/vectorized/{dataset_name}_tensors.pt")
    except FileNotFoundError:
        print(f"Dataset for {dataset_name} not found.")
        return
        
    hole = dataset_dict['hole']
    board = dataset_dict['board']
    context = dataset_dict['context']
    actions = dataset_dict['actions']
    stage_action = dataset_dict['stage_action']
    ev = dataset_dict['ev']
    
    print(f"Data context tensor shape: {context.shape}")
    
    # Split 80/20 train/val
    dataset = TensorDataset(hole, board, context, actions, stage_action, ev)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == 'policy':
        model = PokerPolicyModel().to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        model = PokerEVModelV2().to(device)
        criterion = nn.MSELoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting {model_type.upper()} model training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for b_hole, b_board, b_ctx, b_act, b_stage_act, b_ev in train_loader:
            b_hole, b_board, b_ctx, b_act, b_stage_act, b_ev = (
                b_hole.to(device), b_board.to(device), b_ctx.to(device), 
                b_act.to(device), b_stage_act.to(device), b_ev.to(device)
            )
            
            optimizer.zero_grad()
            preds = model(b_hole, b_board, b_ctx, b_act) # shape: (batch, 3)
            
            if model_type == 'policy':
                loss = criterion(preds, b_stage_act)
            else:
                pred_ev = preds.gather(1, b_stage_act.unsqueeze(1)) # shape: (batch, 1)
                loss = criterion(pred_ev, b_ev)
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * b_hole.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_hole, b_board, b_ctx, b_act, b_stage_act, b_ev in val_loader:
                b_hole, b_board, b_ctx, b_act, b_stage_act, b_ev = (
                    b_hole.to(device), b_board.to(device), b_ctx.to(device), 
                    b_act.to(device), b_stage_act.to(device), b_ev.to(device)
                )
                preds = model(b_hole, b_board, b_ctx, b_act)
                
                if model_type == 'policy':
                    loss = criterion(preds, b_stage_act)
                else:
                    pred_ev = preds.gather(1, b_stage_act.unsqueeze(1))
                    loss = criterion(pred_ev, b_ev)
                    
                val_loss += loss.item() * b_hole.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    print("Training complete!")
    os.makedirs('core/weights', exist_ok=True)
    suffix = 'policy' if model_type == 'policy' else 'ev_v2'
    save_path = f'core/weights/expert_{suffix}.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Saved model weights to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nlh_combined')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--model_type', type=str, default='policy', choices=['policy', 'ev'])
    args = parser.parse_args()
    train(args.dataset, epochs=args.epochs, model_type=args.model_type)

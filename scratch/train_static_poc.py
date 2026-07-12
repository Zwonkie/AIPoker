import os
import sys
import time
from multiprocessing import Pool

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.self_play.train_selfplay import (
    create_opponent_pool, simulate_worker, PokerEVModelV4
)

def card_to_int(card_str):
    if not card_str or len(card_str) != 2:
        return 52
    rank, suit = card_str[0], card_str[1]
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    try:
        r = ranks.index(rank.upper())
        s = suits.index(suit.lower())
        return s * 13 + r
    except ValueError:
        return 52

def map_vpip_to_midpoint(v):
    if v < 0.18: return 0.10     # Blue
    elif v < 0.26: return 0.22   # Green
    elif v < 0.35: return 0.30   # Yellow
    else: return 0.45            # Red

def map_agg_to_midpoint(a):
    if a < 0.36: return 0.18     # Blue
    elif a < 0.56: return 0.46   # Green
    elif a < 0.71: return 0.63   # Yellow
    else: return 0.85            # Red

def vectorize_hand_samples(record, max_seq_len=20):
    """Convert a HandRecordV4 into a list of DT-ready sequence tensors (one per decision point).
    The active decision point is always placed at the final step (index 19) for alignment.
    """
    dps = record.get_training_samples()
    samples = []
    
    hero_cards = record.hero_cards
    hole_ints = [card_to_int(c) for c in hero_cards]
    while len(hole_ints) < 2:
        hole_ints.append(52)
        
    act_map = {0: 7, 1: 3, 2: 6}
    
    for dp in dps:
        # Initialize padded sequence
        board_seq = [[52]*5 for _ in range(max_seq_len)]
        context_seq = [[0.0]*31 for _ in range(max_seq_len)]
        action_seq = [0] * max_seq_len
        action_taken_seq = [0] * max_seq_len
        ev_seq = [[0.0] for _ in range(max_seq_len)]
        loss_mask = [0.0] * max_seq_len
        
        # Current board cards go to the final step (index 19)
        b_ints = [card_to_int(c) for c in dp['board']]
        while len(b_ints) < 5:
            b_ints.append(52)
        board_seq[-1] = b_ints
        
        # Context Vector
        bb = dp['big_blind']
        pot_odds = dp['call_amount'] / (dp['pot_size'] + dp['call_amount']) if (dp['pot_size'] + dp['call_amount']) > 0 else 0.0
        
        ctx = [
            dp['hero_position'] / 5.0,
            (dp['hero_stack'] / bb) / 400.0,
            (dp['pot_size'] / bb) / 1000.0,
            dp['equity'],
            pot_odds,
            sum(dp['active_opponents_mask']) / 10.0,
            dp['street'] / 3.0,
            0.3, # fallback default VPIP
            0.4, # fallback default AGG
            dp['pot_size'] / bb,
            dp['call_amount'] / bb
        ]
        
        for idx in range(5):
            seat_key = f"seat_{idx+1}"
            prof = dp['opponents_profiles'].get(seat_key, {'vpip': 0.3, 'agg': 0.4})
            
            # Map simulation floats to color midpoints to match OCR inference feature space
            vpip_val = map_vpip_to_midpoint(prof.get('vpip', 0.3))
            agg_val = map_agg_to_midpoint(prof.get('agg', 0.4))
            
            ctx.append(float(dp['active_opponents_mask'][idx]))
            ctx.append((dp['opponents_stacks'][idx] / bb) / 400.0)
            ctx.append(vpip_val)
            ctx.append(agg_val)
            
        context_seq[-1] = ctx
        action_seq[-1] = act_map.get(dp['action'], 0)
        action_taken_seq[-1] = dp['action']
        ev_seq[-1] = [dp['target_ev'] / bb]
        loss_mask[-1] = 1.0
        
        samples.append((hole_ints, board_seq, context_seq, action_seq, action_taken_seq, ev_seq, loss_mask))
        
    return samples

def main():
    print("=" * 60)
    print("  STATIC DATASET TRAINING FOR PLURIBUS V4 (ROBUST HUBLOSS + GRADCLIP)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device.type.upper()}")
    
    opponent_pool = create_opponent_pool()
    num_hands = 5000
    equity_sims = 200
    
    print(f"\nSimulating {num_hands:,} hands for training dataset (this will take ~2-3 mins)...")
    t0 = time.time()
    
    num_workers = min(os.cpu_count(), 8)
    hands_per_worker = num_hands // num_workers
    args = [(opponent_pool, 10.0, 10, 400, equity_sims, hands_per_worker) for _ in range(num_workers)]
    
    with Pool(num_workers) as pool:
        results = pool.starmap(simulate_worker, args)
        
    records = []
    for res in results:
        records.extend(res)
        
    sim_time = time.time() - t0
    print(f"Simulated {len(records):,} hands in {sim_time:.1f}s.")
    
    # Vectorize
    print("\nVectorizing hands...")
    X_hole, X_board, X_ctx, X_act, X_sa, Y_ev, X_mask = [], [], [], [], [], [], []
    for rec in records:
        samples = vectorize_hand_samples(rec)
        for h, b, c, a, sa, ev, mask in samples:
            X_hole.append(h)
            X_board.append(b)
            X_ctx.append(c)
            X_act.append(a)
            X_sa.append(sa)
            Y_ev.append(ev)
            X_mask.append(mask)
        
    # Convert to tensors
    X_hole = torch.tensor(X_hole, dtype=torch.long)
    X_board = torch.tensor(X_board, dtype=torch.long)
    X_ctx = torch.tensor(X_ctx, dtype=torch.float32)
    X_act = torch.tensor(X_act, dtype=torch.long)
    X_sa = torch.tensor(X_sa, dtype=torch.long)
    Y_ev = torch.tensor(Y_ev, dtype=torch.float32)
    X_mask = torch.tensor(X_mask, dtype=torch.float32)
    
    print(f"Dataset shape: {X_ctx.shape}")
    
    dataset = TensorDataset(X_hole, X_board, X_ctx, X_act, X_sa, Y_ev, X_mask)
    val_size = int(len(dataset) * 0.15)
    train_size = len(dataset) - val_size
    
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    
    model = PokerEVModelV4().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4) # Smaller stable learning rate
    criterion = nn.HuberLoss(reduction='none', delta=2.0)     # Robust Huber Loss
    
    epochs = 150
    print(f"\nTraining for {epochs} epochs on {device.type.upper()}...")
    t_train_start = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        total_items = 0
        for b_h, b_b, b_c, b_a, b_sa, b_ev, b_m in train_loader:
            b_h = b_h.to(device)
            b_b = b_b.to(device)
            b_c = b_c.to(device)
            b_a = b_a.to(device)
            b_sa = b_sa.to(device)
            b_ev = b_ev.to(device)
            b_m = b_m.to(device)
            
            optimizer.zero_grad()
            preds = model(b_h, b_b, b_c, b_a)
            pred_ev = preds.gather(2, b_sa.unsqueeze(-1))
            loss = criterion(pred_ev, b_ev)
            masked_loss = loss * b_m.unsqueeze(-1)
            final_loss = masked_loss.sum() / b_m.sum().clamp(min=1.0)
            final_loss.backward()
            
            # Gradient clipping to prevent gradient explosion/saturation
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += final_loss.item() * b_h.size(0)
            total_items += b_h.size(0)
            
        train_loss = epoch_loss / max(1, total_items)
        
        # Validation
        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            v_loss = 0.0
            total_val_items = 0
            with torch.no_grad():
                for b_h, b_b, b_c, b_a, b_sa, b_ev, b_m in val_loader:
                    b_h = b_h.to(device)
                    b_b = b_b.to(device)
                    b_c = b_c.to(device)
                    b_a = b_a.to(device)
                    b_sa = b_sa.to(device)
                    b_ev = b_ev.to(device)
                    b_m = b_m.to(device)
                    
                    preds = model(b_h, b_b, b_c, b_a)
                    pred_ev = preds.gather(2, b_sa.unsqueeze(-1))
                    loss = criterion(pred_ev, b_ev)
                    masked_loss = loss * b_m.unsqueeze(-1)
                    val_batch_loss = masked_loss.sum() / b_m.sum().clamp(min=1.0)
                    v_loss += val_batch_loss.item() * b_h.size(0)
                    total_val_items += b_h.size(0)
            val_loss = v_loss / max(1, total_val_items)
            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
    train_time = time.time() - t_train_start
    print(f"\nTraining completed in {train_time:.1f}s.")
    
    # Save
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'core', 'weights')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'expert_v4_selfplay.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {os.path.abspath(save_path)}")

if __name__ == '__main__':
    main()

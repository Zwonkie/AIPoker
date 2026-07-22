"""
Self-Play Reinforcement Learning Training Script with Live Dashboard for V8.
Implements Diversity-Based League Training, Stack Curriculum Learning,
Heuristic Bootstrap Decay, and Corrected Preflop EV Target Calculations.
"""
import os
import sys
import time
import csv
import argparse
import random
from multiprocessing import Pool

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from core.models.poker_transformer import PokerEVModelV4
from tools.self_play.v9.six_max_simulator import SixMaxSimulator

# Must match ml_bridge.py vocabulary
VOCAB = {'<PAD>': 0, 'B': 1, 'b': 2, 'c': 3, 'k': 4, 'K': 5, 'r': 6, 'f': 7, 'A': 8, 'Q': 9}

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
    """Convert a HandRecordV4 into sequence tensors for V8."""
    dps = record.get_training_samples()
    samples = []
    
    hero_cards = record.hero_cards
    hole_ints = [card_to_int(c) for c in hero_cards]
    while len(hole_ints) < 2:
        hole_ints.append(52)
        
    act_map = {0: 7, 1: 3, 2: 6}
    
    for dp in dps:
        board_seq = [[52]*5 for _ in range(max_seq_len)]
        context_seq = [[0.0]*31 for _ in range(max_seq_len)]
        action_seq = [0] * max_seq_len
        action_taken_seq = [0] * max_seq_len
        ev_seq = [[0.0, 0.0, 0.0] for _ in range(max_seq_len)]
        loss_mask = [0.0] * max_seq_len
        
        b_ints = [card_to_int(c) for c in dp['board']]
        while len(b_ints) < 5:
            b_ints.append(52)
        board_seq[-1] = b_ints
        
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
            0.3,
            0.4,
            dp['pot_size'] / bb,
            dp['call_amount'] / bb
        ]
        
        for idx in range(5):
            seat_key = f"seat_{idx+1}"
            prof = dp['opponents_profiles'].get(seat_key, {'vpip': 0.3, 'agg': 0.4})
            
            vpip_val = map_vpip_to_midpoint(prof.get('vpip', 0.3))
            agg_val = map_agg_to_midpoint(prof.get('agg', 0.4))
            
            ctx.append(float(dp['active_opponents_mask'][idx]))
            ctx.append((dp['opponents_stacks'][idx] / bb) / 400.0)
            ctx.append(vpip_val)
            ctx.append(agg_val)
            
        context_seq[-1] = ctx
        action_seq[-1] = act_map.get(dp['action'], 0)
        action_taken_seq[-1] = dp['action']
        
        ev_seq[-1] = [
            dp['target_evs'][0] / bb, 
            dp['target_evs'][1] / bb, 
            dp['target_evs'][2] / bb
        ]
        loss_mask[-1] = 1.0
        
        samples.append((hole_ints, board_seq, context_seq, action_seq, action_taken_seq, ev_seq, loss_mask))
        
    return samples

def simulate_worker(current_hand, bb_size, equity_sims, num_hands, hero_personality,
                    active_model_path=None, maniac_model_path=None,
                    nit_model_path=None, sticky_model_path=None, past_model_path=None,
                    bootstrap_alpha=0.0):
    """Headless simulation worker process for V8."""
    sim = SixMaxSimulator(
        bb_size=bb_size, equity_sims=equity_sims, hero_personality=hero_personality,
        bootstrap_alpha=bootstrap_alpha
    )
    
    # Load model weights on CPU inside worker to avoid serialisation issues
    if active_model_path and os.path.exists(active_model_path):
        try:
            model = PokerEVModelV4()
            model.load_state_dict(torch.load(active_model_path, map_location='cpu'))
            model.eval()
            sim.hero_model = model
        except Exception:
            pass
            
    if maniac_model_path and os.path.exists(maniac_model_path):
        try:
            model = PokerEVModelV4()
            model.load_state_dict(torch.load(maniac_model_path, map_location='cpu'))
            model.eval()
            sim.maniac_model = model
        except Exception:
            pass
            
    if nit_model_path and os.path.exists(nit_model_path):
        try:
            model = PokerEVModelV4()
            model.load_state_dict(torch.load(nit_model_path, map_location='cpu'))
            model.eval()
            sim.nit_model = model
        except Exception:
            pass
            
    if sticky_model_path and os.path.exists(sticky_model_path):
        try:
            model = PokerEVModelV4()
            model.load_state_dict(torch.load(sticky_model_path, map_location='cpu'))
            model.eval()
            sim.sticky_model = model
        except Exception:
            pass
            
    if past_model_path and os.path.exists(past_model_path):
        try:
            model = PokerEVModelV4()
            model.load_state_dict(torch.load(past_model_path, map_location='cpu'))
            model.eval()
            sim.past_model = model
        except Exception:
            pass
            
    records = []
    for i in range(num_hands):
        rec = sim.simulate_hand(current_hand=current_hand + i)
        if rec and rec.decision_points:
            records.append(rec)
            
    return records, sim.seat_histories, sim.global_metrics

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def print_dashboard(hands_done, total_hands, elapsed, hands_per_sec, 
                    train_loss, val_loss, seat_profits, seat_vpips, seat_aggs, epoch, personality,
                    bootstrap_alpha, training_samples, total_hands_simulated, seat_extra_stats, global_metrics):
    pct = (hands_done / total_hands) * 100 if total_hands > 0 else 0
    eta = (total_hands - hands_done) / max(1, hands_per_sec)
    
    # Calculate Phase
    if hands_done < 10000:
        phase = "Phase 1: 100BB Static"
    elif hands_done < 30000:
        phase = "Phase 2: Moderate Stacks"
    else:
        phase = "Phase 3: Extreme Stacks"
        
    seat_labels = ["Hero (Main)", "Opp 1 (Maniac)", "Opp 2 (Nit)", "Opp 3 (Sticky)", "Opp 4 (Past Self)", "Opp 5 (TAG Bot)"]
    seat_lines = []
    for s in range(6):
        profit = seat_profits[s]
        bb100 = (profit / max(1.0, total_hands_simulated)) * 10.0
        vpip = seat_vpips[s] * 100.0
        agg = seat_aggs[s] * 100.0
        extra = seat_extra_stats.get(s, {'raises': 0, 'folds': 0, 'all_ins': 0})
        r, f, a = extra['raises'], extra['folds'], extra['all_ins']
        seat_lines.append(f"|  - Seat {s} {seat_labels[s]:<16}: {bb100:>+6.1f} BB/100 (VPIP:{vpip:>4.1f}% AGG:{agg:>4.1f}%) [R:{r:<5} F:{f:<5} AI:{a:<4}]|")
        
    avg_players = global_metrics.get('flop_players', 0) / max(1, global_metrics.get('flop_count', 0))
        
    lines = [
        "+========================================================================================+",
        "|  SELF-PLAY V8 MULTI-PERSONALITY LEAGUE SYSTEM (RTX 4080 GPU)                           |",
        "+========================================================================================+",
        f"|  Active Personality: {personality.upper():<12}                                                       |",
        f"|  Hands Simulated:    {hands_done:>10,} / {total_hands:>10,}   ({pct:>5.1f}%)                            |",
        f"|  Training Epoch:     {epoch:>4}                                                              |",
        f"|  Bootstrap Alpha:    {bootstrap_alpha:>6.2f}                                                            |",
        f"|  Curriculum Stage:   {phase:<24}                                      |",
        f"|  Elapsed Time:       {format_time(elapsed):<12}                                                     |",
        f"|  Sim Speed:          {hands_per_sec:>6.0f} hands/sec                                                 |",
        f"|  ETA:                {format_time(eta):<12}                                                     |",
        f"|  Training Samples:   {training_samples:>10,}                                                        |",
        "+----------------------------------------------------------------------------------------+",
        "|  CUMULATIVE PERFORMANCE BY SEAT:                                                       |",
    ]
    for s_line in seat_lines:
        lines.append(s_line)
    lines.extend([
        "+----------------------------------------------------------------------------------------+",
        f"|  Global Post-Flop Avg Active Players: {avg_players:<4.2f}                                             |",
        "+----------------------------------------------------------------------------------------+",
        f"|  Train Loss (Hub):   {train_loss:>8.4f}  |  Val Loss: {val_loss:>8.4f}                                |",
        "+========================================================================================+"
    ])
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n".join(lines))
    sys.stdout.flush()

def run_intermediate_sensitivity_check(model, step_label, device):
    """Validation range check inside training."""
    model.eval()
    print(f"\n==================================================")
    preflop_hands = [
        ("7d 2s (Garbage)", ["7d", "2s"], 0.30),
        ("Jh Ts (Medium)", ["Jh", "Ts"], 0.46),
        ("Ad Qo (Strong)", ["Ad", "Qo"], 0.60),
        ("Qd Qs (Premium)", ["Qd", "Qs"], 0.78),
        ("Ah As (Nuts)", ["Ah", "As"], 0.85)
    ]
    
    from core.board_state import BoardState, SeatState, HUDStats
    from core.bridge.contract_v8_v9 import ContractV8V9
    bridge = ContractV8V9()
    
    print(f"--- INTERMEDIATE SENSITIVITY CHECK AT {step_label} ---")
    print(f"| Hand | Equity | Fold EV | Call EV | Raise EV | Action |")
    print(f"| :--- | :---: | :---: | :---: | :---: | :---: |")
    
    for label, cards, equity in preflop_hands:
        state = BoardState(
            community_cards=[],
            hero_cards=cards,
            pot_size=30.0,
            hero_stack=1000.0,
            big_blind=10.0,
            call_amount=20.0,
            equity=equity,
            hero_position=0,
            street="Preflop"
        )
        state.seats["seat_1"] = SeatState(
            name="Opponent 1",
            stack=1000.0,
            is_active=True,
            hud=HUDStats(vpip_color="Green", agg_color="Green")
        )
        
        h_t, b_t, c_t, a_t = bridge.to_tensors(state, action_history_raw=["r"])
        
        with torch.no_grad():
            preds = model(h_t.to(device), b_t.to(device), c_t.to(device), a_t.to(device))
            q_vals = preds.squeeze(0)[-1]
            
        f_ev, c_ev, r_ev = q_vals[0].item(), q_vals[1].item(), q_vals[2].item()
        best_act = "RAISE" if r_ev > c_ev and r_ev > f_ev else "CALL" if c_ev > f_ev else "FOLD"
        print(f"| {label:<20} | {equity:.2f} | {f_ev:>7.2f} | {c_ev:>7.2f} | {r_ev:>8.2f} | **{best_act}** |")
    print("==================================================\n")
    sys.stdout.flush()

def run_training(personality, num_hands=100000, batch_size=256, epochs_per_batch=3,
                 sim_batch_size=2000, lr=1e-3, equity_sims=200,
                 save_name=None, resume_path=None, initial_hands_done=0):
    if save_name is None:
        save_name = f"expert_v8_{personality}.pth"
        
    print("=" * 60)
    print("  SELF-PLAY RL TRAINING SYSTEM (PLURIBUS V8)")
    print("=" * 60)
    print(f"  Target Hands:    {num_hands:,}")
    print(f"  Personality:     {personality.upper()}")
    print(f"  Sim Batch Size:  {sim_batch_size:,}")
    print(f"  Epochs/Batch:    {epochs_per_batch}")
    print(f"  Learning Rate:   {lr}")
    print(f"  Save File Name:  {save_name}")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = PokerEVModelV4().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss(reduction='none', delta=2.0)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Checkpoint directories
    weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'core', 'weights'))
    os.makedirs(weights_dir, exist_ok=True)
    
    # 1. Initialize weights from V7 baseline model to avoid starting from scratch
    v7_baseline_path = os.path.join(weights_dir, 'expert_v7_selfplay.pth')
    if resume_path and os.path.exists(resume_path):
        checkpoint_path = resume_path
    else:
        checkpoint_path = v7_baseline_path
        
    if os.path.exists(checkpoint_path):
        print(f"Initializing V8 weights from checkpoint: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"WARNING: Could not load V7 starting weights ({e}). Initializing random weights.")
    else:
        print("WARNING: V7 baseline weights not found. Initializing random weights.")
        
    # 2. Paths to league personality models for opponents
    maniac_path = os.path.join(weights_dir, 'expert_v8_maniac.pth')
    nit_path = os.path.join(weights_dir, 'expert_v8_nit.pth')
    sticky_path = os.path.join(weights_dir, 'expert_v8_sticky.pth')
    past_path = os.path.join(weights_dir, 'v8_past_checkpoint.pth')
    
    # CSV logger
    log_dir = os.path.dirname(__file__)
    log_path = os.path.join(log_dir, f'training_log_{personality}.csv')
    mode = 'a' if initial_hands_done > 0 else 'w'
    log_file = open(log_path, mode, newline='')
    csv_writer = csv.writer(log_file)
    if initial_hands_done == 0:
        csv_writer.writerow([
            'timestamp', 'hands_done', 'training_samples', 'train_loss', 'val_loss',
            'hero_bb100', 'hands_per_sec', 'elapsed_sec'
        ])
    
    hands_done = initial_hands_done
    batch_records = []
    total_training_samples = 0
    train_loss = 0.0
    val_loss = 0.0
    seat_cumulative_profits = [0.0] * 6
    current_seat_vpips = [0.30] * 6
    current_seat_aggs = [0.40] * 6
    seat_extra_stats = {s: {'raises': 0, 'folds': 0, 'all_ins': 0} for s in range(6)}
    global_metrics = {'flop_players': 0, 'flop_count': 0}
    total_hands_simulated = 0
    t_start = time.time()
    total_epochs_completed = 0
    
    # Flag flags to execute intermediate checks
    check_10k_done = False
    check_25k_done = False
    
    print(f"\nLaunching simulation batch workers...")
    
    num_workers = min(os.cpu_count(), 8)
    pool = Pool(num_workers)
    try:
        while hands_done < num_hands:
            # Check intermediate check stops
            if hands_done >= 10000 and not check_10k_done:
                run_intermediate_sensitivity_check(model, "10k Hands", device)
                check_10k_done = True
                
            if hands_done >= 25000 and not check_25k_done:
                run_intermediate_sensitivity_check(model, "25k Hands", device)
                check_25k_done = True
                
            # Decay bootstrap alpha
            if hands_done < 10000:
                bootstrap_alpha = 1.0
            elif hands_done < 30000:
                bootstrap_alpha = 1.0 - (hands_done - 10000) / 20000.0
            else:
                bootstrap_alpha = 0.0
                
            batch_hands = min(sim_batch_size, num_hands - hands_done)
            
            # Save active model weights temporarily for workers to read
            active_model_path = os.path.join(log_dir, f'temp_active_model_{personality}.pth')
            torch.save(model.state_dict(), active_model_path)
            
            # Save past self checkpoint every 5,000 hands
            if hands_done > 0 and (hands_done // 5000) > ((hands_done - len(batch_records)) // 5000):
                torch.save(model.state_dict(), past_path)
                
            # Worker args
            hands_per_worker = max(1, batch_hands // num_workers)
            
            # Opponent model paths to load (only load if training 'main' or if pre-trained checkpoints exist)
            m_path = maniac_path if os.path.exists(maniac_path) else None
            n_path = nit_path if os.path.exists(nit_path) else None
            s_path = sticky_path if os.path.exists(sticky_path) else None
            p_path = past_path if os.path.exists(past_path) else None
            
            args = [
                (hands_done, 10.0, equity_sims, hands_per_worker, personality,
                 active_model_path, m_path, n_path, s_path, p_path, bootstrap_alpha)
                for _ in range(num_workers)
            ]
            
            results = pool.starmap(simulate_worker, args)
                
            batch_records = []
            recent_vpip = {s: [] for s in range(6)}
            recent_agg = {s: [] for s in range(6)}
            for res, worker_hist, worker_extra in results:
                batch_records.extend(res)
                global_metrics['flop_players'] += worker_extra.get('flop_players', 0)
                global_metrics['flop_count'] += worker_extra.get('flop_count', 0)
                for s in range(6):
                    seat_cumulative_profits[s] += worker_hist[s]['profit']
                    recent_vpip[s].extend(worker_hist[s]['vpip'])
                    recent_agg[s].extend(worker_hist[s]['agg'])
                    seat_extra_stats[s]['raises'] += worker_hist[s].get('raises', 0)
                    seat_extra_stats[s]['folds'] += worker_hist[s].get('folds', 0)
                    seat_extra_stats[s]['all_ins'] += worker_hist[s].get('all_ins', 0)
                    
            for s in range(6):
                v_list = recent_vpip[s]
                a_list = recent_agg[s]
                if len(v_list) > 0:
                    current_seat_vpips[s] = sum(v_list) / len(v_list)
                if len(a_list) > 0:
                    current_seat_aggs[s] = sum(a_list) / len(a_list)
                
            hands_done += len(batch_records)
            total_hands_simulated += len(batch_records)
                
            # Vectorize simulated hands
            X_hole, X_board, X_ctx, X_act, X_sa, Y_ev, X_mask = [], [], [], [], [], [], []
            for rec in batch_records:
                samples = vectorize_hand_samples(rec)
                for h, b, c, a, sa, ev, mask in samples:
                    X_hole.append(h)
                    X_board.append(b)
                    X_ctx.append(c)
                    X_act.append(a)
                    X_sa.append(sa)
                    Y_ev.append(ev)
                    X_mask.append(mask)
                    
            if not X_hole:
                continue
                
            hole_t = torch.tensor(X_hole, dtype=torch.long)
            board_t = torch.tensor(X_board, dtype=torch.long)
            ctx_t = torch.tensor(X_ctx, dtype=torch.float32)
            act_t = torch.tensor(X_act, dtype=torch.long)
            sa_t = torch.tensor(X_sa, dtype=torch.long)
            ev_t = torch.tensor(Y_ev, dtype=torch.float32)
            mask_t = torch.tensor(X_mask, dtype=torch.float32)
            
            total_training_samples += len(X_hole)
            
            # Split train/val
            dataset = TensorDataset(hole_t, board_t, ctx_t, act_t, sa_t, ev_t, mask_t)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            if train_size == 0 or val_size == 0:
                continue
                
            train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
            
            # GPU Training Epochs
            for epoch in range(epochs_per_batch):
                model.train()
                epoch_loss = 0.0
                total_items = 0
                for b_h, b_b, b_c, b_a, b_sa, b_ev, b_m in train_loader:
                    b_h = b_h.to(device, non_blocking=True)
                    b_b = b_b.to(device, non_blocking=True)
                    b_c = b_c.to(device, non_blocking=True)
                    b_a = b_a.to(device, non_blocking=True)
                    b_sa = b_sa.to(device, non_blocking=True)
                    b_ev = b_ev.to(device, non_blocking=True)
                    b_m = b_m.to(device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    
                    with torch.cuda.amp.autocast():
                        preds = model(b_h, b_b, b_c, b_a)
                        
                        # Apply personality reward/loss modifiers to training labels
                        b_ev_mod = b_ev.clone()
                        if personality == 'maniac':
                            b_ev_mod[..., 2] += 0.5  # Raise EV boost
                            b_ev_mod[..., 0] -= 0.2  # Fold EV penalty
                            b_ev_mod[..., 1] -= 0.2  # Call EV penalty
                        elif personality == 'nit':
                            # Scale negative EV targets (losses) by 1.5x to enforce risk aversion
                            b_ev_mod = torch.where(b_ev_mod < 0.0, b_ev_mod * 1.5, b_ev_mod)
                        elif personality == 'sticky':
                            b_ev_mod[..., 0] -= 1.0  # Fold EV penalty (sticky station)
                            
                        loss = criterion(preds, b_ev_mod)
                        masked_loss = loss * b_m.unsqueeze(-1)
                        final_loss = masked_loss.sum() / (b_m.sum().clamp(min=1.0) * 3.0)
                        
                    scaler.scale(final_loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    epoch_loss += final_loss.item() * len(b_h)
                    total_items += len(b_h)
                    
                total_epochs_completed += 1
                train_loss = epoch_loss / total_items if total_items > 0 else 0.0
                
            scheduler.step()
            
            # Validation Epoch
            model.eval()
            val_epoch_loss = 0.0
            val_items = 0
            with torch.no_grad():
                for b_h, b_b, b_c, b_a, b_sa, b_ev, b_m in val_loader:
                    b_h = b_h.to(device)
                    b_b = b_b.to(device)
                    b_c = b_c.to(device)
                    b_a = b_a.to(device)
                    b_ev = b_ev.to(device)
                    b_m = b_m.to(device)
                    
                    with torch.cuda.amp.autocast():
                        preds = model(b_h, b_b, b_c, b_a)
                        
                        # Apply corresponding validation label modifiers
                        b_ev_mod = b_ev.clone()
                        if personality == 'maniac':
                            b_ev_mod[..., 2] += 0.5
                            b_ev_mod[..., 0] -= 0.2
                            b_ev_mod[..., 1] -= 0.2
                        elif personality == 'nit':
                            b_ev_mod = torch.where(b_ev_mod < 0.0, b_ev_mod * 1.5, b_ev_mod)
                        elif personality == 'sticky':
                            b_ev_mod[..., 0] -= 1.0
                            
                        loss = criterion(preds, b_ev_mod)
                        masked_loss = loss * b_m.unsqueeze(-1)
                        final_loss = masked_loss.sum() / (b_m.sum().clamp(min=1.0) * 3.0)
                        
                    val_epoch_loss += final_loss.item() * len(b_h)
                    val_items += len(b_h)
                    
            val_loss = val_epoch_loss / val_items if val_items > 0 else 0.0
            
            # Logging metrics
            elapsed = time.time() - t_start
            hands_per_sec = total_hands_simulated / max(1, elapsed)
            hero_bb100 = (seat_cumulative_profits[0] / 10.0) / max(1, total_hands_simulated) * 100.0
            
            csv_writer.writerow([
                time.strftime('%Y-%m-%d %H:%M:%S'), hands_done, total_training_samples,
                f"{train_loss:.6f}", f"{val_loss:.6f}", f"{hero_bb100:.2f}",
                f"{hands_per_sec:.2f}", f"{elapsed:.2f}"
            ])
            log_file.flush()
            
            # Refresh live dashboard
            print_dashboard(
                hands_done=hands_done, total_hands=num_hands, elapsed=elapsed,
                hands_per_sec=hands_per_sec, train_loss=train_loss, val_loss=val_loss,
                seat_profits=seat_cumulative_profits, seat_vpips=current_seat_vpips,
                seat_aggs=current_seat_aggs, epoch=total_epochs_completed,
                personality=personality, bootstrap_alpha=bootstrap_alpha,
                training_samples=total_training_samples,
                total_hands_simulated=total_hands_simulated,
                seat_extra_stats=seat_extra_stats, global_metrics=global_metrics
            )
            
        # Save final personality model checkpoint
        final_save_path = os.path.join(weights_dir, save_name)
        torch.save(model.state_dict(), final_save_path)
        print(f"\nTraining completed successfully! Saved final weights to: {final_save_path}")
        
    finally:
        pool.close()
        pool.join()
        log_file.close()
        # Clean up temp active weight files
        temp_model_path = os.path.join(log_dir, f'temp_active_model_{personality}.pth')
        if os.path.exists(temp_model_path):
            try:
                os.remove(temp_model_path)
            except Exception:
                pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pluribus V8 self-play training loop.")
    parser.add_argument('--personality', type=str, default='main', choices=['main', 'maniac', 'nit', 'sticky'],
                        help="The playing personality to optimize (shapes the EV loss rewards).")
    parser.add_argument('--num_hands', type=int, default=100000, help="Total hands to simulate.")
    parser.add_argument('--sim_batch_size', type=int, default=2000, help="Size of simulation batches.")
    parser.add_argument('--resume_path', type=str, default=None, help="Resume from an existing checkpoint.")
    parser.add_argument('--save_name', type=str, default=None, help="Output weights filename.")
    parser.add_argument('--hands_done', type=int, default=0, help="Initial hand offset to resume tracking from.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for AdamW optimizer.")
    
    args = parser.parse_args()
    
    run_training(
        personality=args.personality,
        num_hands=args.num_hands,
        sim_batch_size=args.sim_batch_size,
        save_name=args.save_name,
        resume_path=args.resume_path,
        initial_hands_done=args.hands_done,
        lr=args.lr
    )

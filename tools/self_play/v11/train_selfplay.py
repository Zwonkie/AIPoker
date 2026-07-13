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
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from core.models.v11.poker_transformer_v11 import PokerEVModelV4
from tools.self_play.v11.six_max_simulator import SixMaxSimulator

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
    dps = record.decision_points
    if not dps:
        return []
        
    hole_ints = [card_to_int(c) for c in record.hero_cards]
    while len(hole_ints) < 2:
        hole_ints.append(52)
        
    act_map = {0: 7, 1: 3, 2: 6}
    
    board_seq = [[52]*5 for _ in range(max_seq_len)]
    context_seq = [[0.0]*35 for _ in range(max_seq_len)]
    action_seq = [0] * max_seq_len
    action_taken_seq = [0] * max_seq_len
    target_evs_seq = [[0.0, 0.0, 0.0] for _ in range(max_seq_len)]
    loss_mask = [0.0] * max_seq_len
    
    # Aux labels
    opp_bluff_seq = [0.0] * max_seq_len
    opp_strength_seq = [0.0] * max_seq_len
    self_equity_seq = [0.0] * max_seq_len
    
    # Truncate if hand is too long
    dps = dps[-max_seq_len:]
    
    bb = dps[0]['big_blind'] if dps else 10.0
    final_profit = record.final_hero_profit
    
    start_idx = 0
    
    for i, dp in enumerate(dps):
        idx = start_idx + i
        b_ints = [card_to_int(c) for c in dp['board']]
        while len(b_ints) < 5:
            b_ints.append(52)
        board_seq[idx] = b_ints
        
        pot_odds = dp['call_amount'] / (dp['pot_size'] + dp['call_amount']) if (dp['pot_size'] + dp['call_amount']) > 0 else 0.0
        
        active_opps_count = sum(dp['active_opponents_mask'])
        if active_opps_count > 0:
            sum_vpip = 0.0
            sum_agg = 0.0
            for j in range(5):
                if dp['active_opponents_mask'][j] == 1.0:
                    seat_key = f"seat_{j+1}"
                    prof = record.opponents_profiles.get(seat_key, {'vpip': 0.3, 'agg': 0.4})
                    sum_vpip += map_vpip_to_midpoint(prof.get('vpip', 0.3))
                    sum_agg += map_agg_to_midpoint(prof.get('agg', 0.4))
            global_vpip = sum_vpip / active_opps_count
            global_agg = sum_agg / active_opps_count
        else:
            global_vpip = 0.3
            global_agg = 0.4

        ctx = [
            dp['hero_position'] / 5.0,
            (dp['hero_stack'] / bb) / 400.0,
            (dp['pot_size'] / bb) / 1000.0,
            dp['equity'],
            pot_odds,
            active_opps_count / 10.0,
            dp['street'] / 3.0,
            global_vpip, global_agg,
            (dp['call_amount'] / bb) / 400.0
        ]
        
        for j in range(5):
            seat_key = f"seat_{j+1}"
            prof = record.opponents_profiles.get(seat_key, {'vpip': 0.3, 'agg': 0.4})
            
            opp_pos = (j + 1 + dp['hero_position']) % 6
            pos_val = float(opp_pos) / 5.0 if dp['active_opponents_mask'][j] == 1.0 else -1.0
            
            ctx.append(float(dp['active_opponents_mask'][j]))
            ctx.append(pos_val)
            ctx.append((dp['opponents_stacks'][j] / bb) / 400.0)
            ctx.append(map_vpip_to_midpoint(prof.get('vpip', 0.3)))
            ctx.append(map_agg_to_midpoint(prof.get('agg', 0.4)))
            
        context_seq[idx] = ctx
        action_seq[idx] = act_map.get(dp['action'], 0)
        action_taken_seq[idx] = dp['action']
        
        # Multi-Action Targets: [ev_fold, ev_call, ev_raise] from the simulator
        # These come back in RAW CHIPS. We must scale them to BIG BLINDS!
        t_evs = [ev / bb for ev in list(dp.get('target_evs', [0.0, 0.0, 0.0]))]
        
        # Override the true MC return for the action actually taken
        mc_return = (final_profit + dp['committed_before']) / bb
        if dp['action'] in [0, 1, 2]:
            t_evs[dp['action']] = mc_return
            
        # Clip Target EVs to avoid massive gradient updates
        t_evs = [max(-100.0, min(100.0, ev)) for ev in t_evs]
            
        target_evs_seq[idx] = t_evs
        loss_mask[idx] = 1.0
        
        opp_bluff_seq[idx] = float(dp.get('opp_bluff_prob', 0.0))
        opp_strength_seq[idx] = float(dp.get('opp_strength', 0.0))
        self_equity_seq[idx] = float(dp['equity'])
        
    return [(hole_ints, board_seq, context_seq, action_seq, action_taken_seq, target_evs_seq, loss_mask, opp_bluff_seq, opp_strength_seq, self_equity_seq)]

def simulate_worker(current_hand, bb_size, equity_sims, num_hands, hero_personality,
                    active_model_path=None, maniac_model_path=None,
                    nit_model_path=None, sticky_model_path=None, past_model_path=None,
                    bootstrap_alpha=0.0, focus_archetype=None):
    """Headless simulation worker process for V8."""
    sim = SixMaxSimulator(
        bb_size=bb_size, equity_sims=equity_sims, hero_personality=hero_personality,
        bootstrap_alpha=bootstrap_alpha
    )
    sim.focus_archetype = focus_archetype
    
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
            
    return records, sim.seat_histories, sim.global_metrics, sim.global_exploitation_net

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def print_dashboard(hands_done, total_hands, elapsed, hands_per_sec, train_loss, val_loss,
                    seat_profits, seat_vpips, seat_aggs, epoch, personality, bootstrap_alpha,
                    training_samples, total_hands_simulated, seat_extra_stats, global_metrics, 
                    telemetry=None, global_exploitation_net=None,
                    train_loss_q=0.0, train_loss_bluff=0.0, train_loss_str=0.0, train_loss_eq=0.0):
    """Prints a clear, V8 multi-personality training dashboard with telemetry."""
    pct = (hands_done / max(1, total_hands)) * 100
    eta = ((total_hands - hands_done) / max(1, hands_per_sec)) if hands_per_sec > 0 else 0  # Calculate Phase
    if hands_done < 10000:
        phase = "Phase 1: 100BB Static"
    elif hands_done < 30000:
        phase = "Phase 2: Moderate Stacks"
    elif hands_done < 50000:
        phase = "Phase 3: Extreme Stacks"
    elif hands_done < 75000:
        phase = "Phase 4: Dynamic Activity"
    else:
        phase = "Phase 5: Focus Rounds"
        
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
        "+----------------------------------------------------------------------------------------+"
    ])
    
    if global_exploitation_net is not None:
        lines.extend([
            "|  EXPLOITATION SCOREBOARD (Net BB/100 Matrix):                                          |",
            "|  [Winner \\ Loser] | Hero  | S1    | S2    | S3    | S4    | S5    |                    |",
        ])
        for i in range(6):
            row_str = f"|  {seat_labels[i]:<16} |"
            for j in range(6):
                if i == j:
                    row_str += "   -   |"
                else:
                    net = global_exploitation_net.get(i, {}).get(j, 0.0)
                    bb100_net = (net / max(1.0, total_hands_simulated)) * 10.0
                    row_str += f" {bb100_net:>+5.1f} |"
            lines.append(row_str + "                    |")
        lines.append("+----------------------------------------------------------------------------------------+")
    
    if telemetry is not None:
        stats_dict = telemetry.get_matrix_stats()
        ent = telemetry.get_average_entropy()
        lines.extend([
            f"|  Action Entropy:     {ent:>8.4f}                                                        |",
            "+----------------------------------------------------------------------------------------------------------+",
            "|  Equity Matrix     | Fold  | Call  | Raise | RR   | All-In | Avg End Street | Net Chips | Won    | Lost  |",
            "+----------------------------------------------------------------------------------------------------------+"
        ])
        labels = {
            '<20': '<20% (Pure Air)',
            '20-40': '20-40% (Draws)',
            '40-60': '40-60% (Marginal)',
            '60-80': '60-80% (Strong)',
            '>80': '>80% (Nuts)'
        }
        for b in ['<20', '20-40', '40-60', '60-80', '>80']:
            s = stats_dict[b]
            f = s['f_pct'] * 100
            c = s['c_pct'] * 100
            r = s['r_pct'] * 100
            rr = s['rr_pct'] * 100
            ai = s['ai_pct'] * 100
            avg_st = s['avg_street']
            net = s['total_chips']
            won = s.get('won_chips', 0.0)
            lost = s.get('lost_chips', 0.0)
            
            # Formatting won/lost compactly (e.g. +100.0, -50.0). Since it might be large, format to int K if needed, but float is fine for now
            lines.append(f"|  {labels[b]:<17} | {f:>4.1f}% | {c:>4.1f}% | {r:>4.1f}% | {rr:>4.1f}% | {ai:>4.1f}%  |      {avg_st:<4.1f}      | {net:>+7.1f}   | {won:>+6.0f} | {lost:>+6.0f} |")
        
        lines.append("+----------------------------------------------------------------------------------------------------------+")
        
    lines.extend([
        f"|  Train Loss:         {train_loss:>8.4f}  |  Val Loss: {val_loss:>8.4f}                                |",
        f"|  Loss Q: {train_loss_q:>6.4f} | Bluff: {train_loss_bluff:>6.4f} | Str: {train_loss_str:>6.4f} | Eq: {train_loss_eq:>6.4f}                 |",
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
    from core.bridge.v11.contract_v11 import ContractV8V9 as ContractV11
    bridge = ContractV11()
    
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
        
        h_t, b_t, c_t, a_t = bridge.to_tensors(state, hero_actions=[6])
        
        mask_array = [0.0 if i < 1 else 1.0 for i in range(bridge.max_seq_len)]
        key_padding_mask = torch.tensor([mask_array], dtype=torch.bool, device=device)
        
        with torch.no_grad():
            preds = model(h_t.to(device), b_t.to(device), c_t.to(device), a_t.to(device), key_padding_mask=key_padding_mask)
            q_vals = preds['q_vals'].squeeze(0)[0] if isinstance(preds, dict) else preds.squeeze(0)[0]
            
        f_ev, c_ev, r_ev = q_vals[0].item(), q_vals[1].item(), q_vals[2].item()
        best_act = "RAISE" if r_ev > c_ev and r_ev > f_ev else "CALL" if c_ev > f_ev else "FOLD"
        print(f"| {label:<20} | {equity:.2f} | {f_ev:>7.2f} | {c_ev:>7.2f} | {r_ev:>8.2f} | **{best_act}** |")
    print("==================================================\n")
    sys.stdout.flush()

def run_training(personality, num_hands=100000, batch_size=256, epochs_per_batch=3,
                 sim_batch_size=2000, lr=1e-3, equity_sims=200,
                 save_name=None, resume_path=None, initial_hands_done=0,
                 mid_flight_diagnostics_interval=10000):
    if save_name is None:
        save_name = f"expert_v11_{personality}.pth"
        
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
    maniac_path = os.path.join(weights_dir, 'expert_v11_maniac.pth')
    nit_path = os.path.join(weights_dir, 'expert_v11_nit.pth')
    sticky_path = os.path.join(weights_dir, 'expert_v11_sticky.pth')
    past_path = os.path.join(weights_dir, 'v11_past_checkpoint.pth')
    
    # CSV logger
    global_metrics = {'flop_players': 0, 'flop_count': 0}
    global_exploitation_net = {i: {j: 0.0 for j in range(6)} for i in range(6)}
    
    log_dir = os.path.dirname(__file__)
    log_path = os.path.join(log_dir, f'training_log_{personality}.csv')
    mode = 'a' if initial_hands_done > 0 else 'w'
    log_file = open(log_path, mode, newline='')
    csv_writer = csv.writer(log_file)
    if initial_hands_done == 0:
        csv_writer.writerow([
            'timestamp', 'hands_done', 'training_samples', 'train_loss', 'val_loss',
            'train_loss_q', 'train_loss_bluff', 'train_loss_str', 'train_loss_eq',
            'hero_bb100', 'hands_per_sec', 'elapsed_sec'
        ])
    
    hands_done = initial_hands_done
    batch_records = []
    total_training_samples = 0
    train_loss = 0.0
    val_loss = 0.0
    train_loss_q = 0.0
    train_loss_bluff = 0.0
    train_loss_str = 0.0
    train_loss_eq = 0.0
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
    
    from tools.self_play.v10.telemetry import TrainingTelemetry
    telemetry = TrainingTelemetry()
    
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
                
            # Automated Mid-Flight Diagnostics
            if hasattr(run_training, "last_diag") is False:
                run_training.last_diag = initial_hands_done
                
            if hands_done - run_training.last_diag >= mid_flight_diagnostics_interval:
                print(f"\n[DIAGNOSTICS] Mid-flight diagnostics triggered at {hands_done} hands...")
                
                # Save the checkpoint so the diagnostics engine evaluates the latest network
                checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'core', 'weights'))
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, save_name)
                torch.save(model.state_dict(), checkpoint_path)
                
                import subprocess
                diag_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scripts', 'math', 'run_model_diagnostics.py'))
                try:
                    subprocess.run([sys.executable, diag_script, "Herocules (v11 Main)"], check=False)
                except Exception as e:
                    print(f"Diagnostics failed to run: {e}")
                run_training.last_diag = hands_done
                
                
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
            
            # Opponent model paths: DISABLED for fresh V11 retraining.
            # Using fuzzy heuristic bots only (no NN opponent counterparts).
            m_path = None
            n_path = None
            s_path = None
            p_path = None
            
            # Phase 5: Focus Rounds Logic
            focus_archetype = None
            if hands_done >= 75000:
                focus_archetype = random.choice(['maniac', 'nit', 'fish'])
                
            args = [
                (hands_done, 10.0, equity_sims, hands_per_worker, personality,
                 active_model_path, m_path, n_path, s_path, p_path, bootstrap_alpha, focus_archetype)
                for _ in range(num_workers)
            ]
            
            results = pool.starmap(simulate_worker, args)
                
            batch_records = []
            recent_vpip = {s: [] for s in range(6)}
            recent_agg = {s: [] for s in range(6)}
            for res, worker_hist, worker_extra, worker_exploitation in results:
                batch_records.extend(res)
                global_metrics['flop_players'] += worker_extra.get('flop_players', 0)
                global_metrics['flop_count'] += worker_extra.get('flop_count', 0)
                for i in range(6):
                    for j in range(6):
                        global_exploitation_net[i][j] += worker_exploitation.get(i, {}).get(j, 0.0)
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
            X_hole, X_board, X_ctx, X_act, X_sa, Y_mc, X_mask = [], [], [], [], [], [], []
            Y_bluff, Y_strength, Y_equity = [], [], []
            for rec in batch_records:
                samples = vectorize_hand_samples(rec)
                if rec.decision_points:
                    # Record terminal state of the hand
                    final_dp = rec.decision_points[-1]
                    eq = final_dp['equity']
                    street = final_dp['street']
                    action = final_dp['action']
                    call_amount = final_dp['call_amount']
                    is_all_in = final_dp.get('is_all_in', False)
                    net_profit = rec.final_hero_profit
                    telemetry.record_hand_terminal_state(eq, street, action, call_amount, is_all_in, net_profit)

                for h, b, c, a, sa, mc, mask, opp_b, opp_s, self_e in samples:
                    X_hole.append(h)
                    X_board.append(b)
                    X_ctx.append(c)
                    X_act.append(a)
                    X_sa.append(sa)
                    Y_mc.append(mc)
                    X_mask.append(mask)
                    Y_bluff.append(opp_b)
                    Y_strength.append(opp_s)
                    Y_equity.append(self_e)
                    
            if not X_hole:
                continue
                
            hole_t = torch.tensor(X_hole, dtype=torch.long)
            board_t = torch.tensor(X_board, dtype=torch.long)
            ctx_t = torch.tensor(X_ctx, dtype=torch.float32)
            act_t = torch.tensor(X_act, dtype=torch.long)
            sa_t = torch.tensor(X_sa, dtype=torch.long)
            mc_t = torch.tensor(Y_mc, dtype=torch.float32)
            mask_t = torch.tensor(X_mask, dtype=torch.float32)
            bluff_t = torch.tensor(Y_bluff, dtype=torch.float32)
            str_t = torch.tensor(Y_strength, dtype=torch.float32)
            eq_t = torch.tensor(Y_equity, dtype=torch.float32)
            
            total_training_samples += len(X_hole)
            
            # Split train/val
            dataset = TensorDataset(hole_t, board_t, ctx_t, act_t, sa_t, mc_t, mask_t, bluff_t, str_t, eq_t)
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
                epoch_loss_q = 0.0
                epoch_loss_bluff = 0.0
                epoch_loss_str = 0.0
                epoch_loss_eq = 0.0
                total_items = 0
                for b_h, b_b, b_c, b_a, b_sa, b_mc, b_m, b_bluff, b_str, b_eq in train_loader:
                    b_h = b_h.to(device, non_blocking=True)
                    b_b = b_b.to(device, non_blocking=True)
                    b_c = b_c.to(device, non_blocking=True)
                    b_a = b_a.to(device, non_blocking=True)
                    b_sa = b_sa.to(device, non_blocking=True)
                    b_mc = b_mc.to(device, non_blocking=True)
                    b_m = b_m.to(device, non_blocking=True)
                    b_bluff = b_bluff.to(device, non_blocking=True)
                    b_str = b_str.to(device, non_blocking=True)
                    b_eq = b_eq.to(device, non_blocking=True)
                    
                    optimizer.zero_grad()
                    
                    with torch.cuda.amp.autocast():
                        out = model(b_h, b_b, b_c, b_a, key_padding_mask=(b_m == 0.0))
                        if isinstance(out, dict):
                            preds = out['q_vals']
                            pred_bluff = out['bluff']
                            pred_str = out['strength']
                            pred_eq = out['equity']
                        else:
                            preds = out
                        
                        with torch.no_grad():
                            probs = torch.softmax(preds, dim=-1)
                            batch_entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()
                            telemetry.record_entropy_value(batch_entropy)
                        
                        # Multi-Action Target EV Loss
                        # Only calculate Q-loss for the action actually taken to fix heuristic EV distortion
                        action_mask = torch.zeros_like(preds)
                        action_mask.scatter_(2, b_sa.unsqueeze(-1), 1.0)
                        
                        loss_q = criterion(preds, b_mc) * b_m.unsqueeze(-1) * action_mask
                        final_loss_q = loss_q.sum() / (b_m.sum().clamp(min=1.0))
                        
                        # Interpretable Auxiliary Heads Loss
                        if isinstance(out, dict):
                            loss_bluff = nn.functional.mse_loss(pred_bluff, b_bluff, reduction='none') * b_m
                            loss_str = nn.functional.mse_loss(pred_str, b_str, reduction='none') * b_m
                            loss_eq = nn.functional.mse_loss(pred_eq, b_eq, reduction='none') * b_m
                            
                            sc_bluff = loss_bluff.sum() / b_m.sum().clamp(min=1.0)
                            sc_str = loss_str.sum() / b_m.sum().clamp(min=1.0)
                            sc_eq = loss_eq.sum() / b_m.sum().clamp(min=1.0)
                            
                            final_loss_aux = sc_bluff + sc_str + sc_eq
                        else:
                            final_loss_aux = 0.0
                            sc_bluff = 0.0
                            sc_str = 0.0
                            sc_eq = 0.0
                            
                        final_loss = final_loss_q + 10.0 * final_loss_aux
                        
                    scaler.scale(final_loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    epoch_loss += float(final_loss) * len(b_h)
                    epoch_loss_q += float(final_loss_q) * len(b_h)
                    epoch_loss_bluff += float(sc_bluff) * len(b_h)
                    epoch_loss_str += float(sc_str) * len(b_h)
                    epoch_loss_eq += float(sc_eq) * len(b_h)
                    total_items += len(b_h)
                    
                total_epochs_completed += 1
                train_loss = epoch_loss / total_items if total_items > 0 else 0.0
                train_loss_q = epoch_loss_q / total_items if total_items > 0 else 0.0
                train_loss_bluff = epoch_loss_bluff / total_items if total_items > 0 else 0.0
                train_loss_str = epoch_loss_str / total_items if total_items > 0 else 0.0
                train_loss_eq = epoch_loss_eq / total_items if total_items > 0 else 0.0
                
            scheduler.step()
            
            # Validation Epoch
            model.eval()
            val_epoch_loss = 0.0
            val_items = 0
            with torch.no_grad():
                for b_h, b_b, b_c, b_a, b_sa, b_mc, b_m, b_bluff, b_str, b_eq in val_loader:
                    b_h = b_h.to(device)
                    b_b = b_b.to(device)
                    b_c = b_c.to(device)
                    b_a = b_a.to(device)
                    b_sa = b_sa.to(device)
                    b_mc = b_mc.to(device)
                    b_m = b_m.to(device)
                    b_bluff = b_bluff.to(device)
                    b_str = b_str.to(device)
                    b_eq = b_eq.to(device)
                    
                    with torch.cuda.amp.autocast():
                        out = model(b_h, b_b, b_c, b_a, key_padding_mask=(b_m == 0.0))
                        if isinstance(out, dict):
                            preds = out['q_vals']
                            pred_bluff = out['bluff']
                            pred_str = out['strength']
                            pred_eq = out['equity']
                        else:
                            preds = out
                            
                        action_mask = torch.zeros_like(preds)
                        action_mask.scatter_(2, b_sa.unsqueeze(-1), 1.0)
                        
                        loss_q = criterion(preds, b_mc) * b_m.unsqueeze(-1) * action_mask
                        final_loss_q = loss_q.sum() / (b_m.sum().clamp(min=1.0))
                        
                        if isinstance(out, dict):
                            loss_bluff = nn.functional.mse_loss(pred_bluff, b_bluff, reduction='none') * b_m
                            loss_str = nn.functional.mse_loss(pred_str, b_str, reduction='none') * b_m
                            loss_eq = nn.functional.mse_loss(pred_eq, b_eq, reduction='none') * b_m
                            final_loss_aux = (loss_bluff + loss_str + loss_eq).sum() / b_m.sum().clamp(min=1.0)
                        else:
                            final_loss_aux = 0.0
                            
                        final_loss = final_loss_q + 10.0 * final_loss_aux
                        
                    val_epoch_loss += final_loss.item() * len(b_h)
                    val_items += len(b_h)
                    
            val_loss = val_epoch_loss / val_items if val_items > 0 else 0.0
            
            # Logging metrics
            elapsed = time.time() - t_start
            hands_per_sec = total_hands_simulated / max(1, elapsed)
            hero_bb100 = (seat_cumulative_profits[0] / 10.0) / max(1, total_hands_simulated) * 100.0
            
            csv_writer.writerow([
                time.strftime('%Y-%m-%d %H:%M:%S'), hands_done, total_training_samples,
                f"{train_loss:.6f}", f"{val_loss:.6f}", f"{train_loss_q:.6f}",
                f"{train_loss_bluff:.6f}", f"{train_loss_str:.6f}", f"{train_loss_eq:.6f}",
                f"{hero_bb100:.2f}", f"{hands_per_sec:.2f}", f"{elapsed:.2f}"
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
                seat_extra_stats=seat_extra_stats, global_metrics=global_metrics,
                telemetry=telemetry, global_exploitation_net=global_exploitation_net,
                train_loss_q=train_loss_q, train_loss_bluff=train_loss_bluff,
                train_loss_str=train_loss_str, train_loss_eq=train_loss_eq
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
    parser = argparse.ArgumentParser(description="Herocules V11 self-play training loop.")
    parser.add_argument('--personality', type=str, default='main', choices=['main', 'maniac', 'nit', 'sticky'],
                        help="The playing personality to optimize (shapes the EV loss rewards).")
    parser.add_argument('--resume_path', type=str, default=None, help="Resume from an existing checkpoint.")
    parser.add_argument('--save_name', type=str, default=None, help="Output weights filename.")
    parser.add_argument('--hands_done', type=int, default=0, help="Initial hand offset to resume tracking from.")
    
    args = parser.parse_args()
    
    # Load YAML config
    config_path = os.path.join(os.path.dirname(__file__), 'training_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    t_cfg = config.get('training', {})
    
    run_training(
        personality=args.personality,
        num_hands=t_cfg.get('target_hands', 100000),
        batch_size=t_cfg.get('batch_size', 256),
        epochs_per_batch=t_cfg.get('epochs_per_batch', 3),
        sim_batch_size=t_cfg.get('sim_batch_size', 2000),
        lr=float(t_cfg.get('learning_rate', 1e-3)),
        equity_sims=t_cfg.get('equity_sims', 200),
        save_name=args.save_name,
        resume_path=args.resume_path,
        initial_hands_done=args.hands_done,
        mid_flight_diagnostics_interval=t_cfg.get('mid_flight_diagnostics_interval', 10000)
    )

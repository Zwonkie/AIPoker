import json
import os
import torch
import numpy as np
from tqdm import tqdm
import sys
import os

# Add parent directory to path to import core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.evaluator import PokerEvaluator

evaluator = PokerEvaluator()

# Vocabulary for actions
VOCAB = {'<PAD>': 0, 'B': 1, 'b': 2, 'c': 3, 'k': 4, 'K': 5, 'r': 6, 'f': 7, 'A': 8, 'Q': 9}

def card_to_int(card_str):
    if not card_str or len(card_str) != 2:
        return 52 # PAD
    rank, suit = card_str[0], card_str[1]
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    try:
        r = ranks.index(rank)
        s = suits.index(suit)
        return s * 13 + r
    except ValueError:
        return 52

def process_file(in_path, out_path, max_seq_len=20):
    print(f"Processing {in_path}...")
    X_hole_all = []
    X_board_all = []
    X_ctx_all = []
    X_act_all = []
    X_stage_act_all = []
    Y_ev_all = []
    
    with open(in_path, 'r') as f:
        for line in tqdm(f):
            data = json.loads(line)
            board = data.get('board', [])
            pots = {p['stage']: p['size'] for p in data.get('pots', [])}
            
            # Map board cards to stages
            board_p = [52]*5
            board_f = [card_to_int(c) for c in board[:3]] + [52]*2 if len(board) >= 3 else [52]*5
            board_t = [card_to_int(c) for c in board[:4]] + [52]*1 if len(board) >= 4 else board_f
            board_r = [card_to_int(c) for c in board[:5]] if len(board) == 5 else board_t
            
            board_by_stage = {'p': board_p, 'f': board_f, 't': board_t, 'r': board_r}
            pot_by_stage = {
                'p': 0, 
                'f': pots.get('f', 0), 
                't': pots.get('t', pots.get('f', 0)), 
                'r': pots.get('r', pots.get('t', pots.get('f', 0)))
            }
            
            for player_name, pdata in data['players'].items():
                pocket = pdata.get('pocket_cards', [])
                if len(pocket) != 2:
                    continue # Skip players without known hole cards
                    
                hole_ints = [card_to_int(c) for c in pocket]
                ev = pdata.get('total_win', 0) - pdata.get('total_bet', 0) # Already normalized to BB
                bankroll = pdata.get('bankroll', 0)
                position = pdata.get('position', 0)
                
                # Sequence builder
                seq = []
                for b in pdata.get('bets', []):
                    stage = b['stage']
                    
                    # Extract the stage action (first action of the stage)
                    actions_str = b.get('actions', '')
                    if not actions_str:
                        continue # Skip stages where the player did not act
                        
                    first_action = actions_str[0]
                    if first_action == 'f':
                        act_idx = 0
                    elif first_action == 'c':
                        act_idx = 1
                    elif first_action == 'r':
                        act_idx = 2
                    else:
                        continue # Skip invalid action
                        
                    # Calculate Equity for this stage (200 sims for speed/accuracy balance)
                    # Use 1 opponent as a heuristic base for the network to scale off of.
                    hole_strs = pocket
                    board_strs = board[:(0 if stage=='p' else 3 if stage=='f' else 4 if stage=='t' else 5)]
                    
                    equity_raw, _ = evaluator.calculate_equity(board_strs, hole_strs, num_opponents=1, num_simulations=150)
                    # Bucket equity to 1% increments
                    equity = round(equity_raw * 100) / 100.0
                    
                    # Calculate pot_odds based on player's first action in this stage
                    if first_action in ['f', 'c']:
                        pot_odds = 0.25  # Facing standard half-pot bet
                    elif first_action == 'r':
                        pot_odds = 0.20  # Facing standard raise
                    else:
                        pot_odds = 0.0   # Checking or open-betting
                        
                    num_opponents = max(1.0, float(data.get('num_players', 2) - 1))
                    
                    stage_map = {'p': 0.0, 'f': 1.0, 't': 2.0, 'r': 3.0}
                    street_level = stage_map.get(stage, 0.0)
                    
                    # Context is now 7 dimensions: [Position, Bankroll, PotSize, Equity, PotOdds, NumOpponents, StreetLevel] (Scaled)
                    ctx = [
                        position / 10.0,
                        bankroll / 500.0,
                        pot_by_stage[stage] / 500.0,
                        equity,
                        pot_odds,
                        num_opponents / 10.0,
                        street_level / 3.0
                    ]
                    
                    # Pre-pad sequence with 0s
                    padded_seq = seq.copy()
                    if len(padded_seq) < max_seq_len:
                        padded_seq = [0] * (max_seq_len - len(padded_seq)) + padded_seq
                    else:
                        padded_seq = padded_seq[:max_seq_len]
                        
                    X_hole_all.append(hole_ints)
                    X_board_all.append(board_by_stage[stage])
                    X_ctx_all.append(ctx)
                    X_act_all.append(padded_seq)
                    X_stage_act_all.append(act_idx)
                    Y_ev_all.append([ev])
                    
                    # Now add this stage's actions to the sequence for the NEXT stage
                    for char in b.get('actions', ''):
                        if char != '-':
                            seq.append(VOCAB.get(char, 0))

    if not X_hole_all:
        print("No valid hands found!")
        return

    # Convert to tensors
    dataset = {
        'hole': torch.tensor(X_hole_all, dtype=torch.long),
        'board': torch.tensor(X_board_all, dtype=torch.long),
        'context': torch.tensor(X_ctx_all, dtype=torch.float32),
        'actions': torch.tensor(X_act_all, dtype=torch.long),
        'stage_action': torch.tensor(X_stage_act_all, dtype=torch.long),
        'ev': torch.tensor(Y_ev_all, dtype=torch.float32)
    }
    
    torch.save(dataset, out_path)
    print(f"Saved {len(X_hole_all)} examples to {out_path}")

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Specific jsonl input file (e.g. tools/data/parsed/hands_nlh.jsonl)')
    parser.add_argument('--output', type=str, help='Specific output pt file')
    args = parser.parse_args()
    
    os.makedirs('tools/data/vectorized', exist_ok=True)
    
    if args.input and args.output:
        if os.path.exists(args.input):
            process_file(args.input, args.output)
        else:
            print(f"Input file not found: {args.input}")
    else:
        # Default behavior for IRC tiers
        tiers = ['tier1', 'tier2', 'tier3']
        for t in tiers:
            in_p = f'tools/data/parsed/hands_{t}.jsonl'
            out_p = f'tools/data/vectorized/{t}_tensors.pt'
            if os.path.exists(in_p):
                process_file(in_p, out_p)

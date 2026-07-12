import ast
import json
import os
import glob
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from core.evaluator import PokerEvaluator
evaluator = PokerEvaluator()

def process_phhs(filepath, out_file):
    with open(filepath, 'r', encoding='utf-8') as f:
        current_hand = {}
        for line in f:
            line = line.strip()
            if not line: continue
            
            if line.startswith('[') and line.endswith(']'):
                if current_hand:
                    parse_and_write_hand(current_hand, out_file)
                current_hand = {}
                continue
                
            if '=' in line:
                key, val_str = line.split('=', 1)
                key = key.strip()
                val_str = val_str.strip()
                
                if val_str == 'false': val_str = 'False'
                elif val_str == 'true': val_str = 'True'
                
                if val_str.startswith("'") or val_str.startswith("["):
                    try:
                        current_hand[key] = ast.literal_eval(val_str)
                    except Exception:
                        current_hand[key] = val_str
                else:
                    try:
                        if '.' in val_str: current_hand[key] = float(val_str)
                        elif ':' in val_str: current_hand[key] = val_str
                        else: current_hand[key] = int(val_str)
                    except ValueError:
                        current_hand[key] = val_str
                        
        if current_hand:
            parse_and_write_hand(current_hand, out_file)

def parse_and_write_hand(h, out_file):
    players_list = h.get('players', [])
    if not players_list: return
    
    actions_raw = h.get('actions', [])
    if not actions_raw: return
    
    stacks = h.get('starting_stacks', [])
    winnings = h.get('winnings', [0]*len(players_list))
    
    raw_blinds = h.get('blinds_or_straddles', [])
    bb_amount = max([float(b) for b in raw_blinds]) if raw_blinds else 1.0
    if bb_amount <= 0: bb_amount = 1.0
    
    stacks = [float(s) / bb_amount for s in stacks]
    winnings = [float(w) / bb_amount for w in winnings]
    
    # We only care about players who show cards
    known_cards = {}
    for act in actions_raw:
        if ' sm ' in act:
            parts = act.split(' sm ')
            pid_str = parts[0] # e.g. p2
            cards = parts[1].strip()
            if cards != '????':
                try:
                    idx = int(pid_str[1:]) - 1
                    # Parse 8sAh into ['8s', 'Ah']
                    idx = int(pid_str[1:]) - 1
                    known_cards[players_list[idx]] = [cards[0:2], cards[2:4]]
                except: pass
                
    if not known_cards:
        return
        
    active_players = set(players_list)
    for act in actions_raw:
        if ' sm ????' in act:
            pid_str = act.split(' sm ')[0]
            try:
                idx = int(pid_str[1:]) - 1
                active_players.discard(players_list[idx])
            except: pass
        else:
            parts = act.split(' ')
            if len(parts) >= 2 and parts[0].startswith('p') and parts[1] == 'f':
                try:
                    idx = int(parts[0][1:]) - 1
                    active_players.discard(players_list[idx])
                except: pass
        
    board = []
    pots = []
    
    players_data = {}
    for i, p in enumerate(players_list):
        if p in known_cards:
            players_data[p] = {
                "pocket_cards": known_cards[p],
                "bankroll": stacks[i] if i < len(stacks) else 0.0,
                "position": i + 1,
                "total_win": winnings[i] if i < len(winnings) else 0.0,
                "total_bet": 0.0,
                "bets": [{"stage": "p", "actions": ""}, {"stage": "f", "actions": ""}, {"stage": "t", "actions": ""}, {"stage": "r", "actions": ""}]
            }
            
    stage_bets = {p: 0.0 for p in players_list}
    total_bets = {p: 0.0 for p in players_list}
    highest_stage_bet = 0.0
    
    blinds = h.get('blinds_or_straddles', [])
    for i, b_amt in enumerate(blinds):
        if float(b_amt) > 0 and i < len(players_list):
            norm_amt = float(b_amt) / bb_amount
            stage_bets[players_list[i]] = norm_amt
            highest_stage_bet = max(highest_stage_bet, norm_amt)

    def settle_stage():
        nonlocal stage_bets, total_bets, highest_stage_bet
        b_vals = list(stage_bets.values())
        b_vals.sort(reverse=True)
        if len(b_vals) >= 2 and b_vals[0] > b_vals[1]:
            for p, amt in stage_bets.items():
                if amt == b_vals[0]:
                    stage_bets[p] = b_vals[1]
        for p, amt in stage_bets.items():
            total_bets[p] += amt
        
        current_pot = sum(total_bets.values())
        pots.append({"stage": current_stage, "size": current_pot})
            
        stage_bets = {p: 0.0 for p in players_list}
        highest_stage_bet = 0.0

    stage_idx = 0
    stages = ['p', 'f', 't', 'r']
    current_stage = 'p'
    
    for act in actions_raw:
        if act.startswith('d db'):
            settle_stage()
            cards = act.split('d db ')[1]
            if len(cards) == 6: # flop
                board.extend([cards[0:2], cards[2:4], cards[4:6]])
                stage_idx = 1
            elif len(cards) == 2:
                board.append(cards)
                stage_idx += 1
            current_stage = stages[min(stage_idx, 3)]
        else:
            # Player action
            parts = act.split(' ')
            if len(parts) >= 2 and parts[0].startswith('p'):
                try:
                    pidx = int(parts[0][1:]) - 1
                    player = players_list[pidx]
                    action_type = parts[1]
                    
                    if action_type == 'cc':
                        stage_bets[player] = highest_stage_bet
                    elif action_type == 'cbr':
                        if len(parts) >= 3:
                            amount = float(parts[2]) / bb_amount
                            stage_bets[player] = amount
                            highest_stage_bet = max(highest_stage_bet, amount)
                            
                    if player in players_data:
                        b = players_data[player]["bets"][min(stage_idx, 3)]
                        if action_type == 'f':
                            b["actions"] += "f"
                        elif action_type == 'cc':
                            b["actions"] += "c"
                        elif action_type == 'cbr':
                            b["actions"] += "r"
                except:
                    pass

    settle_stage()
    
    current_pot = sum(total_bets.values())
    winners = []
    if len(active_players) == 1:
        winners = list(active_players)
    elif len(active_players) > 1:
        best_score = float('inf')
        for p in active_players:
            if p in known_cards and len(board) >= 5:
                try:
                    score, _ = evaluator.evaluate_hand(board[:5], known_cards[p])
                    if score < best_score:
                        best_score = score
                        winners = [p]
                    elif score == best_score:
                        winners.append(p)
                except: pass
        if not winners:
            winners = list(active_players)
            
    win_amount = current_pot / len(winners) if winners else 0.0
    
    for p in players_data:
        players_data[p]["total_bet"] = total_bets[p]
        if p in winners:
            players_data[p]["total_win"] = win_amount

    output = {
        "_id": str(h.get('hand', 0)),
        "board": board,
        "dealer": 1,
        "game": "NLH",
        "hand_num": h.get('hand', 0),
        "num_players": len(players_list),
        "players": players_data,
        "pots": pots
    }
    
    out_file.write(json.dumps(output) + "\n")

if __name__ == '__main__':
    in_dirs = ["data/raw_human_hands", "data/raw_hands"]
    out_dir = "tools/data/parsed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hands_nlh_combined.jsonl")
    
    files = []
    for d in in_dirs:
        files.extend(glob.glob(os.path.join(d, "*.phhs")))
        files.extend(glob.glob(os.path.join(d, "*.phh")))
        
    print(f"Found {len(files)} PHH/PHHS files across directories.")
    
    with open(out_path, 'w', encoding='utf-8') as out_f:
        for f in tqdm(files):
            process_phhs(f, out_f)
    print(f"Done! Wrote to {out_path}")

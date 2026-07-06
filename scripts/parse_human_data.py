import os
import re
import csv
import io
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pokerkit import HandHistory
from core.evaluator import PokerEvaluator

def clean_card(c):
    s = str(c)
    match = re.search(r'\((.*?)\)', s)
    if match:
        val = match.group(1)
        if len(val) == 3 and val.startswith('10'):
            return 'T' + val[2]
        return val
    return s

def parse_phhs_file(file_path):
    hands = []
    current_hand = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('[') and line.strip().endswith(']'):
                if current_hand:
                    hands.append("\n".join(current_hand))
                    current_hand = []
            else:
                current_hand.append(line)
        if current_hand:
            hands.append("\n".join(current_hand))
    return hands

def main():
    print("[Parsing] Initializing Parser & Evaluator for Human Hands...")
    evaluator = PokerEvaluator()
    
    raw_dir = os.path.join("data", "raw_human_hands")
    output_csv = os.path.join("data", "vectorized_human_hands.csv")
    
    if not os.path.exists(raw_dir):
        print(f"[Error] Raw directory '{raw_dir}' does not exist. Run download_human_data.py first.")
        return
        
    files = [f for f in os.listdir(raw_dir) if f.endswith(".phhs")]
    print(f"[Parsing] Found {len(files)} human history session files in '{raw_dir}'.")
    
    dataset = []
    
    # Define headers
    headers = [
        "is_preflop", 
        "num_opponents", 
        "equity", 
        "pot_odds", 
        "stack_pot_ratio", 
        "bet_raise_available", 
        "check_call_available",
        "action" # target label: 0=FOLD, 1=CHECK, 2=CALL, 3=BET, 4=RAISE
    ]
    
    processed_count = 0
    actions_count = 0
    max_actions = 2500  # Cap to keep parsing fast
    
    # Target action mapping
    action_map = {
        'f': 0,    # FOLD
        'cc': 1,   # CHECK / CALL (mapped based on context below)
        'cbr': 3   # BET / RAISE (mapped based on context below)
    }
    
    for file_name in files:
        if actions_count >= max_actions:
            break
            
        file_path = os.path.join(raw_dir, file_name)
        hand_strings = parse_phhs_file(file_path)
        
        for hand_str in hand_strings:
            if actions_count >= max_actions:
                break
                
            try:
                # Pass 1: Load and run to final state to find revealed hole cards
                f_bytes_1 = io.BytesIO(hand_str.encode('utf-8'))
                hh_1 = HandHistory.load(f_bytes_1)
                
                final_state = None
                for state, _ in hh_1.state_actions:
                    final_state = state
                    
                if final_state is None:
                    continue
                    
                known_hole_cards = {}
                for i in range(final_state.player_count):
                    cards = [clean_card(c) for c in final_state.hole_cards[i]]
                    if len(cards) >= 2 and '????' not in cards and '??' not in cards:
                        known_hole_cards[i] = cards
                
                # Pass 2: Re-load and iterate directly to parse actions
                f_bytes_2 = io.BytesIO(hand_str.encode('utf-8'))
                hh_2 = HandHistory.load(f_bytes_2)
                
                for state, sa_str in hh_2.state_actions:
                    if sa_str is None or actions_count >= max_actions:
                        continue
                    # sa_str is e.g. "p3 f", "p1 cc", "p5 cbr 200"
                    match = re.match(r"^p(\d+)\s+(f|cc|cbr(?:\s+[\d\.]+)?)$", sa_str)
                    if not match:
                        continue
                        
                    actor_index = int(match.group(1)) - 1
                    action_type = match.group(2)
                    
                    # Clean the action type if it has numbers
                    if action_type.startswith('cbr'):
                        action_type = 'cbr'
                    
                    # Get hole cards for this actor
                    hero_cards = known_hole_cards.get(actor_index, [])
                    if len(hero_cards) < 2:
                        continue # Skip if this player's hole cards were never revealed in the hand
                        
                    board_cards = [clean_card(c) for street in state.board_cards for c in street]
                        
                    # State parameters
                    pot = float(state.total_pot_amount)
                    stack = float(state.stacks[actor_index])
                    call_amount = float(state.checking_or_calling_amount) if state.checking_or_calling_amount is not None else 0.0
                    
                    # Active players count
                    num_active = sum(1 for i in range(state.player_count) if state.statuses[i])
                    num_opponents = max(num_active - 1, 1)
                    
                    # Calculate win equity via Monte Carlo (200 sims for speed)
                    equity, _ = evaluator.calculate_equity(
                        board_cards, 
                        hero_cards, 
                        num_opponents=num_opponents, 
                        num_simulations=200
                    )
                    
                    # Features
                    is_preflop = 1.0 if len(board_cards) == 0 else 0.0
                    pot_odds = call_amount / (pot + call_amount) if (pot + call_amount) > 0 else 0.0
                    stack_pot_ratio = stack / pot if pot > 0 else 999.0
                    check_call_available = 1.0 if state.can_check_or_call else 0.0
                    bet_raise_available = 1.0 if state.can_complete_bet_or_raise_to else 0.0
                    
                    # Mapping logic
                    raw_mapped = action_map.get(action_type, 0)
                    
                    # Disambiguate CHECK vs CALL and BET vs RAISE
                    final_mapped = raw_mapped
                    if raw_mapped == 1: # 'cc'
                        if call_amount == 0:
                            final_mapped = 1 # CHECK
                        else:
                            final_mapped = 2 # CALL
                    elif raw_mapped == 3: # 'cbr'
                        if call_amount == 0:
                            final_mapped = 3 # BET
                        else:
                            final_mapped = 4 # RAISE
                            
                    row = [
                        is_preflop,
                        float(num_opponents),
                        equity,
                        pot_odds,
                        stack_pot_ratio,
                        bet_raise_available,
                        check_call_available,
                        final_mapped
                    ]
                    
                    dataset.append(row)
                    actions_count += 1
                    
                    if actions_count % 200 == 0:
                        print(f"[Parsing] Processed {actions_count} action states...")
                        
            except Exception as e:
                # print(f"Error parsing hand: {e}")
                pass
                
        processed_count += 1
        
    print(f"[Parsing] Done. Vectorized {actions_count} action states from {processed_count} files.")
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(dataset)
        
    print(f"[Parsing] Saved vectorized data to '{output_csv}'")

if __name__ == '__main__':
    main()

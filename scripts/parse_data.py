import os
import re
import csv
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

def main():
    print("[Parsing] Initializing Parser & Evaluator...")
    evaluator = PokerEvaluator()
    
    raw_dir = os.path.join("data", "raw_hands")
    output_csv = os.path.join("data", "vectorized_hands.csv")
    
    if not os.path.exists(raw_dir):
        print(f"[Error] Raw directory '{raw_dir}' does not exist. Run download_data.py first.")
        return
        
    files = [f for f in os.listdir(raw_dir) if f.endswith(".phh")]
    print(f"[Parsing] Found {len(files)} hand histories in '{raw_dir}'.")
    
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
    
    for file_name in files:
        file_path = os.path.join(raw_dir, file_name)
        try:
            with open(file_path, 'rb') as f:
                hh = HandHistory.load(f)
                
            # Iterate through state actions generator
            for state, sa_str in hh.state_actions:
                if sa_str is None:
                    continue
                # sa_str is e.g. "p3 f", "p1 cc", "p5 cbr 200"
                match = re.match(r"^p(\d+)\s+(f|cc|cbr(?:\s+\d+)?)", sa_str)
                if not match:
                    continue
                    
                actor_index = int(match.group(1)) - 1
                action_type = match.group(2)
                
                # Extract cards and convert card objects to string (e.g. 'As')
                hero_cards = [clean_card(c) for c in state.hole_cards[actor_index]]
                board_cards = [clean_card(c) for street in state.board_cards for c in street]
                
                if len(hero_cards) < 2:
                    continue # Skip if cards aren't dealt or visible yet
                    
                # State parameters
                pot = float(state.total_pot_amount)
                stack = float(state.stacks[actor_index])
                call_amount = float(state.checking_or_calling_amount) if state.checking_or_calling_amount is not None else 0.0
                
                # Active players count (statuses is True for active players in the hand)
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
                
                # Target action mapping
                # 0=FOLD, 1=CHECK, 2=CALL, 3=BET, 4=RAISE
                action_label = 0
                if action_type.startswith("f"):
                    action_label = 0
                elif action_type.startswith("cc"):
                    action_label = 1 if call_amount == 0 else 2
                elif action_type.startswith("cbr"):
                    action_label = 3 if call_amount == 0 else 4
                    
                row = [
                    is_preflop,
                    float(num_opponents),
                    equity,
                    pot_odds,
                    stack_pot_ratio,
                    bet_raise_available,
                    check_call_available,
                    float(action_label)
                ]
                
                dataset.append(row)
                actions_count += 1
                
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"[Parsing] Processed {processed_count} hands ({actions_count} action points)...")
                
        except Exception as e:
            print(f"[Warning] Failed to parse {file_name}: {e}")
            
    # Write to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(dataset)
        
    print(f"[Parsing] Complete! Generated {len(dataset)} training samples. Saved to: {output_csv}")

if __name__ == '__main__':
    main()

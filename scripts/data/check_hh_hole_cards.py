import io
import os
import re
from pokerkit import HandHistory

def clean_card(c):
    s = str(c)
    match = re.search(r'\((.*?)\)', s)
    return match.group(1) if match else s

raw_dir = os.path.join("data", "raw_human_hands")
files = [f for f in os.listdir(raw_dir) if f.endswith(".phhs")]
if files:
    file_path = os.path.join(raw_dir, files[0])
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Parse first hand block
    blocks = content.split('\n\n')
    hand_str = blocks[0]
    # Remove index header if present
    if hand_str.strip().startswith('['):
        lines = hand_str.strip().split('\n')
        hand_str = '\n'.join(lines[1:])
        
    f_bytes = io.BytesIO(hand_str.encode('utf-8'))
    hh = HandHistory.load(f_bytes)
    
    print("hh.hole_cards:", hh.hole_cards)
    for i, hc in enumerate(hh.hole_cards):
        cleaned = [clean_card(c) for c in hc]
        print(f"  Player {i+1}: {cleaned}")

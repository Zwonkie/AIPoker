import os
import sys
import cv2

sys.path.insert(0, os.path.abspath('.'))
from core.vision import PokerVision

def test_active_vision():
    img_path = r"diagnostics/turn_20260709_205714/screenshot.png"
    if not os.path.exists(img_path):
        print(f"Error: Screenshot not found at {img_path}")
        return
        
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    if abs(w - 1536) > 50 or abs(h - 1090) > 50:
        img = cv2.resize(img, (1536, 1090), interpolation=cv2.INTER_CUBIC)
        
    vision = PokerVision()
    state = vision.read_board_state(img)
    
    print("--- VERIFICATION RESULTS (UPDATED CLASS) ---")
    print(f"Hero Name:  '{state.get('hero_name')}'")
    print(f"Hero Stack: {state.get('hero_stack')}")
    print(f"Hero Cards: {state.get('hero_cards')}")
    print(f"Hero VPIP:  {state.get('hero_vpip_color')}")
    print(f"Hero AGG:   {state.get('hero_agg_color')}")
    print(f"Pot Size:   {state.get('pot_size')}")
    print("\nOpponents:")
    for k, v in state.get('opponents', {}).items():
        print(f"  {k}: {v}")

if __name__ == '__main__':
    test_active_vision()

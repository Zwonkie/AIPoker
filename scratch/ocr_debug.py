import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from core.vision import PokerVision

def run_mock_analysis():
    img_path = r"diagnostics/turn_20260709_205714/screenshot.png"
    if not os.path.exists(img_path):
        print(f"Error: Screenshot not found at {img_path}")
        return
        
    img = cv2.imread(img_path)
    # Scale if needed to standard size
    h, w = img.shape[:2]
    if abs(w - 1536) > 50 or abs(h - 1090) > 50:
        img = cv2.resize(img, (1536, 1090), interpolation=cv2.INTER_CUBIC)
        
    vision = PokerVision()
    
    # Let's monkeypatch resolved_centers calculation inside read_board_state
    # to use static coordinates directly
    default_centers = {
        'seat_1': (320, 757),
        'seat_2': (196, 306),
        'seat_3': (767, 180),
        'seat_4': (1340, 306),
        'seat_5': (1213, 757),
        'hero': (767, 837)
    }
    
    # We can temporarily override the template matching part by hacking the vision method
    original_read = vision.read_board_state
    
    def mock_read_board_state(image):
        # We intercept and modify the resolved centers matching loop
        # We will use original method logic, but force resolved_centers to default_centers
        state = {}
        
        # 1. Community Cards
        card_matches = vision.match_templates_in_roi(
            image, vision.rois['community_cards'], vision.card_templates, threshold=0.90, max_matches=5
        )
        state['community_cards'] = [m[0] for m in card_matches]
        
        # 2. Pot
        pot_text = vision.ocr_roi(image, vision.rois['pot'], whitelist='0123456789.Pulje: ')
        state['pot_size'] = vision.clean_pot_string(pot_text)
        
        # Opponents & Hero
        opponents = {}
        resolved_centers = default_centers.copy()
        
        # Parse Opponents
        for seat_key in ['seat_1', 'seat_2', 'seat_3', 'seat_4', 'seat_5']:
            cx, cy = resolved_centers[seat_key]
            
            # Active check
            seat_top = image[max(0, cy-75):min(image.shape[0], cy-49), max(0, cx-65):min(image.shape[1], cx+65)]
            gray = cv2.cvtColor(seat_top, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
            mean_top_5 = np.mean(thresh)
            is_active = mean_top_5 > 160.0
            
            opp_name = ""
            opp_stack_val = 0
            
            if is_active:
                opp_name = vision.ocr_roi(image, (cx-75, cy+22, 150, 26))
                if len(opp_name) <= 2 and not opp_name.isalnum():
                    opp_name = ""
                    
                opp_stack_text = vision.ocr_roi(image, (cx-65, cy+50, 130, 34), whitelist='0123456789.ALLIN-')
                opp_stack_val = vision.clean_stack_string(opp_stack_text) if opp_stack_text.strip() else 0
                
            opponents[seat_key] = {
                'name': opp_name,
                'stack': opp_stack_val,
                'is_active': is_active,
                'state': 'Active' if is_active else 'Folded',
                'vpip_color': None,
                'agg_color': None
            }
            
        state['opponents'] = opponents
        
        # Hero logic
        hcx, hcy = resolved_centers['hero']
        hname = vision.ocr_roi(image, (hcx-75, hcy+22, 150, 26))
        if len(hname) <= 2 and not hname.isalnum():
            hname = ""
        state['hero_name'] = hname
        
        hero_stack_text = vision.ocr_roi(image, (hcx-65, hcy+50, 130, 34), whitelist='0123456789.ALLIN-')
        state['hero_stack'] = vision.clean_stack_string(hero_stack_text) if hero_stack_text.strip() else 0
        
        dynamic_hero_cards_roi = (max(0, hcx - 127), max(0, hcy - 112), 230, 95)
        hero_matches = vision.match_templates_in_roi(
            image, dynamic_hero_cards_roi, vision.card_templates, threshold=0.90, max_matches=2
        )
        state['hero_cards'] = [m[0] for m in hero_matches]
        
        return state

    print("--- STATIC COORDINATES SIMULATION ---")
    try:
        state = mock_read_board_state(img)
        print(f"Hero Name:  '{state.get('hero_name')}'")
        print(f"Hero Stack: {state.get('hero_stack')}")
        print(f"Hero Cards: {state.get('hero_cards')}")
        print(f"Pot Size:   {state.get('pot_size')}")
        print("\nOpponents:")
        for k, v in state.get('opponents', {}).items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Exception during parse: {e}")

if __name__ == '__main__':
    run_mock_analysis()

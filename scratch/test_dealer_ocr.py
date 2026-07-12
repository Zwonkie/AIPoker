import cv2
import os
import sys
import numpy as np

# Add project root to sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vision import PokerVision
from core.table_state import TableState

def test_dealer_ocr():
    vision = PokerVision()
    
    import glob
    screenshot_paths = glob.glob("diagnostics/*/screenshot.png")
    screenshot_paths.sort()
    
    print(f"Found {len(screenshot_paths)} diagnostic screenshots to test.\n")
    print(f"{'Directory':<30} | {'Score':<6} | {'Coords':<12} | {'Closest Seat':<12} | {'Hero Pos':<8}")
    print("-" * 78)
    
    for path in screenshot_paths:
        img = cv2.imread(path)
        if img is None:
            continue
            
        dirname = os.path.basename(os.path.dirname(path))
        
        # Match dealer button
        gray = vision.preprocess_image(img)
        res = cv2.matchTemplate(gray, vision.dealer_button_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if max_val >= 0.85:
            h, w = vision.dealer_button_template.shape
            dx = max_loc[0] + w // 2
            dy = max_loc[1] + h // 2
            
            # Find closest seat
            default_centers = {
                'seat_1': (320, 757),
                'seat_2': (196, 306),
                'seat_3': (767, 180),
                'seat_4': (1340, 306),
                'seat_5': (1213, 757),
                'hero': (767, 837)
            }
            
            closest_seat = None
            min_dist = float('inf')
            for seat_key, (cx, cy) in default_centers.items():
                dist = np.sqrt((dx - cx) ** 2 + (dy - cy) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_seat = seat_key
            
            # Hero position
            dealer_idx = 0
            if closest_seat != 'hero':
                dealer_idx = int(closest_seat.split('_')[1])
            hero_pos = (0 - dealer_idx) % 6
            
            print(f"{dirname:<30} | {max_val:.4f} | {f'({dx},{dy})':<12} | {closest_seat:<12} | {hero_pos:<8}")
        else:
            print(f"{dirname:<30} | {max_val:.4f} | {'No Match':<12} | {'N/A':<12} | {'N/A':<8}")

if __name__ == '__main__':
    test_dealer_ocr()

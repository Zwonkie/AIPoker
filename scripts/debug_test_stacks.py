import cv2
import pytesseract
import os
import sys

# Configure Tesseract path (Windows specific)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.vision import PokerVision

vision = PokerVision()

test_cases = [
    "board_samples/1_postflop_first_fold_check_raise.png",
    "board_samples/2_postflop_river_fold_call_raise.png",
    "board_samples/3_preflop_fold_allin.png",
    "board_samples/4_postflop_river_fold_call_raise_facing_bet.png"
]

for tc in test_cases:
    img = cv2.imread(tc)
    if img is None:
        continue
    h, w = img.shape[:2]
    if abs(w - 1536) > 50 or abs(h - 1090) > 50:
        img = cv2.resize(img, (1536, 1090), interpolation=cv2.INTER_CUBIC)
        
    res = vision.read_board_state(img)
    print(f"=== File: {tc} ===")
    print(f"  Hero Cards: {res['hero_cards']}")
    print(f"  Hero Stack: {res['hero_stack']}")
    print(f"  Pot Size:   {res['pot_size']}")
    
    # Let's inspect the actual crop of Hero Stack
    hcx, hcy = 767, 837  # default center for hero
    # Find active anchor if match succeeds
    active_template = vision.hero_hexagon_template
    active_mask = vision.hero_hexagon_mask
    cx, cy = hcx, hcy
    if active_template is not None:
        search_area = img[770:770+130, 600:600+300]
        match_res = cv2.matchTemplate(search_area, active_template, cv2.TM_CCORR_NORMED, mask=active_mask)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_res)
        if max_val >= 0.70:
            cx = 600 + max_loc[0] + active_template.shape[1] // 2
            cy = 770 + max_loc[1] + active_template.shape[0] // 2
    
    crop = img[max(0, cy+50):min(img.shape[0], cy+84), max(0, cx-65):min(img.shape[1], cx+65)]
    # Save the crop for debugging
    name_base = os.path.basename(tc).replace('.png', '_crop.png')
    cv2.imwrite(name_base, crop)
    
    # Run Tesseract via ocr_roi
    text = vision.ocr_roi(img, (cx-65, cy+50, 130, 34), whitelist='0123456789.kr$€')
    print(f"  ocr_roi output: '{text}'")
    print(f"  Cleaned: {vision.clean_stack_string(text)}")

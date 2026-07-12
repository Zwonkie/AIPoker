import cv2
import numpy as np
import os
import sys

# Add current path so core.vision works
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.vision import PokerVision

img_path = r'C:\REPO\Antigravity\AIPoker\board_samples\11_flop_7c6s_check.png'
out_path = r'C:\Users\zwonk\.gemini\antigravity-ide\brain\89b04cb8-5f64-40df-a4b7-7e0930f4c127\artifacts\mock_11_dynamic_bboxes.png'

vision = PokerVision()
img = cv2.imread(img_path)

if img is None:
    print("Failed to load image")
    exit(1)

# Draw static ROIs (Pot, Community, Hero Cards, Hero Stack, Buttons)
static_rois = ['community_cards', 'pot', 'buttons']
for roi_name in static_rois:
    if roi_name in vision.rois:
        x, y, w, h = vision.rois[roi_name]
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, roi_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Search regions for template matching
search_regions = {
    'seat_1': (150, 700, 250, 150),
    'seat_2': (50, 260, 250, 150),
    'seat_3': (640, 130, 250, 150),
    'seat_4': (1180, 260, 250, 150),
    'seat_5': (1030, 700, 250, 150),
    'hero': (600, 770, 300, 130)
}

default_centers = {
    'seat_1': (320, 757),
    'seat_2': (196, 306),
    'seat_3': (767, 180),
    'seat_4': (1340, 306),
    'seat_5': (1213, 757),
    'hero': (767, 837)
}

resolved_centers = {}
found_anchors = {}

for key, (rx, ry, rw, rh) in search_regions.items():
    # Draw search region in thin gray
    cv2.rectangle(img, (rx, ry), (rx+rw, ry+rh), (100, 100, 100), 1)
    
    cx, cy = default_centers[key]
    found_anchor = False
    # Determine which template to use
    active_template = vision.hero_hexagon_template if key == 'hero' else vision.hexagon_template
    active_mask = vision.hero_hexagon_mask if key == 'hero' else vision.hexagon_mask
    
    if active_template is not None:
        search_area = img[ry:ry+rh, rx:rx+rw]
        if search_area.shape[0] >= active_template.shape[0] and search_area.shape[1] >= active_template.shape[1]:
            res = cv2.matchTemplate(search_area, active_template, cv2.TM_CCORR_NORMED, mask=active_mask)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val >= 0.70:
                cx = rx + max_loc[0] + active_template.shape[1] // 2
                cy = ry + max_loc[1] + active_template.shape[0] // 2
                found_anchor = True
                # Draw anchor dot
                cv2.circle(img, (cx, cy), 3, (0, 255, 255), -1)
                cv2.putText(img, "Anchor", (cx+5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    resolved_centers[key] = (cx, cy)
    found_anchors[key] = found_anchor

for seat_key in ['seat_1', 'seat_2', 'seat_3', 'seat_4', 'seat_5', 'hero']:
    if not found_anchors.get(seat_key, False):
        continue
    cx, cy = resolved_centers[seat_key]
    
    # VPIP and AGG Crops
    # vpip_crop = img[max(0, cy-12):min(img.shape[0], cy+12), max(0, cx-100):min(img.shape[1], cx-80)]
    # agg_crop = img[max(0, cy-12):min(img.shape[0], cy+12), max(0, cx+80):min(img.shape[1], cx+100)]
    # Name: img[max(0, cy+22):min(img.shape[0], cy+48), max(0, cx-75):min(img.shape[1], cx+75)]
    # Stack: img[max(0, cy+50):min(img.shape[0], cy+84), max(0, cx-65):min(img.shape[1], cx+65)]
    
    # Draw VPIP
    vpip_y1, vpip_y2 = max(0, cy-12), min(img.shape[0], cy+12)
    vpip_x1, vpip_x2 = max(0, cx-100), min(img.shape[1], cx-80)
    cv2.rectangle(img, (vpip_x1, vpip_y1), (vpip_x2, vpip_y2), (255, 0, 255), 2)
    cv2.putText(img, "VPIP", (vpip_x1, vpip_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    
    # Draw AGG
    agg_y1, agg_y2 = max(0, cy-12), min(img.shape[0], cy+12)
    agg_x1, agg_x2 = max(0, cx+80), min(img.shape[1], cx+100)
    cv2.rectangle(img, (agg_x1, agg_y1), (agg_x2, agg_y2), (0, 165, 255), 2) # Orange
    cv2.putText(img, "AGG", (agg_x1, agg_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
    
    
    # Draw Name
    name_y1, name_y2 = max(0, cy+22), min(img.shape[0], cy+48)
    name_x1, name_x2 = max(0, cx-75), min(img.shape[1], cx+75)
    cv2.rectangle(img, (name_x1, name_y1), (name_x2, name_y2), (255, 0, 0), 2) # Blue
    cv2.putText(img, "Name", (name_x1, name_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # If Hero, draw Hero Cards
    if seat_key == 'hero':
        hc_y1, hc_y2 = max(0, cy - 112), max(0, cy - 112) + 95
        hc_x1, hc_x2 = max(0, cx - 127), max(0, cx - 127) + 230
        cv2.rectangle(img, (hc_x1, hc_y1), (hc_x2, hc_y2), (0, 255, 0), 2) # Green
        cv2.putText(img, "Hero Cards", (hc_x1, hc_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Draw Stack
    stack_y1, stack_y2 = max(0, cy+50), min(img.shape[0], cy+84)
    stack_x1, stack_x2 = max(0, cx-65), min(img.shape[1], cx+65)
    cv2.rectangle(img, (stack_x1, stack_y1), (stack_x2, stack_y2), (255, 255, 0), 2) # Cyan
    cv2.putText(img, "Stack", (stack_x1, stack_y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
cv2.imwrite(out_path, img)
print('Saved dynamic bboxes to', out_path)

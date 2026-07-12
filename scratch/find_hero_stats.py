import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from core.vision import PokerVision

def find_stats():
    img_path = r"diagnostics/turn_20260709_205714/screenshot.png"
    if not os.path.exists(img_path):
        print(f"Error: Screenshot not found")
        return
        
    img = cv2.imread(img_path)
    img = cv2.resize(img, (1536, 1090), interpolation=cv2.INTER_CUBIC)
    
    hcx, hcy = 767, 837
    vision = PokerVision()
    
    # Let's test a grid of crops around the center to see where the color is!
    # Looking at the image, Hero's stat pill is just above the name box.
    # The name box is y: 859 to 885.
    # The stat pill should be around y: 840 to 860.
    # Let's do horizontal crops:
    # Left stat (VPIP): x: hcx - 60 to hcx - 15 (e.g. 707 to 752)
    # Right stat (AGG): x: hcx + 15 to hcx + 60 (e.g. 782 to 827)
    
    y1, y2 = hcy + 5, hcy + 25 # y: 842 to 862
    
    vpip_crop = img[y1:y2, hcx-50:hcx-15]
    agg_crop = img[y1:y2, hcx+15:hcx+50]
    
    # Save the crops to verify
    output_dir = r"C:\Users\zwonk\.gemini\antigravity-ide\brain\c68a647c-2540-4757-8bda-15d48c1088de"
    cv2.imwrite(os.path.join(output_dir, "hero_vpip_crop.png"), vpip_crop)
    cv2.imwrite(os.path.join(output_dir, "hero_agg_crop.png"), agg_crop)
    
    vpip_color = vision.classify_color(vpip_crop)
    agg_color = vision.classify_color(agg_crop)
    
    print(f"Proposed Crops Y range: {y1} to {y2}")
    print(f"Hero VPIP Color detected: {vpip_color}")
    print(f"Hero AGG Color detected: {agg_color}")
    
    # Let's also print average HSV/BGR to be sure
    hsv_vpip = cv2.cvtColor(vpip_crop, cv2.COLOR_BGR2HSV)
    hsv_agg = cv2.cvtColor(agg_crop, cv2.COLOR_BGR2HSV)
    print(f"VPIP Mean HSV: {np.mean(hsv_vpip, axis=(0,1))}")
    print(f"AGG Mean HSV: {np.mean(hsv_agg, axis=(0,1))}")

if __name__ == '__main__':
    find_stats()

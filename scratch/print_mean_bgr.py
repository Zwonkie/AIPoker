import os
import sys
import cv2
import numpy as np

def print_bgr():
    output_dir = r"C:\Users\zwonk\.gemini\antigravity-ide\brain\c68a647c-2540-4757-8bda-15d48c1088de"
    vpip_crop = cv2.imread(os.path.join(output_dir, "hero_vpip_crop.png"))
    agg_crop = cv2.imread(os.path.join(output_dir, "hero_agg_crop.png"))
    
    mean_vpip = np.mean(vpip_crop, axis=(0,1))
    mean_agg = np.mean(agg_crop, axis=(0,1))
    
    print(f"VPIP Mean BGR: {mean_vpip}")
    print(f"AGG Mean BGR: {mean_agg}")
    
    # Let's check saturation
    for name, bgr in [("VPIP", mean_vpip), ("AGG", mean_agg)]:
        b, g, r = bgr
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        sat = (max_c - min_c) / max_c if max_c > 0 else 0
        print(f"{name} Saturation: {sat:.4f}")

if __name__ == '__main__':
    print_bgr()

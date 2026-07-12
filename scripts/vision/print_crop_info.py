import cv2
import os

crops = [
    "1_postflop_first_fold_check_raise_crop.png",
    "2_postflop_river_fold_call_raise_crop.png",
    "3_preflop_fold_allin_crop.png",
    "4_postflop_river_fold_call_raise_facing_bet_crop.png"
]

for c in crops:
    if os.path.exists(c):
        img = cv2.imread(c)
        print(f"Crop {c}: shape={img.shape}")

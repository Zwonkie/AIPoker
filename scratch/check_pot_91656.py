import cv2
import os

img_path = r"diagnostics/turn_20260711_091656/screenshot.png"
if os.path.exists(img_path):
    img = cv2.imread(img_path)
    # Pot ROI is (700, 365, 160, 45)
    # Let's crop it and save it.
    pot_crop = img[365:365+45, 700:700+160]
    cv2.imwrite("scratch/pot_crop_91656.png", pot_crop)
    print("Saved scratch/pot_crop_91656.png")
else:
    print("Screenshot not found")

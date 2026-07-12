import cv2
import os

img_path = r"diagnostics/turn_20260711_085556/screenshot.png"
if os.path.exists(img_path):
    img = cv2.imread(img_path)
    
    # Hero center is (767, 837)
    # Let's crop a region of 300x300 around Hero to see where the dealer button is!
    crop = img[837-150:837+150, 767-150:767+150]
    cv2.imwrite("scratch/hero_crop.png", crop)
    print("Saved scratch/hero_crop.png")
else:
    print("Screenshot not found")

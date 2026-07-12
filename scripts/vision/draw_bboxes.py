import cv2
import os

img_path = r'C:\REPO\Antigravity\AIPoker\board_samples\11_flop_7c6s_check.png'
out_path = r'C:\Users\zwonk\.gemini\antigravity-ide\brain\89b04cb8-5f64-40df-a4b7-7e0930f4c127\artifacts\mock_11_bboxes.png'

rois = {
    'community_cards': (500, 395, 500, 95),
    'hero_cards': (640, 725, 230, 95),
    'pot': (700, 365, 160, 45),
    'hero_stack': (650, 850, 200, 60),
    'seat_1_name': (200, 770, 200, 40),
    'seat_1_stack': (200, 805, 200, 40),
    'seat_2_name': (200, 320, 200, 40),
    'seat_2_stack': (200, 355, 200, 40),
    'seat_3_name': (670, 195, 200, 40),
    'seat_3_stack': (670, 230, 200, 40),
    'seat_4_name': (1100, 320, 200, 40),
    'seat_4_stack': (1100, 355, 200, 40),
    'seat_5_name': (1100, 770, 200, 40),
    'seat_5_stack': (1100, 805, 200, 40),
    'buttons': (980, 970, 550, 120)
}

img = cv2.imread(img_path)

for name, (x, y, w, h) in rois.items():
    color = (0, 255, 0)
    if 'seat' in name:
        color = (255, 0, 0)
    elif 'button' in name:
        color = (0, 0, 255)
    
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(img, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

os.makedirs(os.path.dirname(out_path), exist_ok=True)
cv2.imwrite(out_path, img)
print('Saved to', out_path)

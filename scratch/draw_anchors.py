import os
import sys
import cv2

def draw_anchors():
    img_path = r"diagnostics/turn_20260709_205714/screenshot.png"
    if not os.path.exists(img_path):
        print(f"Error: Screenshot not found at {img_path}")
        return
        
    img = cv2.imread(img_path)
    img = cv2.resize(img, (1536, 1090), interpolation=cv2.INTER_CUBIC)
    
    default_centers = {
        'seat_1': (320, 757),
        'seat_2': (196, 306),
        'seat_3': (767, 180),
        'seat_4': (1340, 306),
        'seat_5': (1213, 757),
        'hero': (767, 837)
    }
    
    for key, (cx, cy) in default_centers.items():
        # Draw outline circle in red
        cv2.circle(img, (cx, cy), 15, (0, 0, 255), 3)
        # Draw center point in green
        cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)
        
        # Add text label above/below center
        label = f"{key} ({cx}, {cy})"
        text_y = cy - 22 if "seat" in key else cy + 85
        cv2.putText(img, label, (cx - 80, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, label, (cx - 80, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
    output_dir = r"C:\Users\zwonk\.gemini\antigravity-ide\brain\c68a647c-2540-4757-8bda-15d48c1088de"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "anchors_visualization.png")
    cv2.imwrite(output_path, img)
    print(f"Saved visualization to {output_path}")

if __name__ == '__main__':
    draw_anchors()

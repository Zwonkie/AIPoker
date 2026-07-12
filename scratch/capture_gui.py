import sys
import os
import time
import pygetwindow as gw
from PIL import ImageGrab

def main():
    time.sleep(5.0) # Wait for window to render completely
    all_wins = gw.getAllTitles()
    print("Found windows:", [w for w in all_wins if w.strip()])
    
    wins = gw.getWindowsWithTitle("PHPHelp")
    if not wins:
        print("ERROR: Window PHPHelp not found.")
        return
        
    win = wins[0]
    win.activate()
    time.sleep(1.0)
    
    # Capture window rect
    box = (win.left, win.top, win.right, win.bottom)
    print(f"Capturing window at: {box}")
    
    img = ImageGrab.grab(bbox=box)
    
    # Save to artifacts directory
    art_dir = r"C:\Users\zwonk\.gemini\antigravity-ide\brain\c68a647c-2540-4757-8bda-15d48c1088de"
    img_path = os.path.join(art_dir, "gui_decision_tree.png")
    img.save(img_path)
    print(f"SUCCESS: Screenshot saved to {img_path}")

if __name__ == '__main__':
    main()

import time
import random
import math
import pyautogui
import pydirectinput

# Safety features for pyautogui
pyautogui.FAILSAFE = True  # Move mouse to corner to abort
pyautogui.PAUSE = 0.1

class EmergencyAbortException(Exception):
    """Raised when the user interrupts bot execution with the Escape key."""
    pass

class ActionExecutor:
    def __init__(self):
        pass

    def sleep_random(self, min_sec=0.4, max_sec=1.2):
        """Sleeps for a random duration to mimic human delay."""
        time.sleep(random.uniform(min_sec, max_sec))

    def move_mouse_smooth(self, target_x, target_y, duration=0.8):
        """
        Moves mouse to target coordinates using a smooth cubic Bezier curve,
        organic velocity profiles, and accurate elapsed time tracking.
        Adapts dynamically to Windows scheduler tick rate (no lag).
        """
        start_x, start_y = pyautogui.position()
        
        # If already at the target, do nothing
        if start_x == target_x and start_y == target_y:
            return

        dx = target_x - start_x
        dy = target_y - start_y
        dist = math.hypot(dx, dy)
        
        # If very close, just do a fast linear move
        if dist < 20:
            pydirectinput.moveTo(target_x, target_y)
            return

        # Generate control points for a cubic Bezier curve to add human-like curvature
        deviation_scale = dist * random.uniform(0.08, 0.22)
        
        # Random offsets to control points
        ox1 = int(deviation_scale * random.uniform(-1, 1))
        oy1 = int(deviation_scale * random.uniform(-1, 1))
        ox2 = int(deviation_scale * random.uniform(-1, 1))
        oy2 = int(deviation_scale * random.uniform(-1, 1))
        
        # P1 and P2 control points along the trajectory
        p1_x = start_x + dx * 0.3 + ox1
        p1_y = start_y + dy * 0.3 + oy1
        p2_x = start_x + dx * 0.7 + ox2
        p2_y = start_y + dy * 0.7 + oy2

        # Elapsed-time tracking loop (prevents Windows sleep lag)
        start_time = time.perf_counter()
        
        while True:
            # Check for Esc key (0x1B) press to abort immediately
            import ctypes
            if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:
                raise EmergencyAbortException("Escape key pressed during mouse movement")
                
            elapsed = time.perf_counter() - start_time
            if elapsed >= duration:
                break
                
            t = elapsed / duration
            
            # Eased time to simulate muscle acceleration/deceleration (Sine ease)
            t_eased = (1.0 - math.cos(t * math.pi)) / 2.0
            
            # Cubic Bezier interpolation formula
            mt = 1.0 - t_eased
            x = int(mt**3 * start_x + 3 * mt**2 * t_eased * p1_x + 3 * mt * t_eased**2 * p2_x + t_eased**3 * target_x)
            y = int(mt**3 * start_y + 3 * mt**2 * t_eased * p1_y + 3 * mt * t_eased**2 * p2_y + t_eased**3 * target_y)
            
            pydirectinput.moveTo(x, y)
            
            # Tiny sleep to yield CPU. If Windows sleeps longer, 
            # elapsed time checks will skip frames to finish exactly on time.
            time.sleep(0.001)
            
        pydirectinput.moveTo(target_x, target_y)

    def click_button_relative(self, fold_btn_coord, action_type, window_pos=(0, 0), window_size=None):
        """
        Calculates the absolute position of the target button based on
        the detected fold button coordinate, moves mouse smoothly, and clicks.
        Supports multi-click sequences for bet sizing pot shortcuts.
        """
        fold_x, fold_y = fold_btn_coord
        win_x, win_y = window_pos

        # Handle compound actions like BET_POT_50, RAISE_SLIDER_0.50
        base_action = action_type
        shortcut = None
        slider_fraction = None
        if '_' in action_type:
            parts = action_type.split('_')
            if len(parts) >= 3 and parts[1] == 'POT':
                base_action = parts[0]       # 'BET' or 'RAISE'
                shortcut = f"{parts[1]}_{parts[2]}"  # 'POT_50', 'POT_70', etc.
            elif len(parts) >= 3 and parts[1] == 'SLIDER':
                base_action = parts[0]
                slider_fraction = float(parts[2])

        # Helper to click a relative coordinate offset
        def perform_click_at_offset(offset_x, offset_y, use_jitter=True):
            # Scale coordinates relative to reference 1536x1090 layout
            rel_x = fold_x + offset_x
            rel_y = fold_y + offset_y
            if window_size and window_size[0] > 0 and window_size[1] > 0:
                act_w, act_h = window_size
                rel_x = rel_x * (act_w / 1536.0)
                rel_y = rel_y * (act_h / 1090.0)

            # Random jitter
            if use_jitter:
                rand_x = random.randint(-6, 6)
                rand_y = random.randint(-4, 4)
            else:
                rand_x = 0
                rand_y = 0
                
            target_x = int(win_x + rel_x + rand_x)
            target_y = int(win_y + rel_y + rand_y)

            # Smooth Bezier movement (duration 0.4s to 1.5s)
            move_duration = random.uniform(0.4, 1.5)
            self.move_mouse_smooth(target_x, target_y, duration=move_duration)
            self.sleep_random(0.1, 0.25)
            
            # Click
            pydirectinput.mouseDown()
            time.sleep(random.uniform(0.05, 0.15))
            pydirectinput.mouseUp()
            return True

        # 1. Click dynamic sizing shortcut if specified
        if shortcut:
            if shortcut == 'POT_50':
                sc_off_x, sc_off_y = 65, -55
            elif shortcut == 'POT_70':
                sc_off_x, sc_off_y = 195, -55
            elif shortcut == 'POT_100':
                sc_off_x, sc_off_y = 325, -55
            elif shortcut == 'POT_125':
                sc_off_x, sc_off_y = 455, -55
            else:
                print(f"Unknown shortcut: {shortcut}")
                return False
            
            print(f"Executing dynamic sizing shortcut: Clicking {shortcut}...")
            perform_click_at_offset(sc_off_x, sc_off_y)
            # Human latency between shortcut click and action click
            self.sleep_random(0.15, 0.3)
        elif slider_fraction is not None:
            # Click the slider track at the specified fraction (between 0.0 and 1.0)
            slider_fraction = max(0.0, min(1.0, slider_fraction))
            sc_off_x = int(165 + slider_fraction * 355)
            sc_off_y = -13
            print(f"Executing slider sizing: Clicking slider at fraction {slider_fraction:.2f}...")
            perform_click_at_offset(sc_off_x, sc_off_y, use_jitter=False)
            self.sleep_random(0.2, 0.4)

        # 2. Click the main action button
        if base_action == 'FOLD':
            main_off_x, main_off_y = 90, 45
        elif base_action in ['CHECK', 'CALL']:
            main_off_x, main_off_y = 290, 45
        elif base_action in ['BET', 'RAISE']:
            main_off_x, main_off_y = 460, 45
        else:
            print(f"Unknown action: {base_action}")
            return False

        print(f"Executing main action: Clicking {base_action}...")
        return perform_click_at_offset(main_off_x, main_off_y)

if __name__ == '__main__':
    # Simple dry-run test
    print("Action executor dry-run initialized.")
    ae = ActionExecutor()
    print("Moving mouse to center screen...")
    ae.move_mouse_smooth(960, 540, duration=1.2)
    print("Test finished.")

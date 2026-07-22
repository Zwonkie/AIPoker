"""Mouse primitives: humanized move, click, slider drag. Coordinates are CLIENT-AREA
relative (same frame as capture.py ROIs) and translated to screen here -- callers never
see screen coordinates.

The ONLY place the service touches focus: a click/drag brings the window foreground
first (SPECS: "no SetFocus() except for clicks"). pyautogui FAILSAFE stays on -- slam
the cursor into a screen corner to abort a runaway bot."""
import ctypes
import random
import time

import pyautogui

from live2.service import capture

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05

user32 = ctypes.windll.user32


def _focus(hwnd):
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.15)


def _ease(t):
    return t * t * (3 - 2 * t)          # smoothstep


def move_smooth(sx, sy, duration=None):
    """Humanized cursor move to SCREEN coords: eased path, slight arc, endpoint jitter."""
    x0, y0 = pyautogui.position()
    duration = duration or random.uniform(0.35, 0.7)
    arc = random.uniform(-24, 24)
    steps = max(8, int(duration * 60))
    for i in range(1, steps + 1):
        t = _ease(i / steps)
        x = x0 + (sx - x0) * t
        y = y0 + (sy - y0) * t + arc * (1 - abs(2 * t - 1))   # bulge mid-path
        pyautogui.moveTo(x, y, _pause=False)
        time.sleep(duration / steps)
    pyautogui.moveTo(sx + random.uniform(-1.5, 1.5), sy + random.uniform(-1.5, 1.5), _pause=False)


def click(hwnd, cx, cy, jitter=3):
    """Click at client-relative (cx, cy) with humanized approach + positional jitter."""
    ox, oy = capture.client_origin_on_screen(hwnd)
    sx = ox + cx + random.uniform(-jitter, jitter)
    sy = oy + cy + random.uniform(-jitter, jitter)
    _focus(hwnd)
    move_smooth(sx, sy)
    time.sleep(random.uniform(0.05, 0.18))
    pyautogui.click()
    return {'screen': [round(sx), round(sy)]}


def drag_slider(hwnd, track, frac):
    """Drag along `track` = [x1, y1, x2, y2] (client-relative) to `frac` in [0, 1]:
    grab the handle at the track start, drag to start + frac * (end - start)."""
    frac = max(0.0, min(1.0, float(frac)))
    x1, y1, x2, y2 = [float(v) for v in track]
    ox, oy = capture.client_origin_on_screen(hwnd)
    gx, gy = ox + x1, oy + y1
    tx = ox + x1 + (x2 - x1) * frac
    ty = oy + y1 + (y2 - y1) * frac
    _focus(hwnd)
    move_smooth(gx, gy)
    pyautogui.mouseDown()
    time.sleep(random.uniform(0.05, 0.12))
    move_smooth(tx, ty, duration=random.uniform(0.4, 0.8))
    time.sleep(random.uniform(0.05, 0.12))
    pyautogui.mouseUp()
    return {'frac': frac, 'screen_end': [round(tx), round(ty)]}

"""Mouse primitives on a human motor model. Coordinates are CLIENT-AREA relative (same
frame as capture.py ROIs) and translated to screen here -- callers never see screen
coordinates.

Model (owner spec, 2026-07-22):
  Start ---[ curved trajectory + acceleration/deceleration + micro-jitter ]---> target

- **Fitts's law**: movement duration grows logarithmically with distance/target-size --
  T = a + b * log2(2D / W), gaussian-perturbed. No static time intervals: a far or small
  target takes measurably longer than a near, large one.
- **Trajectory**: quadratic Bezier with a perpendicular-offset control point (natural arc),
  velocity shaped by the minimum-jerk profile s(t) = 10t^3 - 15t^4 + 6t^5 (smooth
  accelerate/decelerate, peak velocity mid-path like real reaching movements).
- **Micro-jitter**: per-sample gaussian tremor that decays as the target is approached.
- **Variable event timing**: samples are emitted at ~125Hz jittered per event (real sensor/
  OS event-loop cadence is not a fixed clock).
- **Overshoot/undershoot + micro-correction**: most fast movements miss the exact center
  and settle with a short second corrective movement (its own Fitts duration).
- **Click hold**: mouse-down -> mouse-up 60-150ms, normal-distributed.

Focus: bet365 raises its own window when it becomes hero's turn, so `click`/`drag_slider`
only VERIFY the window is foreground; SetForegroundWindow is a fallback for the case the
client didn't raise itself (and its success is re-checked -- fail loud, never click into
whatever window happens to be in front).

pyautogui FAILSAFE stays on: slam the cursor into a screen corner to abort a runaway bot.
"""
import ctypes
import math
import random
import time

import pyautogui

from live2.phpserver import capture

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0          # we own all pacing; no library-imposed static sleeps

user32 = ctypes.windll.user32

# Fitts's law constants (human-ish: intercept ~120ms, ~115ms/bit) and motor parameters.
_FITTS_A = 0.120
_FITTS_B = 0.115
_MIN_MOVE_T = 0.14
_POLL_MEAN, _POLL_SD = 0.008, 0.0022         # ~125Hz event cadence, jittered per event
_OVERSHOOT_P = 0.6                           # "frequent slight misses of target center"
_DEFAULT_TARGET_W = 24.0                     # px; typical button height when caller omits


class FocusError(RuntimeError):
    """The table window could not be brought/kept foreground -- refusing to click blind."""


# ------------------------------------------------------------------ planning (pure)

def fitts_duration(dist_px, target_w_px=_DEFAULT_TARGET_W, rng=random):
    """Movement time from Fitts's law with normal variance."""
    bits = math.log2(2.0 * max(1.0, dist_px) / max(4.0, target_w_px) + 1.0)
    t = _FITTS_A + _FITTS_B * bits
    return max(_MIN_MOVE_T, rng.gauss(t, 0.12 * t))


def _min_jerk(t):
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def plan_path(x0, y0, x1, y1, target_w=_DEFAULT_TARGET_W, rng=random):
    """-> list of (x, y, dt) samples from (x0,y0) to ~(x1,y1): Bezier arc, min-jerk pacing,
    decaying tremor, variable inter-event timing. Pure -- does not move anything."""
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist < 1.0:
        return [(x1, y1, rng.uniform(0.004, 0.012))]
    duration = fitts_duration(dist, target_w, rng)
    # control point: midpoint pushed perpendicular to the travel axis (arc handedness random)
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    px, py = -(y1 - y0) / dist, (x1 - x0) / dist
    bow = rng.gauss(0.0, 0.08 * dist)
    bow = max(-45.0, min(45.0, bow))
    cx, cy = mx + px * bow, my + py * bow

    samples, t_acc = [], 0.0
    while t_acc < duration:
        dt = max(0.004, min(0.016, rng.gauss(_POLL_MEAN, _POLL_SD)))
        t_acc += dt
        u = min(1.0, t_acc / duration)
        s = _min_jerk(u)
        bx = (1 - s) * (1 - s) * x0 + 2 * (1 - s) * s * cx + s * s * x1
        by = (1 - s) * (1 - s) * y0 + 2 * (1 - s) * s * cy + s * s * y1
        tremor = (1.0 - s) * 0.9
        samples.append((bx + rng.gauss(0.0, tremor), by + rng.gauss(0.0, tremor), dt))
    samples.append((x1, y1, 0.004))
    return samples


def plan_approach(x0, y0, tx, ty, target_w=_DEFAULT_TARGET_W, rng=random):
    """Full approach: main ballistic movement (usually deliberately off-center) plus the
    micro-correction leg onto the target. -> list of path-sample lists."""
    dist = math.hypot(tx - x0, ty - y0)
    legs = []
    if dist > 60.0 and rng.random() < _OVERSHOOT_P:
        # over- or undershoot along the travel axis, slight perpendicular smear
        ux, uy = (tx - x0) / dist, (ty - y0) / dist
        miss = rng.choice((1.0, 1.0, -1.0)) * rng.uniform(0.03, 0.08) * dist
        miss = max(-25.0, min(25.0, miss))
        perp = rng.gauss(0.0, target_w * 0.18)
        ax, ay = tx + ux * miss - uy * perp, ty + uy * miss + ux * perp
        legs.append(plan_path(x0, y0, ax, ay, target_w, rng))
        legs.append(plan_path(ax, ay, tx, ty, target_w * 0.8, rng))   # micro-correction
    else:
        legs.append(plan_path(x0, y0, tx, ty, target_w, rng))
    return legs


def click_hold_time(rng=random):
    """Mouse-down -> mouse-up interval: 60-150ms, normal variance."""
    return max(0.060, min(0.150, rng.gauss(0.095, 0.022)))


# ------------------------------------------------------------------ execution

def _execute(legs):
    for i, leg in enumerate(legs):
        for x, y, dt in leg:
            pyautogui.moveTo(x, y, _pause=False)
            time.sleep(dt)
        if i < len(legs) - 1:
            time.sleep(random.uniform(0.02, 0.07))   # dwell between ballistic + correction


def _ensure_foreground(hwnd):
    """bet365 raises itself when it becomes hero's turn, so this is normally a pure check.
    Fallback-focus once if not; raise FocusError if the window still isn't in front."""
    if user32.GetForegroundWindow() == hwnd:
        return
    user32.SetForegroundWindow(hwnd)
    time.sleep(random.uniform(0.10, 0.20))
    if user32.GetForegroundWindow() != hwnd:
        raise FocusError(f"window {hwnd} is not foreground and could not be raised")


def move_to(hwnd, cx, cy, target_w=_DEFAULT_TARGET_W):
    """Humanized move to client-relative (cx, cy) with no click. Returns final screen pos."""
    ox, oy = capture.client_origin_on_screen(hwnd)
    x0, y0 = pyautogui.position()
    _execute(plan_approach(x0, y0, ox + cx, oy + cy, target_w))
    fx, fy = pyautogui.position()
    return {'screen': [round(fx), round(fy)]}


def click(hwnd, x, y, target_w=_DEFAULT_TARGET_W, jitter=None):
    """Click at client-relative (x, y): foreground check, human approach with aim scatter
    inside the target, Fitts-paced legs, 60-150ms hold."""
    _ensure_foreground(hwnd)
    ox, oy = capture.client_origin_on_screen(hwnd)
    spread = (jitter if jitter is not None else target_w * 0.14)
    tx = ox + x + random.gauss(0.0, spread)
    ty = oy + y + random.gauss(0.0, spread * 0.8)
    x0, y0 = pyautogui.position()
    _execute(plan_approach(x0, y0, tx, ty, target_w))
    time.sleep(random.uniform(0.03, 0.11))       # target acquired -> decision-to-press lag
    pyautogui.mouseDown(_pause=False)
    time.sleep(click_hold_time())
    pyautogui.mouseUp(_pause=False)
    return {'screen': [round(tx), round(ty)]}


def drag_slider(hwnd, track, frac):
    """Drag along `track` = [x1, y1, x2, y2] (client-relative) to `frac` in [0, 1]: human
    approach to the handle, press, slower guided drag (held drags are more damped than free
    movement), settle, release."""
    frac = max(0.0, min(1.0, float(frac)))
    x1, y1, x2, y2 = [float(v) for v in track]
    _ensure_foreground(hwnd)
    ox, oy = capture.client_origin_on_screen(hwnd)
    gx, gy = ox + x1, oy + y1
    tx = ox + x1 + (x2 - x1) * frac
    ty = oy + y1 + (y2 - y1) * frac
    x0, y0 = pyautogui.position()
    _execute(plan_approach(x0, y0, gx, gy, target_w=16.0))
    pyautogui.mouseDown(_pause=False)
    time.sleep(random.uniform(0.06, 0.14))
    for x, y, dt in plan_path(gx, gy, tx, ty, target_w=12.0):
        pyautogui.moveTo(x, y, _pause=False)
        time.sleep(dt * 1.5)                     # held drags run slower than free moves
    time.sleep(random.uniform(0.08, 0.18))       # verify-the-amount pause before release
    pyautogui.mouseUp(_pause=False)
    return {'frac': frac, 'screen_end': [round(tx), round(ty)]}

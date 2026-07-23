"""Map a decision-engine action string onto humanized clicks (live2.phpserver.interact).

Geometry is the legacy ActionExecutor's, re-expressed in the CLIENT-relative frame the
phpserver primitives use. All offsets are calibrated in the 1536x1090 reference layout
(every capture is vision-processed at that size); `_scale` converts reference pixels to
actual client pixels. fold_xy is the per-frame template-matched fold-button anchor (in
reference pixels) -- the sizing shortcuts and the three action buttons hang off it. The
raise SLIDER is anchored to the window center, NOT the fold button (fixed layout element;
calibrated 2026-07-16: track (1153, 970) -> (1508, 970) at reference size).
"""
import random
import time

from live2.phpserver import interact

# Offsets from the fold-button template anchor (reference 1536x1090), legacy-calibrated.
_MAIN_OFFSET = {
    'FOLD': (90, 45),
    'CHECK': (290, 45),
    'CALL': (290, 45),
    'BET': (460, 45),
    'RAISE': (460, 45),
}
_POT_SHORTCUT_OFFSET = {
    'POT_50': (65, -65), 'POT_70': (195, -65), 'POT_100': (325, -65), 'POT_125': (455, -65),
}
_SLIDER_TRACK_REF = (1153.0, 970.0, 1508.0, 970.0)
_BUTTON_TARGET_W = 46.0        # px, approximate action-button hit height in the reference frame


def parse_action(action_type):
    """'RAISE_SLIDER_0.50' -> ('RAISE', None, 0.5); 'BET_POT_70' -> ('BET', 'POT_70', None);
    'CALL' -> ('CALL', None, None). Unknown base -> (None, None, None)."""
    base, shortcut, slider = action_type, None, None
    parts = action_type.split('_')
    if len(parts) >= 3 and parts[1] == 'POT':
        base, shortcut = parts[0], f"{parts[1]}_{parts[2]}"
    elif len(parts) >= 3 and parts[1] == 'SLIDER':
        base = parts[0]
        try:
            slider = max(0.0, min(1.0, float(parts[2])))
        except ValueError:
            return None, None, None
    if base not in _MAIN_OFFSET:
        return None, None, None
    return base, shortcut, slider


def _scale(client_wh):
    cw, ch = client_wh
    return cw / 1536.0, ch / 1090.0


def execute(hwnd, action_type, fold_xy, client_wh, log=print):
    """Execute one decided action on the real client. Returns True on success. Raises
    interact.FocusError when the table window cannot be made foreground (never clicks
    blind) and pyautogui.FailSafeException on the corner-slam abort."""
    base, shortcut, slider = parse_action(action_type)
    if base is None:
        log(f"[pilot.actions] unknown action {action_type!r} -- not clicking")
        return False
    sx, sy = _scale(client_wh)
    fx, fy = fold_xy

    if shortcut:
        ox, oy = _POT_SHORTCUT_OFFSET.get(shortcut, (None, None))
        if ox is None:
            log(f"[pilot.actions] unknown pot shortcut {shortcut!r} -- not clicking")
            return False
        log(f"[pilot.actions] sizing shortcut {shortcut}")
        interact.click(hwnd, (fx + ox) * sx, (fy + oy) * sy, target_w=_BUTTON_TARGET_W * sx)
        time.sleep(random.uniform(0.15, 0.35))
    elif slider is not None:
        x1, y1, x2, y2 = _SLIDER_TRACK_REF
        track = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        log(f"[pilot.actions] slider drag to {slider:.2f} (track {track})")
        interact.drag_slider(hwnd, track, slider)
        time.sleep(random.uniform(0.15, 0.35))

    ox, oy = _MAIN_OFFSET[base]
    log(f"[pilot.actions] clicking {base}")
    interact.click(hwnd, (fx + ox) * sx, (fy + oy) * sy, target_w=_BUTTON_TARGET_W * sx)
    return True

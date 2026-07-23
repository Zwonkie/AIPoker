"""Occlusion-proof window capture via PrintWindow(PW_RENDERFULLCONTENT), ctypes-only
(no pywin32 dependency). Returns PIL Images; ROI crops are client-area relative.

Why PrintWindow and not mss/ImageGrab: the legacy dashboard grabs the MONITOR, so the
table window must be visible and unobstructed. PrintWindow asks the window to render
itself into our DC, which works while occluded (bet365 is a Direct2D-composited app --
PW_RENDERFULLCONTENT, flag 0x2, is what makes composited content come through).
Windows.Graphics.Capture is the documented fallback if a client update breaks this
(not implemented yet -- fail loud instead)."""
import ctypes
from ctypes import wintypes

from PIL import Image

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

PW_RENDERFULLCONTENT = 0x00000002
SRCCOPY = 0x00CC0020
BI_RGB = 0
DIB_RGB_COLORS = 0


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [('biSize', wintypes.DWORD), ('biWidth', ctypes.c_long),
                ('biHeight', ctypes.c_long), ('biPlanes', wintypes.WORD),
                ('biBitCount', wintypes.WORD), ('biCompression', wintypes.DWORD),
                ('biSizeImage', wintypes.DWORD), ('biXPelsPerMeter', ctypes.c_long),
                ('biYPelsPerMeter', ctypes.c_long), ('biClrUsed', wintypes.DWORD),
                ('biClrImportant', wintypes.DWORD)]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [('bmiHeader', BITMAPINFOHEADER), ('bmiColors', wintypes.DWORD * 3)]


def list_windows(title_contains=None):
    """[{hwnd, title}] of visible top-level windows, optionally filtered by substring."""
    out = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd, _l):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                title = buf.value
                if not title_contains or title_contains.lower() in title.lower():
                    out.append({'hwnd': hwnd, 'title': title})
        return True

    user32.EnumWindows(_enum, 0)
    return out


def find_window(title_contains):
    """First matching hwnd or None."""
    hits = list_windows(title_contains)
    return hits[0]['hwnd'] if hits else None


def client_size(hwnd):
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    return rect.right - rect.left, rect.bottom - rect.top


def client_origin_on_screen(hwnd):
    """Screen coordinates of the client area's top-left (for translating clicks)."""
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y


def capture_window(hwnd, roi=None):
    """PIL RGB image of the window's client area (or `roi` = [x, y, w, h] within it).
    Raises RuntimeError when PrintWindow reports failure -- callers must not mistake a
    black frame for a read."""
    w, h = client_size(hwnd)
    if w <= 0 or h <= 0:
        raise RuntimeError(f"window {hwnd} has empty client area ({w}x{h})")

    hdc_win = user32.GetDC(hwnd)
    hdc_mem = gdi32.CreateCompatibleDC(hdc_win)
    bmp = gdi32.CreateCompatibleBitmap(hdc_win, w, h)
    gdi32.SelectObject(hdc_mem, bmp)
    try:
        ok = user32.PrintWindow(hwnd, hdc_mem, PW_RENDERFULLCONTENT)
        if not ok:
            raise RuntimeError(f"PrintWindow failed for hwnd {hwnd}")
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h            # top-down rows
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB
        buf = ctypes.create_string_buffer(w * h * 4)
        got = gdi32.GetDIBits(hdc_mem, bmp, 0, h, buf, ctypes.byref(bmi), DIB_RGB_COLORS)
        if got != h:
            raise RuntimeError(f"GetDIBits returned {got} of {h} rows")
    finally:
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(hwnd, hdc_win)

    img = Image.frombuffer('RGB', (w, h), buf, 'raw', 'BGRX', 0, 1)
    if roi:
        x, y, rw, rh = [int(v) for v in roi]
        img = img.crop((x, y, x + rw, y + rh))
    return img

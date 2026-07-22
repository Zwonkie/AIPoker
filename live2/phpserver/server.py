"""PHPserver WS/JSON API. Localhost only.

Protocol: one JSON object per message.
  request:  {"id": <any>, "method": "<name>", "params": {...}}
  response: {"id": <same>, "ok": true, "result": ...} | {"id", "ok": false, "error": "..."}

Methods (v1):
  ping          {}                          -> "pong"
  list_windows  {title?}                    -> [{hwnd, title}]
  bind          {title}                     -> {hwnd, title, size}   # remember the table window
  capture       {roi?: [x,y,w,h]}          -> {png_b64, size}       # HEADLESS (unfocused ok)
  move_to       {x, y, target_w?}          -> {screen}              # human move, no click
  click         {x, y, target_w?}          -> {screen}              # client-relative
  move_slider   {track: [x1,y1,x2,y2], frac} -> {frac, screen_end}

Capture never needs focus (PrintWindow). Clicks assume bet365 raised itself for hero's
turn -- interact.py verifies foreground and errors rather than clicking blind.

Scoping decision (v1): the service returns PIXELS -- no OCR here. The assembler owns all
interpretation (CV/OCR included), so "read(rois)" from the SPECS collapses into capture(roi).
Revisit only if shipping frames over the socket proves too slow (localhost: it won't).
"""
import asyncio
import base64
import io
import json

import websockets

from live2.phpserver import capture, interact

HOST, PORT = '127.0.0.1', 8766


class Service:
    def __init__(self):
        self.hwnd = None
        self.title = None

    # -- methods ------------------------------------------------------------
    def ping(self):
        return 'pong'

    def list_windows(self, title=None):
        return capture.list_windows(title)

    def bind(self, title):
        hwnd = capture.find_window(title)
        if hwnd is None:
            raise ValueError(f"no visible window matching {title!r}")
        self.hwnd, self.title = hwnd, title
        return {'hwnd': hwnd, 'title': title, 'size': capture.client_size(hwnd)}

    def _bound(self):
        if self.hwnd is None:
            raise ValueError("no window bound -- call bind first")
        return self.hwnd

    def capture(self, roi=None):
        img = capture.capture_window(self._bound(), roi=roi)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return {'png_b64': base64.b64encode(buf.getvalue()).decode('ascii'),
                'size': list(img.size)}

    def move_to(self, x, y, target_w=None):
        kw = {'target_w': float(target_w)} if target_w else {}
        return interact.move_to(self._bound(), float(x), float(y), **kw)

    def click(self, x, y, target_w=None):
        kw = {'target_w': float(target_w)} if target_w else {}
        return interact.click(self._bound(), float(x), float(y), **kw)

    def move_slider(self, track, frac):
        return interact.drag_slider(self._bound(), track, frac)


async def _handle(ws, svc):
    async for raw in ws:
        rid = None
        try:
            msg = json.loads(raw)
            rid = msg.get('id')
            method = getattr(svc, str(msg.get('method')), None)
            if method is None or str(msg.get('method')).startswith('_'):
                raise ValueError(f"unknown method {msg.get('method')!r}")
            params = msg.get('params') or {}
            # click/drag block the loop briefly (humanized timing) -- run off-thread so a
            # concurrent capture request isn't starved.
            result = await asyncio.to_thread(method, **params)
            await ws.send(json.dumps({'id': rid, 'ok': True, 'result': result}))
        except Exception as e:
            await ws.send(json.dumps({'id': rid, 'ok': False, 'error': f"{type(e).__name__}: {e}"}))


async def main():
    svc = Service()
    async with websockets.serve(lambda ws: _handle(ws, svc), HOST, PORT):
        print(f"PHPserver listening on ws://{HOST}:{PORT}")
        await asyncio.Future()


if __name__ == '__main__':
    asyncio.run(main())

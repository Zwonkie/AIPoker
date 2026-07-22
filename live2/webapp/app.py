"""FastAPI app for the view-only dashboard. Localhost bind only.

Transport: the /ws endpoint pushes a full live snapshot whenever the active session's
turns.jsonl grows (1s poll of file size -- the assembler will later push directly and
the poll disappears). REST endpoints serve the browse views. No endpoint mutates state.

Run:  .venv/Scripts/python.exe -m live2.webapp
"""
import asyncio
import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from live2.webapp import sources

STATIC = os.path.join(os.path.dirname(__file__), 'static')

app = FastAPI(title='AIPoker live2 dashboard', docs_url=None, redoc_url=None)


@app.get('/')
def index():
    return FileResponse(os.path.join(STATIC, 'index.html'))


@app.get('/api/live')
def api_live():
    return JSONResponse(sources.live_snapshot())


@app.get('/api/opponents')
def api_opponents(window: int = 100, min_hands: int = 10):
    return JSONResponse(sources.opponent_profiles(window=window, min_hands=min_hands))


@app.get('/api/sessions')
def api_sessions():
    return JSONResponse(sources.list_sessions())


@app.get('/api/hands')
def api_hands(sessioncode: str = None, limit: int = 100):
    return JSONResponse(sources.list_hands(sessioncode=sessioncode, limit=limit))


@app.get('/api/hand/{sessioncode}/{hand_id}')
def api_hand(sessioncode: str, hand_id: str):
    h = sources.get_hand(sessioncode, hand_id)
    if h is None:
        return JSONResponse({'error': 'not found'}, status_code=404)
    return JSONResponse(h)


@app.get('/api/flags')
def api_flags(limit: int = 50):
    return JSONResponse(sources.flagged_turns(limit=limit))


@app.websocket('/ws')
async def ws_live(ws: WebSocket):
    """Push the latest turn record whenever the active session's file grows, and a
    heartbeat snapshot every 10s so the client can show feed staleness."""
    await ws.accept()
    last_sig = None
    idle = 0
    try:
        while True:
            board_id, path = sources.latest_board()
            sig = (board_id, os.path.getsize(path)) if path else None
            if sig != last_sig or idle >= 10:
                last_sig, idle = sig, 0
                await ws.send_text(json.dumps(sources.live_snapshot()))
            await asyncio.sleep(1.0)
            idle += 1
    except (WebSocketDisconnect, RuntimeError):
        pass


app.mount('/static', StaticFiles(directory=STATIC), name='static')


def main():
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8765, log_level='warning')


if __name__ == '__main__':
    main()

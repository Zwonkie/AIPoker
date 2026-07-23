"""FastAPI app for the dashboard. Localhost bind only.

Transport: the /ws endpoint pushes a full live snapshot whenever the active session's
turns.jsonl grows (1s poll of file size -- the assembler will later push directly and
the poll disappears). REST endpoints serve the browse views.

Game state is READ-ONLY. The single mutation surface is /api/pilot/* (live2/webapp/
pilotctl.py): start/stop/probe of the pilot PROCESS -- the webapp is the always-on
listening service, the pilot a detached subprocess that survives webapp restarts.

Run:  .venv/Scripts/python.exe -m live2.webapp
"""
import asyncio
import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from live2.webapp import pilotctl, sources

STATIC = os.path.join(os.path.dirname(__file__), 'static')

app = FastAPI(title='PHPHelper dashboard', docs_url=None, redoc_url=None) ## owner: do not change title


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


@app.get('/api/shadow')
def api_shadow(limit: int = 12):
    return JSONResponse(sources.shadow_snapshot(limit=limit))


@app.get('/api/table_log')
def api_table_log(limit: int = 15):
    return JSONResponse(sources.table_log(limit=limit))


# ------------------------------------------------------------------ pilot control

@app.get('/api/pilot/status')
def api_pilot_status():
    st = pilotctl.status()
    st['log'] = pilotctl.log_tail(30)
    return JSONResponse(st)


@app.post('/api/pilot/start')
async def api_pilot_start(request: dict = None):
    mode = (request or {}).get('mode', 'recommend')
    return JSONResponse(pilotctl.start(mode=mode))


@app.post('/api/pilot/stop')
def api_pilot_stop():
    return JSONResponse(pilotctl.stop())


@app.post('/api/pilot/stop_all')
def api_pilot_stop_all():
    """Release everything: stop the pilot AND kill any orphaned pilot/probe processes,
    returning the box to a fresh, nothing-running state."""
    return JSONResponse(pilotctl.stop_all())


@app.post('/api/pilot/probe')
def api_pilot_probe():
    return JSONResponse(pilotctl.probe())


@app.post('/api/shadow/clear')
def api_shadow_clear():
    """Truncate the active board's shadow mirror (derived output, rebuildable)."""
    return JSONResponse(sources.clear_shadow())


@app.post('/api/flag')
def api_flag():
    """Flag the newest decided turn for review (the old F12). Annotation only -- appends
    a pointer + artifacts under the session folder, never touches game state."""
    return JSONResponse(sources.flag_latest_turn())


@app.get('/api/pilot/probe.png')
def api_pilot_probe_png():
    if not os.path.exists(pilotctl.PROBE_PNG):
        return JSONResponse({'error': 'no probe frame yet'}, status_code=404)
    return FileResponse(pilotctl.PROBE_PNG)


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

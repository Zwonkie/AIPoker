"""Pilot process control for the webapp -- the one deliberate exception to the webapp's
"no endpoint mutates state" rule, scoped to PROCESS control only (start/stop/probe the
pilot); game state stays read-only.

The webapp is the always-on listening service; the pilot runs as a DETACHED subprocess
(its own process group + console), so a webapp restart never kills a live session. The
contract with a possibly-restarted webapp is the pidfile:

  history/pilot.pid    {pid, mode, started}     written on start, removed on stop

stdout/stderr stream to history/pilot.log (truncated per start). stop() sends
CTRL_BREAK_EVENT to the pilot's process group (KeyboardInterrupt -> clean '[pilot]
stopped'), escalating to terminate after a grace period. probe() runs synchronously and
returns the captured output; the frame lands in diagnostics/pilot_probe.png.
"""
import json
import os
import signal
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HISTORY = os.path.join(REPO, 'history')
PIDFILE = os.path.join(HISTORY, 'pilot.pid')
LOGFILE = os.path.join(HISTORY, 'pilot.log')
PROBE_PNG = os.path.join(REPO, 'diagnostics', 'pilot_probe.png')

_PYTHON = os.path.join(REPO, '.venv', 'Scripts', 'python.exe')
if not os.path.exists(_PYTHON):
    _PYTHON = sys.executable


def _pid_alive(pid):
    try:
        out = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'],
                             capture_output=True, text=True, timeout=10).stdout
        return f'"{pid}"' in out
    except Exception:
        return False


def _read_pidfile():
    try:
        with open(PIDFILE, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _table_state():
    """('at-table', '1171769878') | ('waiting', None) | (None, None), from the newest
    state-transition line in the pilot log -- the pilot keeps running between tables."""
    import re
    for line in reversed(log_tail(300)):
        if '[pilot] table window:' in line:
            m = re.search(r'(\d{6,})', line)
            return 'at-table', (m.group(1) if m else None)
        if 'table gone' in line or 'waiting for a table window' in line:
            return 'waiting', None
    return None, None


def status():
    info = _read_pidfile()
    if info and _pid_alive(info.get('pid')):
        table, table_id = _table_state()
        return {'running': True, 'table': table, 'table_id': table_id, **info}
    if info:                                  # stale pidfile (crash / hard kill)
        try:
            os.remove(PIDFILE)
        except OSError:
            pass
    return {'running': False}


def log_tail(lines=40):
    try:
        with open(LOGFILE, encoding='utf-8', errors='replace') as f:
            return f.readlines()[-lines:]
    except OSError:
        return []


def start(mode='recommend'):
    """mode: 'recommend' | 'auto'. Refuses when already running."""
    st = status()
    if st['running']:
        return {'ok': False, 'error': f"pilot already running (pid {st['pid']}, {st['mode']})"}
    args = [_PYTHON, '-u', '-m', 'live2.pilot']
    if mode == 'auto':
        args.append('--auto')
    elif mode != 'recommend':
        return {'ok': False, 'error': f'unknown mode {mode!r}'}
    os.makedirs(HISTORY, exist_ok=True)
    log_f = open(LOGFILE, 'w', encoding='utf-8')
    proc = subprocess.Popen(
        args, cwd=REPO, stdout=log_f, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        # own process group (for CTRL_BREAK) + own hidden console, detached from the
        # webapp: restarting the dashboard must never take the bot down with it.
        creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                       | subprocess.CREATE_NO_WINDOW))
    info = {'pid': proc.pid, 'mode': mode,
            'started': time.strftime('%Y-%m-%dT%H:%M:%S')}
    with open(PIDFILE, 'w', encoding='utf-8') as f:
        json.dump(info, f)
    return {'ok': True, **info}


def stop(grace_s=5.0):
    st = status()
    if not st['running']:
        return {'ok': True, 'note': 'pilot was not running'}
    pid = st['pid']
    try:
        os.kill(pid, signal.CTRL_BREAK_EVENT)       # -> KeyboardInterrupt in the pilot
    except Exception:
        # Best-effort graceful nudge only. Sending a console ctrl event can fail (e.g.
        # SystemError when the caller has no attached console); the grace-period +
        # taskkill /T /F below is the real guarantee, so never let this abort the stop.
        pass
    deadline = time.time() + grace_s
    while time.time() < deadline:
        if not _pid_alive(pid):
            break
        time.sleep(0.25)
    if _pid_alive(pid):
        subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'],
                       capture_output=True, timeout=15)
    try:
        os.remove(PIDFILE)
    except OSError:
        pass
    return {'ok': True, 'stopped': pid, 'forced': _pid_alive(pid) is False and None or True}


def _all_pilot_pids():
    """Every python process whose command line runs `live2.pilot` -- the detached pilot
    AND any in-flight `--probe` child -- regardless of the pidfile. PowerShell CIM (wmic
    is gone on current Win11). Never matches the webapp itself ('live2.webapp')."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR "
          "Name='pythonw.exe'\" | Where-Object { $_.CommandLine -like '*live2.pilot*' } "
          "| Select-Object -ExpandProperty ProcessId")
    try:
        out = subprocess.run(['powershell', '-NoProfile', '-Command', ps],
                             capture_output=True, text=True, timeout=15,
                             creationflags=subprocess.CREATE_NO_WINDOW).stdout
        return [int(x) for x in out.split() if x.strip().isdigit()]
    except Exception:
        return []


def stop_all(grace_s=5.0):
    """Full release / return-to-fresh-state: stop the tracked pilot, THEN hard-kill any
    orphaned pilot-or-probe processes the pidfile doesn't know about, and clear the
    pidfile. Idempotent and safe to hit when nothing is running (reports 0 killed)."""
    tracked = stop(grace_s=grace_s)                 # pidfile pilot: CTRL_BREAK -> taskkill /T
    killed, failed = [], []
    for pid in _all_pilot_pids():
        if not _pid_alive(pid):
            continue
        try:
            subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'],
                           capture_output=True, timeout=15)
            (killed if not _pid_alive(pid) else failed).append(pid)
        except Exception:
            failed.append(pid)
    try:
        os.remove(PIDFILE)
    except OSError:
        pass
    return {'ok': not failed, 'tracked': tracked, 'orphans_killed': killed,
            'failed': failed}


def probe(timeout_s=90):
    """Synchronous one-shot capture+vision pass. Safe alongside a running pilot (both
    only read the screen)."""
    try:
        out = subprocess.run(
            [_PYTHON, '-u', '-m', 'live2.pilot', '--probe'], cwd=REPO,
            capture_output=True, text=True, timeout=timeout_s,
            creationflags=subprocess.CREATE_NO_WINDOW)
        text = (out.stdout or '') + (out.stderr or '')
        return {'ok': out.returncode == 0, 'output': text.strip().splitlines(),
                'png': os.path.exists(PROBE_PNG),
                'png_mtime': os.path.getmtime(PROBE_PNG) if os.path.exists(PROBE_PNG) else None}
    except subprocess.TimeoutExpired:
        return {'ok': False, 'output': [f'probe timed out after {timeout_s}s'], 'png': False}

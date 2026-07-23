"""Shadow replay + parity reporting: run recorded sessions through the assembler and show
exactly what it would have changed, turn by turn. This is migration gate 1's harness --
during live shadow sessions the same code runs against the tail of the live turns.jsonl.

Writes history/<board_id>/shadow_turns.jsonl (one AssembledTurn per record; the legacy
turns.jsonl is never touched) and prints a per-board summary.

Run:  .venv/Scripts/python.exe -m live2.assembler --replay Double_Or_Nothing_1171681859
      .venv/Scripts/python.exe -m live2.assembler --replay-all [--limit 10]
      .venv/Scripts/python.exe -m live2.assembler --watch        # live shadow mode
"""
import argparse
import glob
import json
import os
import time
from dataclasses import asdict

from live2.assembler.assemble import Assembler

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HISTORY = os.path.join(REPO, 'history')


def _read_records(turns_path):
    out = []
    with open(turns_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def replay_board(board_id, write=True, verbose=True):
    turns_path = os.path.join(HISTORY, board_id, 'turns.jsonl')
    if not os.path.exists(turns_path):
        print(f"{board_id}: no turns.jsonl")
        return None
    asm = Assembler(board_id)
    records = _read_records(turns_path)
    results = [asm.process_turn(r) for r in records]

    n_corr = sum(1 for r in results if r.corrections)
    n_quar = sum(1 for r in results for c in r.corrections if 'QUARANTINED' in c)
    n_contra = sum(len(r.contradictions) for r in results)
    n_price = sum(1 for r in results if r.provenance.get('call_amount') == 'derived')
    gt = len(asm.carry.hands) if asm.carry else 0
    print(f"{board_id}: {len(records)} turns | ground-truth hands {gt} | "
          f"turns with corrections {n_corr} | quarantines {n_quar} | "
          f"derived prices {n_price} | contradictions {n_contra}")
    if verbose:
        for r in results:
            for c in r.corrections:
                print(f"   turn {r.turn:>3}: {c}")
            for c in r.contradictions:
                print(f"   turn {r.turn:>3}: CONTRADICTION {c['field']}: {c}")
    if write:
        out_path = os.path.join(HISTORY, board_id, 'shadow_turns.jsonl')
        with open(out_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + '\n')
    return results


def _start_ingest_thread():
    """Run the blob ingester inside the shadow process (SPECS: ingest_watch runs as an
    assembler-managed thread during play). Keeps carry-over/roster fresh MID-SESSION and
    harvests the ephemeral TempData blobs; a failure here degrades shadow quality but must
    never kill the watch."""
    import threading

    def _loop():
        try:
            from live2.historydb import ingest_watch as iw
            db = iw.load_db()
            done = set(db['processed_hands'])
            print(f"[ingest] thread up ({len(done)} hands already in ledger)", flush=True)
            while True:
                new = sorted((h, p) for h, p in iw.find_blobs(iw.DEFAULT_ROOT).items()
                             if h not in done)
                for hid, path in new:
                    try:
                        iw.ingest(path, db)
                    except Exception as e:
                        print(f"[ingest] ! {hid}: {e} (will retry)", flush=True)
                        continue
                    done.add(hid)
                    db['processed_hands'].append(hid)
                if new:
                    iw.save_db(db)
                time.sleep(2.0)
        except Exception as e:
            print(f"[ingest] thread DOWN: {e} -- shadow continues without live carry-over refresh",
                  flush=True)

    threading.Thread(target=_loop, daemon=True, name='ingest').start()


def watch(poll=1.0):
    """Live shadow mode: follow whichever board's turns.jsonl is newest, run each new turn
    through the assembler as it lands, append to shadow_turns.jsonl, print corrections and
    contradictions as they happen. Survives session changes (a new board dir takes over).
    This is the process that runs alongside PHPHelp during the gate-1 shadow sessions.
    Runs the blob ingester as an internal thread so carry-over stays current in-session."""
    _start_ingest_thread()
    print("assembler shadow watch: waiting for live turns ...", flush=True)
    cur_board, asm, offset, out_f = None, None, 0, None
    while True:
        dirs = [d for d in glob.glob(os.path.join(HISTORY, '*'))
                if os.path.isdir(d) and os.path.exists(os.path.join(d, 'turns.jsonl'))]
        if dirs:
            newest = os.path.basename(max(dirs, key=lambda d: os.path.getmtime(
                os.path.join(d, 'turns.jsonl'))))
            if newest != cur_board:
                if out_f:
                    out_f.close()
                cur_board, asm, offset = newest, Assembler(newest), 0
                # 'w' not 'a': we re-process the board from offset 0, so truncate --
                # appending on watcher restart duplicated every already-shadowed turn.
                out_f = open(os.path.join(HISTORY, newest, 'shadow_turns.jsonl'),
                             'w', encoding='utf-8')
                print(f"[shadow] following {newest}", flush=True)
            path = os.path.join(HISTORY, cur_board, 'turns.jsonl')
            try:
                size = os.path.getsize(path)
                if size > offset:
                    with open(path, encoding='utf-8') as f:
                        f.seek(offset)
                        chunk = f.read()
                    consumed = 0
                    for line in chunk.splitlines(keepends=True):
                        if not line.endswith('\n'):
                            break              # partial write in flight -- next poll
                        consumed += len(line.encode('utf-8'))
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        r = asm.process_turn(rec)
                        out_f.write(json.dumps(asdict(r), ensure_ascii=False) + '\n')
                        out_f.flush()
                        for c in r.corrections:
                            print(f"[shadow] turn {r.turn}: {c}", flush=True)
                        for c in r.contradictions:
                            print(f"[shadow] turn {r.turn}: CONTRADICTION {c}", flush=True)
                    offset += consumed
            except OSError:
                pass
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replay', help='board_id to replay')
    ap.add_argument('--replay-all', action='store_true')
    ap.add_argument('--watch', action='store_true', help='live shadow mode (tails newest board)')
    ap.add_argument('--limit', type=int, default=10, help='newest N boards for --replay-all')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    if args.replay:
        replay_board(args.replay, verbose=not args.quiet)
    elif args.replay_all:
        dirs = [d for d in glob.glob(os.path.join(HISTORY, '*'))
                if os.path.isdir(d) and os.path.exists(os.path.join(d, 'turns.jsonl'))]
        dirs.sort(key=os.path.getmtime, reverse=True)
        for d in dirs[:args.limit]:
            replay_board(os.path.basename(d), verbose=not args.quiet)
    elif args.watch:
        watch()
    else:
        ap.print_help()


if __name__ == '__main__':
    main()

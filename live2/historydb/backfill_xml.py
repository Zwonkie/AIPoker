"""Backfill the hand store from every persisted session XML.

Walks History\\Data\\Tournaments\\*.xml, parses each session (parse_xml.py), and appends any
hand not already in the ledger to history/handhistory/<sessioncode>/hands.jsonl. Hands
already ingested from the richer protobuf blobs (same hand_id in the ledger) are skipped --
blob provenance wins.

Run:  .venv/Scripts/python.exe tools/handhistory/backfill_xml.py
      [--dir "C:/.../History/Data/Tournaments"]
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from live2.historydb.parse_xml import parse_session          # noqa: E402
from live2.historydb.ingest_watch import load_db, save_db, OUT_DIR  # noqa: E402

DEFAULT_DIR = r"C:\Users\zwonk\AppData\Local\Poker at Bet365.DK\data\Zwonkie\History\Data\Tournaments"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=DEFAULT_DIR)
    args = ap.parse_args()

    db = load_db()
    done = set(db['processed_hands'])
    added = skipped = failed = 0
    files = sorted(glob.glob(os.path.join(args.dir, '*.xml')))
    for i, path in enumerate(files):
        try:
            hands = parse_session(path)
        except Exception as e:
            print(f"! {os.path.basename(path)}: {e}")
            failed += 1
            continue
        by_sess = {}
        for h in hands:
            hid = str(h['hand_id'])
            if hid in done:
                skipped += 1
                continue
            by_sess.setdefault(str(h['game_id']), []).append(h)
            done.add(hid)
            db['processed_hands'].append(hid)
            added += 1
        for sess, hs in by_sess.items():
            os.makedirs(os.path.join(OUT_DIR, sess), exist_ok=True)
            with open(os.path.join(OUT_DIR, sess, 'hands.jsonl'), 'a', encoding='utf-8') as f:
                for h in hs:
                    f.write(json.dumps(h, ensure_ascii=False) + '\n')
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(files)} sessions ({added} hands added)")
    save_db(db)
    print(f"backfill done: {added} hands added, {skipped} already ingested, "
          f"{failed} files failed, {len(files)} sessions scanned")
    if added:
        from live2.historydb import sqlindex
        print(f"sqlindex rebuilt: {sqlindex.rebuild()} hands")   # bulk op -> full rebuild is simplest


if __name__ == '__main__':
    main()

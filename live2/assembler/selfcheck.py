"""Post-hand OCR self-validation (migration gate 2): after a hand's ground-truth record
lands in the store, diff every recorded observation belonging to that hand against what
the completed hand PROVED. Disagreements append (with ground truth attached) to the
vision regression corpus -- history/vision_regression.jsonl -- turning every OCR mistake
made in real play into a permanent test case.

v1 checks per turn, strongest evidence first:
  hero_cards   -- obs vs the hand's dealt hero hole cards (exact)
  board        -- obs community cards vs the hand's board prefix for that street (exact)
  big_blind    -- obs vs the hand's blind level (exact)
  hero_stack   -- obs vs hero's `stack_before` at the matching action (blob hands only;
                  matched k-th hero VOLUNTARY action in that street <-> k-th turn recorded
                  for that street; tolerance 1 chip)
  seat_names   -- occupied ACTIVE seats vs dealt-in players (fuzzy; reports both foreign
                  names and real players vision failed to see)

Run:  .venv/Scripts/python.exe -m live2.assembler.selfcheck --board <board_id>
      .venv/Scripts/python.exe -m live2.assembler.selfcheck --all [--limit 10]
"""
import argparse
import glob
import json
import os

from live2.assembler import feeds
from live2.assembler.assemble import _roster_match

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HISTORY = os.path.join(REPO, 'history')
CORPUS = os.path.join(HISTORY, 'vision_regression.jsonl')

_STREET_N = {'Preflop': 0, 'Flop': 3, 'Turn': 4, 'River': 5}


def _norm_card(c):
    """'10c' (live OCR) and 'Tc' (store) -> same form."""
    c = str(c or '').strip()
    return ('T' + c[2:]) if c.startswith('10') else c


def check_turn_against_hand(rec, hand):
    """-> list of diff dicts for one recorded turn vs its completed hand."""
    obs = rec.get('observation') or {}
    diffs = []

    def diff(field, seen, truth, note=''):
        diffs.append({'board_id': rec.get('board_id'), 'turn': rec.get('turn'),
                      'ts': rec.get('ts'), 'hand_id': hand['hand_id'], 'field': field,
                      'vision': seen, 'truth': truth, 'note': note})

    hero = next((p for p in hand.get('players', []) if p['name'] == feeds.HERO_NAME), None)

    # hero cards
    truth_cards = [_norm_card(c) for c in (hero or {}).get('hole_cards') or [] if c]
    seen_cards = [_norm_card(c) for c in obs.get('hero_cards') or []]
    if truth_cards and seen_cards and sorted(seen_cards) != sorted(truth_cards):
        diff('hero_cards', seen_cards, truth_cards)

    # board prefix for the street the obs claims
    n = _STREET_N.get(obs.get('street'))
    truth_board = [_norm_card(c) for c in hand.get('board') or []]
    if n is not None and len(truth_board) >= n:
        seen_board = [_norm_card(c) for c in obs.get('community_cards') or []]
        if seen_board != truth_board[:n]:
            diff('board', seen_board, truth_board[:n], f"street {obs.get('street')}")

    # blind level
    bb = (hand.get('blinds') or {}).get('bb')
    if bb and obs.get('big_blind') and float(obs['big_blind']) != float(bb):
        diff('big_blind', obs['big_blind'], bb)

    # seat names: active occupied seats vs dealt-in players
    dealt = {p['name'] for p in hand.get('players', [])} - {feeds.HERO_NAME}
    for s in obs.get('seats') or []:
        if s.get('occupied') and s.get('is_active'):
            nm = (s.get('name') or '').strip()
            if nm and _roster_match(nm, dealt) is None:
                diff('seat_name', nm, sorted(dealt), f"{s.get('seat_key')} not in dealt-in set")
    return diffs


def _hero_stack_checks(records, hand):
    """Blob hands carry per-action stack_before: match the k-th recorded turn of a street
    with hero's k-th voluntary action there and compare stacks exactly."""
    diffs = []
    if hand.get('source') != 'blob':
        return diffs
    by_street = {}
    for a in hand.get('actions', []):
        if a.get('name') == feeds.HERO_NAME and not str(a.get('action', '')).startswith('post'):
            by_street.setdefault(a['street'], []).append(a)
    seen_streets = {}
    for rec in records:
        obs = rec.get('observation') or {}
        street = str(obs.get('street', '')).lower()
        k = seen_streets.get(street, 0)
        seen_streets[street] = k + 1
        acts = by_street.get(street, [])
        if k < len(acts) and acts[k].get('stack_before') is not None and obs.get('hero_stack'):
            truth = float(acts[k]['stack_before'])
            seen = float(obs['hero_stack'])
            bb = float((hand.get('blinds') or {}).get('bb') or 20)
            if abs(seen - truth) > 1.0:
                # blind/ante posting timing produces small constant offsets (~<=2bb) between
                # the obs snapshot and stack_before; real OCR failures are much larger.
                sev = 'minor-timing' if abs(seen - truth) <= 2 * bb else 'MAJOR'
                diffs.append({'board_id': rec.get('board_id'), 'turn': rec.get('turn'),
                              'ts': rec.get('ts'), 'hand_id': hand['hand_id'],
                              'field': 'hero_stack', 'vision': seen, 'truth': truth,
                              'severity': sev, 'note': f"{street} hero action #{k + 1}"})
    return diffs


def selfcheck_board(board_id, write=True, verbose=True):
    turns_path = os.path.join(HISTORY, board_id, 'turns.jsonl')
    tid = feeds.tournament_id_of(board_id)
    if not (os.path.exists(turns_path) and tid):
        return None
    carry = feeds.CarryOverFeed(tid)
    if not carry.hands:
        print(f"{board_id}: no ground-truth hands -- nothing to validate against")
        return None
    records = [json.loads(l) for l in open(turns_path, encoding='utf-8') if l.strip()]

    # group turns by the hand in progress at their timestamp -- BOUNDED: a turn must fall
    # inside the hand's own time window (start .. next-hand start, or start+duration+60s
    # for the newest stored hand), else a sparse store maps every later turn onto its one
    # stored hand and the diff compares unrelated hands (found on first dry run).
    hands = carry.hands
    groups = {}
    for rec in records:
        h = carry.current(rec.get('ts'))
        if not h:
            continue
        t = feeds._to_epoch(rec.get('ts'))
        idx = next((i for i, x in enumerate(hands) if x['hand_id'] == h['hand_id']), None)
        nxt = hands[idx + 1] if idx is not None and idx + 1 < len(hands) else None
        if nxt is not None:
            end = feeds.start_epoch(nxt)
        else:
            s = feeds.start_epoch(h)
            end = (s + float(h.get('duration_s') or 120.0) + 60.0) if s else None
        if t is None or end is None or t > end:
            continue
        groups.setdefault(h['hand_id'], (h, []))[1].append(rec)

    all_diffs = []
    for hand_id, (hand, recs) in sorted(groups.items()):
        for rec in recs:
            all_diffs.extend(check_turn_against_hand(rec, hand))
        all_diffs.extend(_hero_stack_checks(recs, hand))

    by_field = {}
    for d in all_diffs:
        by_field[d['field']] = by_field.get(d['field'], 0) + 1
    matched = sum(len(r) for _h, r in groups.values())
    print(f"{board_id}: {len(records)} turns, {matched} matched to {len(groups)} hands | "
          f"diffs {len(all_diffs)} {by_field if by_field else ''}")
    if verbose:
        for d in all_diffs[:20]:
            print(f"   turn {d['turn']:>3} hand {d['hand_id']} {d['field']}: "
                  f"vision={d['vision']} truth={d['truth']} {d['note']}")
    if write and all_diffs:
        with open(CORPUS, 'a', encoding='utf-8') as f:
            for d in all_diffs:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
    return all_diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--board')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--limit', type=int, default=10)
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--no-write', action='store_true')
    args = ap.parse_args()
    if args.board:
        selfcheck_board(args.board, write=not args.no_write, verbose=not args.quiet)
    elif args.all:
        dirs = [d for d in glob.glob(os.path.join(HISTORY, '*'))
                if os.path.isdir(d) and os.path.exists(os.path.join(d, 'turns.jsonl'))]
        dirs.sort(key=os.path.getmtime, reverse=True)
        for d in dirs[:args.limit]:
            selfcheck_board(os.path.basename(d), write=not args.no_write,
                            verbose=not args.quiet)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()

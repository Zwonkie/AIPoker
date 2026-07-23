"""Read-only feed accessors for the assembler: carry-over ground truth (hand store) and
opponent profiles (windowed stats). Vision arrives as turns.jsonl records via shadow.py.

Carry-over lookup is BY TOURNAMENT: the live board_id's numeric suffix is the client's
tournament id, and every decoded hand carries `tournament_id` -- that join is what lets
the previous completed hand of THIS SAME table vouch for roster/stacks/blinds."""
import glob
import json
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HANDSTORE = os.path.join(REPO, 'history', 'handhistory')

HERO_NAME = 'Zwonkie'


def _to_epoch(ts):
    """Turn ts / start_local string ('YYYY-MM-DD[ T]HH:MM:SS', local) -> epoch seconds."""
    from datetime import datetime
    try:
        return datetime.strptime(str(ts).replace('T', ' '), '%Y-%m-%d %H:%M:%S').timestamp()
    except (ValueError, TypeError):
        return None


def start_epoch(hand):
    """Hand start as epoch seconds: blob records carry start_utc_ms (wire ms-epoch), XML
    records carry start_local. None when neither parses."""
    if hand.get('start_utc_ms'):
        return hand['start_utc_ms'] / 1000.0
    return _to_epoch(hand.get('start_local'))


def tournament_id_of(board_id):
    """'Double_Or_Nothing_1171681859' -> 1171681859; None if no numeric suffix."""
    m = re.search(r'_(\d+)$', str(board_id))
    return int(m.group(1)) if m else None


class CarryOverFeed:
    """Hands of one tournament, ordered by hand_id. `latest_before(ts_epoch)` returns the
    newest COMPLETED hand whose record the client had written before the given moment --
    replay-correct: a live assembler at turn time can only know hands already on disk.
    (v1 approximation: records carry no write-time, so we use start_local ordering and the
    one-hand-behind rule -- the hand in progress is never in the store.)"""

    def __init__(self, tournament_id):
        self.tournament_id = tournament_id
        self.hands = []
        self._mtimes = {}
        self.refresh()

    def refresh(self):
        changed = False
        for p in glob.glob(os.path.join(HANDSTORE, '*', 'hands.jsonl')):
            mt = os.path.getmtime(p)
            if self._mtimes.get(p) == mt:
                continue
            self._mtimes[p] = mt
            changed = True
        if not (changed or not self.hands):
            return
        hands = []
        for p in self._mtimes:
            try:
                with open(p, encoding='utf-8') as f:
                    for line in f:
                        try:
                            h = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if h.get('tournament_id') == self.tournament_id:
                            hands.append(h)
            except OSError:
                continue
        # blob + xml duplicates share hand_id -- keep the richer blob record
        by_id = {}
        for h in sorted(hands, key=lambda x: (x['hand_id'], x.get('source') == 'blob')):
            by_id[h['hand_id']] = h
        self.hands = [by_id[k] for k in sorted(by_id)]

    def latest(self, before_ts=None):
        """Newest hand COMPLETED before `before_ts` ('YYYY-MM-DD[ T]HH:MM:SS' local).

        Live mode (before_ts=None): the store physically contains only completed hands
        (records are written when the cycle ends), so the newest record is correct.
        Replay mode: the whole tournament is on disk, so a turn must not see hands from
        its own future. Hand i counts as completed by T when hand i+1 had already
        STARTED before T (the client deals the next hand immediately); the tournament's
        final hand is completed only for turns after its start + a settle margin."""
        if before_ts is None:
            return self.hands[-1] if self.hands else None
        t = _to_epoch(before_ts)
        if t is None:
            return None
        completed = None
        for i, h in enumerate(self.hands):
            nxt = self.hands[i + 1] if i + 1 < len(self.hands) else None
            if nxt is not None:
                ns = start_epoch(nxt)
                if ns is not None and ns <= t:
                    completed = h
            else:
                s = start_epoch(h)
                if s is not None and s + 90 <= t:
                    completed = h
        return completed

    def current(self, at_ts):
        """The hand IN PROGRESS at `at_ts` (replay adjudication only -- live code must
        never use this; that hand is unknowable live). Newest hand started before at_ts."""
        t = _to_epoch(at_ts)
        if t is None:
            return None
        cur = None
        for h in self.hands:
            s = start_epoch(h)
            if s is not None and s <= t:
                cur = h
        return cur

    def roster(self, before_ts=None):
        """Names seated in hands COMPLETED before `before_ts` (None -> all). Aligned like
        latest(): replay must not learn the roster from a turn's future, and live code
        genuinely has no roster until the first hand completes."""
        cutoff = self.latest(before_ts=before_ts) if before_ts is not None else (
            self.hands[-1] if self.hands else None)
        names = set()
        if cutoff is None:
            return names
        for h in self.hands:
            for p in h.get('players', []):
                names.add(p['name'])
            if h is cutoff:
                break
        return names


_PROFILES_CACHE = None


def opponent_profiles(window=100, min_hands=10):
    """{lower-cased name: profile} from the windowed stats engine. Cached per process."""
    global _PROFILES_CACHE
    if _PROFILES_CACHE is None:
        from live2.historydb import stats
        built, _total = stats.build(window=window, min_hands=min_hands)
        _PROFILES_CACHE = {rec['name'].lower(): rec for rec in built.values()}
    return _PROFILES_CACHE

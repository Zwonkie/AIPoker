"""The fusion core. `Assembler.process_turn(record)` takes one recorded vision observation
(the LiveObservation dict inside a turns.jsonl record) and returns an AssembledTurn:

  observation    -- corrected LiveObservation dict (schema-compatible, replayable)
  provenance     -- {field: source} for every field the assembler changed
                    ('carry-over' | 'derived' | 'quarantine' | 'opponent-db')
  contradictions -- feed disagreements that were NOT auto-resolved, each with both values
  corrections    -- human-readable list of what changed and why

Rules (v1), in order of trust:
0. SEAT MAP (carry-over, PRIMARY once one hand is stored): ground truth records every
   player's ABSOLUTE seat id (hero included), and the client draws the ring rotated so
   hero sits at the bottom -- so screen slot seat_k = k-th ring seat clockwise
   (ascending id, circular) from hero. Validated against 958 clean OCR reads across 8
   tournaments with zero true disagreements (all 65 "mismatches" were OCR typos of the
   expected name). Names come FROM THE MAP; OCR is demoted to bootstrap (rule 1, first
   hand only) + a cross-check that surfaces a contradiction if a clean read ever names a
   DIFFERENT roster player than the map expects. A slot whose expected occupant BUSTED
   (final_stack 0, or absent from the last completed hand) is force-vacated -- an
   occupied read there is a stale-pod misread.
1. ROSTER (carry-over, tournament play): the field is fixed once the tournament starts.
   An occupied seat whose OCR name matches nothing in the roster is handled by SEAT
   IDENTITY STICKINESS (first replay of the JJ session showed why both directions are
   needed):
     - seat has an ESTABLISHED identity (its name roster-matched on an earlier turn):
       a garbage/timer read there is an OCR failure over a REAL player -> REPAIR the
       name to the last-known identity, keep the seat. Dropping it would delete a real
       opponent from the equity field ("new false fact", forbidden by the gate).
     - seat has NO established identity: a timer-pattern ('Tid: 18') or foreign name
       first-registering it is a PHANTOM -> QUARANTINE (occupied=False, not counted).
       This is the JJ-fold killer, and mirrors the 2026-07-22 live hotfix.
   Fuzzy matching absorbs OCR noise before a name is declared foreign.
2. STACKS: vision is live truth mid-hand, but an UNREAD stack (0.0 on an active seat) is
   filled from the carry-over final stack minus this hand's committed chips.
2b. HERO STACK/POT SANITY: hero_committed > pot_size is impossible (hero's committed
   chips ARE in the pot; pot display includes current-street bets). Real causes seen in
   shadow session #1: pot misread, hero stack misread absorbed by the committed tracker
   (970 read as 380, 720 as 15), or the tracker carrying corrupt state across a hand
   boundary. Flagged as a composite contradiction with all evidence attached (in-stream
   attribution is unreliable); no guessed repair.
3. PRICE: an unknown call_amount (call_amount_known=False) is DERIVED from the street bet
   level minus hero's committed chips (floored at the big blind when facing a bet) --
   never encoded as 'free' (the JJ-fold price bug). call_amount_known stays False:
   downstream consumers keep seeing an estimate, not a read.
4. Blinds: carry-over blinds that disagree with vision are surfaced as a contradiction
   (level changes are real; vision wins, but the disagreement is visible).
Nothing else is touched: every unlisted field passes through as vision reported it.
"""
import difflib
import re
from dataclasses import dataclass, field

from live2.assembler import feeds

_TIMER_RE = re.compile(r'(?i)^tid\s*[:.]?\s*\d*$')
_PLAUSIBLE_RE = re.compile(r'^[\w .\-\[\]]{2,20}$')   # one line, name-shaped -- not OCR garbage
_FUZZY_MIN = 0.72


@dataclass
class AssembledTurn:
    turn: int
    observation: dict
    provenance: dict = field(default_factory=dict)
    contradictions: list = field(default_factory=list)
    corrections: list = field(default_factory=list)


def _roster_match(name, roster):
    """Exact -> case-insensitive -> fuzzy. Returns the roster name or None."""
    if not name:
        return None
    if name in roster:
        return name
    low = {r.lower(): r for r in roster}
    if name.lower() in low:
        return low[name.lower()]
    best, best_r = None, 0.0
    for r in roster:
        ratio = difflib.SequenceMatcher(None, name.lower(), r.lower()).ratio()
        if ratio > best_r:
            best, best_r = r, ratio
    return best if best_r >= _FUZZY_MIN else None


class Assembler:
    def __init__(self, board_id):
        self.board_id = board_id
        self.tournament_id = feeds.tournament_id_of(board_id)
        self.carry = feeds.CarryOverFeed(self.tournament_id) if self.tournament_id else None
        self.profiles = feeds.opponent_profiles()
        self.seat_identity = {}     # seat_key -> last roster-matched name (sticky)
        self._map_cache = None      # (last_hand_id, seat map) -- see _seat_map

    def _seat_map(self, last_hand):
        """{seat_key: {'name', 'alive'}} for screen slots, from ground truth (rule 0).
        Ring/ownership accumulate over hands up to last_hand (no future leakage in
        replay); aliveness comes from last_hand itself. {} until a hand is stored."""
        if not (self.carry and last_hand):
            return {}
        if self._map_cache and self._map_cache[0] == last_hand.get('hand_id'):
            return self._map_cache[1]
        ring, hero_seat, owner = set(), None, {}
        for h in self.carry.hands:
            for p in h.get('players', []):
                sid = p.get('seat')
                if sid is None:
                    continue
                ring.add(sid)
                owner[sid] = p['name']
                if p['name'] == feeds.HERO_NAME:
                    hero_seat = sid
            if h.get('hand_id') == last_hand.get('hand_id'):
                break
        if hero_seat is None or len(ring) < 2:
            return {}
        in_last = {p['name']: p for p in last_hand.get('players', [])}
        ring = sorted(ring)
        hi = ring.index(hero_seat)
        out = {}
        for j in range(1, len(ring)):
            name = owner[ring[(hi + j) % len(ring)]]
            p = in_last.get(name)
            alive = p is not None and (p.get('final_stack') is None
                                       or float(p['final_stack']) > 0)
            out[f'seat_{j}'] = {'name': name, 'alive': alive}
        self._map_cache = (last_hand.get('hand_id'), out)
        return out

    def process_turn(self, record):
        obs = dict(record.get('observation') or {})
        out = AssembledTurn(turn=record.get('turn'), observation=obs)
        if not obs:
            return out
        if self.carry:
            self.carry.refresh()
        turn_ts = record.get('ts')
        last_hand = self.carry.latest(before_ts=turn_ts) if self.carry else None
        # Roster is aligned to completed hands and AUTHORITATIVE only once one exists --
        # before that (or with no ground truth at all) we run vision-trust mode: plausible
        # names establish unverified identities, foreign-name quarantine stands down.
        roster = self.carry.roster(before_ts=turn_ts) if self.carry else set()
        roster_authoritative = last_hand is not None and len(roster) > 1
        roster.add(feeds.HERO_NAME)

        seats = [dict(s) for s in (obs.get('seats') or [])]
        final_stacks = {}
        if last_hand:
            for p in last_hand.get('players', []):
                if p.get('final_stack') is not None:
                    final_stacks[p['name']] = p['final_stack']

        # -- rule 0: seat map from ground truth (primary once a hand is stored) --
        seat_map = self._seat_map(last_hand)
        mapped_keys = set()
        if seat_map:
            for s in seats:
                if not s.get('occupied'):
                    continue
                key = s['seat_key']
                nm = (s.get('name') or '').strip()
                exp = seat_map.get(key)
                if exp is None:
                    # authoritative map has NO ring seat for this slot -> phantom pod
                    self._quarantine(out, s, f"no ring seat maps to this slot ({nm[:24]!r})")
                    mapped_keys.add(key)
                    continue
                if not exp['alive']:
                    self._quarantine(
                        out, s, f"mapped occupant {exp['name']!r} busted; stale pod read "
                                f"({nm[:24]!r})")
                    mapped_keys.add(key)
                    continue
                # cross-check: a CLEAN read naming a different roster player than the map
                # expects would mean the rotation is off -- surface loudly, keep the map.
                if nm and _roster_match(nm, roster - {exp['name']}) and \
                        _roster_match(nm, {exp['name']}) is None:
                    out.contradictions.append({
                        'field': f'{key}.name', 'vision': nm, 'seat_map': exp['name'],
                        'note': 'clean OCR names a different roster player than the '
                                'seat map -- check ring rotation'})
                if nm != exp['name']:
                    s['name'] = exp['name']
                    out.provenance[f'{key}.name'] = 'carry-over'
                    if _roster_match(nm, {exp['name']}) is None:
                        out.corrections.append(
                            f"{key}: name {nm[:24]!r} set from seat map -> {exp['name']!r}")
                self.seat_identity[key] = exp['name']
                mapped_keys.add(key)

        # -- rule 1: roster + sticky seat identity (bootstrap: no stored hand yet) --
        # pass A: resolve what can be resolved directly (roster match, vision-trust
        # plausible name, sticky identity); collect the rest for pass B.
        assigned = set()
        unresolved = []
        for s in seats:
            if not s.get('occupied') or s['seat_key'] in mapped_keys:
                continue
            key = s['seat_key']
            nm = (s.get('name') or '').strip()
            is_timer = bool(_TIMER_RE.match(nm))
            hit = _roster_match(nm, roster) if roster_authoritative and not is_timer else None
            if hit:
                self.seat_identity[key] = hit        # identity (re)established, verified
                assigned.add(hit)
                if hit != nm:
                    s['name'] = hit
                    out.provenance[f"{key}.name"] = 'carry-over'
                    out.corrections.append(f"{key}: OCR name {nm!r} -> roster {hit!r}")
                continue
            if not roster_authoritative and not is_timer and _PLAUSIBLE_RE.match(nm):
                self.seat_identity[key] = nm         # unverified identity (vision-trust mode)
                assigned.add(nm)
                continue
            known = self.seat_identity.get(key)
            if known and last_hand:
                # sticky identity EXPIRES on proven bust: if carry-over shows this player
                # finished a completed hand with 0 chips (or vanished from the dealt-in
                # set while the roster is authoritative), the seat is vacated -- keeping
                # the repair would hold a dead player in the equity field. (Found on the
                # first pilot session: orelno27 busted, the vacated seat kept OCR-reading
                # 'f'+occupied, and a continuous sticky map would have repaired it back
                # to orelno27 for 50+ turns.)
                p = next((x for x in last_hand.get('players', []) if x['name'] == known), None)
                busted = ((p is not None and p.get('final_stack') is not None
                           and float(p['final_stack']) <= 0)
                          or (p is None and roster_authoritative))
                if busted:
                    self.seat_identity.pop(key, None)
                    known = None
            if known:
                # established seat, unreadable/garbage/timer name -> OCR failure over a
                # REAL player: repair, never drop (dropping = new false fact).
                s['name'] = known
                assigned.add(known)
                out.provenance[f"{key}.name"] = 'sticky-identity'
                out.corrections.append(
                    f"{key}: unreadable name {nm[:24]!r} repaired to known occupant {known!r}")
            else:
                unresolved.append((s, nm, is_timer))

        # pass B: roster deduction before any quarantine. If EXACTLY ONE occupied seat is
        # unresolved and EXACTLY ONE not-busted player from the last completed hand is
        # unaccounted for, that player must be this seat -- repair. (Shadow session #3
        # showed the gap: seat_3 read as 'f' from the first turn had no sticky identity,
        # so it was quarantined every turn while the roster plainly had one member with
        # no seat. The revised JJ adjudication -- 'Tid: 18' WAS real occupant Paul6969 --
        # is the same shape. A full-accounted roster still quarantines phantoms: zero
        # candidates means the extra seat is not real.)
        if unresolved and last_hand:
            live = {p['name'] for p in last_hand.get('players', [])
                    if p.get('final_stack') is None or float(p['final_stack']) > 0}
            candidates = live - assigned - {feeds.HERO_NAME}
            if len(unresolved) == 1 and len(candidates) == 1:
                s, nm, _t = unresolved.pop()
                name = candidates.pop()
                s['name'] = name
                self.seat_identity[s['seat_key']] = name
                assigned.add(name)
                out.provenance[f"{s['seat_key']}.name"] = 'carry-over'
                out.corrections.append(
                    f"{s['seat_key']}: unreadable name {nm[:24]!r} deduced to be {name!r} "
                    f"(only unassigned live roster player)")
        for s, nm, is_timer in unresolved:
            if is_timer:
                self._quarantine(out, s, f"timer pattern first-registering seat ({nm!r})")
            elif roster_authoritative:
                self._quarantine(out, s, f"name {nm[:24]!r} not in tournament roster "
                                         f"({len(roster)} known players)")

        # -- rule 2: unread stacks filled from carry-over ------------------------
        for s in seats:
            if s.get('occupied') and s.get('is_active') and not s.get('stack'):
                base = final_stacks.get(s.get('name'))
                if base is not None:
                    filled = max(0.0, float(base) - float(s.get('committed') or 0.0))
                    s['stack'] = filled
                    out.provenance[f"{s['seat_key']}.stack"] = 'carry-over'
                    out.corrections.append(
                        f"{s['seat_key']}: unread stack filled {filled:.0f} "
                        f"(prev-hand final {base} - committed {s.get('committed', 0)})")

        # -- rule 2b: hero stack/pot sanity --------------------------------------
        # hero_committed > pot_size is impossible: hero's committed chips ARE in the pot
        # (verified: this table's pot display includes current-street bets). Three real
        # upstream causes seen in shadow data: pot misread (often 0), hero stack misread
        # absorbed by the committed tracker (970 read as 380, 720 as 15), or the tracker
        # carrying corrupt state across a hand boundary (committed frozen at 1040 on a
        # fresh preflop). In-stream attribution is NOT reliable -- stack+committed always
        # reconstructs the tracker's own start by construction -- so flag the composite
        # state with all evidence attached and let ground-truth adjudication classify.
        # Only pot==0 with chips committed is unambiguous. Flag only, no guessed repair.
        committed = float(obs.get('hero_committed') or 0.0)
        pot = float(obs.get('pot_size') or 0.0)
        if committed > pot + 1.0:
            suspect = 'pot_size' if pot == 0.0 else 'pot_vs_committed'
            out.contradictions.append({
                'field': suspect, 'hero_stack': obs.get('hero_stack'),
                'hero_committed': committed, 'pot_size': pot,
                'carry_over_prev_final': final_stacks.get(feeds.HERO_NAME),
                'note': 'hero_committed exceeds pot (impossible) -- pot misread, hero '
                        'stack misread, or committed-tracker corruption upstream'})
            out.provenance[suspect] = 'contradiction'

        obs['seats'] = seats
        occupied = [s for s in seats if s.get('occupied')]
        new_count = len(occupied) + 1                # + hero
        if new_count != obs.get('seated_count'):
            out.provenance['seated_count'] = 'quarantine'
            out.corrections.append(f"seated_count {obs.get('seated_count')} -> {new_count}")
            obs['seated_count'] = new_count

        # -- rule 3: unknown price derived, never 'free' -------------------------
        if not obs.get('call_amount_known'):
            level = float(obs.get('current_street_bet_level') or 0.0)
            hero_in = float(obs.get('hero_committed') or 0.0)
            derived = max(0.0, level - hero_in)
            if derived == 0.0 and not obs.get('check_call_available', True):
                derived = float(obs.get('big_blind') or 0.0)   # facing SOMETHING unreadable
            if obs.get('call_amount') != derived:
                out.contradictions.append({
                    'field': 'call_amount', 'vision': obs.get('call_amount'),
                    'derived': derived,
                    'note': 'price not positively read; derived from street bet level'})
            obs['call_amount'] = derived
            out.provenance['call_amount'] = 'derived'

        # -- rule 4: blinds cross-check ------------------------------------------
        if last_hand:
            cb = (last_hand.get('blinds') or {}).get('bb')
            if cb and obs.get('big_blind') and float(cb) != float(obs['big_blind']):
                out.contradictions.append({
                    'field': 'big_blind', 'vision': obs['big_blind'], 'carry_over': cb,
                    'note': 'level change or misread; vision kept'})

        return out

    @staticmethod
    def _quarantine(out, seat, reason):
        seat['occupied'] = False
        seat['is_active'] = False
        out.provenance[f"{seat['seat_key']}.occupied"] = 'quarantine'
        out.corrections.append(f"{seat['seat_key']} QUARANTINED: {reason}")

"""Turn recording for the pilot: same two-layer format-2 schema the legacy recorder wrote
(history/<board_id>/turns.jsonl -- observation / board_state / evaluation / action), plus
an 'assembler' layer (corrections, contradictions, provenance) since the pilot decides
from the CORRECTED observation. The webapp, selfcheck, and replay tooling consume these
records unchanged.

The stored 'observation' is the corrected one -- the record's contract is "what the
adapter decided from", and with the assembler in the decision path that is the corrected
state. When the assembler changed anything, the raw vision dict is preserved under
'observation_raw' so OCR regressions remain measurable against ground truth.
"""
import datetime
import json
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HISTORY = os.path.join(REPO, 'history')

_STREETS = {0: 'Preflop', 3: 'Flop', 4: 'Turn', 5: 'River'}
_ACTION_DIAG_ORDER = ('FOLD', 'CALL', 'RAISE', 'RAISE_33', 'RAISE_66', 'RAISE_POT', 'ALLIN')


def board_id_from_title(title):
    """Same rule as the legacy recorder: anchor on the >=6-digit tournament id so stakes /
    level / client-version churn in the title never forks the session folder."""
    if not title:
        return 'unknown'
    nums = re.findall(r'\d{6,}', title)
    if nums:
        match_id = max(nums, key=len)
        prefix = re.sub(r'[^A-Za-z]+', '_', re.match(r'\D*', title).group(0)).strip('_')[:30]
        return f"{prefix}_{match_id}".strip('_') if prefix else match_id
    s = re.sub(r'(?:€|\$|£)?\d+(?:\.\d+)?\s*/\s*(?:€|\$|£)?\d+(?:\.\d+)?', '', title)
    s = re.sub(r'[^A-Za-z0-9._-]+', '_', s).strip('_')
    return s[:60] or 'table'


def decode_model_input(ev, engine):
    """The scalars the model ACTUALLY consumed, decoded from its recorded input tensor with
    the active contract's own scales (see legacy _decode_model_input for the two incidents
    that justify never hand-maintaining these). {} if no tensor was captured."""
    try:
        last = (ev or {}).get('model_input', {}).get('ctx')[0][-1]
        scales = engine.context_scales()
        return {
            'position': round(last[0] * 5.0, 2),
            'hero_stack_bb': round(last[1] * scales['stack'], 1),
            'pot_bb': round(last[2] * scales['pot'], 2),
            'equity': round(last[3], 3),
            'pot_odds': round(last[4], 3),
            'to_call_bb': round(last[9] * scales['call'], 2),
        }
    except Exception:
        return {}


class TurnWriter:
    """Owns the session folder + turn counter; one instance per pilot run."""

    def __init__(self):
        self.board_id = None
        self.dir = None
        self.turn = 0

    def ensure_session(self, window_title):
        board_id = board_id_from_title(window_title)
        if board_id != self.board_id:
            self.board_id = board_id
            self.dir = os.path.join(HISTORY, board_id)
            os.makedirs(os.path.join(self.dir, 'flagged'), exist_ok=True)
            # RESUME an existing session (pilot restart mid-tournament): continue the turn
            # numbering after the last recorded turn and keep the shadow mirror -- a reset
            # counter wrote duplicate turn numbers into the same turns.jsonl (found on the
            # first real pilot session). Fresh boards start at 0 with a clean shadow file.
            self.turn = self._last_recorded_turn()
            if self.turn == 0:
                open(os.path.join(self.dir, 'shadow_turns.jsonl'), 'w', encoding='utf-8').close()
            print(f"[pilot] recording -> {self.dir}\\turns.jsonl"
                  + (f" (resuming after turn {self.turn})" if self.turn else ''))
        return self.dir

    def _last_recorded_turn(self):
        path = os.path.join(self.dir, 'turns.jsonl')
        last = 0
        try:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        last = max(last, int(json.loads(line).get('turn') or 0))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
        except OSError:
            pass
        return last

    def build_record(self, *, window_title, obs_dict, obs_raw_dict, assembled, decision,
                     ev_dict, engine):
        em = decision.equity_meta or {}
        bb = float(obs_dict.get('big_blind') or 0) or None

        def _bb(x):
            try:
                return round(float(x) / bb, 2) if bb else None
            except (TypeError, ValueError):
                return None

        board = obs_dict.get('community_cards') or []
        seen = decode_model_input(ev_dict, engine)
        to_call_bb = seen.get('to_call_bb')
        to_call = round(to_call_bb * bb, 2) if (to_call_bb is not None and bb) else \
            obs_dict.get('call_amount')
        if to_call_bb is None:
            to_call_bb = _bb(to_call)
        pot = obs_dict.get('pot_size')
        pot_odds = seen.get('pot_odds')
        if pot_odds is None:
            try:
                denom = float(pot or 0) + float(to_call or 0)
                pot_odds = round(float(to_call or 0) / denom, 3) if denom > 0 else 0.0
            except (TypeError, ValueError):
                pot_odds = None

        action, reason, bet_size = decision.as_tuple()[:3]
        record = {
            'format': 2,
            'turn': self.turn,
            'ts': datetime.datetime.now().isoformat(timespec='seconds'),
            'board_id': self.board_id,
            'window_title': window_title,
            'flagged': False,
            'recorder': 'live2-pilot',
            'observation': obs_dict,
            'board_state': {
                'street': _STREETS.get(len(board), f'{len(board)}cards'),
                'hero_cards': obs_dict.get('hero_cards'),
                'board': board,
                'hero_position': obs_dict.get('hero_position'),
                'hero_stack': obs_dict.get('hero_stack'),
                'hero_stack_bb': _bb(obs_dict.get('hero_stack')),
                'pot': pot,
                'pot_bb': _bb(pot),
                'to_call': to_call,
                'to_call_bb': to_call_bb,
                'pot_odds': pot_odds,
                'big_blind': obs_dict.get('big_blind'),
                'num_opponents': em.get('num_opponents'),
                'opponents': [
                    {'seat': s.get('seat_key'), 'is_active': s.get('is_active'),
                     'vpip_color': s.get('vpip_color'), 'agg_color': s.get('agg_color'),
                     'stack': s.get('stack')}
                    for s in (obs_dict.get('seats') or []) if s.get('occupied')],
            },
            'evaluation': {
                'model': getattr(engine, 'active_model_name', None),
                'equity': decision.equity,
                'equity_method': em.get('method'),
                'equity_opp_colors': em.get('opp_colors'),
                'equity_opp_colors_in_pot': em.get('opp_colors_in_pot'),
                'equity_opp_colors_still_to_act': em.get('opp_colors_still_to_act'),
                'hand_strength': em.get('hand_strength'),
                'equity_edge': em.get('equity_edge'),
                'actor_policy': {k: ev_dict.get(k) for k in _ACTION_DIAG_ORDER
                                 if k in (ev_dict or {})},
                'critic_q': (ev_dict or {}).get('q_vals'),
                'model_input': (ev_dict or {}).get('model_input'),
            },
            'action': {'chosen': action, 'bet_size': bet_size, 'reason': reason},
            'assembler': {
                'corrections': assembled.corrections,
                'contradictions': assembled.contradictions,
                'provenance': assembled.provenance,
            },
        }
        if assembled.corrections or assembled.provenance:
            record['observation_raw'] = obs_raw_dict
        return record

    def write(self, record, assembled):
        self.turn += 1
        record['turn'] = self.turn
        with open(os.path.join(self.dir, 'turns.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, default=str) + '\n')
        shadow = {'turn': self.turn, 'observation': record['observation'],
                  'provenance': assembled.provenance,
                  'contradictions': assembled.contradictions,
                  'corrections': assembled.corrections}
        with open(os.path.join(self.dir, 'shadow_turns.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(shadow, default=str) + '\n')

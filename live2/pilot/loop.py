"""The pilot's turn cycle -- the headless successor to PHPHelp.bot_worker_loop.

Frame flow per tick (cadence ~1s):
  capture (PrintWindow, unfocused-safe) -> resize to the 1536x1090 reference frame ->
  vision.read_board_state -> TableState stabilization (+XML baseline seeding on hand
  reset) -> hero-turn detection (button templates) -> price/button-availability reads ->
  LiveObservation -> ASSEMBLER corrections -> decision_engine.decide -> record ->
  (--auto) motor-model clicks.

Legacy behaviours deliberately preserved: partial-board wait (1-2 community cards is a
mid-deal frame with no street encoding), <2 hero cards guard, unknown-price semantics
(call_amount_known=False, FOLD never masked), post-action debounce (require one
"not hero's turn" frame before acting again). New: a decide-once fingerprint -- in
recommend-only mode the legacy loop re-decided the same turn every ~1.5s and recorded
duplicates; the pilot decides once per distinct decision point.
"""
import os
import re
import time

import cv2
import numpy as np

from core.vision import PokerVision
from core.table_state import TableState
from core.evaluator import PokerEvaluator
from core.decision import PokerDecisionEngine
from core.live_observation import LiveObservation
from core.xml_tracker import XMLTracker
from live2.assembler.assemble import Assembler
from live2.assembler.shadow import _start_ingest_thread
from live2.pilot import actions, capture
from live2.pilot.recorder import TurnWriter

REF_W, REF_H = 1536, 1090


def find_table_window(title_contains=None):
    """Best-candidate poker table hwnd: prefers windows whose title carries a >=6-digit
    tournament id AND a hold'em marker (lobby windows have neither). Windows with an
    empty client area (closing/minimized table shells) are skipped -- a dead table would
    otherwise be re-found in a capture-fail loop."""
    wins = capture.list_windows(title_contains)
    best, best_score = None, -1
    for w in wins:
        t = w['title']
        score = 0
        if re.search(r'\d{6,}', t):
            score += 2
        if 'hold' in t.lower():
            score += 2
        if re.search(r'\d+\s*/\s*\d+', t):
            score += 1
        if score > best_score:
            try:
                cw, ch = capture.client_size(w['hwnd'])
            except Exception:
                continue
            if cw <= 0 or ch <= 0:
                continue
            best, best_score = w, score
    return (best['hwnd'], best['title']) if best and best_score >= 2 else (None, None)


def parse_blinds_from_title(title):
    """bb in the pipeline's cents denomination (decimal stakes x100), or None."""
    m = re.search(r'(?:€|\$|£)?(\d+\.\d+)/(?:€|\$|£)?(\d+\.\d+)', title or '')
    if m:
        return float(m.group(2)) * 100.0
    m = re.search(r'(?:€|\$|£)?(\d+)/(?:€|\$|£)?(\d+)', title or '')
    if m:
        return float(m.group(2))
    return None


def parse_button_money(text_upper):
    """Digit-strip the first number ('0.20' -> 20.0), mirroring vision's cent semantics
    (see the legacy _parse_button_money incident log). None when no number is present."""
    m = re.search(r'(\d+(?:[.,]\d+)?)', text_upper)
    if not m:
        return None
    digits = ''.join(c for c in m.group(1) if c.isdigit())
    return float(int(digits)) if digits else None


def window_title(hwnd):
    import ctypes
    n = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
    if not n:
        return ''
    buf = ctypes.create_unicode_buffer(n + 1)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


class Pilot:
    def __init__(self, auto=False, window=None, board_size='6-Max', sims=2000, log=print):
        self.auto = auto
        self.window_hint = window
        self.board_size = board_size
        self.sims = sims
        self.log = log
        self.vision = PokerVision()
        self.evaluator = PokerEvaluator()
        self._engine = None      # lazy: --probe and startup don't pay the model load
        self.table_state = TableState()
        self.xml_tracker = XMLTracker()
        self.writer = TurnWriter()
        self.assembler = None            # rebuilt per board_id
        self.big_blind = 20.0
        self.pending_baseline = None
        self.awaiting_turn_clear = False
        self.last_fingerprint = None

    def _reset_session(self):
        """Clear per-table state when the table window goes away (tournament over, window
        closed). The recorder/assembler re-key themselves on the next board_id; the hand
        store keeps everything durable, so a reset loses nothing."""
        self.table_state.reset(big_blind=self.big_blind)
        self.last_fingerprint = None
        self.awaiting_turn_clear = False
        self.pending_baseline = None
        self.assembler = None

    @property
    def engine(self):
        if self._engine is None:
            self.log('[pilot] loading decision engine ...')
            self._engine = PokerDecisionEngine(game_type='nlh')
            self.log(f"[pilot] model: {getattr(self._engine, 'active_model_name', '?')}")
        return self._engine

    # ------------------------------------------------------------------ frame helpers

    def _grab(self, hwnd):
        pil = capture.capture_window(hwnd)
        client_wh = pil.size
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        if img.shape[1] != REF_W or img.shape[0] != REF_H:
            img = cv2.resize(img, (REF_W, REF_H), interpolation=cv2.INTER_CUBIC)
        return img, client_wh

    def _read_price(self, img, fold_xy):
        """(call_amount, call_amount_known, check_call_available, bet_raise_available) --
        the legacy availability crops + Check/Call OCR, unchanged semantics."""
        fx, fy = fold_xy
        check_call_available, bet_raise_available = True, True
        cc_crop = img[fy + 20:fy + 70, fx + 210:fx + 350]
        if cc_crop.size > 0 and float(np.mean(cc_crop, axis=(0, 1))[1]) < 60.0:
            check_call_available = False
        btn_crop = img[fy + 20:fy + 70, fx + 380:fx + 520]
        if btn_crop.size > 0 and float(np.mean(btn_crop, axis=(0, 1))[1]) < 60.0:
            bet_raise_available = False

        street_price = float(getattr(self.table_state, 'current_street_bet_level', 0.0) or 0.0)
        if not check_call_available:
            amount = float(self.table_state.hero_stack or 0.0) or street_price
            return amount, False, check_call_available, bet_raise_available

        text = self.vision.ocr_roi(img, (fx + 190, fy + 15, 160, 60)).upper().replace(',', '.')
        if any(w in text for w in ('KALD', 'CALL', 'KLD', 'KND')):
            parsed = parse_button_money(text)
            if parsed is not None:
                return parsed, True, check_call_available, bet_raise_available
            return street_price, False, check_call_available, bet_raise_available
        if any(w in text for w in ('CHECK', 'CHEC', 'CHE', 'TJEK', 'TJK', 'KOLLA', 'PASS')):
            return 0.0, True, check_call_available, bet_raise_available
        return street_price, False, check_call_available, bet_raise_available

    # ------------------------------------------------------------------ the loop

    def run(self):
        mode = 'AUTO (clicks live)' if self.auto else 'recommend-only'
        self.log(f"[pilot] mode={mode}")
        _ = self.engine                  # load the model up front, not on the first turn
        _start_ingest_thread()
        hwnd = None
        scanning_announced = False
        while True:
            try:
                if hwnd is None:
                    hwnd, title = find_table_window(self.window_hint)
                    if hwnd is None:
                        if not scanning_announced:
                            self.log('[pilot] waiting for a table window ...')
                            scanning_announced = True
                        time.sleep(2.0)
                        continue
                    scanning_announced = False
                    self.log(f"[pilot] table window: {title!r}")
                    self._reset_session()
                self.tick(hwnd)
                time.sleep(1.0)
            except KeyboardInterrupt:
                self.log('[pilot] stopped (Ctrl+C)')
                return
            except RuntimeError as e:
                # capture_window fail-loud path: table over / window closed -- reset the
                # per-table state and go back to scanning for the next table
                self.log(f"[pilot] table gone ({e}) -- session state reset, waiting for next table")
                hwnd = None
                self._reset_session()
                time.sleep(2.0)
            except Exception as e:
                import traceback
                self.log(f"[pilot] tick error: {e}\n{traceback.format_exc()}")
                time.sleep(3.0)

    def tick(self, hwnd):
        title = window_title(hwnd)
        bb = parse_blinds_from_title(title)
        if bb and bb != self.big_blind:
            self.log(f"[pilot] blind level from title: bb={bb:.0f}")
            self.big_blind = bb

        img, client_wh = self._grab(hwnd)
        raw_state = self.vision.read_board_state(img, board_size=self.board_size)

        if self.table_state.detect_hand_reset(raw_state):
            self.log('[pilot] new hand -- table state reset')
            self.table_state.reset(big_blind=self.big_blind)
            self.last_fingerprint = None
            try:
                baseline = self.xml_tracker.get_baseline_stacks()
                self.pending_baseline = baseline if baseline and baseline[0] else None
            except Exception:
                self.pending_baseline = None
        self.table_state.update(raw_state)
        if self.pending_baseline:
            self.table_state.seed_stacks(*self.pending_baseline)
            self.pending_baseline = None

        buttons = self.vision.match_templates_in_roi(
            img, self.vision.rois['buttons'], self.vision.button_templates,
            threshold=0.85, max_matches=1)
        if not buttons:
            self.awaiting_turn_clear = False
            self.last_fingerprint = None
            return
        if self.awaiting_turn_clear:
            return                          # stale buttons from the action we just took

        state = self.table_state.to_dict()
        if len(state['hero_cards']) < 2:
            self.log('[pilot] hero turn but cards unread -- waiting')
            return
        if len(state['community_cards']) in (1, 2):
            self.log('[pilot] partial board (mid-deal frame) -- waiting')
            return

        fold_xy = buttons[0][1]
        call_amount, known, cc_avail, br_avail = self._read_price(img, fold_xy)

        fingerprint = (tuple(state['hero_cards']), tuple(state['community_cards']),
                       round(float(state.get('pot_size') or 0)), round(float(call_amount or 0)),
                       cc_avail, br_avail)
        if fingerprint == self.last_fingerprint:
            return                          # same decision point, already decided
        self.last_fingerprint = fingerprint

        obs = self.table_state.to_observation(
            call_amount=call_amount, call_amount_known=known,
            check_call_available=cc_avail, bet_raise_available=br_avail,
            big_blind=self.big_blind, ts_epoch=time.time(), source='live')
        obs_raw_dict = obs.to_json_dict()

        # -- assembler in the decision path --------------------------------------
        self.writer.ensure_session(title)
        if self.assembler is None or self.assembler.board_id != self.writer.board_id:
            self.assembler = Assembler(self.writer.board_id)
        assembled = self.assembler.process_turn({
            'observation': dict(obs_raw_dict), 'ts': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'turn': self.writer.turn + 1, 'board_id': self.writer.board_id})
        for c in assembled.corrections:
            self.log(f"[assembler] {c}")
        for c in assembled.contradictions:
            self.log(f"[assembler] CONTRADICTION {c}")
        obs_dict = assembled.observation
        decide_obs = LiveObservation.from_json_dict(obs_dict)

        active = [s for s in (obs_dict.get('seats') or [])
                  if s.get('occupied') and s.get('is_active')]
        decision = self.engine.decide(
            decide_obs, evaluator=self.evaluator, fallback_sims=self.sims,
            fallback_num_opponents=max(1, len(active)), log_fn=lambda *_a, **_k: None)
        action, reason, bet_size = decision.as_tuple()[:3]
        ev_dict = decision.as_tuple()[3] if len(decision.as_tuple()) > 3 else {}
        self.log(f"[pilot] TURN {self.writer.turn + 1}: {action} ({reason})")

        record = self.writer.build_record(
            window_title=title, obs_dict=obs_dict, obs_raw_dict=obs_raw_dict,
            assembled=assembled, decision=decision, ev_dict=ev_dict or {}, engine=self.engine)
        self.writer.write(record, assembled)
        # rolling frame of the last DECIDED turn -- the webapp's flag endpoint copies it
        # into flagged/turn_N_<ts>/screenshot.png so a flag captures what vision saw
        try:
            cv2.imwrite(os.path.join(self.writer.dir, 'last_turn.png'), img)
        except Exception:
            pass

        if self.auto:
            ok = actions.execute(hwnd, action, fold_xy, client_wh, log=self.log)
            self.log(f"[pilot] click {'done' if ok else 'FAILED'}")
            self.awaiting_turn_clear = True

    # ------------------------------------------------------------------ probe

    def probe(self, out_png):
        hwnd, title = find_table_window(self.window_hint)
        if hwnd is None:
            self.log('[pilot] no table window found')
            return False
        img, client_wh = self._grab(hwnd)
        cv2.imwrite(out_png, img)
        raw = self.vision.read_board_state(img, board_size=self.board_size)
        buttons = self.vision.match_templates_in_roi(
            img, self.vision.rois['buttons'], self.vision.button_templates,
            threshold=0.85, max_matches=1)
        self.log(f"[probe] window={title!r} client={client_wh}")
        self.log(f"[probe] frame saved -> {out_png}")
        self.log(f"[probe] hero_cards={raw.get('hero_cards')} community={raw.get('community_cards')} "
                 f"pot={raw.get('pot_size')} hero_stack={raw.get('hero_stack')}")
        opps = raw.get('opponents') or {}
        for k, v in (opps.items() if isinstance(opps, dict) else []):
            self.log(f"[probe]   {k}: {v.get('name')!r} stack={v.get('stack')} active={v.get('is_active')}")
        self.log(f"[probe] hero-turn buttons visible: {bool(buttons)}"
                 + (f" fold_anchor={buttons[0][1]}" if buttons else ''))
        return True

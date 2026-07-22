"""[v46_legacySweep] Verification: the legacy dispatch surfaces are GONE and every registered
engine is fully self-declaring -- so "add a model" can never again mean "N hand-synchronized
ladder edits that fail silently" (the H4 lesson, demonstrated three times in two days).

Run: python versions/v46_legacySweep/verify_legacy_sweep.py
"""
import io
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

RESULTS = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"\n         {detail}" if detail and not ok else ""))
    RESULTS.append(ok)
    return ok


def main():
    print("1. Source-level: the ladders are actually gone")
    src = io.open('core/decision.py', encoding='utf-8').read()
    for needle in ("is_v13_model =", "is_v29_model =", "_LEGACY_LIVE_FEATURES = [",
                   "bridge_v9 =", "bridge_v25 =", "Herocules (v9 Main)",
                   "use_math_engine and", "is_actor_policy"):
        check(f"decision.py contains no '{needle}'", needle not in src)
    php = io.open('PHPHelp.py', encoding='utf-8').read()
    for needle in ("SEAT_ORDER_CLOCKWISE = [", "layer_math_var = ctk", "last_valid_hero_stack ="):
        check(f"PHPHelp.py contains no '{needle}'", needle not in php)

    print("\n2. Registry: pre-v30 deregistered, every engine fully self-declaring")
    from core.decision import PokerDecisionEngine
    eng = PokerDecisionEngine()
    expected = {'Herocules (v44)', 'Herocules (v43)', 'Herocules (v41)', 'Herocules (v40)'}
    check(f"registry is exactly {sorted(expected)}", set(eng.models) == expected,
          detail=f"got {sorted(eng.models)}")
    check("active model is v44", eng.active_model_name == 'Herocules (v44)')
    for name, m in eng.models.items():
        ok = (getattr(m, 'loaded', False) is True
              and getattr(m, 'is_sized', False) is True
              and bool(getattr(m, 'display_tag', None))
              and getattr(m, 'has_aux', None) is not None
              and name in eng._engine_bridges)
        check(f"{name}: loaded + is_sized + display_tag + has_aux + own bridge", ok)
        spec, source, err = eng._resolve_live_spec(name)
        check(f"{name}: live_features resolves from the ENGINE", source == 'engine' and err is None,
              detail=f"source={source} err={err}")
        scales = eng.context_scales(name)
        check(f"{name}: context_scales from its own contract", scales['source'].endswith(':contract'),
              detail=str(scales))

    print("\n3. Every registered model still produces an executable action")
    from core.table_state import TableState

    def make_ts():
        ts = TableState()
        ts.reset(big_blind=20.0)
        ts.dealer_idx = 2
        ts._dealer_seen = True
        ts.hero_cards = ['Qs', 'Qd']
        ts.pot_size = 60.0
        ts.hero_stack = 1480.0
        ts.opponents = {f"seat_{i}": {'name': chr(64 + i), 'stack': 1480.0, 'is_active': True,
                                      'state': 'Active', 'vpip_color': 'Yellow',
                                      'agg_color': 'Green'} for i in range(1, 6)}
        ts.hand_start_stacks = {'Hero': 1500.0, **{f"seat_{i}": 1500.0 for i in range(1, 6)}}
        ts._recompute_positions()
        return ts

    stub = {'equity_fn': (lambda hero, board, colors, sims=250, front_colors=None: 0.62),
            'hand_strength_fn': (lambda c1, c2: 0.71), 'effective_field_fn': None,
            'use_front_colors': True, 'source': 'stub', 'error': None}
    eng.live_feature_providers = lambda name=None: dict(stub)
    executable = ('FOLD', 'CALL')
    for name in sorted(eng.models):
        eng.set_active_model(name)
        eng.hand_history_buffer, eng.hero_action_buffer = [], []
        eng._last_street, eng._last_hole_cards = None, None
        obs = make_ts().to_observation(call_amount=20.0, call_amount_known=True,
                                       check_call_available=True, bet_raise_available=True,
                                       big_blind=20.0)
        random.seed(7)
        dec = eng.decide(obs)
        ok = dec.action in executable or dec.action.startswith('RAISE_SLIDER_')
        tag = eng.models[name].display_tag
        check(f"{name}: executable action ({dec.action}), reason tagged '{tag}'",
              ok and dec.reason.startswith(tag), detail=dec.reason)

    print("\n4. Fail-loud paths")
    eng.set_active_model('Herocules (v999)')          # unknown -> keeps current, no V20 fallback
    check("unknown model name keeps the active model", eng.active_model_name in expected)

    class _Undeclared:                                 # engine with no declarations at all
        loaded = True
    eng.models['Undeclared (test)'] = _Undeclared()
    eng.set_active_model('Undeclared (test)')
    a, r, _, _ = eng.make_decision(make_ts().to_board_state(call_amount=20.0, equity=0.5,
                                                            big_blind=20.0))
    check("undeclared engine refuses loudly (FOLD + is_sized named)",
          a == 'FOLD' and 'is_sized' in r, detail=r)
    del eng.models['Undeclared (test)']

    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

import os
from core.models.engine import ModelEngine
from core.bridge.contract_v8_v9 import ContractV8V9
from core.bridge.v11.contract_v11 import ContractV8V9 as ContractV11
from core.board_state import BoardState
from core.action import PokerAction

class PokerDecisionEngine:
    def __init__(self, game_type="limit"):
        self.game_type = game_type
        # Only loading V8/V9 supported architectures
        self.models = {
            'Herocules (v8 Self-Play)': ModelEngine(expert_name="expert_v8_selfplay.pth"),
            'Herocules (v11 Nit)': ModelEngine(expert_name="expert_v11_nit.pth", is_v11=True),
            'Herocules (v11 Maniac)': ModelEngine(expert_name="expert_v11_maniac.pth", is_v11=True),
            'Herocules (v11 Sticky)': ModelEngine(expert_name="expert_v11_sticky.pth", is_v11=True),
            'Herocules (v9 Main)': ModelEngine(expert_name="expert_v9_main.pth"),
            'Herocules (v10 Main)': ModelEngine(expert_name="v10_100k_main.pt"),
            'v10_100k_main.pt': ModelEngine(expert_name="v10_100k_main.pt"),
            'v10_50k_main.pt': ModelEngine(expert_name="v10_50k_main.pt"),
            'V10 100k Final': ModelEngine(expert_name="expert_v8_main.pth"),
            'V11 Maniac': ModelEngine(expert_name="expert_v11_maniac.pth", is_v11=True),
            'V11 Nit': ModelEngine(expert_name="expert_v11_nit.pth", is_v11=True),
            'V11 Sticky': ModelEngine(expert_name="expert_v11_sticky.pth", is_v11=True),
            'Herocules (v11 Main)': ModelEngine(expert_name="expert_v11_main.pth", is_v11=True),
            'herocules_v11_fuzzyHeuristicsOpp.pth': ModelEngine(expert_name="herocules_v11_fuzzyHeuristicsOpp.pth", is_v11=True),
        }
        self.active_model_name = 'Herocules (v11 Main)'
        self.bridge_v9 = ContractV8V9()
        self.bridge_v11 = ContractV11()

    def set_active_model(self, model_name: str, tree_file: str = None):
        if model_name in self.models:
            self.active_model_name = model_name
        else:
            print(f"Warning: {model_name} is not loaded or supported. Falling back to V9 Main.")
            self.active_model_name = 'Herocules (v9 Main)'

    def make_decision(self, board_state: BoardState, 
                      use_preflop_chart: bool = True,
                      use_math_engine: bool = True,
                      use_bluff_engine: bool = True,
                      use_dynamic_sizing: bool = True,
                      bet_raise_available: bool = True,
                      check_call_available: bool = True,
                      action_history_raw: list = None):
        
        active_model = self.models.get(self.active_model_name)
        if not active_model:
            return 'FOLD', "Model not found", 0.0, {}

        try:
            if getattr(active_model, 'is_v11', False) or 'v11' in self.active_model_name.lower():
                hole, board, ctx, act = self.bridge_v11.to_tensors(board_state, action_history_raw)
            else:
                hole, board, ctx, act = self.bridge_v9.to_tensors(board_state, action_history_raw)
            evs = active_model.predict_ev(hole, board, ctx, act)
        except Exception as e:
            return 'FOLD', f"Fatal decision engine crash: {e}", 0.0, {}

        # 1. Base Model Decision
        best_action_str = max(evs, key=evs.get)
        action = best_action_str
        reason = f"Model Output: {evs}"
        
        # Determine basic bet size
        bet_size = 0.0
        if action == 'RAISE':
            if use_dynamic_sizing:
                bet_size = board_state.pot_size * 0.75
            else:
                bet_size = board_state.big_blind * 3

        # 2. V9 River Air Guardrail
        if board_state.street == 'River' and board_state.equity < 0.35 and board_state.call_amount > 0:
             if self.active_model_name == 'Herocules (v9 Main)' and action in ('RAISE', 'ALL_IN'):
                  action = 'FOLD'
                  reason = "Guardrail: V9 River Air Defense (Equity too low to bluff shove)"
                  bet_size = 0.0

        # 3. Apply Post-flop Math Engine Guardrail (Pot Odds Check)
        math_engine_status = "Passed"
        math_engine_details = "Math checks out OK"
        if use_math_engine and board_state.street != 'Preflop' and board_state.call_amount > 0:
            pot_odds = board_state.call_amount / (board_state.pot_size + board_state.call_amount)
            
            buffer_offset = -0.05
            profile_desc = "Default"
            
            # Very simplistic HUD aggregation for the math buffer
            is_maniac = any(s.hud.agg_color == 'Red' or s.hud.vpip_color == 'Red' for s in board_state.seats.values() if s.is_active)
            is_loose = any(s.hud.agg_color == 'Yellow' or s.hud.vpip_color == 'Yellow' for s in board_state.seats.values() if s.is_active)
            is_nit = any(s.hud.vpip_color == 'Blue' and s.hud.agg_color in ('Blue', None) for s in board_state.seats.values() if s.is_active)
            is_tight = any(s.hud.vpip_color == 'Blue' for s in board_state.seats.values() if s.is_active)
            
            if is_maniac:
                buffer_offset = -0.12
                profile_desc = "Maniac"
            elif is_loose:
                buffer_offset = -0.09
                profile_desc = "Loose"
            elif is_nit:
                buffer_offset = 0.02
                profile_desc = "Nit"
            elif is_tight:
                buffer_offset = 0.00
                profile_desc = "Tight"
            
            if board_state.equity < pot_odds + buffer_offset:
                if action not in ('FOLD', 'CHECK'):
                    action = 'FOLD'
                    reason = f"Math Override ({profile_desc} HUD): Equity ({board_state.equity:.1%}) < Pot Odds ({pot_odds:.1%}) + Buffer ({buffer_offset:+.1%}). Overriding to FOLD."
                    bet_size = 0.0
                    math_engine_status = "Triggered (Active)"
                    math_engine_details = f"Overridden to FOLD: Equity < Pot Odds + Buffer"

        # Construct decision path for UI
        decision_path = {
            'preflop_chart': {'status': "Bypassed", 'details': "Deprecated in V8/V9"},
            'active_model': {'status': "Active", 'details': f"Loaded {self.active_model_name}"},
            'bluff_engine': {'status': "Passed", 'details': "Handled natively by V8/V9"},
            'math_engine': {'status': math_engine_status, 'details': math_engine_details}
        }
        
        ev_dict = evs.copy()
        ev_dict['decision_path'] = decision_path

        return action, reason, bet_size, ev_dict

import os

class PokerDecisionEngine:
    def __init__(self):
        from core.models import HeuristicEngine, XGBoostModel, PyTorchMLPModel, MixedXGBoostModel
        self.models = {
            'Heuristic (Rules)': HeuristicEngine(),
            'XGBoost Classifier': XGBoostModel(),
            'XGBoost Mixed (Pro + Human)': MixedXGBoostModel(self),
            'PyTorch Neural Net': PyTorchMLPModel()
        }
        self.active_model_name = 'Heuristic (Rules)'
        self.mixed_ratio = 0.5  # 0.0 = 100% Pro, 1.0 = 100% Human

    def set_active_model(self, model_name: str):
        if model_name in self.models:
            self.active_model_name = model_name
            # If it's XGBoost, PyTorch or Mixed, reload binary in case it was just trained/updated!
            if model_name == 'XGBoost Classifier':
                from core.models import XGBoostModel
                self.models[model_name] = XGBoostModel()
            elif model_name == 'XGBoost Mixed (Pro + Human)':
                from core.models import MixedXGBoostModel
                self.models[model_name] = MixedXGBoostModel(self)
            elif model_name == 'PyTorch Neural Net':
                from core.models import PyTorchMLPModel
                self.models[model_name] = PyTorchMLPModel()

    def make_decision(self, board: list, hand: list, equity: float, pot_size: float, 
                      call_amount: float, hero_stack: float, num_opponents: int = 1,
                      is_preflop: bool = True,
                      use_preflop_chart: bool = True,
                      use_math_engine: bool = True,
                      use_bluff_engine: bool = True,
                      use_dynamic_sizing: bool = True,
                      bet_raise_available: bool = True,
                      check_call_available: bool = True,
                      active_opponents: list = None):
        
        if active_opponents is None:
            active_opponents = []
            
        # Force pre-flop chart fallback if enabled to ensure tight/correct pre-flop ranges
        # (avoiding ML mucking/showdown biases pre-flop)
        if is_preflop and use_preflop_chart:
            try:
                fallback = self.models['Heuristic (Rules)']
                return fallback.predict_action(
                    board, hand, equity, pot_size, call_amount, hero_stack, num_opponents,
                    is_preflop, use_preflop_chart, use_math_engine, use_bluff_engine,
                    use_dynamic_sizing, bet_raise_available, check_call_available, active_opponents
                )
            except Exception as e:
                pass

        # Get active model
        active_model = self.models.get(self.active_model_name)
        if active_model is None:
            active_model = self.models['Heuristic (Rules)']

        try:
            return active_model.predict_action(
                board, hand, equity, pot_size, call_amount, hero_stack, num_opponents,
                is_preflop, use_preflop_chart, use_math_engine, use_bluff_engine,
                use_dynamic_sizing, bet_raise_available, check_call_available, active_opponents
            )
        except Exception as e:
            # Fallback safety
            fallback = self.models['Heuristic (Rules)']
            try:
                action, reason, bet_size = fallback.predict_action(
                    board, hand, equity, pot_size, call_amount, hero_stack, num_opponents,
                    is_preflop, use_preflop_chart, use_math_engine, use_bluff_engine,
                    use_dynamic_sizing, bet_raise_available, check_call_available, active_opponents
                )
                return action, f"Fallback on Heuristic due to model error ({e}). (Heur: {reason})", bet_size
            except Exception as e_fallback:
                return 'FOLD', f"Fatal decision engine crash: {e_fallback}", 0.0

import os
import numpy as np
from core.models.base import PokerModelInterface

class MixedXGBoostModel(PokerModelInterface):
    def __init__(self, engine=None):
        self.engine = engine  # reference to the decision engine to access dynamic ratio
        self.action_map = {0: 'FOLD', 1: 'CHECK', 2: 'CALL', 3: 'BET', 4: 'RAISE'}
        
        # Load Pro model
        pro_path = os.path.join(os.path.dirname(__file__), "binaries", "xgboost_poker.json")
        self.pro_model = None
        if os.path.exists(pro_path):
            try:
                import xgboost as xgb
                self.pro_model = xgb.XGBClassifier()
                self.pro_model.load_model(pro_path)
            except Exception as e:
                print(f"[MixedXGBoostModel] Error loading Pro binary: {e}")
                
        # Load Human model
        human_path = os.path.join(os.path.dirname(__file__), "binaries", "xgboost_human.json")
        self.human_model = None
        if os.path.exists(human_path):
            try:
                import xgboost as xgb
                self.human_model = xgb.XGBClassifier()
                self.human_model.load_model(human_path)
            except Exception as e:
                print(f"[MixedXGBoostModel] Error loading Human binary: {e}")

    def predict_action(self, board: list, hand: list, equity: float, pot_size: float, 
                       call_amount: float, hero_stack: float, num_opponents: int,
                       is_preflop: bool, use_preflop_chart: bool, use_math_engine: bool,
                       use_bluff_engine: bool, use_dynamic_sizing: bool,
                       bet_raise_available: bool, check_call_available: bool,
                       active_opponents: list = None) -> tuple:
                       
        if self.pro_model is None or self.human_model is None:
            return 'FOLD', "Mixed model binaries not fully loaded.", 0.0
            
        # Construct feature vector matching train schema
        is_pre = 1.0 if is_preflop else 0.0
        stack_pot_ratio = hero_stack / pot_size if pot_size > 0 else 999.0
        br_avail = 1.0 if bet_raise_available else 0.0
        cc_avail = 1.0 if check_call_available else 0.0
        
        features = [[
            is_pre,
            float(num_opponents),
            equity,
            call_amount / (pot_size + call_amount) if (pot_size + call_amount) > 0 else 0.0, # pot odds
            stack_pot_ratio,
            br_avail,
            cc_avail
        ]]
        
        # Get ratio from engine
        ratio = 0.5
        if self.engine and hasattr(self.engine, 'mixed_ratio'):
            ratio = self.engine.mixed_ratio
            
        try:
            pro_probs = self.pro_model.predict_proba(features)[0]
            human_probs = self.human_model.predict_proba(features)[0]
            
            # 1. Keep Pro FOLD probability baseline to ensure tight/correct folding behavior
            # (since human fold data is unrevealed and biased)
            fold_prob = pro_probs[0]
            
            # 2. Extract active probabilities (1=CHECK, 2=CALL, 3=BET, 4=RAISE)
            pro_active = pro_probs[1:]
            human_active = human_probs[1:]
            
            sum_pro_active = sum(pro_active)
            sum_human_active = sum(human_active)
            
            # Normalize active probabilities
            pro_norm = pro_active / sum_pro_active if sum_pro_active > 0 else np.array([0.25, 0.25, 0.25, 0.25])
            human_norm = human_active / sum_human_active if sum_human_active > 0 else np.array([0.25, 0.25, 0.25, 0.25])
            
            # Blend normalized active probabilities
            mixed_active_norm = (1.0 - ratio) * pro_norm + ratio * human_norm
            
            # Scale active probabilities back to the remaining non-fold probability mass
            mixed_active = mixed_active_norm * (1.0 - fold_prob)
            
            # Construct final mixed probability vector
            mixed_probs = np.zeros(5)
            mixed_probs[0] = fold_prob
            mixed_probs[1:] = mixed_active
            
            # Apply Temperature Scaling (T = 2.5) to soften/smooth the distribution
            temp = 2.5
            mixed_probs = np.power(mixed_probs, 1.0 / temp)
            
            # Ensure probabilities sum to exactly 1.0 (for np.random.choice numerical tolerance)
            mixed_probs = mixed_probs / np.sum(mixed_probs)
            
            # Weighted random sampling to match human strategy variance (non-deterministic)
            pred_idx = int(np.random.choice([0, 1, 2, 3, 4], p=mixed_probs))
            
            action = self.action_map.get(pred_idx, 'FOLD')
            dist_str = ", ".join([f"{self.action_map[i]}:{mixed_probs[i]:.1%}" for i in range(5)])
            reason = f"Mixed XGBoost Sampled (Ratio {ratio*100:.0f}% Human, Class {pred_idx}). Dist: [{dist_str}]"
        except Exception as e:
            return 'FOLD', f"Mixed inference error: {e}", 0.0
            
        # Post-process sizing if action is BET or RAISE
        bet_size = 0.0
        if action in ['BET', 'RAISE']:
            if use_dynamic_sizing:
                from core.evaluator import PokerEvaluator
                pe = PokerEvaluator()
                
                if is_preflop:
                    target_bet = max(3.0 * 20.0, call_amount * 3.0)
                    reason += " (Pre-flop open/3-bet sizing)"
                else:
                    texture = pe.analyze_board_texture(board)
                    wetness = texture['wetness']
                    bet_pct = 0.80 if wetness >= 0.5 else 0.40
                    target_bet = pot_size * bet_pct
                    reason += f" (Post-flop slider sizing, wetness={wetness:.1f})"
                    
                min_bet = 2.0 * call_amount if call_amount > 0 else 20.0
                clamped_bet = max(min_bet, min(target_bet, hero_stack))
                
                min_slider_val = 10.0
                max_slider_val = hero_stack
                
                if max_slider_val > min_slider_val:
                    slider_fraction = (clamped_bet - min_slider_val) / (max_slider_val - min_slider_val)
                    slider_fraction = max(0.0, min(1.0, slider_fraction))
                    action = f"{action}_SLIDER_{slider_fraction:.2f}"
                    bet_size = clamped_bet
                else:
                    action = f"{action}_SLIDER_1.0"
                    bet_size = hero_stack
            else:
                if action == 'BET':
                    bet_size = min(max(pot_size * 0.5, 2.0), hero_stack)
                else:
                    bet_size = min(call_amount + max(pot_size * 0.5, call_amount * 2.0), hero_stack)
                    
        # Safeguards
        if not bet_raise_available and (action.startswith('BET') or action.startswith('RAISE')):
            if call_amount == 0:
                action = 'CHECK'
                reason = f"Mixed predicted BET/RAISE but Bet/Raise button is unavailable. Checking."
                bet_size = 0.0
            else:
                action = 'FOLD'
                reason = f"Mixed predicted BET/RAISE but Bet/Raise button is unavailable. Folding."
                bet_size = 0.0
                
        if not check_call_available and action == 'CALL':
            action = 'RAISE'
            reason = f"Mixed predicted CALL but Call button is unavailable. Shoving ALL-IN. (Original: {reason})"
            
        return action, reason, bet_size

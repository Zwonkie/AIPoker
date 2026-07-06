import os
from core.models.decision_parser import DecisionParser
from core.models.base import PokerModelInterface

class DecisionTreeEngine(PokerModelInterface):
    def __init__(self, json_filepath=None):
        if json_filepath is None:
            # Default to a file in the models directory
            json_filepath = os.path.join(os.path.dirname(__file__), 'decision_rules.json')
            
        self.parser = DecisionParser(json_filepath) if os.path.exists(json_filepath) else None
        
    def predict_action(self, board, hand, equity, pot_size, call_amount, hero_stack,
                       num_opponents, is_preflop, use_preflop_chart, use_math_engine,
                       use_bluff_engine, use_dynamic_sizing, bet_raise_available,
                       check_call_available, active_opponents):
        
        valid_actions = []
        if check_call_available:
            valid_actions.extend(['CHECK', 'CALL'] if call_amount > 0 else ['CHECK'])
        if bet_raise_available:
            valid_actions.extend(['RAISE'] if call_amount > 0 else ['BET'])
        
        if not self.parser:
            print("[DecisionTree] No rules loaded. Defaulting to Check/Fold.")
            return self._fallback_action(valid_actions)

        # 1. Determine Context (e.g. preflop, flop, turn, river)
        # 2. Find matching context node
        context_str = 'preflop' if is_preflop else ('flop' if len(board) == 3 else ('turn' if len(board) == 4 else 'river'))
        
        # Override with bet context if more specific root exists
        if call_amount > hero_stack:
            context_str = 'facing_allin'
        elif call_amount > 0:
            context_str = 'facing_bet'
            
        start_node = None
        # Try specific first, then street
        for root in self.parser.find_context_roots():
            if root['data'].get('contextType') == context_str:
                start_node = root
                break
                
        # Fallback to street if facing_bet etc not found
        if not start_node and context_str in ['facing_bet', 'facing_allin']:
            street_ctx = 'preflop' if is_preflop else ('flop' if len(board) == 3 else ('turn' if len(board) == 4 else 'river'))
            for root in self.parser.find_context_roots():
                if root['data'].get('contextType') == street_ctx:
                    start_node = root
                    break

        if not start_node:
            print(f"[DecisionTree] No matching root for context.")
            return self._fallback_action(valid_actions)

        # 3. Traverse the tree
        current_node_id = start_node['id']
        next_node_id = self.parser.get_next_node_id(current_node_id, 'a')
        
        while next_node_id:
            current_node = self.parser.get_node(next_node_id)
            if not current_node:
                break
                
            node_type = current_node['type']
            data = current_node['data']
            
            if node_type == 'actionNode':
                return self._execute_action(data, valid_actions, pot_size)
                
            elif node_type == 'conditionNode':
                result = self._evaluate_condition(data, equity, hero_stack, pot_size)
                handle = 'true' if result else 'false'
                next_node_id = self.parser.get_next_node_id(current_node['id'], handle)
                
            elif node_type == 'profileNode':
                # Simplified for now: just checks first active opponent stats
                opp_stats = active_opponents[0] if active_opponents else {}
                result = self._evaluate_profile(data, opp_stats)
                handle = 'true' if result else 'false'
                next_node_id = self.parser.get_next_node_id(current_node['id'], handle)
            else:
                break

        print("[DecisionTree] Tree traversal ended without an action.")
        return self._fallback_action(valid_actions)

    def _evaluate_condition(self, data, equity, hero_stack, pot_size):
        metric = data.get('metric', 'equity')
        operator = data.get('operator', '>')
        val_threshold = float(data.get('value', 0))
        
        val_actual = 0
        if metric == 'equity':
            val_actual = equity * 100 
        elif metric == 'ev':
            val_actual = 0 # Dummy EV, would need true EV calc
        elif metric == 'spr':
            val_actual = hero_stack / pot_size if pot_size > 0 else 0
            
        if operator == '>': return val_actual > val_threshold
        if operator == '<': return val_actual < val_threshold
        if operator == '>=': return val_actual >= val_threshold
        if operator == '<=': return val_actual <= val_threshold
        return False

    def _evaluate_profile(self, data, opponent_stats):
        if not opponent_stats:
            return False
            
        stat = data.get('stat', 'vpip_color')
        if stat in ['vpip_color', 'agg_color']:
            color = data.get('value', 'grey')
            return opponent_stats.get(stat, 'grey') == color
            
        return False

    def _execute_action(self, data, valid_actions, pot_size):
        action = data.get('action', 'FOLD')
        amount_pct = float(data.get('amount', 50))
        
        bet_size = (amount_pct / 100.0) * pot_size if action in ['BET', 'RAISE'] else 0
        
        if action == 'CHECK' and 'CHECK' not in valid_actions:
            action = 'FOLD'
        if action in ['BET', 'RAISE'] and action not in valid_actions:
            action = 'CALL' if 'CALL' in valid_actions else 'FOLD'
            
        return action, "Visual Decision Tree Route", bet_size

    def _fallback_action(self, valid_actions):
        if 'CHECK' in valid_actions:
            return 'CHECK', 'Decision Tree Fallback', 0.0
        return 'FOLD', 'Decision Tree Fallback', 0.0

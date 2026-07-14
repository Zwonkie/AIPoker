import numpy as np

class TrainingTelemetry:
    """External telemetry tracker for V10 to monitor Action Entropy and an Expanded Bluff/Equity Matrix."""
    
    def __init__(self):
        # Equity Buckets: <20, 20-40, 40-60, 60-80, >80
        self.buckets = ['<20', '20-40', '40-60', '60-80', '>80']
        self.equity_matrix = {
            b: {
                'total': 0,
                'actions': {'fold': 0, 'call': 0, 'raise': 0, 'rr': 0, 'allin': 0},
                'street_sum': 0.0,
                'net_chips': 0.0,
                'won_chips': 0.0,
                'lost_chips': 0.0
            } for b in self.buckets
        }
        
        # Action Entropy
        self.entropy_history = []
        
    def _get_bucket(self, equity):
        if equity < 0.20: return '<20'
        if equity < 0.40: return '20-40'
        if equity < 0.60: return '40-60'
        if equity < 0.80: return '60-80'
        return '>80'

    def record_entropy(self, action_probs):
        """Record the raw entropy of a decision point."""
        probs = np.array(action_probs)
        probs = probs[probs > 0] # Avoid log(0)
        entropy = -np.sum(probs * np.log(probs))
        self.record_entropy_value(entropy)

    def record_entropy_value(self, entropy):
        self.entropy_history.append(entropy)
        if len(self.entropy_history) > 1000:
            self.entropy_history.pop(0)

    def record_hand_terminal_state(self, equity, street, action, call_amount, is_all_in, net_profit):
        """
        Record the final, terminal state of a hand.
        street: 0=Preflop, 1=Flop, 2=Turn, 3=River
        action: 0=Fold, 1=Call, 2=Raise
        """
        bucket = self._get_bucket(equity)
        stats = self.equity_matrix[bucket]
        
        stats['total'] += 1
        stats['street_sum'] += street
        stats['net_chips'] += net_profit
        if net_profit > 0:
            stats['won_chips'] += net_profit
        elif net_profit < 0:
            stats['lost_chips'] += net_profit
        
        # Determine exact last action
        if action == 0:
            stats['actions']['fold'] += 1
        elif is_all_in and action > 0:
            # If hero went all-in
            stats['actions']['allin'] += 1
        elif action == 1:
            stats['actions']['call'] += 1
        elif action == 2:
            if call_amount > 0:
                stats['actions']['rr'] += 1
            else:
                stats['actions']['raise'] += 1
                
    def get_average_entropy(self):
        if not self.entropy_history:
            return 0.0
        return sum(self.entropy_history) / len(self.entropy_history)

    def get_matrix_stats(self):
        """Returns the fully computed matrix for dashboard rendering."""
        results = {}
        for b in self.buckets:
            stats = self.equity_matrix[b]
            total = stats['total']
            if total == 0:
                results[b] = {
                    'f_pct': 0.0, 'c_pct': 0.0, 'r_pct': 0.0, 'rr_pct': 0.0, 'ai_pct': 0.0,
                    'avg_street': 0.0, 'total_chips': 0.0, 'total_hands': 0,
                    'won_chips': 0.0, 'lost_chips': 0.0
                }
            else:
                results[b] = {
                    'f_pct': stats['actions']['fold'] / total,
                    'c_pct': stats['actions']['call'] / total,
                    'r_pct': stats['actions']['raise'] / total,
                    'rr_pct': stats['actions']['rr'] / total,
                    'ai_pct': stats['actions']['allin'] / total,
                    'avg_street': stats['street_sum'] / total,
                    'total_chips': stats['net_chips'],
                    'total_hands': total,
                    'won_chips': stats['won_chips'],
                    'lost_chips': stats['lost_chips']
                }
        return results

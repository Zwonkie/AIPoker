import numpy as np

class TrainingTelemetry:
    """External telemetry tracker for V10 to monitor Action Entropy and an Expanded Bluff/Equity Matrix."""
    
    def __init__(self):
        # Equity Buckets: <20, 20-40, 40-60, 60-80, >80
        self.buckets = ['<20', '20-40', '40-60', '60-80', '>80']
        self.equity_matrix = {
            b: {
                'total': 0,
                # V14 6-action space: fold, call, raise 0.33/0.66/1.0 pot, all-in. A raise that
                # committed the whole stack is counted as all-in (de-facto shove) regardless of size.
                'actions': {'fold': 0, 'call': 0, 'r33': 0, 'r66': 0, 'rpot': 0, 'allin': 0},
                'street_sum': 0.0,
                'net_chips': 0.0,
                'won_chips': 0.0,
                'lost_chips': 0.0,
                # [V17] free_checks: decisions where call_amount<=0 (checking is free -- correctly
                # never folded, and "call" here just means "check", not paying to continue). These
                # were previously silently mixed into 'actions'/'total' above, which made the
                # Fold%/Call% columns uninterpretable for judging continue-vs-fold QUALITY (a bucket
                # showing e.g. 65% fold / 23% call couldn't say how much of that 23% was a free,
                # mandatory check vs an actual paid continuation with air). `actions`/`total` above
                # are now FACING-A-BET ONLY; free checks are tallied here instead.
                'free_checks': 0,
            } for b in self.buckets
        }
        
        # Action Entropy
        self.entropy_history = []

        # V14 action-usage + adaptation metrics ----------------------------------------------
        self.action_hist = [0, 0, 0, 0, 0, 0]   # size-selection over ALL hero decisions [F,C,r33,r66,rP,AI]
        self.allin_wins = 0                       # all-in hands the hero won (terminal, net>0)
        self.allin_total = 0
        # jam (all-in) frequency bucketed by the tightest active opponent's VPIP colour
        self.jam_by_color = {c: {'jam': 0, 'total': 0} for c in ['Blue', 'Green', 'Yellow', 'Red']}

    def record_decision(self, action_idx, is_all_in, opp_color=None):
        """Per hero decision: size-selection histogram + jam-by-opponent-colour (jam == all-in)."""
        if 0 <= action_idx < len(self.action_hist):
            self.action_hist[action_idx] += 1
        jammed = bool(is_all_in) or action_idx == 5
        b = self.jam_by_color.get(opp_color)
        if b is not None:
            b['total'] += 1
            if jammed:
                b['jam'] += 1

    def get_action_usage(self):
        total = sum(self.action_hist) or 1
        return [n / total for n in self.action_hist]

    def get_allin_winrate(self):
        return ((self.allin_wins / self.allin_total) if self.allin_total else 0.0), self.allin_total

    def get_jam_by_color(self):
        return {c: ((v['jam'] / v['total'] if v['total'] else 0.0), v['total'])
                for c, v in self.jam_by_color.items()}
        
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
        action: 0=Fold, 1=Call, 2=raise_33, 3=raise_66, 4=raise_pot, 5=all-in

        [V17] `call_amount` now actually used: a decision with call_amount<=0 was a FREE check
        (fold is dominated/never chosen there, and 'call'==1 just means "checked", not "paid to
        continue") -- tallied separately in `free_checks`, excluded from `actions`/`total` so the
        Fold%/Call%/raise/all-in percentages reported below mean "given the hero FACED A BET",
        which is what actually answers "is the continue-vs-fold decision any good" (previously the
        free-check volume silently diluted every bucket's Call% with correct, mandatory checks).
        """
        bucket = self._get_bucket(equity)
        stats = self.equity_matrix[bucket]

        if call_amount is not None and call_amount <= 0:
            stats['free_checks'] += 1
            return

        stats['total'] += 1
        stats['street_sum'] += street
        stats['net_chips'] += net_profit
        if is_all_in:
            self.allin_total += 1
            if net_profit > 0:
                self.allin_wins += 1
        if net_profit > 0:
            stats['won_chips'] += net_profit
        elif net_profit < 0:
            stats['lost_chips'] += net_profit

        # Determine exact last action (V14 6-action space). A raise that committed the whole stack
        # is bucketed as all-in (de-facto shove) regardless of the chosen size; otherwise by size.
        if action == 0:
            stats['actions']['fold'] += 1
        elif action == 1:
            stats['actions']['call'] += 1
        elif is_all_in:
            stats['actions']['allin'] += 1
        elif action == 2:
            stats['actions']['r33'] += 1
        elif action == 3:
            stats['actions']['r66'] += 1
        elif action == 4:
            stats['actions']['rpot'] += 1
        else:  # action == 5 (explicit all-in that wasn't stack-flagged)
            stats['actions']['allin'] += 1
                
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
                    'f_pct': 0.0, 'c_pct': 0.0, 'r33_pct': 0.0, 'r66_pct': 0.0, 'rpot_pct': 0.0,
                    'ai_pct': 0.0, 'avg_street': 0.0, 'total_chips': 0.0, 'total_hands': 0,
                    'won_chips': 0.0, 'lost_chips': 0.0, 'free_checks': stats['free_checks']
                }
            else:
                results[b] = {
                    'f_pct': stats['actions']['fold'] / total,
                    'c_pct': stats['actions']['call'] / total,
                    'r33_pct': stats['actions']['r33'] / total,
                    'r66_pct': stats['actions']['r66'] / total,
                    'rpot_pct': stats['actions']['rpot'] / total,
                    'ai_pct': stats['actions']['allin'] / total,
                    'avg_street': stats['street_sum'] / total,
                    'total_chips': stats['net_chips'],
                    'total_hands': total,               # FACING-A-BET hands only (see docstring above)
                    'won_chips': stats['won_chips'],
                    'lost_chips': stats['lost_chips'],
                    'free_checks': stats['free_checks'],  # excluded from total/f_pct/c_pct/etc above
                }
        return results

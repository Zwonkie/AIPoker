"""
Parameterized opponent bots for self-play training.
Each bot has hardcoded behavioral constraints that guarantee specific VPIP/AGG profiles.
"""
import random


class OpponentBot:
    """Base class for parameterized opponent bots."""
    
    def __init__(self, name, style):
        self.name = name
        self.style = style  # 'nit', 'fish', 'maniac', 'tag'
        
        # HUD tracking stats
        self.hands_played = 0
        self.vpip_count = 0  # Voluntarily Put $ In Pot
        self.pfr_count = 0   # Pre-Flop Raise
        self.agg_bets = 0    # Post-flop bets + raises
        self.agg_calls = 0   # Post-flop calls
        self.agg_folds = 0   # Post-flop folds
        
    @property
    def vpip(self):
        return self.vpip_count / max(1, self.hands_played)
    
    @property
    def agg_factor(self):
        return self.agg_bets / max(1, self.agg_calls)
        
    @property
    def agg_frequency(self):
        total = self.agg_bets + self.agg_calls + self.agg_folds
        return self.agg_bets / max(1, total)
    
    @property
    def vpip_normalized(self):
        """Normalized VPIP for neural network input (0.0 to 1.0)."""
        return min(1.0, self.vpip)
    
    @property
    def agg_normalized(self):
        """Normalized AGG frequency for neural network input (0.0 to 1.0)."""
        return min(1.0, self.agg_frequency)
    
    def decide_preflop(self, equity, pot_odds, is_blind=False):
        """Returns 'fold', 'call', or 'raise' for pre-flop action."""
        raise NotImplementedError
        
    def decide_postflop(self, equity, pot_odds, pot_size, stack, street):
        """Returns 'fold', 'call', or 'raise' for post-flop action."""
        raise NotImplementedError
    
    def record_preflop(self, action):
        """Track HUD stats for a pre-flop decision."""
        self.hands_played += 1
        if action in ('call', 'raise'):
            self.vpip_count += 1
        if action == 'raise':
            self.pfr_count += 1
    
    def record_postflop(self, action):
        """Track HUD stats for a post-flop decision."""
        if action in ('raise', 'bet'):
            self.agg_bets += 1
        elif action == 'call':
            self.agg_calls += 1
        elif action == 'fold':
            self.agg_folds += 1


class NitBot(OpponentBot):
    """
    Tight-Passive rock. Only plays premium hands.
    Target: VPIP ~12%, AGG Factor ~0.8
    """
    def __init__(self, name="Nit"):
        super().__init__(name, 'nit')
        
    def decide_preflop(self, equity, pot_odds, is_blind=False):
        # Very tight: only premium hands (AA, KK, QQ, AKs type equities)
        if equity >= 0.95:
            return 'raise'
        elif equity >= 0.90:
            return 'call'
        else:
            return 'fold'
    
    def decide_postflop(self, equity, pot_odds, pot_size, stack, street):
        if equity >= 0.80:
            return 'raise'
        elif equity >= 0.70:
            return 'call'
        elif equity >= pot_odds and pot_odds > 0:
            # Will call if getting correct price, but barely
            return 'call' if random.random() < 0.4 else 'fold'
        else:
            return 'fold'


class FishBot(OpponentBot):
    """
    Loose-Passive calling station. Calls too much, rarely raises.
    Target: VPIP ~55%, AGG Factor ~1.2
    """
    def __init__(self, name="Fish"):
        super().__init__(name, 'fish')
        
    def decide_preflop(self, equity, pot_odds, is_blind=False):
        if equity >= 0.70:
            return 'raise'
        elif equity >= 0.35:
            return 'call'
        else:
            # Still calls sometimes with garbage
            return 'call' if random.random() < 0.15 else 'fold'
    
    def decide_postflop(self, equity, pot_odds, pot_size, stack, street):
        if equity >= 0.75:
            return 'raise'
        elif equity >= 0.25:
            # Calling station: calls with almost anything
            return 'call'
        elif random.random() < 0.05:
            # Rare bluff
            return 'raise'
        else:
            return 'fold' if random.random() < 0.6 else 'call'


class ManiacBot(OpponentBot):
    """
    Loose-Aggressive maniac. Raises constantly, bluffs heavily.
    Target: VPIP ~63%, AGG Factor ~3.5
    """
    def __init__(self, name="Maniac"):
        super().__init__(name, 'maniac')
        
    def decide_preflop(self, equity, pot_odds, is_blind=False):
        if equity >= 0.45:
            return 'raise'
        elif equity >= 0.30:
            # Coin flip between raise and call
            return 'raise' if random.random() < 0.55 else 'call'
        else:
            return 'call' if random.random() < 0.25 else 'fold'
    
    def decide_postflop(self, equity, pot_odds, pot_size, stack, street):
        if equity >= 0.40:
            return 'raise'
        elif equity >= 0.20:
            # Aggressive: raises more than calls
            return 'raise' if random.random() < 0.45 else 'call'
        else:
            # Bluffs frequently even with air
            r = random.random()
            if r < 0.30:
                return 'raise'
            elif r < 0.50:
                return 'call'
            else:
                return 'fold'


class TAGBot(OpponentBot):
    """
    Tight-Aggressive regular. Plays solid, standard poker.
    Target: VPIP ~28%, AGG Factor ~2.1
    """
    def __init__(self, name="TAG"):
        super().__init__(name, 'tag')
        
    def decide_preflop(self, equity, pot_odds, is_blind=False):
        if equity >= 0.75:
            return 'raise'
        elif equity >= 0.65:
            return 'call'
        else:
            return 'fold'
    
    def decide_postflop(self, equity, pot_odds, pot_size, stack, street):
        if equity >= 0.65:
            return 'raise'
        elif equity >= 0.50:
            # Value bet or check-call
            return 'raise' if random.random() < 0.35 else 'call'
        elif equity >= pot_odds and pot_odds > 0:
            # Getting correct pot odds
            return 'call'
        elif random.random() < 0.10:
            # Occasional bluff on good texture
            return 'raise'
        else:
            return 'fold'


# Factory for creating bot pools
BOT_CLASSES = {
    'nit': NitBot,
    'fish': FishBot,
    'maniac': ManiacBot,
    'tag': TAGBot,
}

def create_opponent_pool(distribution=None):
    """
    Creates a weighted pool of opponent bots.
    
    Args:
        distribution: dict mapping style to weight, e.g. {'fish': 0.30, 'tag': 0.25, ...}
                      Defaults to standard low-stakes distribution.
    Returns:
        List of (bot_instance, weight) tuples.
    """
    if distribution is None:
        distribution = {
            'fish': 0.30,
            'tag': 0.25,
            'maniac': 0.25,
            'nit': 0.20,
        }
    
    pool = []
    for style, weight in distribution.items():
        bot_class = BOT_CLASSES[style]
        bot = bot_class(name=f"{style.capitalize()}")
        pool.append((bot, weight))
    
    return pool

def sample_opponent(pool):
    """Randomly select an opponent from the weighted pool."""
    bots, weights = zip(*pool)
    return random.choices(bots, weights=weights, k=1)[0]

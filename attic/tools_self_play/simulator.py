"""
Headless 6-max No-Limit Hold'em Poker Simulator for self-play training.
Deals cards, manages blinds/pots/betting, and produces vectorized training records.
"""
import random
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from treys import Card, Deck, Evaluator
from core.evaluator import PokerEvaluator
from tools.self_play.opponent_bots import sample_opponent


# Shared evaluator instances
_treys_evaluator = Evaluator()
_poker_evaluator = PokerEvaluator()

# Card conversion helpers
RANKS = '23456789TJQKA'
SUITS = 'cdhs'

def int_to_str(card_int):
    """Convert a treys Card int back to a 2-char string like 'Ah'."""
    return Card.int_to_str(card_int)

def str_to_int(card_str):
    """Convert a 2-char string like 'Ah' to a treys Card int."""
    if len(card_str) == 2:
        return Card.new(card_str[0].upper() + card_str[1].lower())
    raise ValueError(f"Invalid card: {card_str}")


class HandRecord:
    """Stores all decision points from a single hand for training."""
    
    def __init__(self):
        self.decision_points = []  # List of dicts, one per Hero decision
        self.final_hero_profit = 0.0  # Net chips won/lost by Hero for the whole hand
    
    def add_decision(self, street, equity, pot_size, call_amount, hero_stack,
                     num_opponents, action_taken, chips_committed_before,
                     opp_vpip_norm, opp_agg_norm, hero_cards_str, board_str,
                     action_history):
        """Record a single decision point."""
        self.decision_points.append({
            'street': street,           # 0=preflop, 1=flop, 2=turn, 3=river
            'equity': equity,
            'pot_size': pot_size,
            'call_amount': call_amount,
            'hero_stack': hero_stack,
            'num_opponents': num_opponents,
            'action': action_taken,     # 0=fold, 1=call, 2=raise
            'committed_before': chips_committed_before,
            'opp_vpip': opp_vpip_norm,
            'opp_agg': opp_agg_norm,
            'hero_cards': hero_cards_str,
            'board': board_str,
            'action_history': list(action_history),
        })
    
    def get_training_samples(self):
        """
        Convert decision points to training samples with sunk-cost-corrected EV targets.
        """
        samples = []
        for dp in self.decision_points:
            if dp['action'] == 0:  # Fold
                target_ev = 0.0  # Folding costs nothing going forward
            else:
                # Forward-looking EV = final profit - what was already committed
                # total profit = final_hero_profit (which is total_win - total_bet)
                # But we want: future_profit = total_win - future_investment
                # future_investment = total_bet - committed_before
                # future_profit = total_win - (total_bet - committed_before)
                #               = (total_win - total_bet) + committed_before
                #               = final_hero_profit + committed_before
                target_ev = self.final_hero_profit + dp['committed_before']
            
            samples.append({
                **dp,
                'target_ev': target_ev,
            })
        return samples


class HeadlessPokerSimulator:
    """
    Fast headless 6-max NLH simulator.
    Hero (seat 0) uses a neural network. Opponents use parameterized bots.
    """
    
    def __init__(self, opponent_pool, hero_model=None, bb_size=1.0, 
                 min_stack_bb=10, max_stack_bb=400, equity_sims=50):
        """
        Args:
            opponent_pool: List of (bot, weight) tuples from create_opponent_pool()
            hero_model: Optional neural network model for Hero decisions.
                        If None, Hero uses a simple equity-based heuristic.
            bb_size: Big blind size in chips
            min_stack_bb: Minimum starting stack in BB
            max_stack_bb: Maximum starting stack in BB
            equity_sims: Number of MC simulations for equity calculation (speed vs accuracy)
        """
        self.opponent_pool = opponent_pool
        self.hero_model = hero_model
        self.bb_size = bb_size
        self.min_stack_bb = min_stack_bb
        self.max_stack_bb = max_stack_bb
        self.equity_sims = equity_sims
        
        # Stats tracking
        self.total_hands = 0
        self.hero_total_profit = 0.0
        
    def _random_stack(self):
        """Generate a random starting stack between min and max BB."""
        return random.randint(self.min_stack_bb, self.max_stack_bb) * self.bb_size
    
    def _calculate_equity(self, hero_cards_str, board_str, num_opponents):
        """Calculate Hero's equity using MC simulation."""
        eq, _ = _poker_evaluator.calculate_equity(
            board_str, hero_cards_str,
            num_opponents=num_opponents,
            num_simulations=self.equity_sims
        )
        return round(eq * 100) / 100.0  # Bucket to 1%
    
    def _hero_decide(self, equity, pot_size, call_amount, hero_stack, 
                     num_opponents, is_preflop, opp_vpip, opp_agg):
        """Hero decision logic. Uses simple equity-based heuristic (bootstrapping phase)."""
        pot_odds = call_amount / (pot_size + call_amount) if (pot_size + call_amount) > 0 else 0.0
        
        if is_preflop:
            if equity >= 0.68:
                return 'raise'
            elif equity >= 0.52:
                return 'call'
            else:
                return 'fold'
        else:
            # Post-flop: equity-aware with some aggression
            if equity >= 0.70:
                return 'raise'
            elif equity >= 0.55:
                return 'raise' if random.random() < 0.30 else 'call'
            elif equity >= pot_odds:
                return 'call'
            elif equity >= pot_odds - 0.05 and random.random() < 0.15:
                # Occasional bluff/semi-bluff
                return 'raise'
            else:
                return 'fold'
    
    def simulate_hand(self):
        """
        Simulate a single hand of heads-up NLH.
        Returns a HandRecord with all Hero decision points, or None if hand is trivial.
        """
        # 1. Setup
        hero_stack = self._random_stack()
        opponent = sample_opponent(self.opponent_pool)
        opp_stack = self._random_stack()
        
        sb = self.bb_size * 0.5
        bb = self.bb_size
        
        # Deal cards
        deck = Deck()
        hero_cards_int = deck.draw(2)
        opp_cards_int = deck.draw(2)
        
        hero_cards_str = [Card.int_to_str(c) for c in hero_cards_int]
        opp_cards_str = [Card.int_to_str(c) for c in opp_cards_int]
        
        # Post blinds (Hero=SB, Opp=BB for simplicity, alternating isn't critical for training)
        hero_committed = min(sb, hero_stack)
        opp_committed = min(bb, opp_stack)
        hero_stack -= hero_committed
        opp_stack -= opp_committed
        
        pot = hero_committed + opp_committed
        
        record = HandRecord()
        action_history = []
        board_cards_int = []
        board_str = []
        
        hero_folded = False
        opp_folded = False
        
        streets = [
            ('preflop', 0, 0),
            ('flop', 1, 3),
            ('turn', 2, 1),
            ('river', 3, 1),
        ]
        
        for street_name, street_idx, num_cards in streets:
            if hero_folded or opp_folded:
                break
            if hero_stack <= 0 and opp_stack <= 0:
                break  # Both all-in
                
            # Deal community cards
            if num_cards > 0:
                new_cards = deck.draw(num_cards)
                if isinstance(new_cards, int):
                    new_cards = [new_cards]
                board_cards_int.extend(new_cards)
                board_str = [Card.int_to_str(c) for c in board_cards_int]
            
            # Calculate equities
            hero_equity = self._calculate_equity(hero_cards_str, board_str, 1)
            opp_equity = self._calculate_equity(opp_cards_str, board_str, 1)
            
            is_preflop = (street_idx == 0)
            
            # Determine action order (pre-flop: opp acts first as BB checks/raises)
            # Simplified: Hero always acts first post-flop, opp acts first pre-flop
            if is_preflop:
                actors = [('hero', hero_equity), ('opp', opp_equity)]
            else:
                actors = [('hero', hero_equity), ('opp', opp_equity)]
            
            current_bet = 0.0  # Current street bet to call
            if is_preflop:
                current_bet = bb - sb  # Hero (SB) needs to call the BB difference
            
            for actor_name, actor_equity in actors:
                if hero_folded or opp_folded:
                    break
                    
                if actor_name == 'hero':
                    if hero_stack <= 0:
                        continue  # All-in already
                    
                    call_amt = min(current_bet, hero_stack)
                    pot_odds = call_amt / (pot + call_amt) if (pot + call_amt) > 0 else 0.0
                    
                    # Record decision point BEFORE acting
                    record.add_decision(
                        street=street_idx,
                        equity=hero_equity,
                        pot_size=pot,
                        call_amount=call_amt,
                        hero_stack=hero_stack,
                        num_opponents=1,
                        action_taken=-1,  # Placeholder, filled below
                        chips_committed_before=hero_committed,
                        opp_vpip_norm=opponent.vpip_normalized,
                        opp_agg_norm=opponent.agg_normalized,
                        hero_cards_str=hero_cards_str,
                        board_str=list(board_str),
                        action_history=action_history,
                    )
                    
                    # Hero decides
                    decision = self._hero_decide(
                        hero_equity, pot, call_amt, hero_stack, 1, is_preflop,
                        opponent.vpip_normalized, opponent.agg_normalized
                    )
                    
                    # Map decision to action index
                    if decision == 'fold':
                        action_idx = 0
                        hero_folded = True
                        action_history.append('f')
                    elif decision == 'call':
                        action_idx = 1
                        hero_stack -= call_amt
                        hero_committed += call_amt
                        pot += call_amt
                        current_bet = 0.0
                        action_history.append('c')
                    else:  # raise
                        action_idx = 2
                        raise_amt = min(pot * 0.75, hero_stack)  # 3/4 pot raise
                        raise_amt = max(raise_amt, call_amt + bb)  # Min raise
                        raise_amt = min(raise_amt, hero_stack)
                        hero_stack -= raise_amt
                        hero_committed += raise_amt
                        pot += raise_amt
                        current_bet = raise_amt - call_amt  # New bet for opponent
                        action_history.append('r')
                    
                    # Update the last decision point with actual action
                    record.decision_points[-1]['action'] = action_idx
                    
                    if is_preflop:
                        opponent.record_preflop(decision)  # Track for HUD (opponent's perspective not relevant, but we track Hero's opponent)
                    
                else:  # Opponent
                    if opp_stack <= 0:
                        continue
                    
                    call_amt = min(current_bet, opp_stack)
                    pot_odds = call_amt / (pot + call_amt) if (pot + call_amt) > 0 else 0.0
                    
                    if is_preflop:
                        decision = opponent.decide_preflop(opp_equity, pot_odds)
                        opponent.record_preflop(decision)
                    else:
                        decision = opponent.decide_postflop(
                            opp_equity, pot_odds, pot, opp_stack, street_idx
                        )
                        opponent.record_postflop(decision)
                    
                    if decision == 'fold':
                        opp_folded = True
                    elif decision == 'call':
                        opp_stack -= call_amt
                        opp_committed += call_amt
                        pot += call_amt
                        current_bet = 0.0
                    else:  # raise
                        raise_amt = min(pot * 0.75, opp_stack)
                        raise_amt = max(raise_amt, call_amt + bb)
                        raise_amt = min(raise_amt, opp_stack)
                        opp_stack -= raise_amt
                        opp_committed += raise_amt
                        pot += raise_amt
                        current_bet = raise_amt - call_amt
                        
                        # Opponent raised — Hero needs to respond if not already acted
                        # (simplified: allow one re-action per street)
                        if not hero_folded and hero_stack > 0:
                            hero_call_amt = min(current_bet, hero_stack)
                            hero_equity_2 = hero_equity  # Re-use same equity
                            
                            record.add_decision(
                                street=street_idx,
                                equity=hero_equity_2,
                                pot_size=pot,
                                call_amount=hero_call_amt,
                                hero_stack=hero_stack,
                                num_opponents=1,
                                action_taken=-1,
                                chips_committed_before=hero_committed,
                                opp_vpip_norm=opponent.vpip_normalized,
                                opp_agg_norm=opponent.agg_normalized,
                                hero_cards_str=hero_cards_str,
                                board_str=list(board_str),
                                action_history=action_history,
                            )
                            
                            resp = self._hero_decide(
                                hero_equity_2, pot, hero_call_amt, hero_stack, 1, is_preflop,
                                opponent.vpip_normalized, opponent.agg_normalized
                            )
                            
                            if resp == 'fold':
                                record.decision_points[-1]['action'] = 0
                                hero_folded = True
                                action_history.append('f')
                            elif resp == 'call':
                                record.decision_points[-1]['action'] = 1
                                hero_stack -= hero_call_amt
                                hero_committed += hero_call_amt
                                pot += hero_call_amt
                                current_bet = 0.0
                                action_history.append('c')
                            else:  # re-raise
                                record.decision_points[-1]['action'] = 2
                                reraise = min(pot * 0.75, hero_stack)
                                reraise = max(reraise, hero_call_amt + bb)
                                reraise = min(reraise, hero_stack)
                                hero_stack -= reraise
                                hero_committed += reraise
                                pot += reraise
                                current_bet = 0.0  # Cap at one re-raise per street
                                action_history.append('r')
        
        # 2. Showdown or fold resolution
        if hero_folded:
            hero_profit = -hero_committed
        elif opp_folded:
            hero_profit = pot - hero_committed
        else:
            # Showdown
            if len(board_cards_int) < 5:
                # Deal remaining cards for showdown
                remaining = 5 - len(board_cards_int)
                extra = deck.draw(remaining)
                if isinstance(extra, int):
                    extra = [extra]
                board_cards_int.extend(extra)
            
            hero_score = _treys_evaluator.evaluate(board_cards_int, hero_cards_int)
            opp_score = _treys_evaluator.evaluate(board_cards_int, opp_cards_int)
            
            if hero_score < opp_score:  # Lower is better in treys
                hero_profit = pot - hero_committed
            elif hero_score > opp_score:
                hero_profit = -hero_committed
            else:
                hero_profit = (pot / 2.0) - hero_committed  # Split pot
        
        # Normalize profit to BB
        record.final_hero_profit = hero_profit / self.bb_size
        
        # Update stats
        self.total_hands += 1
        self.hero_total_profit += record.final_hero_profit
        
        # Filter out hands with no Hero decisions
        if not record.decision_points:
            return None
        
        # Filter out records where action was never set (shouldn't happen, but safety)
        record.decision_points = [dp for dp in record.decision_points if dp['action'] >= 0]
        if not record.decision_points:
            return None
            
        return record
    
    @property
    def hero_bb_per_100(self):
        """Hero's win rate in BB/100 hands."""
        if self.total_hands == 0:
            return 0.0
        return (self.hero_total_profit / self.total_hands) * 100


if __name__ == '__main__':
    from tools.self_play.opponent_bots import create_opponent_pool
    
    pool = create_opponent_pool()
    sim = HeadlessPokerSimulator(pool, equity_sims=50)
    
    print("Running 1000 test hands...")
    total_samples = 0
    for i in range(1000):
        record = sim.simulate_hand()
        if record:
            samples = record.get_training_samples()
            total_samples += len(samples)
    
    print(f"Hands: {sim.total_hands}")
    print(f"Training samples: {total_samples}")
    print(f"Hero win rate: {sim.hero_bb_per_100:.1f} BB/100")
    
    # Print opponent HUD stats
    for bot, _ in pool:
        print(f"  {bot.name}: VPIP={bot.vpip*100:.1f}%, AGG={bot.agg_factor:.1f}")

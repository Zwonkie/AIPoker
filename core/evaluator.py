import random
from treys import Card, Evaluator, Deck

class PokerEvaluator:
    def __init__(self):
        self.evaluator = Evaluator()

    def parse_card(self, card_str: str) -> int:
        """
        Converts standard 2-char string (e.g. 'Ah', 'Kd') to treys Card int.
        Handles rank conversion for '10' -> 'T'.
        """
        if len(card_str) == 3 and card_str.startswith('10'):
            card_str = 'T' + card_str[2]
        # Ensure correct capitalization (e.g., Ah, Kd, 3s)
        if len(card_str) == 2:
            rank = card_str[0].upper()
            suit = card_str[1].lower()
            return Card.new(f"{rank}{suit}")
        raise ValueError(f"Invalid card string: {card_str}")

    def evaluate_hand(self, board: list, hand: list):
        """
        Evaluates a 5, 6, or 7 card hand. Returns score and rank description.
        Lower score is better.
        """
        board_ints = [self.parse_card(c) if isinstance(c, str) else c for c in board]
        hand_ints = [self.parse_card(c) if isinstance(c, str) else c for c in hand]
        
        score = self.evaluator.evaluate(board_ints, hand_ints)
        rank_class = self.evaluator.get_rank_class(score)
        class_str = self.evaluator.class_to_string(rank_class)
        percentage = self.evaluator.get_five_card_rank_percentage(score)
        
        return score, class_str, percentage

    def calculate_equity(self, board: list, hand: list, num_opponents: int = 1, num_simulations: int = 2000, specific_opponents: list = None):
        """
        Calculates hand equity using Monte Carlo simulations.
        If specific_opponents is provided (list of 2-card lists), it evaluates exactly against those hands.
        """
        try:
            board_ints = [self.parse_card(c) if isinstance(c, str) else c for c in board]
            hand_ints = [self.parse_card(c) if isinstance(c, str) else c for c in hand]
            
            specific_opp_ints = []
            if specific_opponents is not None:
                for opp_hand in specific_opponents:
                    specific_opp_ints.append([self.parse_card(c) if isinstance(c, str) else c for c in opp_hand])
        except Exception as e:
            return 0.0, f"Error parsing cards: {e}"

        known_cards = board_ints + hand_ints
        if specific_opp_ints:
            for opp_hand in specific_opp_ints:
                known_cards.extend(opp_hand)
                
        # Build the deck, excluding known cards
        full_deck = Deck.GetFullDeck()
        remaining_deck = [c for c in full_deck if c not in known_cards]
        
        wins = 0
        ties = 0
        losses = 0
        
        needed_board = 5 - len(board_ints)
        
        if specific_opp_ints:
            num_opponents = len(specific_opp_ints)
            needed_opponents_cards = 0
        else:
            needed_opponents_cards = 2 * num_opponents
            
        total_needed = needed_board + needed_opponents_cards
        
        if len(remaining_deck) < total_needed:
            return 0.0, "Not enough cards in deck for simulation"

        # Pre-convert list for faster indexing
        for _ in range(num_simulations):
            # Draw random cards from the remaining deck
            sim_cards = random.sample(remaining_deck, total_needed)
            
            # Distribute cards
            sim_board = board_ints + sim_cards[:needed_board]
            
            if specific_opp_ints:
                opponents_hands = specific_opp_ints
            else:
                opponents_hands = []
                idx = needed_board
                for _ in range(num_opponents):
                    opponents_hands.append(sim_cards[idx:idx+2])
                    idx += 2
                
            # Evaluate hero score
            hero_score = self.evaluator.evaluate(sim_board, hand_ints)
            
            # Evaluate opponents scores
            best_opp_score = float('inf')
            opp_scores = []
            for opp_hand in opponents_hands:
                opp_score = self.evaluator.evaluate(sim_board, opp_hand)
                opp_scores.append(opp_score)
                if opp_score < best_opp_score:
                    best_opp_score = opp_score
            
            # Compare (lower score is better)
            if hero_score < best_opp_score:
                wins += 1
            elif hero_score == best_opp_score:
                # Count how many tied for the best score
                num_ties = opp_scores.count(hero_score) + 1
                ties += 1
            else:
                losses += 1
                
        win_pct = (wins / num_simulations) * 100
        tie_pct = (ties / num_simulations) * 100
        loss_pct = (losses / num_simulations) * 100
        equity = (wins / num_simulations) + ((ties / num_simulations) / (num_opponents + 1))
        
        return equity, f"Simulated {num_simulations} hands: W={win_pct:.1f}%, D={tie_pct:.1f}%, L={loss_pct:.1f}%"

    def analyze_board_texture(self, board: list) -> dict:
        """
        Analyzes the community board cards for texture properties:
        - wetness: float (0.0 = dry, 1.0 = wet)
        - has_flush_draw: bool
        - has_straight_draw: bool
        - num_suited: int
        - max_connected: int
        """
        if len(board) < 3:
            return {'wetness': 0.0, 'has_flush_draw': False, 'has_straight_draw': False, 'num_suited': 0, 'max_connected': 0}
            
        ranks = []
        suits = []
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        for card_str in board:
            if len(card_str) == 3 and card_str.startswith('10'):
                rank = 'T'
                suit = card_str[2]
            else:
                rank = card_str[0].upper()
                suit = card_str[1].lower()
            ranks.append(rank_values.get(rank, 0))
            suits.append(suit)
            
        # 1. Flush potential
        suit_counts = {}
        for s in suits:
            suit_counts[s] = suit_counts.get(s, 0) + 1
        max_suited = max(suit_counts.values()) if suit_counts else 0
        has_flush_draw = (max_suited >= 3)
        
        # 2. Straight potential
        unique_ranks = sorted(list(set(ranks)))
        if 14 in unique_ranks:
            unique_ranks = [1] + unique_ranks  # Add Ace as 1
            unique_ranks = sorted(list(set(unique_ranks)))
            
        max_connected = 1
        current_connected = 1
        for i in range(1, len(unique_ranks)):
            if unique_ranks[i] == unique_ranks[i-1] + 1:
                current_connected += 1
            else:
                max_connected = max(max_connected, current_connected)
                current_connected = 1
        max_connected = max(max_connected, current_connected)
        has_straight_draw = (max_connected >= 3)
        
        # 3. Calculate wetness score (0.0 to 1.0)
        wetness = 0.0
        if max_suited >= 3:
            wetness += 0.4
        if max_suited >= 4:
            wetness += 0.3
        if max_connected >= 3:
            wetness += 0.2
        if max_connected >= 4:
            wetness += 0.1
            
        wetness = min(1.0, wetness)
        
        return {
            'wetness': wetness,
            'has_flush_draw': has_flush_draw,
            'has_straight_draw': has_straight_draw,
            'num_suited': max_suited,
            'max_connected': max_connected
        }

if __name__ == '__main__':
    # Simple test
    pe = PokerEvaluator()
    hand = ['As', 'Ks']
    board = ['Js', 'Qs', '2d']
    eq, log_msg = pe.calculate_equity(board, hand, num_opponents=1, num_simulations=5000)
    print(f"Hand: {hand}, Board: {board}")
    print(f"Equity: {eq * 100:.2f}%")
    print(log_msg)

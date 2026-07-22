import torch
import os
import itertools

class CudaPokerEvaluator:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        tensor_path = os.path.join(os.path.dirname(__file__), 'cuda_evaluator', 'treys_tensors.pt')
        if not os.path.exists(tensor_path):
            raise FileNotFoundError(f"Treys lookup tensors not found at {tensor_path}.")
            
        data = torch.load(tensor_path, map_location=self.device)
        self.flush_lookup = data['flush'].to(self.device)
        self.unflush_lookup = data['unflush'].to(self.device)
        self.deck_ints = data['deck_ints'].to(self.device)
        self.card_strs = data['card_strs']
        
        self.str_to_idx = {s: i for i, s in enumerate(self.card_strs)}
        
        self.combos = {
            5: torch.tensor(list(itertools.combinations(range(5), 5)), dtype=torch.long, device=self.device),
            6: torch.tensor(list(itertools.combinations(range(6), 5)), dtype=torch.long, device=self.device),
            7: torch.tensor(list(itertools.combinations(range(7), 5)), dtype=torch.long, device=self.device)
        }
        
    def _parse_cards(self, cards_list):
        cleaned = []
        for hand in cards_list:
            c_hand = []
            for c in hand:
                if len(c) == 3 and c.startswith('10'):
                    c = 'T' + c[2]
                rank = c[0].upper()
                suit = c[1].lower()
                c_hand.append(f"{rank}{suit}")
            cleaned.append(c_hand)
            
        indices = [[self.str_to_idx[c] for c in hand] for hand in cleaned]
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
        return self.deck_ints[idx_tensor]
        
    def evaluate_batched(self, hands_ints):
        N, num_cards = hands_ints.shape
        if num_cards < 5:
            raise ValueError("Need at least 5 cards to evaluate a poker hand.")
            
        c_idx = self.combos[num_cards]
        expanded = hands_ints[:, c_idx] # [N, C, 5]
        
        suits_and = expanded[:, :, 0] & expanded[:, :, 1] & expanded[:, :, 2] & expanded[:, :, 3] & expanded[:, :, 4]
        is_flush = (suits_and & 0xF000) != 0
        
        flush_q = (expanded[:, :, 0] | expanded[:, :, 1] | expanded[:, :, 2] | expanded[:, :, 3] | expanded[:, :, 4]) >> 16
        primes = (expanded[:, :, 0] & 0xFF) * (expanded[:, :, 1] & 0xFF) * (expanded[:, :, 2] & 0xFF) * (expanded[:, :, 3] & 0xFF) * (expanded[:, :, 4] & 0xFF)
        
        flush_q = flush_q.to(torch.long)
        primes = primes.to(torch.long)
        
        scores = torch.where(is_flush, self.flush_lookup[flush_q], self.unflush_lookup[primes])
        best_scores, _ = scores.min(dim=1)
        
        return best_scores

    def calculate_equity_batched(self, hero_hands, boards, num_opponents=1, num_simulations=2000):
        N = len(hero_hands)
        B = len(boards[0]) if N > 0 else 0
        
        h_ints = self._parse_cards(hero_hands)
        b_ints = self._parse_cards(boards) if B > 0 else torch.empty((N, 0), dtype=torch.int32, device=self.device)
        
        needed_board = 5 - B
        needed_opponents_cards = 2 * num_opponents
        total_needed = needed_board + needed_opponents_cards
        
        deck_mask = torch.ones((N, 52), dtype=torch.bool, device=self.device)
        
        h_idx = torch.tensor([[self.str_to_idx[c] for c in hand] for hand in hero_hands], device=self.device)
        if B > 0:
            b_idx = torch.tensor([[self.str_to_idx[c] for c in board] for board in boards], device=self.device)
        else:
            b_idx = torch.empty((N, 0), dtype=torch.long, device=self.device)
            
        deck_mask.scatter_(1, h_idx, False)
        if B > 0:
            deck_mask.scatter_(1, b_idx, False)
            
        wins = torch.zeros(N, device=self.device, dtype=torch.float32)
        ties = torch.zeros(N, device=self.device, dtype=torch.float32)
        
        chunk_size = min(num_simulations, 2000)
        num_chunks = (num_simulations + chunk_size - 1) // chunk_size
        
        hero_known = torch.cat([h_ints, b_ints], dim=1)
        
        for _ in range(num_chunks):
            r = torch.rand((N, chunk_size, 52), device=self.device)
            r = torch.where(deck_mask.unsqueeze(1), r, torch.tensor(2.0, device=self.device))
            sorted_indices = r.argsort(dim=-1)
            
            drawn_indices = sorted_indices[:, :, :total_needed]
            drawn_cards = self.deck_ints[drawn_indices]
            
            sim_board = drawn_cards[:, :, :needed_board]
            
            hero_expanded = hero_known.unsqueeze(1).expand(-1, chunk_size, -1)
            hero_final = torch.cat([hero_expanded, sim_board], dim=2)
            
            hero_scores = self.evaluate_batched(hero_final.reshape(-1, 7)).view(N, chunk_size)
            
            best_opp_scores = torch.full((N, chunk_size), 9999, device=self.device, dtype=torch.int16)
            
            idx = needed_board
            b_expanded = b_ints.unsqueeze(1).expand(-1, chunk_size, -1)
            for _ in range(num_opponents):
                opp_hole = drawn_cards[:, :, idx:idx+2]
                opp_final = torch.cat([opp_hole, b_expanded, sim_board], dim=2)
                opp_scores = self.evaluate_batched(opp_final.reshape(-1, 7)).view(N, chunk_size)
                best_opp_scores = torch.minimum(best_opp_scores, opp_scores)
                idx += 2
                
            wins += (hero_scores < best_opp_scores).float().sum(dim=1)
            ties += (hero_scores == best_opp_scores).float().sum(dim=1)
            
        actual_sims = num_chunks * chunk_size
        equity = (wins + (ties / (num_opponents + 1))) / actual_sims
        return equity.cpu().tolist()

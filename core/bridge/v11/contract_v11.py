import torch
from core.board_state import BoardState
from core.bridge.data_contract import DataContract
from typing import Tuple

# Must match exactly with the training vocabulary
VOCAB = {'<PAD>': 0, 'B': 1, 'b': 2, 'c': 3, 'k': 4, 'K': 5, 'r': 6, 'f': 7, 'A': 8, 'Q': 9}

def card_to_int(card_str: str) -> int:
    if not card_str or len(card_str) != 2:
        return 52 # PAD
    rank, suit = card_str[0], card_str[1]
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    try:
        r = ranks.index(rank)
        s = suits.index(suit)
        return s * 13 + r
    except ValueError:
        return 52

class ContractV8V9(DataContract):
    """
    Implements the 31-feature context extraction for Pluribus V8 and V9 models.
    (PokerEVModelV4 architecture)
    """
    
    def __init__(self, max_seq_len: int = 20):
        self.max_seq_len = max_seq_len

    def to_tensors(self, states, action_history_raw: list = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(states, list):
            states = [states]
            
        states = states[-self.max_seq_len:]
        start_idx = self.max_seq_len - len(states)
        
        # 1. Hole Cards (from final state)
        hole_ints = [card_to_int(c) for c in states[-1].hero_cards]
        while len(hole_ints) < 2:
            hole_ints.append(52)
            
        # 2. Board Cards Sequence (padded to max_seq_len)
        board_seq = [[52]*5 for _ in range(self.max_seq_len)]
        
        # 3. Context (31 features) Sequence
        context_seq = [[0.0]*31 for _ in range(self.max_seq_len)]
        
        for i, state in enumerate(states):
            idx = start_idx + i
            
            b_ints = [card_to_int(c) for c in state.community_cards]
            while len(b_ints) < 5:
                b_ints.append(52)
            board_seq[idx] = b_ints
            
            pot_odds = state.call_amount / (state.pot_size + state.call_amount) if (state.pot_size + state.call_amount) > 0 else 0.0
            
            board_len = len(state.community_cards)
            if board_len == 0: street_level = 0.0
            elif board_len == 3: street_level = 1.0
            elif board_len == 4: street_level = 2.0
            else: street_level = 3.0
                
            # HUD Maps
            vpip_map = {'Red': 0.45, 'Yellow': 0.30, 'Green': 0.22, 'Blue': 0.10}
            agg_map = {'Red': 0.85, 'Yellow': 0.63, 'Green': 0.46, 'Blue': 0.18}
            
            # Determine active opponents mask
            active_mask = []
            for j in range(5):
                seat_key = f"seat_{j+1}"
                if seat_key in state.seats:
                    active_mask.append(1.0 if state.seats[seat_key].is_active else 0.0)
                else:
                    active_mask.append(0.0)
                
            # Dynamically calculate global VPIP/AGG norms from active opponents
            total_active = sum(active_mask)
            if total_active > 0:
                sum_vpip = 0.0
                sum_agg = 0.0
                for j in range(5):
                    if active_mask[j] == 1.0:
                        seat_key = f"seat_{j+1}"
                        opp = state.seats[seat_key]
                        vpip_col = opp.hud.vpip_color
                        agg_col = opp.hud.agg_color
                        sum_vpip += vpip_map.get(vpip_col, 0.3)
                        sum_agg += agg_map.get(agg_col, 0.4)
                opp_vpip_norm = sum_vpip / total_active
                opp_agg_norm = sum_agg / total_active
            else:
                opp_vpip_norm = 0.3 
                opp_agg_norm = 0.4
            
            ctx = [
                float(state.hero_position) / 5.0,
                (state.hero_stack / state.big_blind) / 400.0,
                (state.pot_size / state.big_blind) / 1000.0,
                state.equity,
                pot_odds,
                sum(active_mask) / 10.0,
                street_level / 3.0,
                opp_vpip_norm,
                opp_agg_norm,
                
                # BB Ratios
                state.pot_size / state.big_blind,
                state.call_amount / state.big_blind
            ]
            
            # Opponents' seats HUD
            for j in range(5):
                seat_key = f"seat_{j+1}"
                opp = state.seats.get(seat_key)
                if opp:
                    opp_stack = opp.stack
                    vpip_col = opp.hud.vpip_color
                    agg_col = opp.hud.agg_color
                else:
                    opp_stack = 0.0
                    vpip_col = "Blue"
                    agg_col = "Blue"
                
                ctx.append(active_mask[j])
                ctx.append((opp_stack / state.big_blind) / 400.0)
                ctx.append(vpip_map.get(vpip_col, 0.3))
                ctx.append(agg_map.get(agg_col, 0.4))
                
            context_seq[idx] = ctx
            
        # 4. Action Sequence
        act_ints = [0] * self.max_seq_len
        if action_history_raw:
            # We only do this if we are replaying sequences (e.g. for ML training/diagnostics)
            acts = [VOCAB.get(char, 0) for char in action_history_raw]
            if len(acts) < self.max_seq_len:
                act_ints = [0] * (self.max_seq_len - len(acts)) + acts
            else:
                act_ints = acts[-self.max_seq_len:]
        
        # Convert to batch-first tensors [1, ...]
        hole_tensor = torch.tensor([hole_ints], dtype=torch.long)
        board_tensor = torch.tensor([board_seq], dtype=torch.long)
        ctx_tensor = torch.tensor([context_seq], dtype=torch.float32)
        act_tensor = torch.tensor([act_ints], dtype=torch.long)
        
        return hole_tensor, board_tensor, ctx_tensor, act_tensor

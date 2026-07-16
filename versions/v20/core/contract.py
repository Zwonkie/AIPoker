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

class ContractV12(DataContract):
    """
    Implements the 35-feature context extraction for Pluribus V12 models.
    (PokerEVModelV4 architecture)
    """
    
    def __init__(self, max_seq_len: int = 20):
        self.max_seq_len = max_seq_len

    def to_tensors(self, states, hero_actions: list = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(states, list):
            states = [states]
            
        states = states[-self.max_seq_len:]
        # Left-padding: start index is at the end of the array
        start_idx = self.max_seq_len - len(states)
        
        # 1. Hole Cards (from final state)
        hole_ints = [card_to_int(c) for c in states[-1].hero_cards]
        while len(hole_ints) < 2:
            hole_ints.append(52)
            
        # 2. Board Cards Sequence (padded to max_seq_len)
        board_seq = [[52]*5 for _ in range(self.max_seq_len)]
        
        # 3. Context (35 features) Sequence
        context_seq = [[0.0]*35 for _ in range(self.max_seq_len)]
        
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
            
            # V20 LIVE-SAFETY CLAMP: `stack_depth_mix` caps TRAINING stacks at 50bb (never higher --
            # this simulator has never generated a training example past that depth). Real live
            # tables routinely run 80-150bb+. Rescaling hero_stack/opp_stack to /100 (from /400)
            # gives 4x more resolution INSIDE the trained 5-50bb band, but the flip side is 4x LESS
            # headroom for anything past it -- confirmed via live smoke-test: a flopped SET OF ACES
            # facing a bet started folding more as simulated stack depth rose past the training cap
            # (fold% 10%->18%->25%->28%->32% at 45/60/80/100/150bb), a real, worsening OOD
            # extrapolation artifact, not a deliberate model preference. Clamping the STACK-DERIVED
            # features (not the real stack used for bet sizing elsewhere) to the training ceiling
            # keeps every live query in-distribution instead of extrapolating past what any weights
            # in this line have ever seen -- a no-op during training (stack_depth_mix already never
            # exceeds this) and strictly safer at serve time. Pot/call_amount clamped proportionally
            # (2x/1x the stack ceiling) for the same reason.
            _stack_ceil_bb = 50.0
            _pot_ceil_bb = 100.0
            _call_ceil_bb = 50.0
            _hero_stack_bb = min(state.hero_stack / state.big_blind, _stack_ceil_bb)
            _pot_bb = min(state.pot_size / state.big_blind, _pot_ceil_bb)
            _call_bb = min(state.call_amount / state.big_blind, _call_ceil_bb)

            ctx = [
                float(state.hero_position) / 5.0,
                # V20: rescaled 400->100 (stack) / 1000->250 (pot). This training line's
                # `stack_depth_mix` has capped stacks at 5-50bb since V15, but this normalization
                # was still calibrated for a hypothetical ~400bb range -- meaningfully different
                # stacks (15bb vs 40bb) landed only 0.0625 apart on a [0,1] feature, most of the
                # representable range never touched by any training example. Same issue as [P5]'s
                # already-flagged call_amount compression, just also hitting stack/pot. See
                # versions/v20/SPECS.md. NOT backward-compatible with older checkpoints' learned
                # scale -- contract_version bumped, requires fresh training + its own live bridge.
                _hero_stack_bb / 100.0,
                _pot_bb / 250.0,
                state.equity,
                pot_odds,
                sum(active_mask) / 10.0,
                street_level / 3.0,
                opp_vpip_norm,
                opp_agg_norm,

                # BB Ratios
                _call_bb / 100.0
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
                    vpip_col = "Yellow"
                    agg_col = "Green"
                
                opp_pos = (j + 1 + state.hero_position) % 6
                pos_val = float(opp_pos) / 5.0 if active_mask[j] == 1.0 else -1.0
                
                ctx.append(active_mask[j])
                ctx.append(pos_val)
                ctx.append(min(opp_stack / state.big_blind, _stack_ceil_bb) / 100.0)   # V20: rescaled + clamped, see ctx[1] comment above
                ctx.append(vpip_map.get(vpip_col, 0.3))
                ctx.append(agg_map.get(agg_col, 0.4))
                
            context_seq[idx] = ctx
            
        # 4. Action Sequence
        act_ints = [0] * self.max_seq_len
        if hero_actions is not None:
            # Transformer shifts actions by 1 internally. act_ints[i] should be the action taken AT states[i]
            for i in range(min(len(hero_actions), len(states))):
                act_ints[start_idx + i] = hero_actions[i]
        
        # Convert to batch-first tensors [1, ...]
        hole_tensor = torch.tensor([hole_ints], dtype=torch.long)
        board_tensor = torch.tensor([board_seq], dtype=torch.long)
        ctx_tensor = torch.tensor([context_seq], dtype=torch.float32)
        act_tensor = torch.tensor([act_ints], dtype=torch.long)
        
        return hole_tensor, board_tensor, ctx_tensor, act_tensor

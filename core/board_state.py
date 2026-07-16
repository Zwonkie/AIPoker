from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class HUDStats:
    vpip: float = 0.0
    agg: float = 0.0
    vpip_color: str = "Blue"
    agg_color: str = "Blue"

@dataclass
class SeatState:
    name: str = ""
    stack: float = 0.0
    is_active: bool = True
    state_label: str = "Active"  # e.g., "Folded", "All-In", "Active"
    hud: HUDStats = field(default_factory=HUDStats)

@dataclass
class BoardState:
    """
    Pure mathematical representation of the poker table at a single point in time.
    Single Source of Truth for Dashboards, Data Contracts, and Evaluators.
    """
    community_cards: List[str] = field(default_factory=list)
    hero_cards: List[str] = field(default_factory=list)
    pot_size: float = 0.0
    hero_stack: float = 0.0
    seats: Dict[str, SeatState] = field(default_factory=dict) # e.g., "seat_1": SeatState
    active_buttons: List[str] = field(default_factory=list)
    dealer_idx: int = 0
    hero_position: int = 0
    street: str = "Preflop"
    call_amount: float = 0.0
    equity: float = 0.0
    big_blind: float = 10.0
    # [V20_preflopEq] Field-independent hand-quality signal (0.5 = neutral default): preflop an
    # O(1) lookup into preflop_equities.csv's 169-hand vs-1-random table, postflop a cheap live
    # vs-1-random MC call -- see versions/v20_preflopEq/core/contract.py. Populated by the CALLER
    # (mirrors `equity` itself), read by ContractV12.to_tensors. Optional/additive: every other
    # version's contract simply never reads this field, so this default is inert for them.
    hand_strength: float = 0.5
    
    @property
    def num_active_players(self) -> int:
        # +1 assuming Hero is included in the active players count if playing
        # Actually, let's strictly count active seats + hero if hero is active
        return sum(1 for s in self.seats.values() if s.is_active) + 1

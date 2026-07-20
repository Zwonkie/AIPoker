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
    # [V22] Chips this opponent has already put into THIS hand's pot (raw chips, same convention
    # as `stack` -- contract.py scales it via the same scaled_stack_bb() helper). Distinguishes two
    # opponents with identical remaining stack/HUD color who got there via very different lines
    # this hand (e.g. limped-in vs 3-bet) -- `stack` alone (a remaining-stack snapshot) can't tell
    # them apart. Optional/additive, mirrors `hand_strength`'s pattern below: every earlier
    # version's contract simply never reads this field, so the 0.0 default is inert for them.
    committed: float = 0.0
    # [V29, OPP-2] Whether THIS specific opponent has raised at least once so far this hand /
    # on the current betting street -- distinguishes a specific seat's own in-hand aggression from
    # the static, cross-hand VPIP/AGG HUD colors above, which is all a decision previously had to
    # go on for "who specifically did what" (see .agents/skills/OFK/references/
    # known-shortcomings-backlog.md [OPP-2]). Optional/additive, same convention as `committed`:
    # every earlier version's contract simply never reads these fields, so the False default is
    # inert for them.
    raised_this_hand: bool = False
    raised_this_street: bool = False

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
    # [V22] Chips HERO has already put into THIS hand's pot (raw chips, same convention as
    # `hero_stack`) -- see SeatState.committed above for the same idea on the opponent side.
    # Optional/additive: inert (0.0) for every version whose contract doesn't read it.
    hero_committed: float = 0.0
    # [V23] Whole-hand raise count so far, bucketed: 0=limped/unraised, 1=single-raised,
    # 2=3-bet-or-more. A hand-level property (not per-seat like `committed` above) -- distinguishes
    # a pot that's seen one raise from one that's been 3-bet+, which `committed`/`call_amount`
    # alone don't cleanly capture (a big call_amount can arise from one big bet OR from a raise war,
    # and those are different situations). Optional/additive: inert (0) for every version whose
    # contract doesn't read it.
    pot_type: int = 0

    @property
    def num_active_players(self) -> int:
        # +1 assuming Hero is included in the active players count if playing
        # Actually, let's strictly count active seats + hero if hero is active
        return sum(1 for s in self.seats.values() if s.is_active) + 1

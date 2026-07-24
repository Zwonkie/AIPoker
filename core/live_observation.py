"""LiveObservation -- the frozen, model-agnostic handover object between the live dashboard
(vision -> TableState) and whatever model is serving.

[V45_liveHandover] This is the boundary the live layer publishes and the ONLY thing a model-side
adapter (core/live_adapter.py) is allowed to read. Design rules, in order of importance:

1. RAW FACTS ONLY. No equity, no hand_strength, no scaled features, no per-version defaults --
   those are the *model's* interpretation of the table and live in the adapter for whichever
   version is active. The observation says what the screen said, stabilized by TableState.

2. SENTINELS ARE None, NEVER PLAUSIBLE VALUES. An unread HUD badge is `vpip_color=None`, not
   'Blue' or 'Yellow'; an unread price is `call_amount_known=False`, not 0.0. The adapter decides
   what its training used as the default (V42_liveFixes finding #8-CE is the canonical example of
   why: live defaulted an unknown opponent to super-nit while training defaulted to average).

3. APPEND-ONLY EVOLUTION, like the tensor contract itself: new fields get defaults that mean
   "absent", existing fields never change meaning or units. `schema` bumps when a field is added
   so recorded observations stay replayable forever.

4. FROZEN + JSON-SERIALIZABLE. Adapters cannot mutate the live state through it, and the turn
   recorder can persist it verbatim (history/<board_id>/turns.jsonl) for offline replay of any
   hand through any adapter.

Units: all money fields are raw table chips (the same denomination TableState/vision use --
NB: on decimal-stake tables that is cents, see V42_liveFixes A2). Positions are button-relative
over the OCCUPIED ring (BU=0, SB=1, BB=2, ...), the V42_liveFixes C3 convention.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

# The physical table layout convention this codebase has always used: seat index increases in the
# direction of action, hero at index 0. (Same constant as PHPHelp.SEAT_ORDER_CLOCKWISE -- defined
# here so the model-side classifier needs nothing from the dashboard module.)
SEAT_ORDER_CLOCKWISE = ('hero', 'seat_1', 'seat_2', 'seat_3', 'seat_4', 'seat_5')

OBSERVATION_SCHEMA = 1


@dataclass(frozen=True)
class SeatObservation:
    """One opponent seat, exactly as the live layer knows it. `occupied=False` means the seat has
    not been seen at all this hand -- every other field is then meaningless and left at its
    'absent' default."""
    seat_key: str                          # physical seat: 'seat_1'..'seat_5'
    occupied: bool = False
    name: str = ""
    is_active: bool = False                # still contesting THIS hand (fold-latched, monotonic)
    state_label: str = "Folded"            # vision's label: 'Active' | 'Folded' | 'All-In'
    stack: float = 0.0                     # chips, monotonic-decay stabilized
    committed: float = 0.0                 # chips into this hand's pot (start stack - current)
    raised_this_hand: bool = False         # [OPP-2] live raise attribution
    raised_this_street: bool = False
    vpip_color: Optional[str] = None       # RAW HUD read; None == not classified yet (rule 2)
    agg_color: Optional[str] = None
    position: Optional[int] = None         # true button-relative position over the occupied ring
    contract_slot: Optional[str] = None    # 'seat_N' slot whose contract-derived position == true
                                           # position (V42_liveFixes C3); None if position unknown
    is_small_blind: bool = False
    is_big_blind: bool = False


@dataclass(frozen=True)
class LiveObservation:
    """A complete, self-contained snapshot of one hero decision point."""
    schema: int = OBSERVATION_SCHEMA

    # --- board ---
    community_cards: Tuple[str, ...] = ()
    street: str = "Preflop"                # derived from card count: 0/3/4/5; else 'Unknown'
    pot_size: float = 0.0                  # chips, median-filtered + monotonic
    big_blind: float = 10.0

    # --- hero ---
    hero_cards: Tuple[str, ...] = ()
    hero_stack: float = 0.0                # chips; 0.0 == never read this hand (see quality flags)
    hero_committed: float = 0.0
    hero_position: int = 0                 # BU=0, SB=1, BB=2 ... over the occupied ring
    hero_is_small_blind: bool = False
    hero_is_big_blind: bool = False
    hero_raised_this_hand: bool = False
    hero_raised_this_street: bool = False

    # --- table ---
    dealer_idx: int = 0                    # 0 == hero holds the button
    dealer_detected: bool = False          # False == dealer_idx is the fallback default, not a read
    seated_count: int = 6                  # players actually seen at the table (occupied ring size)
    raise_count: int = 0                   # whole-hand raise EVENTS (source for pot_type buckets)
    current_street_bet_level: float = 0.0  # largest single-player contribution seen this street
    seats: Tuple[SeatObservation, ...] = ()  # always 5 entries, seat_1..seat_5, occupied or not

    # --- the price hero faces, honestly ---
    call_amount: Optional[float] = None    # chips; None == no reading AND no estimate available
    call_amount_known: bool = False        # True only when the price was positively read
    check_call_available: bool = True      # buttons actually on screen this frame
    bet_raise_available: bool = True
    active_buttons: Tuple[str, ...] = ()

    # --- history ---
    hero_action_history: Tuple[str, ...] = ()   # hero's own line this hand: 'f'/'c'/'r' tokens

    # --- provenance ---
    ts_epoch: float = 0.0                  # caller-supplied wall clock (0.0 = not stamped)
    source: str = "live"                   # 'live' | 'mock' | 'replay'

    # ------------------------------------------------------------------ #
    def seat(self, seat_key: str) -> Optional[SeatObservation]:
        for s in self.seats:
            if s.seat_key == seat_key:
                return s
        return None

    @property
    def active_seats(self) -> Tuple[SeatObservation, ...]:
        return tuple(s for s in self.seats if s.occupied and s.is_active)

    @property
    def num_active_opponents(self) -> int:
        return len(self.active_seats)

    @property
    def is_preflop(self) -> bool:
        return len(self.community_cards) == 0

    def to_json_dict(self) -> dict:
        """Plain-dict form for the turn recorder / replay harnesses. Round-trips via
        `from_json_dict` -- tuples become lists in JSON and are restored on load."""
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: dict) -> "LiveObservation":
        d = dict(d or {})
        # Forward-compat: ignore fields a NEWER schema added (append-only rule keeps this safe).
        # Applied to the nested seat dicts as well as the top-level record -- a file recorded by a
        # newer SeatObservation schema must not blow up on an unexpected keyword argument, or the
        # "replayable forever" guarantee above only holds for half the record.
        seat_known = set(SeatObservation.__dataclass_fields__)
        seats = tuple(
            SeatObservation(**{k: v for k, v in s.items() if k in seat_known})
            for s in (d.pop('seats', None) or [])
        )
        for key in ('community_cards', 'hero_cards', 'active_buttons', 'hero_action_history'):
            if key in d and d[key] is not None:
                d[key] = tuple(d[key])
        known = set(cls.__dataclass_fields__)
        d = {k: v for k, v in d.items() if k in known}
        return cls(seats=seats, **d)


def street_from_board(community_cards) -> str:
    """The one street derivation, shared with TableState.to_dict/to_board_state: 0/3/4/5 cards map
    to a real street, anything else is 'Unknown' (a mid-deal/partial frame -- callers must treat it
    as not decidable, see V42_liveFixes C4)."""
    n = len(community_cards or [])
    if n == 0:
        return 'Preflop'
    if n == 3:
        return 'Flop'
    if n == 4:
        return 'Turn'
    if n == 5:
        return 'River'
    return 'Unknown'

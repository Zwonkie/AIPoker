import difflib

class TableState:
    """
    Data model that tracks and stabilizes the poker table state over time.
    Filters out transient OCR errors (noise) using history, monotonicity constraints,
    and median filtering.
    """
    def __init__(self):
        self.reset()
        
    def reset(self, big_blind: float = None):
        """Clears state history for a new hand.

        [V29 live-info expansion] `big_blind`: if provided, persists across the reset (a table's
        blinds don't change hand-to-hand) and seeds `current_street_bet_level` to it -- preflop's
        real opening price to call is the BB, not 0. Without this seed, the very first voluntary
        entry (e.g. UTG limping in for exactly the BB) would be misread as a "raise" by the
        stack-drop diff logic below (any diff > the previous street_bet_level, which defaulted to
        0, looks like a raise) -- a real train/live mismatch, since training never treats a limp as
        a raise. Falls back to the last-known `self.big_blind` (or 10.0 on the very first call)
        if the caller doesn't pass one, so existing call sites that don't update immediately still
        get a sane, non-zero preflop seed rather than reverting to the old (wrong) 0.0 behavior.
        """
        self.community_cards = []
        self.hero_cards = []
        self.pot_size = 0.0
        self.hero_stack = 0
        self.opponents = {}
        self.active_buttons = []
        self.dealer_name = ""
        self.dealer_idx = 0
        self.hero_position = 0

        if big_blind is not None:
            self.big_blind = big_blind
        elif not hasattr(self, 'big_blind'):
            self.big_blind = 10.0

        # Action tracking for ML
        self.action_history = []
        # [V29] Preflop's real opening price is the big blind, not 0 -- see docstring above.
        self.current_street_bet_level = self.big_blind
        self.last_known_stacks = {}
        self.last_street = 'Preflop'

        # [V29, OPP-2 live] Per-seat ('Hero' + 'seat_N') in-hand raise attribution -- see
        # `_generate_timeline_actions`'s stack-drop-diff classifier below.
        # `raised_this_hand` persists the whole hand; `raised_this_street` resets every street.
        self.raised_this_hand = {}
        self.raised_this_street = {}
        # [V29] Whole-hand raise EVENT count (every raise, including repeat raises by the same
        # seat e.g. a 4-bet) -- source for `pot_type` (bucketed 0/1/2+), mirrors the simulator's
        # own `raise_count` (see versions/v29/self_play/simulator.py).
        self.raise_count = 0
        # [V29] Chips each player had at the START of this hand (first stack ever observed for
        # them this hand) -- `committed` is derived as start_stack - current_stack. Populated
        # lazily the first time each player's stack is seen (see `update()`), not here, since
        # stacks aren't known yet at reset time.
        self.hand_start_stacks = {}

        # Internal buffers for temporal filtering
        self._pot_buffer = []
        
    def detect_hand_reset(self, raw_state: dict) -> bool:
        """
        Detects if a new hand has started based on raw vision state.
        Returns True if a reset is detected.
        """
        # 1. Community cards disappeared (e.g., transition from River to Preflop)
        if len(raw_state.get('community_cards', [])) == 0 and len(self.community_cards) > 0:
            return True
            
        # 2. Pot size dropped significantly (payout occurred)
        raw_pot = raw_state.get('pot_size', 0.0)
        if raw_pot < self.pot_size * 0.5 and self.pot_size > 5.0:
            return True
            
        # 3. Hero cards changed completely
        raw_hero = raw_state.get('hero_cards', [])
        if len(raw_hero) == 2 and len(self.hero_cards) == 2:
            if set(raw_hero) != set(self.hero_cards):
                return True
                
        return False
        
    def update(self, raw_state: dict):
        """
        Updates the stabilized table state using the latest raw OCR reading.
        Applies monotonicity and history constraints to reject noise.
        """
        # --- Community Cards (Monotonic Growth) ---
        raw_comm = raw_state.get('community_cards', [])
        if len(raw_comm) >= len(self.community_cards):
            self.community_cards = raw_comm
            
        # --- Hero Cards (Lock once detected) ---
        raw_hero = raw_state.get('hero_cards', [])
        if len(self.hero_cards) < 2 and len(raw_hero) == 2:
            self.hero_cards = raw_hero
            
        # --- Pot Size (Monotonic Growth & Median Filter) ---
        raw_pot = raw_state.get('pot_size', 0.0)
        self._pot_buffer.append(raw_pot)
        if len(self._pot_buffer) > 3:
            self._pot_buffer.pop(0)
            
        # Take the median of the last 3 reads to filter out single-frame OCR glitches
        sorted_buffer = sorted(self._pot_buffer)
        median_pot = sorted_buffer[len(sorted_buffer) // 2]
        
        # Pot size can only grow within a hand
        self.pot_size = max(self.pot_size, median_pot)
        
        # --- Hero Stack (Monotonic Decay) ---
        raw_hero_stack = raw_state.get('hero_stack', 0)
        # [V29 live-info expansion] `raw_hero_stack == 0` is normally an OCR non-read sentinel (the
        # `> 0` guard exists so a failed digit read doesn't zero out hero's real stack) -- but
        # `core/vision.py`'s `hero_all_in` flag (set via an explicit 'ALL'/'IN' text match, mirroring
        # the already-reliable opponent-side signal) distinguishes a genuine hero all-in from that
        # ambiguous case. Without this, hero's own all-in was silently never tracked (see
        # SeatState.committed's own docstring / [OPP-2] backlog entry for the equivalent opponent
        # fix this mirrors).
        if raw_hero_stack > 0 or raw_state.get('hero_all_in', False):
            if self.hero_stack == 0:
                self.hero_stack = raw_hero_stack
            else:
                self.hero_stack = min(self.hero_stack, raw_hero_stack)
                
        # --- Opponents (Monotonic Decay & State Persistence) ---
        raw_opps = raw_state.get('opponents', {})
        for seat_key, raw_opp in raw_opps.items():
            if seat_key not in self.opponents:
                # First time seeing this opponent this hand
                self.opponents[seat_key] = raw_opp.copy()
            else:
                # Update existing opponent
                tracked_opp = self.opponents[seat_key]
                raw_stack = raw_opp.get('stack', 0)
                raw_state_lbl = raw_opp.get('state', 'Folded')
                raw_active = bool(raw_opp.get('is_active', False))
                
                # If they folded or have 0 stack (and are not all-in), they stay folded
                if not raw_active or raw_state_lbl == 'Folded':
                    # Only accept a fold if the raw read is highly confident they are inactive
                    # (To prevent obscuring timers from falsely folding them, we might be cautious.
                    # But the vision module handles this via `is_active` check.)
                    tracked_opp['state'] = raw_state_lbl
                    tracked_opp['is_active'] = raw_active
                    if raw_stack > 0:
                         tracked_opp['stack'] = raw_stack
                else:
                    # They are active or all-in
                    tracked_opp['state'] = raw_state_lbl
                    tracked_opp['is_active'] = raw_active

                    # [V29 live-info expansion] `raw_stack == 0` is normally treated as an OCR
                    # non-read (the whole `if raw_stack > 0` guard exists so a transient failed
                    # digit read doesn't zero out a real stack). BUT `core/vision.py`'s own
                    # `read_board_state` (lines ~417-419/426-428) sets `state='All-In'` ONLY via an
                    # explicit 'ALL'/'IN' text match (or a lone '0' digit) BEFORE any digit-mangling
                    # -- a genuinely reliable, purpose-built signal, not the same ambiguous 0 the
                    # guard above is protecting against. Without this, an opponent's `committed`
                    # (this change) and [OPP-2]'s raised_this_hand/street would silently stop
                    # updating the instant they shove their last chips -- exactly the scenario
                    # where the model most needs to see committed/raise info correctly.
                    if raw_stack > 0 or raw_state_lbl == 'All-In':
                        # Stack can only decrease
                        if tracked_opp['stack'] == 0:
                            tracked_opp['stack'] = raw_stack
                        else:
                            tracked_opp['stack'] = min(tracked_opp['stack'], raw_stack)
                            
                # Persist VPIP and AGG colors (once detected, keep them)
                raw_vpip = raw_opp.get('vpip_color')
                if raw_vpip:
                    tracked_opp['vpip_color'] = raw_vpip
                    
                raw_agg = raw_opp.get('agg_color')
                if raw_agg:
                    tracked_opp['agg_color'] = raw_agg
                            
        # --- Active Buttons (Always take raw as they appear/disappear on Hero's turn) ---
        self.active_buttons = raw_state.get('active_buttons', [])
        
        # --- Dealer Button & Hero Position (Update from OCR if detected) ---
        raw_dealer_idx = raw_state.get('dealer_idx', -1)
        if raw_dealer_idx != -1:
            self.dealer_idx = raw_dealer_idx
            self.dealer_name = raw_state.get('dealer_name', '')
            self.hero_position = (0 - self.dealer_idx) % 6
        
        # --- Timeline Generation ---
        self._generate_timeline_actions()
        
    def _generate_timeline_actions(self):
        """Infers chronological betting actions from stabilized state differences."""
        # 1. Determine current street
        num_comm = len(self.community_cards)
        current_street = 'Preflop'
        if num_comm >= 3: current_street = 'Flop'
        if num_comm >= 4: current_street = 'Turn'
        if num_comm == 5: current_street = 'River'

        # Reset street bet level if street changed
        if current_street != self.last_street:
            self.current_street_bet_level = 0.0
            self.last_street = current_street
            # [V29, OPP-2 live] Per-street raise attribution resets at every new street, mirroring
            # `current_street_bet_level`'s own reset (a raise reopens things fresh each street).
            self.raised_this_street = {}

        # 2. Check for Folds and Stack Changes
        # Build current stacks map
        current_stacks = {'Hero': self.hero_stack}
        for seat, opp in self.opponents.items():
            current_stacks[seat] = opp.get('stack', 0)

        # [V29] Seed `hand_start_stacks` the FIRST time each player's stack is observed this hand
        # (mirrors `last_known_stacks`'s own "first frame just caches" pattern below) -- source for
        # `committed` (start_stack - current_stack), see `to_board_state()`.
        for player_key, current_stack in current_stacks.items():
            if player_key not in self.hand_start_stacks and current_stack > 0:
                self.hand_start_stacks[player_key] = current_stack

        # 3. Detect Bets, Calls, Raises via Stack Drops
        # [V29] `street_bet_before` is captured ONCE before this tick's diffs and updated locally
        # as each player's diff is processed (in case a rare missed-frame tick contains more than
        # one player's stack drop at once) -- classifies each diff as a RAISE only if it exceeds
        # the bet level THAT PLAYER actually faced, not just "any stack drop", matching how
        # training's own simulator only calls a raise a raise when it strictly increases
        # `highest_bet` (versions/v29/self_play/simulator.py). A small epsilon (1% of a big blind,
        # floor 0.01) absorbs OCR rounding noise without misreading a real (larger) raise as a call.
        street_bet_before = self.current_street_bet_level
        epsilon = max(0.01, self.big_blind * 0.01)
        for player_key, current_stack in current_stacks.items():
            if player_key in self.last_known_stacks:
                last_data = self.last_known_stacks[player_key]
                last_stack = last_data['stack']

                # If stack dropped, money went in!
                if last_stack > 0 and current_stack < last_stack:
                    diff = last_stack - current_stack

                    if diff > street_bet_before + epsilon:
                        # This player's contribution now EXCEEDS what they needed to just call --
                        # a genuine raise (or an opening bet, from a street_bet_before of 0), not a
                        # call/limp. [OPP-2] live tracking.
                        self.raised_this_hand[player_key] = True
                        self.raised_this_street[player_key] = True
                        self.raise_count += 1
                        street_bet_before = diff

                    # Update the highest bet to call
                    self.current_street_bet_level = max(self.current_street_bet_level, diff)

        # Update cached state for next tick
        self.last_known_stacks = {'Hero': {'stack': self.hero_stack, 'is_active': True}}
        for seat, opp in self.opponents.items():
            self.last_known_stacks[seat] = {'stack': opp.get('stack', 0), 'is_active': opp.get('is_active', False)}

    def to_dict(self) -> dict:
        """Serializes to the standard dictionary format expected by the evaluator and GUI."""
        # Calculate active opponents
        num_active = len([o for o in self.opponents.values() if o.get('is_active', True)])
        
        # Determine street
        num_comm = len(self.community_cards)
        if num_comm == 0:
            street = 'Preflop'
        elif num_comm == 3:
            street = 'Flop'
        elif num_comm == 4:
            street = 'Turn'
        elif num_comm == 5:
            street = 'River'
        else:
            street = 'Unknown'
            
        return {
            'community_cards': self.community_cards,
            'hero_cards': self.hero_cards,
            'pot_size': self.pot_size,
            'hero_stack': self.hero_stack,
            'opponents': self.opponents,
            'num_active_players': num_active,
            'active_buttons': self.active_buttons,
            'street': street,
            'dealer_name': self.dealer_name,
            'dealer_idx': self.dealer_idx,
            'hero_position': self.hero_position,
            'action_history': self.action_history.copy()
        }

    def to_board_state(self, call_amount: float = 0.0, equity: float = 0.0, big_blind: float = 10.0) -> 'BoardState':
        """Generates the pure decoupled mathematical model of the table state."""
        from core.board_state import BoardState, SeatState, HUDStats
        
        num_comm = len(self.community_cards)
        if num_comm == 0: street = 'Preflop'
        elif num_comm == 3: street = 'Flop'
        elif num_comm == 4: street = 'Turn'
        elif num_comm == 5: street = 'River'
        else: street = 'Unknown'

        # [V29 live-info expansion] These three (`hero_committed`/`pot_type` here, `committed`
        # per-seat below) were previously ALWAYS 0/inert in live play for every version since
        # V22/V23 introduced them -- to_board_state() simply never set them (discovered while
        # wiring [OPP-2]'s live tracking; see .agents/skills/OFK/references/
        # known-shortcomings-backlog.md [OPP-2]). Now sourced from real tracked state.
        hero_start_stack = self.hand_start_stacks.get('Hero', self.hero_stack)
        hero_committed = max(0.0, hero_start_stack - self.hero_stack)
        pot_type = min(2, self.raise_count)

        bs = BoardState(
            community_cards=self.community_cards,
            hero_cards=self.hero_cards,
            pot_size=self.pot_size,
            hero_stack=self.hero_stack,
            active_buttons=self.active_buttons,
            dealer_idx=self.dealer_idx,
            hero_position=self.hero_position,
            street=street,
            call_amount=call_amount,
            equity=equity,
            big_blind=big_blind,
            hero_committed=hero_committed,
            pot_type=pot_type,
        )

        for seat_key, opp_dict in self.opponents.items():
            opp_stack = opp_dict.get('stack', 0.0)
            start_stack = self.hand_start_stacks.get(seat_key, opp_stack)
            bs.seats[seat_key] = SeatState(
                name=opp_dict.get('name', ''),
                stack=opp_stack,
                is_active=opp_dict.get('is_active', False),
                state_label=opp_dict.get('state', 'Active'),
                hud=HUDStats(
                    vpip_color=opp_dict.get('vpip_color', 'Blue'),
                    agg_color=opp_dict.get('agg_color', 'Blue')
                ),
                committed=max(0.0, start_stack - opp_stack),
                raised_this_hand=self.raised_this_hand.get(seat_key, False),
                raised_this_street=self.raised_this_street.get(seat_key, False),
            )
        return bs

    def seed_stacks(self, baseline_stacks: dict, hero_name: str, dealer_name: str = ""):
        """
        Seeds player stacks using the baseline XML data.
        Performs fuzzy matching on names to map XML names to OCR-recognized seats.
        """
        if not baseline_stacks:
            return

        self.dealer_name = dealer_name

        # 1. Seed Hero's stack
        if hero_name in baseline_stacks:
            self.hero_stack = baseline_stacks[hero_name]
            
        # 2. Seed Opponents' stacks
        xml_opp_names = [name for name in baseline_stacks.keys() if name != hero_name]
        
        for seat_key, tracked_opp in self.opponents.items():
            ocr_name = tracked_opp.get('name', '')
            if not ocr_name:
                continue
                
            # Find best match from XML opponent names
            best_name = None
            best_ratio = 0.0
            for xml_name in xml_opp_names:
                ratio = difflib.SequenceMatcher(None, ocr_name.lower(), xml_name.lower()).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_name = xml_name
                    
            # If match is reliable (ratio > 0.6), overwrite the OCR stack and correct the name
            if best_ratio > 0.6:
                tracked_opp['name'] = best_name
                tracked_opp['stack'] = baseline_stacks[best_name]
                # If they have a positive stack, reset their state to Active
                if baseline_stacks[best_name] > 0:
                    tracked_opp['state'] = 'Active'
                    tracked_opp['is_active'] = True
                else:
                    # If stack is 0 in baseline, they might be All-In. Keep OCR state if it was All-In.
                    if tracked_opp.get('state') == 'All-In':
                        tracked_opp['state'] = 'All-In'
                        tracked_opp['is_active'] = True
                    else:
                        tracked_opp['state'] = 'Folded'
                        tracked_opp['is_active'] = False

        # 3. Calculate Hero's GTO Position relative to the Button
        # BU = 0, SB = 1, BB = 2, UTG = 3, MP = 4, CO = 5
        self.dealer_idx = 0  # Default to Hero
        if self.dealer_name and self.dealer_name != hero_name:
            # Check which seat key matches the dealer name
            for seat_key, tracked_opp in self.opponents.items():
                if tracked_opp.get('name') == self.dealer_name:
                    # Extract the digit index: seat_1 -> 1, seat_5 -> 5
                    try:
                        self.dealer_idx = int(seat_key.split('_')[1])
                    except (IndexError, ValueError):
                        pass
                    break
        
        self.hero_position = (0 - self.dealer_idx) % 6


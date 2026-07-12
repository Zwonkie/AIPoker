import difflib

class TableState:
    """
    Data model that tracks and stabilizes the poker table state over time.
    Filters out transient OCR errors (noise) using history, monotonicity constraints,
    and median filtering.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Clears state history for a new hand."""
        self.community_cards = []
        self.hero_cards = []
        self.pot_size = 0.0
        self.hero_stack = 0
        self.opponents = {}
        self.active_buttons = []
        self.dealer_name = ""
        self.dealer_idx = 0
        self.hero_position = 0
        
        # Action tracking for ML
        self.action_history = []
        self.current_street_bet_level = 0.0
        self.last_known_stacks = {}
        self.last_street = 'Preflop'
        
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
        if raw_hero_stack > 0:
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
                    
                    if raw_stack > 0:
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
            
        # 2. Check for Folds and Stack Changes
        # Build current stacks map
        current_stacks = {'Hero': self.hero_stack}
        for seat, opp in self.opponents.items():
            current_stacks[seat] = opp.get('stack', 0)
                    
        # 3. Detect Bets, Calls, Raises via Stack Drops
        for player_key, current_stack in current_stacks.items():
            if player_key in self.last_known_stacks:
                last_data = self.last_known_stacks[player_key]
                last_stack = last_data['stack']
                
                # If stack dropped, money went in!
                if last_stack > 0 and current_stack < last_stack:
                    diff = last_stack - current_stack
                        
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
            big_blind=big_blind
        )
        
        for seat_key, opp_dict in self.opponents.items():
            bs.seats[seat_key] = SeatState(
                name=opp_dict.get('name', ''),
                stack=opp_dict.get('stack', 0.0),
                is_active=opp_dict.get('is_active', False),
                state_label=opp_dict.get('state', 'Active'),
                hud=HUDStats(
                    vpip_color=opp_dict.get('vpip_color', 'Blue'),
                    agg_color=opp_dict.get('agg_color', 'Blue')
                )
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


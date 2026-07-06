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
                raw_active = raw_opp.get('is_active', False)
                
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
            'street': street
        }

    def seed_stacks(self, baseline_stacks: dict, hero_name: str):
        """
        Seeds player stacks using the baseline XML data.
        Performs fuzzy matching on names to map XML names to OCR-recognized seats.
        """
        if not baseline_stacks:
            return

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
                    tracked_opp['state'] = 'Folded'
                    tracked_opp['is_active'] = False


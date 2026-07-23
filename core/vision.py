import cv2
import numpy as np
import pytesseract
import os
import glob

# [v49 live2/ocr, owner-approved 2026-07-23] Chip-count (stack/pot) NUMBERS are read by
# the gated template reader, not Tesseract -- accept-or-abstain, zero wrong acceptances
# on the whole labeled corpus. Tesseract remains ONLY for text fields (names) and the
# ALL-IN / '-' textual state detection fallback when the template reader abstains.
from live2.ocr import harvest_digits as chip_ocr

# Configure Tesseract path (Windows specific)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class PokerVision:
    def __init__(self, templates_dir='card_templates'):
        self.templates_dir = templates_dir
        self.card_templates = {}
        self.button_templates = {}
        self.load_templates()

        # [v49 live2/ocr] Gated chip-count reader templates (soft/aliased, canonical
        # transforms). FAIL-LOUD: an incomplete soft alphabet would make read_chips()
        # abstain on EVERY frame -- stacks silently frozen at their first value is the
        # exact failure class this reader exists to kill, so refuse to start instead.
        self.chip_templates = chip_ocr.load_templates()
        for pfx, fname in (('soft_s', 'stack'), ('soft_p', 'pot')):
            missing = [d for d in '0123456789' if f'{pfx}{d}' not in self.chip_templates]
            if missing:
                raise RuntimeError(
                    f"chip-template alphabet incomplete for {fname} font (missing "
                    f"{missing}) -- run `python -m live2.ocr.harvest_digits` and review")
        print(f"Loaded {len(self.chip_templates)} chip templates (gated reader active).")

        # Coordinates of Regions of Interest (ROI) for 1536x1090 resolution
        # format: (x, y, width, height)
        self.rois = {
            'community_cards': (500, 395, 500, 95),
            'pot': (700, 365, 160, 45),
            
            # Opponent Seats (1 to 5)
            'seat_1_name': (200, 770, 200, 40),
            'seat_1_stack': (200, 805, 200, 40),
            'seat_2_name': (200, 320, 200, 40),
            'seat_2_stack': (200, 355, 200, 40),
            'seat_3_name': (670, 195, 200, 40),
            'seat_3_stack': (670, 230, 200, 40),
            'seat_4_name': (1100, 320, 200, 40),
            'seat_4_stack': (1100, 355, 200, 40),
            'seat_5_name': (1100, 770, 200, 40),
            'seat_5_stack': (1100, 805, 200, 40),
            
            # Legacy aliases
            'seat_left_name': (200, 770, 200, 40),
            'seat_left_stack': (200, 805, 200, 40),
            'seat_right_name': (1100, 770, 200, 40),
            'seat_right_stack': (1100, 805, 200, 40),
            'seat_top_name': (670, 195, 200, 40),
            'seat_top_stack': (670, 230, 200, 40),
            
            # Top edge moved down 8px then a further 10px 2026-07-16 (bottom edge held fixed at
            # y=1090) -- the ROI was capturing extra area above the actual fold/check/raise buttons.
            'buttons': (980, 988, 550, 102)
        }

    def load_templates(self):
        """Loads all card, button, and digit templates from the templates directories."""
        if not os.path.exists(self.templates_dir):
            print(f"Warning: templates directory '{self.templates_dir}' not found.")
            return

        # Load card templates (e.g., Ah.png, 2c.png)
        for path in glob.glob(os.path.join(self.templates_dir, '*.png')):
            filename = os.path.basename(path)
            name = filename.replace('.png', '')
            
            # Load in grayscale
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if name.startswith('button_'):
                    self.button_templates[name] = img
                else:
                    self.card_templates[name] = img
        print(f"Loaded {len(self.card_templates)} cards and {len(self.button_templates)} button templates.")

        # Extract dealer button template specifically
        self.dealer_button_template = self.card_templates.pop('dealer_button', None)
        if self.dealer_button_template is not None:
            print("Loaded dealer button template successfully.")
        else:
            print("Warning: dealer_button.png template not found in card_templates.")

        # Load Hexagon Template and Mask for seat anchoring
        self.hexagon_template = None
        self.hexagon_mask = None
        tpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'binaries', 'hexagon_anchor.png')
        if os.path.exists(tpl_path):
            try:
                self.hexagon_template = cv2.imread(tpl_path)
                if self.hexagon_template is not None:
                    b, g, r_channel = cv2.split(self.hexagon_template)
                    self.hexagon_mask = np.ones(self.hexagon_template.shape[:2], dtype=np.uint8) * 255
                    red_mask = (r_channel > 200) & (g < 50) & (b < 50)
                    self.hexagon_mask[red_mask] = 0
            except Exception as e:
                print(f"Error loading hexagon anchor template: {e}")
        else:
            print("Warning: hexagon_anchor.png not found.")

        # Load Hero Hexagon Template for seat anchoring
        self.hero_hexagon_template = None
        self.hero_hexagon_mask = None
        hero_tpl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'binaries', 'hero_hexagon_anchor.png')
        if os.path.exists(hero_tpl_path):
            try:
                self.hero_hexagon_template = cv2.imread(hero_tpl_path)
                if self.hero_hexagon_template is not None:
                    b, g, r_channel = cv2.split(self.hero_hexagon_template)
                    self.hero_hexagon_mask = np.ones(self.hero_hexagon_template.shape[:2], dtype=np.uint8) * 255
                    # Mask out light blue/grey inside
                    blue_mask = (b > 100) & (g > 100) & (r_channel < 100)
                    self.hero_hexagon_mask[blue_mask] = 0
            except Exception as e:
                print(f"Error loading hero hexagon anchor template: {e}")
        else:
            print("Warning: hero_hexagon_anchor.png not found.")

    def preprocess_image(self, img):
        """Preprocesses the image (e.g. resize, grayscale) if needed."""
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def match_templates_in_roi(self, img, roi, templates, threshold=0.90, max_matches=5):
        """
        Extracts an ROI from the image and performs template matching
        against the given dictionary of templates.
        """
        x, y, w, h = roi
        roi_img = img[y:y+h, x:x+w]
        roi_gray = self.preprocess_image(roi_img)
        
        matches = []
        for name, tpl in templates.items():
            if tpl.shape[0] > roi_gray.shape[0] or tpl.shape[1] > roi_gray.shape[1]:
                continue
                
            res = cv2.matchTemplate(roi_gray, tpl, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            
            for pt in zip(*loc[::-1]):
                score = res[pt[1], pt[0]]
                # Map coordinate back to the full image space
                global_x = x + pt[0]
                global_y = y + pt[1]
                matches.append((name, (global_x, global_y), score))
                
        # Filter overlapping matches (keep highest score within 15 pixels)
        matches = sorted(matches, key=lambda val: val[2], reverse=True)
        filtered = []
        for m in matches:
            name, pt, score = m
            too_close = False
            for f in filtered:
                fx, fy = f[1]
                if abs(fx - pt[0]) < 15 and abs(fy - pt[1]) < 15:
                    too_close = True
                    break
            if not too_close:
                filtered.append(m)
                if len(filtered) >= max_matches:
                    break
                    
        return sorted(filtered, key=lambda val: val[1][0]) # sort by X coordinate

    def find_dealer_button(self, img):
        """
        Scans the entire image for the dealer button template.
        Returns the (x, y) center coordinates of the button if found above threshold, else None.
        """
        if self.dealer_button_template is None:
            return None
            
        gray = self.preprocess_image(img)
        res = cv2.matchTemplate(gray, self.dealer_button_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        threshold = 0.85 # Highly robust for normalized cross-correlation
        if max_val >= threshold:
            h, w = self.dealer_button_template.shape
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            return (center_x, center_y)
        return None

    def ocr_roi(self, img, roi, whitelist=None, single_line=True):
        """Extracts an ROI and runs Tesseract OCR using a dual-pass grayscale/Otsu strategy."""
        x, y, w, h = roi
        crop = img[y:y+h, x:x+w]
        
        # Convert to gray
        gray = self.preprocess_image(crop)
        
        # Scale up (Tesseract performs much better on larger text)
        resized = cv2.resize(gray, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        
        config = ''
        if single_line:
            config += '--psm 6 '
        if whitelist:
            config += f'-c tessedit_char_whitelist={whitelist}'
            
        # Pass 1: Standard grayscale resized (best for standard anti-aliased text)
        text = pytesseract.image_to_string(resized, config=config.strip()).strip()
        
        # Pass 2: Fallback to Otsu thresholding + Inversion if empty or lacking digits (when expected)
        has_digit_whitelist = whitelist and any(c.isdigit() for c in whitelist)
        has_digits_parsed = text and any(c.isdigit() for c in text)
        
        if not text or (has_digit_whitelist and not has_digits_parsed):
            # Apply Otsu's thresholding with inversion (assuming light text on dark background)
            processed = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            fallback_text = pytesseract.image_to_string(processed, config=config.strip()).strip()
            if fallback_text:
                text = fallback_text
            
        return text



    def clean_pot_string(self, text):
        """Cleans and extracts integer value from pot OCR text with translation mapping."""
        # Isolate value from label if colon is present
        if ":" in text:
            text = text.split(":")[-1].strip()
            
        # Custom character replacement for Tesseract misreads
        trans = str.maketrans({
            'T': '7', 'J': '7', 'j': '7',
            'E': '6', 'I': '4', 'i': '4',
            'N': '0', 'O': '0', 'o': '0',
            'S': '5', 's': '5', 'B': '8',
            'G': '6', 'g': '9', 'A': '4'
        })
        cleaned = text.translate(trans)
        digits = "".join([c for c in cleaned if c.isdigit()])
        try:
            return int(digits)
        except ValueError:
            return 0

    def clean_stack_string(self, text):
        """Cleans and extracts integer stack value from OCR text."""
        # If there are multiple lines (e.g. Tid: 10 \n stack_value), take the last line
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return 0
        last_line = lines[-1]
        
        # Custom character replacement for Tesseract misreads
        trans = str.maketrans({
            'T': '7', 'J': '7', 'j': '7',
            'E': '6', 'I': '4', 'i': '4',
            'N': '0', 'O': '0', 'o': '0',
            'S': '5', 's': '5', 'B': '8',
            'G': '6', 'g': '9', 'A': '4'
        })
        cleaned = last_line.translate(trans)
        digits = "".join([c for c in cleaned if c.isdigit()])
        try:
            return int(digits)
        except ValueError:
            return 0

    def classify_color(self, bgr_crop):
        """Classify dominant color in a small BGR crop into Blue, Green, Yellow, Red, or None using average BGR."""
        if bgr_crop.size == 0:
            return None
        
        # Calculate mean BGR color across all pixels in the crop
        mean_bgr = np.mean(bgr_crop, axis=(0,1))
        
        # Calculate HSV-like saturation
        b, g, r = mean_bgr
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        sat = (max_c - min_c) / max_c if max_c > 0 else 0
        
        # Enforce minimum saturation threshold of 30% to filter out grey stats
        if sat < 0.30:
            return None
            
        centers = {
            'Blue': np.array([185, 155, 90]),    # Calibrated Blue
            'Green': np.array([85, 150, 85]),    # Calibrated Green
            'Yellow': np.array([50, 180, 210]),  # Calibrated Yellow
            'Red': np.array([85, 50, 175])       # Calibrated Red
        }
        
        min_dist = float('inf')
        best_color = None
        
        for name, center in centers.items():
            dist = np.linalg.norm(mean_bgr - center)
            if dist < min_dist:
                min_dist = dist
                best_color = name
                
        # Enforce maximum distance threshold of 85 to filter out background leaks
        if min_dist > 85:
            return None
            
        return best_color

    def read_board_state(self, img, board_size="6-Max"):
        """
        Parses the full board image and returns a dict with:
        - community_cards: list of strings
        - hero_cards: list of strings
        - pot_size: float/int
        - hero_stack: int
        - opponents: dict of seat -> {name, stack, is_active, state, vpip_color, agg_color}
        - active_buttons: list of active button names
        """
        state = {}
        
        # 1. Detect cards
        comm_matches = self.match_templates_in_roi(
            img, self.rois['community_cards'], self.card_templates, threshold=0.90, max_matches=5
        )
        state['community_cards'] = [m[0] for m in comm_matches]
        
        # 2. Pot chips -- gated template reader (canonical pot transform: colon anchor
        # @55% + truncation below POT_TRUNC_GRAY, soft/aliased matching). Abstain -> 0,
        # the no-read sentinel TableState's monotonic pot filter already tolerates.
        px, py, pw, ph = chip_ocr.POT_ROI
        pot_crop = img[py:py + ph, px:px + pw]
        pot_text, _pd, _pm, pot_ok = chip_ocr.read_chips(pot_crop, self.chip_templates,
                                                         font='pot')
        state['pot_size'] = int(pot_text) if pot_ok else 0
        
        
        
        # 4. OCR Opponents & Hero using Dynamic Hexagon Anchor System
        opponents = {}
        
        # Default centers in case template matching fails
        default_centers = {
            'seat_1': (320, 757),
            'seat_2': (196, 306),
            'seat_3': (767, 180),
            'seat_4': (1340, 306),
            'seat_5': (1213, 757),
            'hero': (767, 837)
        }
        
        # Search regions for template matching
        search_regions = {
            'seat_1': (150, 700, 250, 150),
            'seat_2': (50, 260, 250, 150),
            'seat_3': (640, 130, 250, 150),
            'seat_4': (1180, 260, 250, 150),
            'seat_5': (1030, 700, 250, 150),
            'hero': (600, 770, 300, 130)
        }
        
        if board_size == "6-Max":
            # Use static coordinate layout directly to avoid false template matching
            resolved_centers = default_centers.copy()
            found_anchors = {k: True for k in default_centers.keys()}
        else:
            # Fallback to template matching scan for other sizes (like 10-Max)
            resolved_centers = {}
            found_anchors = {}
            for key, (rx, ry, rw, rh) in search_regions.items():
                cx, cy = default_centers[key]
                found_anchor = False
                
                active_template = self.hero_hexagon_template if key == 'hero' else self.hexagon_template
                active_mask = self.hero_hexagon_mask if key == 'hero' else self.hexagon_mask
                
                if active_template is not None:
                    search_area = img[ry:ry+rh, rx:rx+rw]
                    if search_area.shape[0] >= active_template.shape[0] and search_area.shape[1] >= active_template.shape[1]:
                        res = cv2.matchTemplate(search_area, active_template, cv2.TM_CCORR_NORMED, mask=active_mask)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        if max_val >= 0.70:
                            cx = rx + max_loc[0] + active_template.shape[1] // 2
                            cy = ry + max_loc[1] + active_template.shape[0] // 2
                            found_anchor = True
                resolved_centers[key] = (cx, cy)
                found_anchors[key] = found_anchor

        # Parse Opponents
        for seat_key in ['seat_1', 'seat_2', 'seat_3', 'seat_4', 'seat_5']:
            if not found_anchors.get(seat_key, False):
                continue
            cx, cy = resolved_centers[seat_key]
            
            # VPIP and AGG Crops
            vpip_crop = img[max(0, cy-12):min(img.shape[0], cy+12), max(0, cx-100):min(img.shape[1], cx-80)]
            agg_crop = img[max(0, cy-12):min(img.shape[0], cy+12), max(0, cx+80):min(img.shape[1], cx+100)]
            
            # Name and Stack Crops (Name shifted down: top down 5 [cy+22], bottom down 8 [cy+48]. Stack rolled back to [cy+50, cy+84])
            name_crop = img[max(0, cy+22):min(img.shape[0], cy+48), max(0, cx-75):min(img.shape[1], cx+75)]
            stack_crop = img[max(0, cy+50):min(img.shape[0], cy+84), max(0, cx-65):min(img.shape[1], cx+65)]
            
            # Run OCR on Name using robust ocr_roi
            name_text = self.ocr_roi(img, (cx-75, cy+22, 150, 26))
            
            import re
            name = name_text.strip()
            name = re.sub(r'^[^a-zA-Z0-9]+', '', name) # remove leading symbols
            name = re.sub(r'[^a-zA-Z0-9_\'\s]+$', '', name) # remove trailing noise
            if len(name) <= 2 and not name.isalnum():
                name = ""
                
            if not name:
                continue
                
            # Determine if player is active (5% brightest pixels in name_gray > 160.0)
            name_gray = self.preprocess_image(name_crop)
            flat = name_gray.flatten()
            flat_sorted = np.sort(flat)
            top_5_percent_idx = int(len(flat_sorted) * 0.95)
            top_5_percent_pixels = flat_sorted[top_5_percent_idx:]
            mean_top_5 = np.mean(top_5_percent_pixels) if len(top_5_percent_pixels) > 0 else 0.0
            
            is_active = mean_top_5 > 160.0
            stack_val = 0
            state_label = "Active"
            
            if is_active:
                # [v49 live2/ocr] Gated template read of the chip count (canonical
                # STACK transform: ROI top +6 / gray trunc 160 / baseline scrub).
                chip_crop = img[max(0, cy + 56):min(img.shape[0], cy + 84),
                                max(0, cx - 65):min(img.shape[1], cx + 65)]
                chips, _cd, _cm, chips_ok = chip_ocr.read_chips(
                    chip_crop, self.chip_templates, font='stack')
                if chips_ok:
                    stack_val = int(chips)
                    # a displayed literal 0 is the client's all-in rendering (chips in
                    # the middle), same semantics the legacy lone-'0' branch used
                    state_label = "All-In" if stack_val == 0 else "Active"
                else:
                    # Abstain -> the NUMBER stays 0 (TableState's no-read sentinel:
                    # keeps the last stabilized value). Tesseract runs ONLY to detect
                    # the textual states the template alphabet cannot express.
                    stack_text = self.ocr_roi(img, (cx-65, cy+50, 130, 34), whitelist='0123456789.ALLIN-')
                    stack_line = stack_text.strip()
                    stack_val = 0
                    state_label = "Active"
                    if stack_line:
                        stack_upper = stack_line.upper()
                        if 'ALL' in stack_upper or 'IN' in stack_upper:
                            state_label = "All-In"
                        elif stack_upper == '-' or '-' in stack_upper:
                            state_label = "Folded"
                            is_active = False
            else:
                state_label = "Folded"
                stack_val = 0
                
            # Classify VPIP and AGG Colors
            vpip_color = self.classify_color(vpip_crop) if is_active else None
            agg_color = self.classify_color(agg_crop) if is_active else None
            
            opponents[seat_key] = {
                'name': name,
                'stack': stack_val,
                'is_active': bool(is_active),
                'state': state_label,
                'vpip_color': vpip_color,
                'agg_color': agg_color
            }
            
        state['opponents'] = opponents
        
        # Parse Hero VPIP/AGG/Stack/Name/Cards
        hcx, hcy = resolved_centers['hero']
        
        # Hero Name using robust ocr_roi
        hero_name_text = self.ocr_roi(img, (hcx-75, hcy+22, 150, 26))
        import re
        hname = hero_name_text.strip()
        hname = re.sub(r'^[^a-zA-Z0-9]+', '', hname)
        hname = re.sub(r'[^a-zA-Z0-9_\'\s]+$', '', hname)
        if len(hname) <= 2 and not hname.isalnum():
            hname = ""
        state['hero_name'] = hname

        hero_vpip_crop = img[max(0, hcy-12):min(img.shape[0], hcy+12), max(0, hcx-100):min(img.shape[1], hcx-80)]
        hero_agg_crop = img[max(0, hcy-12):min(img.shape[0], hcy+12), max(0, hcx+80):min(img.shape[1], hcx+100)]
        
        state['hero_vpip_color'] = self.classify_color(hero_vpip_crop)
        state['hero_agg_color'] = self.classify_color(hero_agg_crop)
        
        # [v49 live2/ocr] Hero chip count via the gated template reader (canonical
        # STACK transform), mirroring the opponent-seat wiring above: an ACCEPTED read
        # is the displayed value (a literal 0 = the client's all-in rendering); an
        # abstain leaves the number at the 0 no-read sentinel and falls back to
        # Tesseract ONLY for the textual ALL-IN detection (see the [V29] note it
        # replaces: 'ALL IN' text must be recognized BEFORE any digit mangling, else
        # hero's own all-in is indistinguishable from a failed read).
        hero_chip_crop = img[max(0, hcy + 56):min(img.shape[0], hcy + 84),
                             max(0, hcx - 65):min(img.shape[1], hcx + 65)]
        hchips, _hd, _hm, hchips_ok = chip_ocr.read_chips(
            hero_chip_crop, self.chip_templates, font='stack')
        hero_stack_val = 0
        hero_all_in = False
        if hchips_ok:
            hero_stack_val = int(hchips)
            hero_all_in = (hero_stack_val == 0)
        else:
            hero_stack_text = self.ocr_roi(img, (hcx-65, hcy+50, 130, 34), whitelist='0123456789.ALLIN-')
            h_stack_line = hero_stack_text.strip()
            if h_stack_line:
                h_stack_upper = h_stack_line.upper()
                if 'ALL' in h_stack_upper or 'IN' in h_stack_upper:
                    hero_all_in = True

        state['hero_stack'] = hero_stack_val
        state['hero_all_in'] = hero_all_in
                
        # Parse Hero Cards dynamically based on anchor
        dynamic_hero_cards_roi = (max(0, hcx - 127), max(0, hcy - 112), 230, 95)
        hero_matches = self.match_templates_in_roi(
            img, dynamic_hero_cards_roi, self.card_templates, threshold=0.90, max_matches=2
        )
        state['hero_cards'] = [m[0] for m in hero_matches]
        
        # 5. Detect Active Buttons
        button_matches = self.match_templates_in_roi(
            img, self.rois['buttons'], self.button_templates, threshold=0.85, max_matches=3
        )
        state['active_buttons'] = [m[0] for m in button_matches]
        
        # 6. Detect Dealer Button
        state['dealer_idx'] = -1
        state['dealer_name'] = ""
        
        dealer_coords = self.find_dealer_button(img)
        if dealer_coords is not None:
            dx, dy = dealer_coords
            
            # Find the closest seat center (anchor)
            closest_seat = None
            min_dist = float('inf')
            
            for seat_key, (cx, cy) in resolved_centers.items():
                dist = np.sqrt((dx - cx) ** 2 + (dy - cy) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest_seat = seat_key
                    
            if closest_seat is not None:
                # Map seat key to integer index: hero -> 0, seat_i -> i
                if closest_seat == 'hero':
                    state['dealer_idx'] = 0
                    state['dealer_name'] = "Hero"
                else:
                    try:
                        idx = int(closest_seat.split('_')[1])
                        state['dealer_idx'] = idx
                        # Find player name at this seat
                        opp = opponents.get(closest_seat, {})
                        state['dealer_name'] = opp.get('name', f"Player {idx}")
                    except (IndexError, ValueError):
                        pass

        # Calculate active players count
        # In this 6-max game:
        active_opponents = [opp for opp in opponents.values() if opp.get('is_active', True)]
        state['num_active_players'] = 1 + len(active_opponents)
        
        return state

if __name__ == '__main__':
    # Simple test on board3 and board4
    vision = PokerVision()
    for board_file in ['board3.png', 'board4.png']:
        if os.path.exists(board_file):
            img = cv2.imread(board_file)
            res = vision.read_board_state(img)
            print(f"\n--- Extracted State for {board_file} ---")
            print(f"Community: {res['community_cards']}")
            print(f"Hero Cards:{res['hero_cards']}")
            print(f"Pot Size:  {res['pot_size']}")
            print(f"Hero Stack:{res['hero_stack']}")
            print(f"Opponents: {res['opponents']}")
            print(f"Buttons:   {res['active_buttons']}")

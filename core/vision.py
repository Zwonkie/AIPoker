import cv2
import numpy as np
import pytesseract
import os
import glob

# Configure Tesseract path (Windows specific)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class PokerVision:
    def __init__(self, templates_dir='card_templates'):
        self.templates_dir = templates_dir
        self.card_templates = {}
        self.button_templates = {}
        self.load_templates()

        # Coordinates of Regions of Interest (ROI) for 1536x1090 resolution
        # format: (x, y, width, height)
        self.rois = {
            'community_cards': (500, 395, 500, 95),
            'hero_cards': (640, 725, 230, 95),
            'pot': (700, 365, 160, 45),
            'hero_stack': (650, 850, 200, 60),
            
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
            
            'buttons': (980, 970, 550, 120)
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

        # Load digit classifier ML model
        self.digit_classifier = None
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'binaries', 'digit_classifier.pkl')
        if not os.path.exists(model_path):
            model_path = os.path.join('core', 'models', 'binaries', 'digit_classifier.pkl')
        if os.path.exists(model_path):
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    self.digit_classifier = pickle.load(f)
                print("Loaded ML digit classifier model successfully.")
            except Exception as e:
                print(f"Error loading ML digit classifier: {e}")
        else:
            print("Warning: digit_classifier.pkl model not found.")

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

    def parse_digits_template_matching(self, img, roi, is_pot=False):
        """
        Parses numerical digits from an ROI using scale-invariant ML classification.
        Falls back to empty string if classification fails or model is not loaded.
        """
        if not hasattr(self, 'digit_classifier') or self.digit_classifier is None:
            return ""
            
        x, y, w, h = roi
        crop = img[y:y+h, x:x+w]
        
        # Preprocess
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()
            
        # Vertical slice to focus on digits and discard lines/names
        if not is_pot:
            if h == 60:
                slice_img = gray[38:59, :]
            elif h == 40:
                slice_img = gray[10:35, :]
            else:
                slice_img = gray
        else:
            slice_img = gray[10:35, :]
            
        # Binarize: use Otsu for Pot, fixed threshold of 100 for Stacks
        if is_pot:
            _, thresh = cv2.threshold(slice_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(slice_img, 100, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find colon if it's a pot ROI to skip label prefix (e.g. "Pulje: ")
        colon_x = -1
        if is_pot:
            dots = []
            for c in contours:
                bx, by, bw, bh = cv2.boundingRect(c)
                if bw <= 5 and bh <= 6:
                    dots.append((bx, by))
            for i in range(len(dots)):
                for j in range(i + 1, len(dots)):
                    x1, y1 = dots[i]
                    x2, y2 = dots[j]
                    if abs(x1 - x2) <= 3 and 5 <= abs(y1 - y2) <= 15:
                        colon_x = max(x1, x2)
                        break
                if colon_x != -1:
                    break
                    
        detected = []
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw < 2 or bh < 2:
                continue
                
            # Filter out table border noise at the left side of stack crops (x < 50)
            if not is_pot and bx < 50:
                continue
                
            # Filter out underlines / horizontal line noise
            if bw / bh > 1.8 or bh < 6:
                continue
                
            # Filter out prefix label
            if is_pot and colon_x != -1 and bx <= colon_x:
                continue
            elif is_pot and colon_x == -1 and bx < 68:
                continue
                
            if bh >= 8:
                # Handle merged/touching digits if width is too large (typically >= 18 pixels)
                if bw >= 18:
                    w_half = bw // 2
                    
                    # Left digit
                    char_crop_l = thresh[by:by+bh, bx:bx+w_half]
                    char_l, score_l = self.match_digit_crop(char_crop_l)
                    if score_l >= 0.50:
                        detected.append((char_l, bx))
                        
                    # Right digit
                    char_crop_r = thresh[by:by+bh, bx+w_half:bx+bw]
                    char_r, score_r = self.match_digit_crop(char_crop_r)
                    if score_r >= 0.50:
                        detected.append((char_r, bx + w_half))
                else:
                    # Single digit
                    char_crop = thresh[by:by+bh, bx:bx+bw]
                    char, score = self.match_digit_crop(char_crop)
                    if score >= 0.50:
                        detected.append((char, bx))
            elif bh <= 7 and bw <= 7:
                detected.append(('.', bx))
                
        detected = sorted(detected, key=lambda d: d[1])
        return "".join([d[0] for d in detected])

    def match_digit_crop(self, char_crop):
        """Classifies a cropped character contour using the trained ML model."""
        if not hasattr(self, 'digit_classifier') or self.digit_classifier is None:
            return '?', -1.0
            
        # Resize to exactly 16x16 and binarize using Otsu
        resized = cv2.resize(char_crop, (16, 16), interpolation=cv2.INTER_AREA)
        _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        flat = (binary.flatten() / 255.0).reshape(1, -1)
        
        # Predict digit class and probability
        probs = self.digit_classifier.predict_proba(flat)[0]
        best_class_idx = np.argmax(probs)
        best_score = probs[best_class_idx]
        best_char = str(best_class_idx)
        
        return best_char, best_score

    def clean_pot_string(self, text):
        """Cleans and extracts integer/float value from pot OCR text with translation mapping."""
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
        digits = ""
        for c in cleaned:
            if c.isdigit() or c in ['.', ',']:
                digits += c
        digits = digits.replace(',', '.')
        try:
            return float(digits) if '.' in digits else int(digits)
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

    def read_board_state(self, img):
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
        
        hero_matches = self.match_templates_in_roi(
            img, self.rois['hero_cards'], self.card_templates, threshold=0.90, max_matches=2
        )
        state['hero_cards'] = [m[0] for m in hero_matches]
        
        # 2. OCR Pot
        pot_val = 0
        pot_parsed = self.parse_digits_template_matching(img, self.rois['pot'], is_pot=True)
        if pot_parsed:
            pot_val = self.clean_pot_string(pot_parsed)
        if pot_val == 0:
            pot_text = self.ocr_roi(img, self.rois['pot'], whitelist='0123456789.Pulje: ')
            pot_val = self.clean_pot_string(pot_text)
        state['pot_size'] = pot_val
        
        # 3. OCR Hero Stack
        hero_val = 0
        hero_parsed = self.parse_digits_template_matching(img, self.rois['hero_stack'], is_pot=False)
        if hero_parsed:
            hero_val = self.clean_stack_string(hero_parsed)
        if hero_val == 0:
            hero_text = self.ocr_roi(img, self.rois['hero_stack'])
            hero_val = self.clean_stack_string(hero_text)
        state['hero_stack'] = hero_val
        
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
            'hero': (600, 750, 300, 150)
        }
        
        # Match all seats and Hero
        resolved_centers = {}
        for key, (rx, ry, rw, rh) in search_regions.items():
            cx, cy = default_centers[key] # fallback
            if self.hexagon_template is not None:
                search_area = img[ry:ry+rh, rx:rx+rw]
                if search_area.shape[0] >= self.hexagon_template.shape[0] and search_area.shape[1] >= self.hexagon_template.shape[1]:
                    res = cv2.matchTemplate(search_area, self.hexagon_template, cv2.TM_CCORR_NORMED, mask=self.hexagon_mask)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    if max_val >= 0.70:
                        cx = rx + max_loc[0] + self.hexagon_template.shape[1] // 2
                        cy = ry + max_loc[1] + self.hexagon_template.shape[0] // 2 # Rolled back to template center
            resolved_centers[key] = (cx, cy)

        # Parse Opponents
        for seat_key in ['seat_1', 'seat_2', 'seat_3', 'seat_4', 'seat_5']:
            cx, cy = resolved_centers[seat_key]
            
            # VPIP and AGG Crops
            vpip_crop = img[max(0, cy-12):min(img.shape[0], cy+12), max(0, cx-100):min(img.shape[1], cx-80)]
            agg_crop = img[max(0, cy-12):min(img.shape[0], cy+12), max(0, cx+80):min(img.shape[1], cx+100)]
            
            # Name and Stack Crops (Name shifted down: top down 5 [cy+22], bottom down 8 [cy+48]. Stack rolled back to [cy+50, cy+84])
            name_crop = img[max(0, cy+22):min(img.shape[0], cy+48), max(0, cx-75):min(img.shape[1], cx+75)]
            stack_crop = img[max(0, cy+50):min(img.shape[0], cy+84), max(0, cx-65):min(img.shape[1], cx+65)]
            
            # Run OCR on Name
            name_gray = self.preprocess_image(name_crop)
            name_resized = cv2.resize(name_gray, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            name_text = pytesseract.image_to_string(name_resized, config='--psm 6')
            
            import re
            name = name_text.strip()
            name = re.sub(r'^[^a-zA-Z0-9]+', '', name) # remove leading symbols
            name = re.sub(r'[^a-zA-Z0-9_\'\s]+$', '', name) # remove trailing noise
            if len(name) <= 2 and not name.isalnum():
                name = ""
                
            if not name:
                continue
                
            # Run OCR on Stack
            stack_gray = self.preprocess_image(stack_crop)
            stack_resized = cv2.resize(stack_gray, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            stack_text = pytesseract.image_to_string(stack_resized, config='--psm 6')
            
            stack_line = stack_text.strip()
            stack_val = 0
            is_active = True
            state_label = "Active"
            
            if stack_line:
                stack_upper = stack_line.upper()
                if 'ALL' in stack_upper or 'IN' in stack_upper:
                    state_label = "All-In"
                    stack_val = 0
                elif stack_upper == '-' or '-' in stack_upper:
                    state_label = "Folded"
                    stack_val = 0
                    is_active = False
                else:
                    stack_val = self.clean_stack_string(stack_line)
                    if stack_val == 0:
                        state_label = "Folded"
                        is_active = False
            else:
                state_label = "Folded"
                is_active = False
                
            # Classify VPIP and AGG Colors
            vpip_color = self.classify_color(vpip_crop) if is_active else None
            agg_color = self.classify_color(agg_crop) if is_active else None
            
            opponents[seat_key] = {
                'name': name,
                'stack': stack_val,
                'is_active': is_active,
                'state': state_label,
                'vpip_color': vpip_color,
                'agg_color': agg_color
            }
            
        state['opponents'] = opponents
        
        # Parse Hero VPIP/AGG/Stack
        hcx, hcy = resolved_centers['hero']
        hero_vpip_crop = img[max(0, hcy-12):min(img.shape[0], hcy+12), max(0, hcx-100):min(img.shape[1], hcx-80)]
        hero_agg_crop = img[max(0, hcy-12):min(img.shape[0], hcy+12), max(0, hcx+80):min(img.shape[1], hcx+100)]
        
        state['hero_vpip_color'] = self.classify_color(hero_vpip_crop)
        state['hero_agg_color'] = self.classify_color(hero_agg_crop)
        
        # Update hero stack if dynamic matching found a better value
        hero_stack_parsed = self.parse_digits_template_matching(img, (hcx-65, hcy+50, 130, 34), is_pot=False)
        if hero_stack_parsed:
            hero_stack_val = self.clean_stack_string(hero_stack_parsed)
            if hero_stack_val > 0:
                state['hero_stack'] = hero_stack_val
        
        # 5. Detect Active Buttons
        button_matches = self.match_templates_in_roi(
            img, self.rois['buttons'], self.button_templates, threshold=0.85, max_matches=3
        )
        state['active_buttons'] = [m[0] for m in button_matches]
        
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

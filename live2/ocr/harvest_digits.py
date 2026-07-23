"""Harvest binarized digit glyph templates for the chip-count ROIs (stacks and pot are tournament CHIPS, not money).

Sources: every frame we can pair with its own turn record --
  history/<board>/flagged/turn_*/screenshot.png  (turn number in the dir name)
  history/<board>/last_turn.png                  (the board's newest recorded turn)
Both are saved at decision time in the normalized 1536x1090 reference frame, so ROI
geometry and font scale are constant (mirrors core/vision.py exactly).

Labels come from the record's observation (hero_stack / seat stacks / pot_size). Those
are STABILIZED values and can be stale vs the pixels (the monotonic-ratchet bug is why
this harvester exists), so labeling is defensive:
  1. a ROI is used only when its segmented digit count equals the label's digit count;
  2. after a first template build, every source ROI is re-read with the templates and
     samples from ROIs whose consensus read disagrees with their label are dropped, then
     templates rebuild (one self-cleaning round);
  3. the review sheet shows EVERY sample -- the owner reviews before anything goes live.

Binarization: grayscale -> Otsu threshold (glyphs are bright-on-dark; stored glyph=255
on 0). Matching (validation + the future live reader) happens in the same binary space.
"""
import glob
import json
import os
import re
import shutil

import cv2
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HISTORY = os.path.join(REPO, 'history')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

# geometry mirrored from core/vision.py (1536x1090 reference frame)
CENTERS = {'seat_1': (320, 757), 'seat_2': (196, 306), 'seat_3': (767, 180),
           'seat_4': (1340, 306), 'seat_5': (1213, 757), 'hero': (767, 837)}
# Pot line is a fixed element anchored to the window center (owner, 2026-07-23): text
# "Pulje: <amount>" centered on x=768 at y~385. Wider than the legacy ROI so long
# amounts never clip; the label is handled by the trailing-suffix read, not the crop.
POT_ROI = (628, 363, 280, 46)

# glyph geometry filters (reference-frame font)
MIN_H, MAX_H = 10, 32
MIN_W, MAX_W = 2, 26
SEP_MAX_H_FRAC = 0.55          # a component this much shorter than the digit line = separator

# Live-read acceptance gates, tuned on board_samples (2026-07-23): every wrong read in
# the corpus fails at least one -- 4595-for-4505 (0-vs-9) had margin 0.023, 710-for-740
# (4-vs-1) scored 0.25 -- while every correct read passes both. A failed gate = ABSTAIN
# (TableState keeps its last value), never a guess.
READ_MAX_DIST = 0.24           # worst per-glyph XOR distance allowed
READ_MIN_MARGIN = 0.03         # best-vs-second-best distance separation required

# Gates for the SOFT (aliased) pot templates -- different metric, different scale:
# cost = sum|glyph - template/255| / max(ink) at the best integer shift (no resizing),
# so a gray fringe pixel adds/detracts only its uncertainty instead of a full mismatch.
SOFT_MAX_DIST = 0.20
SOFT_MIN_MARGIN = 0.05

# Gates for the GRAYSCALE stack path (aliased glyph vs aliased template): the metric
# accumulates small aliasing differences over every pixel, so correct reads sit higher
# than the binary metrics -- corpus-measured (54 labeled ROIs, every read pixel-
# verified): correct band dist 0.29..0.58 margin >=0.162, impostor distances 0.60+.
STACK_SOFT_MAX_DIST = 0.60
STACK_SOFT_MIN_MARGIN = 0.12

# THE canonical pot transform (owner decision 2026-07-23): truncate the stretched gray
# below this CONSTANT before segmenting/matching pot digits -- always, harvest and live
# alike, so templates and reads share one transform. Chosen just above the 'Pulje'
# label ink band (measured 167..174 vs amount digits 200..255): the label, colon glow
# and halo all go black; only the bright amount digits survive. The colon is still
# located first at the 55% pass purely as a boundary/presence anchor.
POT_TRUNC_GRAY = 176
POT_TRUNC_TH = POT_TRUNC_GRAY / 255.0

# STACK transform (owner-directed 2026-07-23 eve): truncate the stretched gray below
# this constant and SKIP binarization -- glyphs keep their natural anti-aliased gray
# edges (measured on the seat_3 pod: bar body ~106..137, digit ink 155..255; 160 kills
# bar/streak/sheen while preserving glyph aliasing). Templates are built from these
# grayscale glyphs so the aliasing lives in the template, not a threshold.
STACK_TRUNC_GRAY = 160
# Baseline-artifact scrub (owner rule, 2026-07-23): the pod's gloss/timer line can
# survive truncation as a full-width streak at the digits' baseline, welding all
# glyphs into ONE component (hero '680' active pod: one 130x21 blob -> 0 boxes).
# Within the crop's LAST 8 LINES, the topmost line with >50% white pixels AND every
# line below it are deleted. Digit rows measure 9..18% white -- far from the trigger.
STACK_LINE_WIN = 8
STACK_LINE_FRAC = 0.50


def stack_gray(crop_bgr):
    """The canonical STACK transform: contrast-stretched gray truncated below
    STACK_TRUNC_GRAY (NO binarization), then the owner's last-8-lines baseline scrub.
    Applied identically at harvest and read time. Near-flat crops return zeros."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    p_lo, p_hi = np.percentile(gray, (5, 99.5))
    if p_hi - p_lo < 10:
        return np.zeros_like(gray)
    st = np.clip((gray.astype(np.float32) - p_lo) * (255.0 / (p_hi - p_lo)), 0, 255)
    st[st < STACK_TRUNC_GRAY] = 0
    g = st.astype(np.uint8)
    for r in range(max(0, g.shape[0] - STACK_LINE_WIN), g.shape[0]):
        if np.count_nonzero(g[r]) > STACK_LINE_FRAC * g.shape[1]:
            g[r:] = 0
            break
    return g


def accept(text, worst, margin, templates=None, font='stack'):
    """The one gate the live reader applies to a read_number() result.

    ALPHABET-COMPLETENESS: when the templates dict is passed, the font's full digit
    set '0'..'9' must exist -- an incomplete alphabet cannot certify ANY read, because
    a digit with no template silently matches its nearest neighbour instead of failing
    (measured: pot '390' read as an ACCEPTED '300' at dist 0.207 while p9 was missing).
    Live callers must pass templates; harvest-internal label comparison doesn't use
    accept() and is unaffected.

    Gate scale follows the matcher read_chips() picked: a complete soft (aliased) set
    for the font means the scores came from _soft_match_glyph -> SOFT_* gates apply."""
    if text is None:
        return False
    soft_prefix = 'soft_p' if font == 'pot' else 'soft_s'
    is_soft = templates is not None and \
        all(f'{soft_prefix}{d}' in templates for d in '0123456789')
    if not is_soft:
        max_d, min_m = READ_MAX_DIST, READ_MIN_MARGIN
    elif font == 'pot':
        max_d, min_m = SOFT_MAX_DIST, SOFT_MIN_MARGIN
    else:
        max_d, min_m = STACK_SOFT_MAX_DIST, STACK_SOFT_MIN_MARGIN
    if worst > max_d or margin < min_m:
        return False
    if templates is not None and not is_soft:
        prefix = 'p' if font == 'pot' else ''
        if any(prefix + d not in templates for d in '0123456789'):
            return False
    return True


def load_templates():
    """Load every template file for the live reader: keys '0'..'9' (stack font),
    'p0'..'p9' (hard pot medians), 'soft_p0'..'soft_p9' (aliased pot maps)."""
    out = {}
    for p in glob.glob(os.path.join(OUT, 'digit_*.png')):
        out[os.path.basename(p)[6:-4]] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    for p in glob.glob(os.path.join(OUT, 'soft_*.png')):
        out[os.path.basename(p)[:-4]] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return {k: v for k, v in out.items() if v is not None}


def chip_rois(record):
    """[(roi_key, (x, y, w, h), label_int)] for one turn record."""
    obs = record.get('observation_raw') or record.get('observation') or {}
    out = []

    def stack_roi(cx, cy):
        # owner 2026-07-23: top edge +6px (cuts the pod's top bloom/bar), bottom fixed
        return (cx - 65, cy + 56, 130, 28)

    v = obs.get('hero_stack')
    if v and float(v) > 0:
        out.append(('hero_stack', stack_roi(*CENTERS['hero']), int(float(v))))
    for s in obs.get('seats') or []:
        v = s.get('stack')
        key = s.get('seat_key')
        if v and float(v) > 0 and key in CENTERS:
            out.append((key + '_stack', stack_roi(*CENTERS[key]), int(float(v))))
    v = obs.get('pot_size')
    if v and float(v) > 0:
        out.append(('pot', POT_ROI, int(float(v))))
    return out


def binarize(crop_bgr, th=0.55):
    """Percentile contrast-stretch then FIXED threshold (default 55%). Not Otsu: Otsu is
    invariant to the stretch and on dim FOLDED pods (gray digits on dark gray) it either
    drowns the glyphs or merges them -- measured on board_samples, Otsu read a merged
    '1240' as a confident '12' while this variant reads all four digits, and it recovers
    folded-pod stacks legacy OCR returned garbage for. Near-flat crops return empty.
    th is raised (0.70) by read_chips's glow retry: a pot-update glow halo bridges
    adjacent digits at 55% while the bright amount digits survive far higher."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    p_lo, p_hi = np.percentile(gray, (5, 99.5))
    if p_hi - p_lo < 10:
        return np.zeros_like(gray)
    stretched = np.clip((gray.astype(np.float32) - p_lo) * (255.0 / (p_hi - p_lo)), 0, 255)
    binimg = (stretched >= th * 255).astype(np.uint8) * 255
    # glyphs must be the minority-bright class; flip if the pod itself is bright
    if np.count_nonzero(binimg) > binimg.size * 0.5:
        binimg = 255 - binimg
    return binimg


def segment(binimg):
    """-> (digit_boxes, sep_boxes), each [(x, y, w, h)], left-to-right."""
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(binimg, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        # area floor 3, not 6: v25's colon dots are 2x3 with area 5 and were being
        # killed as noise before sep classification (no colon -> whole read abstained).
        # Digits are protected by their own MIN_H/MIN_W below; a 3-5px speck can only
        # become a sep candidate, and find_colon() demands a same-x stacked PAIR.
        if area < 3 or w > MAX_W or h > MAX_H:
            continue
        boxes.append((x, y, w, h))
    if not boxes:
        return [], []
    tall = [b[3] for b in boxes if b[3] >= MIN_H]
    if not tall:
        return [], []
    line_h = float(np.median(tall))
    digits, seps = [], []
    for b in boxes:
        if b[3] >= MIN_H and b[2] >= MIN_W and b[3] > line_h * SEP_MAX_H_FRAC:
            digits.append(b)
        elif b[3] < line_h * SEP_MAX_H_FRAC:
            seps.append(b)
    digits.sort(key=lambda b: b[0])
    seps.sort(key=lambda b: b[0])
    return digits, seps


def glyph(binimg, box):
    x, y, w, h = box
    return binimg[y:y + h, x:x + w]


def find_colon(seps):
    """The pot line's 'Pulje:' colon segments as two tiny boxes vertically stacked at the
    same x (measured signature, e.g. 2x3 @ (133,19) + 3x3 @ (133,27)). -> right edge x of
    the rightmost such pair, or None. The colon is the ONLY safe label/amount boundary --
    per-glyph match failure is not one (truncated '2021'->'21' incident)."""
    best = None
    for i, (x1, y1, w1, h1) in enumerate(seps):
        for x2, y2, w2, h2 in seps[i + 1:]:
            if abs(x1 - x2) <= 3 and 2 <= abs((y1 + h1 / 2) - (y2 + h2 / 2)) <= 14:
                edge = max(x1 + w1, x2 + w2)
                if best is None or edge > best:
                    best = edge
    return best


def pot_boxes(crop_bgr):
    """The canonical two-threshold pot segmentation. -> (bin_t, boxes, seps, ok).
    Pass 1 (55%): locate the 'Pulje:' colon -- presence gate (no colon = this is not
    the pot pill -> abstain, never guess a boundary) and x-boundary. Pass 2 (the
    POT_TRUNC_GRAY constant): truncate; only amount ink survives; segment it. Digit
    boxes right of the colon are the amount; seps (e.g. the thousands comma, which is
    amount-bright and survives) are returned for the ink-completeness guard."""
    b55 = binarize(crop_bgr)
    colon_x = find_colon(segment(b55)[1])
    if colon_x is None:
        return None, [], [], False
    bin_t = binarize(crop_bgr, th=POT_TRUNC_TH)
    digits, seps = segment(bin_t)
    return bin_t, [b for b in digits if b[0] > colon_x], \
        [s for s in seps if s[0] > colon_x], True


def collect_frames():
    """[(frame_path, record)] -- every frame paired with its own turn record."""
    pairs = []
    for shot in glob.glob(os.path.join(HISTORY, '*', 'flagged', 'turn_*', 'screenshot.png')):
        m = re.search(r'turn_(\d+)_', os.path.basename(os.path.dirname(shot)))
        board_dir = os.path.dirname(os.path.dirname(os.path.dirname(shot)))
        if not m:
            continue
        turn = int(m.group(1))
        rec = None
        try:
            for line in open(os.path.join(board_dir, 'turns.jsonl'), encoding='utf-8'):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get('turn') == turn:
                    rec = r                      # last wins (duplicate-turn era files)
        except OSError:
            continue
        if rec:
            pairs.append((shot, rec))
    for png in glob.glob(os.path.join(HISTORY, '*', 'last_turn.png')):
        try:
            lines = [l for l in open(os.path.join(os.path.dirname(png), 'turns.jsonl'),
                                     encoding='utf-8') if l.strip()]
            rec = json.loads(lines[-1])
            pairs.append((png, rec))
        except (OSError, json.JSONDecodeError, IndexError):
            continue
    return pairs


def _diag_pot_candidates(v):
    """Displayed digit-string candidates for an old-era telemetry pot_size. That era
    parsed the Danish-formatted pot ('Pulje: 1.040') as a FLOAT -> 1.04, so fractional
    values recover the displayed digits as v*1000. Integral values are ambiguous
    ('60' vs '60.000' -> 60.0) -> both candidates; the segmented digit count picks."""
    if v <= 0:
        return []
    if abs(v - round(v)) < 1e-9:
        return [str(int(round(v))), str(int(round(v * 1000)))]
    return [str(int(round(v * 1000)))]


def collect_diag_pot():
    """[(frame_path, record)] from diagnostics/turn_*/telemetry.json -- POT ONLY (that
    era's seat layout keys don't map to current CENTERS; stack classes are already fat
    from pilot frames). The label candidate is resolved HERE by segmenting the frame's
    pot ROI and keeping the candidate whose digit count matches, then emitted as a
    normal record so harvest/validate/self-clean treat it like any other source."""
    pairs = []
    for tj in glob.glob(os.path.join(REPO, 'diagnostics', 'turn_*', 'telemetry.json')):
        shot = os.path.join(os.path.dirname(tj), 'screenshot.png')
        img = cv2.imread(shot)
        if img is None:
            continue
        try:
            with open(tj, encoding='utf-8') as f:
                v = float(json.load(f)['table_state']['pot_size'])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if img.shape[1] != 1536 or img.shape[0] != 1090:
            img = cv2.resize(img, (1536, 1090), interpolation=cv2.INTER_CUBIC)
        x, y, w, h = POT_ROI
        _bt, boxes, _seps, ok = pot_boxes(img[y:y + h, x:x + w])
        if not ok:
            continue
        matches = [c for c in _diag_pot_candidates(v) if len(c) == len(boxes)]
        if len(matches) == 1:
            pairs.append((shot, {'observation': {'pot_size': int(matches[0])}}))
    return pairs


def harvest(pairs):
    """-> samples: {label_char: [ {img, src} ]}, one entry per segmented glyph."""
    samples = {}
    used = skipped = 0
    for path, rec in pairs:
        img = cv2.imread(path)
        if img is None:
            continue
        if img.shape[1] != 1536 or img.shape[0] != 1090:
            img = cv2.resize(img, (1536, 1090), interpolation=cv2.INTER_CUBIC)
        for key, (x, y, w, h), label in chip_rois(rec):
            crop = img[max(0, y):y + h, max(0, x):x + w]
            if crop.size == 0:
                continue
            text = str(label)
            if key == 'pot':
                # pot renders in its OWN (smaller) font -> separate 'p<d>' template
                # classes, harvested under the CANONICAL pot transform (colon-anchored
                # boundary from the 55% pass, glyphs from the POT_TRUNC truncation) so
                # templates and live reads share one transform
                binimg, boxes, seps, ok = pot_boxes(crop)
                prefix = 'p'
                seps = []
                gtr = None
            else:
                # canonical STACK transform for harvest too: segment on the gray
                # transform's ink mask, exactly what read_chips(font='stack') does
                gtr = stack_gray(crop)
                binimg = (gtr > 0).astype(np.uint8) * 255
                boxes, seps = segment(binimg)
                ok, prefix = True, ''
            if not ok or len(boxes) != len(text):
                skipped += 1
                continue
            used += 1
            frame_id = (path, key)             # unique identity for the cleaning pass
            for ch, box in zip(text, boxes):
                s = {'img': glyph(binimg, box), 'id': frame_id, 'roi': key, 'label': label}
                if gtr is not None:
                    s['gray'] = glyph(gtr, box)
                samples.setdefault(prefix + ch, []).append(s)
            for box in seps:
                samples.setdefault('sep', []).append({'img': glyph(binimg, box), 'id': frame_id,
                                                      'roi': key, 'label': label})
    return samples, used, skipped


def load_manual(samples):
    """Merge hand-labeled glyphs from templates/manual/<class>/*.png into the harvest.
    These exist because record-labeled harvesting SKIPS frames whose stale label
    disagrees with the pixels -- which is exactly where the rare digits hid (every pot
    '7'/'9' in the 2026-07 corpus sat on a mislabeled frame). Human-labeled, so the
    self-cleaning round must never drop them: their id is the manual path, which
    validate() can never emit as a bad frame id."""
    n = 0
    for p in glob.glob(os.path.join(OUT, 'manual', '*', '*.png')):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        ch = os.path.basename(os.path.dirname(p))
        samples.setdefault(ch, []).append(
            {'img': (img >= 128).astype(np.uint8) * 255, 'id': ('manual', p),
             'roi': 'manual', 'label': ch})
        n += 1
    return n


def build_soft_templates(samples, pad=3, key='img'):
    """Owner-spec'd 'aliased' templates (2026-07-23, refined same day): every sample of
    a class is aligned by integer translation -- no resizing -- to the position of best
    overall fit against the running class mean, where fit is SYMMETRIC: white matching
    white pulls the offset in, and white landing on black (either direction) pushes it
    away (minimized |mean - sample| over the whole canvas). The aligned stack is then
    averaged per pixel, so the two effects the owner asked for happen together: pixels
    all renders share stay full white (the centered core); minority whites accumulate
    as gray anti-aliasing fringe; and whites that other samples vote BLACK on are toned
    down toward gray -- the result is the average pixel color of the font at native
    scale. Seed = the sample closest to the class's median shape, pasted centered."""
    out = {}
    for ch, items in samples.items():
        # 0/255 binary samples and grayscale (aliased) samples both normalize to 0..1;
        # alignment and per-pixel averaging generalize unchanged
        imgs = [s[key].astype(np.float32) / 255.0 for s in items if key in s]
        if not imgs:
            continue
        med_area = float(np.median([g.shape[0] * g.shape[1] for g in imgs]))
        imgs.sort(key=lambda g: abs(g.shape[0] * g.shape[1] - med_area))
        H = max(g.shape[0] for g in imgs) + 2 * pad
        W = max(g.shape[1] for g in imgs) + 2 * pad
        acc = np.zeros((H, W), np.float32)
        for n, g in enumerate(imgs):
            gh, gw = g.shape
            if n == 0:
                oy, ox = (H - gh) // 2, (W - gw) // 2
            else:
                ref, best, (oy, ox) = acc / n, None, (0, 0)
                for dy in range(H - gh + 1):
                    for dx in range(W - gw + 1):
                        canvas = np.zeros((H, W), np.float32)
                        canvas[dy:dy + gh, dx:dx + gw] = g
                        d = float(np.abs(ref - canvas).sum())
                        if best is None or d < best:
                            best, (oy, ox) = d, (dy, dx)
            acc[oy:oy + gh, ox:ox + gw] += g
        soft = np.rint(acc / len(imgs) * 255).astype(np.uint8)
        ys, xs = np.nonzero(soft)
        out[ch] = soft[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return out


def build_templates(samples):
    """Median-stack each glyph class into one canonical binary template (native scale)."""
    templates = {}
    for ch, items in samples.items():
        if not items:
            continue
        hh = int(np.median([s['img'].shape[0] for s in items]))
        ww = int(np.median([s['img'].shape[1] for s in items]))
        stack = []
        for s in items:
            stack.append(cv2.resize(s['img'], (ww, hh), interpolation=cv2.INTER_NEAREST))
        med = np.median(np.stack(stack), axis=0)
        templates[ch] = (med >= 128).astype(np.uint8) * 255
    return templates


def _soft_match_glyph(binimg, box, soft_tset):
    """Match one segmented glyph against the SOFT (aliased) templates: the glyph is
    slid over each template (integer shifts, never resized) and scored by the mean
    absolute difference against the 0..1 probability map -- a gray pixel adds or
    detracts only its confidence fraction. Works for binary (0/255) AND grayscale
    glyphs: both normalize to 0..1, so an aliased glyph edge is compared against the
    template's aliased edge at fractional weight (owner's no-binarization stack path).
    -> (best_char, best_dist, second_dist)."""
    g = glyph(binimg, box).astype(np.float32) / 255.0
    gh, gw = g.shape
    best_ch, best_d, second_d = None, 1.0, 1.0
    for ch, tpl in soft_tset.items():
        t = tpl.astype(np.float32) / 255.0
        th, tw = t.shape
        H, W = max(gh, th) + 4, max(gw, tw) + 4
        T = np.zeros((H, W), np.float32)
        T[2:2 + th, 2:2 + tw] = t
        norm = max(float(T.sum()), float(g.sum()), 1.0)
        d_ch = 1.0
        for dy in range(H - gh + 1):
            for dx in range(W - gw + 1):
                G = np.zeros((H, W), np.float32)
                G[dy:dy + gh, dx:dx + gw] = g
                d_ch = min(d_ch, float(np.abs(G - T).sum()) / norm)
        if d_ch < best_d:
            best_ch, best_d, second_d = ch, d_ch, best_d
        elif d_ch < second_d:
            second_d = d_ch
    return best_ch, best_d, second_d


def _match_glyph(binimg, box, templates):
    """-> (best_char, best_dist, second_dist) for one segmented box."""
    g = glyph(binimg, box)
    best_ch, best_d, second_d = None, 1.0, 1.0
    for ch, tpl in templates.items():
        if ch == 'sep':
            continue
        gr = cv2.resize(g, (tpl.shape[1], tpl.shape[0]), interpolation=cv2.INTER_NEAREST)
        d = float(np.count_nonzero(gr != tpl)) / tpl.size
        if d < best_d:
            best_ch, best_d, second_d = ch, d, best_d
        elif d < second_d:
            second_d = d
    return best_ch, best_d, second_d


def _read_boxes(binimg, digits, seps, tset, matcher, max_stray_frac=0.10):
    """Shared gated glyph-sequence read -> (string, worst_score, min_margin).

    A failing glyph ANYWHERE -> the whole read abstains (a truncated suffix once read
    2021 as a confident 21 -- partial chip values are never returned). Separators
    ('.'/',') are segmented out by height and ignored, matching the cents convention.

    INK-COMPLETENESS GUARD: ink inside the read span that no segmented box claims
    means glyphs were dropped (merged past the width filter, broken strokes) -- reading
    the survivors yields a confidently WRONG number ('1240' -> '12'; a lone noise blob
    -> '7'). More than max_stray_frac unclaimed ink = abstain."""
    if not digits or not tset:
        return None, 1.0, 0.0
    x_left = min(b[0] for b in digits) - 2
    y0 = min(b[1] for b in digits)
    y1 = max(b[1] + b[3] for b in digits)
    covered = np.zeros(binimg.shape, dtype=bool)
    for x, y, w, h in digits + seps:
        covered[y:y + h, x:x + w] = True
    band = np.zeros(binimg.shape, dtype=bool)
    band[max(0, y0 - 2):y1 + 2, max(0, x_left):] = True
    total = int(np.count_nonzero((binimg > 0) & band))
    stray = int(np.count_nonzero((binimg > 0) & band & ~covered))
    if total and stray / total > max_stray_frac:
        return None, 1.0, 0.0

    out, worst, min_margin = [], 0.0, 1.0
    for box in digits:
        ch, d, s = matcher(binimg, box, tset)
        out.append(ch or '?')
        worst = max(worst, d)
        min_margin = min(min_margin, s - d)
    return ''.join(out), worst, min_margin


def read_number(binimg, templates, max_stray_frac=0.10, font='stack'):
    """Template-read a binarized STACK-font chip ROI -> (string, worst, min_margin).
    Score per glyph = normalized XOR distance in [0,1]; lower = better. Pot ROIs go
    through read_chips() -- the pot path owns its own canonical binarization and must
    start from the raw crop, not a pre-binarized image."""
    if font == 'pot':
        raise ValueError("pot ROIs are read by read_chips(crop) -- the canonical pot "
                         "transform starts from the raw crop")
    digits, seps = segment(binimg)
    tset = {k: v for k, v in templates.items()
            if not k.startswith('p') and not k.startswith('soft_') and k != 'sep'}
    return _read_boxes(binimg, digits, seps, tset, _match_glyph, max_stray_frac)


def read_chips(crop_bgr, templates, font='stack'):
    """THE crop-level money reader -> (text, worst, margin, accepted).

    font='stack' (owner-canonicalized 2026-07-23 eve): stack_gray() -- stretch,
    truncate below STACK_TRUNC_GRAY, NO binarization -- segment on the ink mask,
    then match the GRAYSCALE glyphs against the aliased soft_s templates: real
    font anti-aliasing on both sides of the comparison, at fractional weight.
    font='pot' (owner-canonicalized 2026-07-23): pot_boxes() applies the ONE pot
    transform -- colon presence/boundary at 55%, then truncation below POT_TRUNC_GRAY
    so only amount ink survives -- matched against the soft_p templates harvested
    under that same transform.
    Either font: an incomplete soft alphabet -> abstain (never fall back to a
    mismatched template basis). All _read_boxes guards (truncation-forbidden,
    ink-completeness) apply."""
    if font != 'pot':
        gtr = stack_gray(crop_bgr)
        digits, seps = segment((gtr > 0).astype(np.uint8) * 255)
        soft_tset = {k[6:]: v for k, v in templates.items() if k.startswith('soft_s')}
        if not all(d in soft_tset for d in '0123456789'):
            return None, 1.0, 0.0, False
        got, worst, margin = _read_boxes(gtr, digits, seps, soft_tset, _soft_match_glyph)
        return got, worst, margin, accept(got, worst, margin, templates, font)

    bin_t, boxes, seps, ok = pot_boxes(crop_bgr)
    if not ok:
        return None, 1.0, 0.0, False
    soft_tset = {k[6:]: v for k, v in templates.items() if k.startswith('soft_p')}
    if not all(d in soft_tset for d in '0123456789'):
        return None, 1.0, 0.0, False
    got, worst, margin = _read_boxes(bin_t, boxes, seps, soft_tset, _soft_match_glyph)
    return got, worst, margin, accept(got, worst, margin, templates, 'pot')


def validate(pairs, templates):
    """Re-read the HARVESTABLE ROIs (segmented digit count == label length -- the rest
    are the stale-label class, counted separately as unlabelable).
    -> (n_ok, n_bad, n_unlabelable, mismatches list with frame ids)."""
    ok, bad, unlabelable, mism = 0, 0, 0, []
    for path, rec in pairs:
        img = cv2.imread(path)
        if img is None:
            continue
        if img.shape[1] != 1536 or img.shape[0] != 1090:
            img = cv2.resize(img, (1536, 1090), interpolation=cv2.INTER_CUBIC)
        for key, (x, y, w, h), label in chip_rois(rec):
            crop = img[max(0, y):y + h, max(0, x):x + w]
            if crop.size == 0:
                continue
            if key == 'pot':
                # canonical pot transform; matched against the HARD pot medians
                # (the soft set is only built after cleaning) -- label comparison
                # is all this pass needs
                binimg, boxes, seps, seg_ok = pot_boxes(crop)
                tset = {k[1:]: v for k, v in templates.items() if k.startswith('p')}
            else:
                binimg = (stack_gray(crop) > 0).astype(np.uint8) * 255
                boxes, seps = segment(binimg)
                seg_ok = True
                tset = {k: v for k, v in templates.items() if not k.startswith('p')
                        and k != 'sep'}
            if not seg_ok or len(boxes) != len(str(label)):
                unlabelable += 1
                continue
            got, score, margin = _read_boxes(binimg, boxes, seps, tset, _match_glyph)
            if got == str(label):
                ok += 1
            else:
                bad += 1
                mism.append({'id': (path, key), 'src': os.path.basename(os.path.dirname(path)),
                             'roi': key, 'label': label, 'read': got,
                             'score': round(score, 3), 'margin': round(margin, 3)})
    return ok, bad, unlabelable, mism


def main():
    pairs = collect_frames()
    diag = collect_diag_pot()
    print(f"frames with records: {len(pairs)} + {len(diag)} diagnostics pot frames")
    pairs += diag
    samples, used, skipped = harvest(pairs)
    n_manual = load_manual(samples)
    total = sum(len(v) for v in samples.values())
    print(f"ROIs used: {used} (skipped {skipped} count-mismatch) + {n_manual} manual "
          f"-> {total} glyph samples")
    for ch in sorted(samples):
        print(f"  '{ch}': {len(samples[ch])} samples")

    templates = build_templates(samples)

    # self-cleaning round: drop samples whose whole-ROI read disagrees with its label
    ok, bad, unlab, mism = validate(pairs, templates)
    print(f"round 1 validation: {ok} ok / {bad} mismatched / {unlab} unlabelable")
    bad_ids = {tuple(m['id']) for m in mism}
    cleaned = {ch: [s for s in items if s['id'] not in bad_ids]
               for ch, items in samples.items()}
    templates = build_templates(cleaned)
    ok2, bad2, unlab2, mism2 = validate(pairs, templates)
    print(f"round 2 validation (cleaned): {ok2} ok / {bad2} mismatched / {unlab2} unlabelable")

    os.makedirs(OUT, exist_ok=True)
    # RESET generated outputs before writing: a run with fewer samples/classes than the
    # last one must not leave stale files behind (a leftover samples/p5/003.png -- a '0'
    # mislabeled '5' by the first fully-label-trusting run -- haunted the review sheet,
    # and a stale digit_p8.png once satisfied the alphabet-completeness gate).
    # templates/manual/ is hand-curated input, never touched.
    for p in glob.glob(os.path.join(OUT, 'digit_*.png')) + \
            glob.glob(os.path.join(OUT, 'soft_*.png')):
        os.remove(p)
    shutil.rmtree(os.path.join(OUT, 'samples'), ignore_errors=True)
    for ch, tpl in templates.items():
        cv2.imwrite(os.path.join(OUT, f"digit_{ch}.png"), tpl)
    # aliased/soft pot templates (owner spec) -- alignment-averaged probability maps
    soft = build_soft_templates({ch: v for ch, v in cleaned.items() if ch.startswith('p')})
    for ch, tpl in soft.items():
        cv2.imwrite(os.path.join(OUT, f"soft_{ch}.png"), tpl)
    # aliased STACK templates (owner-directed): built from the GRAYSCALE glyphs of the
    # truncate-below-160 no-binarization transform -- files soft_s<d>.png
    soft_stack = build_soft_templates(
        {ch: v for ch, v in cleaned.items() if ch in '0123456789'}, key='gray')
    for ch, tpl in soft_stack.items():
        cv2.imwrite(os.path.join(OUT, f"soft_s{ch}.png"), tpl)
    for ch, items in cleaned.items():
        d = os.path.join(OUT, 'samples', ch)
        os.makedirs(d, exist_ok=True)
        for i, s in enumerate(items):
            cv2.imwrite(os.path.join(d, f"{i:03d}.png"), s['img'])
    with open(os.path.join(OUT, 'review.json'), 'w', encoding='utf-8') as f:
        json.dump({'frames': len(pairs), 'rois_used': used, 'rois_skipped': skipped,
                   'samples': {ch: len(v) for ch, v in cleaned.items()},
                   'validation': {'ok': ok2, 'bad': bad2, 'unlabelable': unlab2,
                                  'mismatches': [{k: v for k, v in m.items() if k != 'id'}
                                                 for m in mism2[:40]]}},
                  f, indent=2)
    print(f"templates + samples + review.json -> {OUT}")
    return templates, cleaned, (ok2, bad2, mism2)


if __name__ == '__main__':
    main()

"""Harvest binarized digit glyph templates for the money-field ROIs.

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

import cv2
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HISTORY = os.path.join(REPO, 'history')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

# geometry mirrored from core/vision.py (1536x1090 reference frame)
CENTERS = {'seat_1': (320, 757), 'seat_2': (196, 306), 'seat_3': (767, 180),
           'seat_4': (1340, 306), 'seat_5': (1213, 757), 'hero': (767, 837)}
POT_ROI = (700, 365, 160, 45)

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


def accept(text, worst, margin):
    """The one gate the live reader applies to a read_number() result."""
    return text is not None and worst <= READ_MAX_DIST and margin >= READ_MIN_MARGIN


def money_rois(record):
    """[(roi_key, (x, y, w, h), label_int)] for one turn record."""
    obs = record.get('observation_raw') or record.get('observation') or {}
    out = []

    def stack_roi(cx, cy):
        return (cx - 65, cy + 50, 130, 34)

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


def binarize(crop_bgr):
    """Percentile contrast-stretch then FIXED 55% threshold. Not Otsu: Otsu is invariant
    to the stretch and on dim FOLDED pods (gray digits on dark gray) it either drowns the
    glyphs or merges them -- measured on board_samples, Otsu read a merged '1240' as a
    confident '12' while this variant reads all four digits, and it recovers folded-pod
    stacks legacy OCR returned garbage for. Near-flat crops (no text) return empty."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    p_lo, p_hi = np.percentile(gray, (5, 99.5))
    if p_hi - p_lo < 10:
        return np.zeros_like(gray)
    stretched = np.clip((gray.astype(np.float32) - p_lo) * (255.0 / (p_hi - p_lo)), 0, 255)
    binimg = (stretched >= 0.55 * 255).astype(np.uint8) * 255
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
        if area < 6 or w > MAX_W or h > MAX_H:
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
        for key, (x, y, w, h), label in money_rois(rec):
            crop = img[max(0, y):y + h, max(0, x):x + w]
            if crop.size == 0:
                continue
            binimg = binarize(crop)
            digits, seps = segment(binimg)
            text = str(label)
            if len(digits) != len(text):
                skipped += 1
                continue
            used += 1
            frame_id = (path, key)             # unique identity for the cleaning pass
            for ch, box in zip(text, digits):
                samples.setdefault(ch, []).append({'img': glyph(binimg, box), 'id': frame_id,
                                                   'roi': key, 'label': label})
            for box in seps:
                samples.setdefault('sep', []).append({'img': glyph(binimg, box), 'id': frame_id,
                                                      'roi': key, 'label': label})
    return samples, used, skipped


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


def read_number(binimg, templates, max_stray_frac=0.10):
    """Template-read a binarized money ROI -> (string, worst_score). Score per glyph =
    normalized XOR distance in [0,1] after resize to the template box; lower = better.
    This is the same reader the live path will use; abstention = high worst_score.

    INK-COMPLETENESS GUARD: ink inside the digit line band that no segmented box claims
    means glyphs were dropped (merged past the width filter, broken strokes) -- reading
    the survivors yields a confidently WRONG number ('1240' -> '12'; a lone noise blob
    -> '7'). More than max_stray_frac unclaimed ink = abstain."""
    digits, seps = segment(binimg)
    if not digits:
        return None, 1.0, 0.0
    y0 = min(b[1] for b in digits)
    y1 = max(b[1] + b[3] for b in digits)
    covered = np.zeros(binimg.shape, dtype=bool)
    for x, y, w, h in digits + seps:
        covered[y:y + h, x:x + w] = True
    band = np.zeros(binimg.shape, dtype=bool)
    band[max(0, y0 - 2):y1 + 2, :] = True
    total = int(np.count_nonzero((binimg > 0) & band))
    stray = int(np.count_nonzero((binimg > 0) & band & ~covered))
    if total and stray / total > max_stray_frac:
        return None, 1.0, 0.0

    out, worst, min_margin = [], 0.0, 1.0
    for box in digits:
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
        out.append(best_ch or '?')
        worst = max(worst, best_d)
        min_margin = min(min_margin, second_d - best_d)
    return ''.join(out), worst, min_margin


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
        for key, (x, y, w, h), label in money_rois(rec):
            crop = img[max(0, y):y + h, max(0, x):x + w]
            if crop.size == 0:
                continue
            binimg = binarize(crop)
            digits, _seps = segment(binimg)
            if len(digits) != len(str(label)):
                unlabelable += 1
                continue
            got, score, margin = read_number(binimg, templates)
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
    print(f"frames with records: {len(pairs)}")
    samples, used, skipped = harvest(pairs)
    total = sum(len(v) for v in samples.values())
    print(f"ROIs used: {used} (skipped {skipped} count-mismatch) -> {total} glyph samples")
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
    for ch, tpl in templates.items():
        cv2.imwrite(os.path.join(OUT, f"digit_{ch}.png"), tpl)
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

"""Read text: labels written along the ribbons (straightened into flat strips first) and free text
around them (title, names inside the loops, badge numbers). Uses Tesseract."""
import os
import re
import shutil

import cv2
import numpy as np
import pytesseract

from colors import tangents

for cand in (shutil.which('tesseract'), r'C:\Program Files\Tesseract-OCR\tesseract.exe',
             r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
             os.path.expandvars(r'%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe')):
    if cand and os.path.exists(cand):
        pytesseract.pytesseract.tesseract_cmd = cand
        break

WORDS_PATH = os.path.join(os.path.dirname(__file__), '..', 'datagen', 'words.txt')
STRIP_H = 64            # label strips are scaled to this height before reading


def tesseract_ok():
    try:
        return str(pytesseract.get_tesseract_version())
    except Exception:
        return None


def load_words():
    try:
        with open(WORDS_PATH, encoding='utf-8') as f:
            return {w.strip().upper() for w in f if w.strip()}
    except OSError:
        return set()


VOCAB = load_words()


def unroll(img, pts, closed, ids, T):
    """Straighten the band around chain points ids into a flat strip (left → right along the chain)."""
    p = pts[ids]
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] < 4:
        return None
    s = np.arange(0, cum[-1], 1.0)
    cx = np.interp(s, cum, p[:, 0])
    cy = np.interp(s, cum, p[:, 1])
    tg = tangents(np.column_stack([cx, cy]), False)
    nr = np.column_stack([-tg[:, 1], tg[:, 0]])
    h = int(T * 1.0)
    off = np.linspace(-h / 2, h / 2, h)
    # rows run across the band so that text written along the direction of travel reads upright
    mx = (cx[None, :] + nr[None, :, 0] * off[:, None]).astype(np.float32)
    my = (cy[None, :] + nr[None, :, 1] * off[:, None]).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def text_layer(strip):
    """Light letters on a colored band → black letters on white, plus the letter mask and color."""
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)
    lightness = v - 0.6 * s
    _, m = cv2.threshold(np.clip(lightness, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    letters = m > 0
    # light pixels on more than half the strip = we picked the background, not letters
    if letters.mean() > 0.5:
        letters = ~letters
    h = strip.shape[0]
    letters[: int(h * 0.12)] = False
    letters[-int(h * 0.12):] = False
    out = np.full(strip.shape[:2], 255, np.uint8)
    out[letters] = 0
    # a soft version too (dark letters on light, not thresholded): Tesseract's own binarisation
    # sometimes keeps thin strokes that Otsu loses
    soft = np.clip(lightness, 0, 255)
    lo, hi = np.percentile(soft, 5), np.percentile(soft, 99)
    soft = np.clip((soft - lo) / max(1, hi - lo) * 255, 0, 255).astype(np.uint8)
    soft = soft if (m > 0).mean() <= 0.5 else 255 - soft
    soft = 255 - soft
    color = np.median(strip[letters], axis=0) if letters.sum() > 10 else np.array([255, 255, 255])
    rows = np.nonzero(letters.mean(1) > 0.02)[0]
    height = (rows[-1] - rows[0] + 1) if len(rows) else 0
    return out, letters, color, height, soft


def fix_word(t):
    """Snap a misread label onto a known process word when it is close (datagen/words.txt).
    Returns (text, how): how = 'exact', 'close' (snapped) or ''."""
    t = re.sub(r'[^A-Za-z0-9&/ -]', '', t).strip()
    if not VOCAB or not t:
        return t, ''
    u = t.upper()
    if u in VOCAB:
        return u, 'exact'
    # "PLAN EGY": a known word plus scraps of a neighbouring arrow or label
    words = u.split()
    known = [w for w in words if w in VOCAB]
    if len(words) > 1 and len(known) == 1 and all(len(w) <= 3 for w in words if w not in VOCAB):
        return known[0], 'close'
    if len(u) < 3:
        return u, ''
    best, bd = None, 99
    for w in VOCAB:
        if abs(len(w) - len(u)) > 3:
            continue
        d = _lev(u, w)
        # a label partly hidden under a crossing reads as the word's start ("COLLE" → COLLECT)
        if len(u) >= 4 and w.startswith(u):
            d = min(d, 1)
        if w[0] != u[0]:
            d += 0.6                     # first letters are rarely misread
        if d < bd:
            best, bd = w, d
    if best and bd <= max(1, round(len(u) * 0.4)):
        return best, 'close'
    return u, ''


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def read_strips(strips, extra=''):
    """OCR many one-line strips with one Tesseract call: stack them with white space between and
    sort the words back to their rows. strips: list of black-on-white images (any height)."""
    if not strips:
        return []
    gap = 30
    rows, y = [], gap
    scaled = []
    for s in strips:
        k = STRIP_H / max(1, s.shape[0])
        im = cv2.resize(s, (max(1, int(s.shape[1] * k)), STRIP_H), interpolation=cv2.INTER_CUBIC)
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        scaled.append(im)
        rows.append((y, y + STRIP_H))
        y += STRIP_H + gap
    W = max(im.shape[1] for im in scaled) + 2 * gap
    sheet = np.full((y, W), 255, np.uint8)
    for im, (a, b) in zip(scaled, rows):
        sheet[a:b, gap:gap + im.shape[1]] = im
    data = pytesseract.image_to_data(sheet, config='--psm 6 ' + extra, output_type=pytesseract.Output.DICT)
    out = [{'words': [], 'conf': []} for _ in strips]
    for i, word in enumerate(data['text']):
        word = word.strip()
        conf = float(data['conf'][i])
        if not word or conf < 0:
            continue
        cy = data['top'][i] + data['height'][i] / 2
        for r, (a, b) in enumerate(rows):
            if a - gap / 2 <= cy <= b + gap / 2:
                out[r]['words'].append((data['left'][i], word))
                out[r]['conf'].append(conf)
                break
    res = []
    for o in out:
        words = [w for _, w in sorted(o['words'])]
        res.append({'text': ' '.join(words), 'conf': float(np.mean(o['conf'])) if o['conf'] else 0.0})
    return res


def ribbon_labels(img, chains, analysis, T, scale_to_canvas=1.0):
    """For every piece: its label text, text color, font size (work px) and confidence."""
    jobs = []
    for ci, (ch, an) in enumerate(zip(chains, analysis)):
        for pi, pc in enumerate(an['pieces']):
            ids = pc['ids']
            if len(ids) < 4:
                continue
            strip = unroll(img, ch['pts'], ch['closed'], ids, T)
            if strip is None or strip.shape[1] < T * 0.6:
                continue
            # leave out the ends, where the white arrow tips are
            cut = int(T * 0.4)
            if strip.shape[1] > 2 * cut + T * 0.5:
                strip = strip[:, cut:strip.shape[1] - cut]
            bw, letters, color, height, soft = text_layer(strip)
            if letters.sum() < 15 or height < 4:
                continue
            # trim to the letters, with a margin
            cols = np.nonzero(letters.any(0))[0]
            a, b = max(0, cols[0] - 8), min(bw.shape[1], cols[-1] + 9)
            jobs.append((ci, pi, [bw[:, a:b], soft[:, a:b]], color, height))
    if not jobs:
        return
    # every strip both ways round (we don't know which way the text runs), thresholded and soft
    imgs = []
    for _, _, versions, _, _ in jobs:
        for v in versions:
            imgs += [v, cv2.rotate(v, cv2.ROTATE_180)]
    try:
        reads = read_strips(imgs)
    except Exception as e:
        print('Label reading failed:', e)
        return
    per = 4
    for k, (ci, pi, _, color, height) in enumerate(jobs):
        cand = []
        for r in reads[per * k:per * k + per]:
            t = re.sub(r'\s+', ' ', r['text']).strip()
            letters = sum(ch.isalnum() for ch in t)
            fixed, how = fix_word(t)
            score = r['conf'] * {'exact': 1.6, 'close': 1.25, '': 1.0}[how] * min(1, letters / 3)
            cand.append((score, fixed, r['conf'], how))
        cand.sort(key=lambda c: -c[0])
        score, text, conf, known = cand[0]
        if not text or sum(ch.isalnum() for ch in text) < 2 or conf < 35:
            continue
        pc = analysis[ci]['pieces'][pi]
        pc['text'] = text
        pc['textColor'] = '#%02X%02X%02X' % tuple(int(v) for v in color[::-1])
        pc['fontSize'] = float(height / 0.72)          # cap height ≈ 0.72 × font size
        pc['textConf'] = round(min(0.99, conf / 100 * {'exact': 1.1, 'close': 0.8, '': 0.9}[known]), 2)


def free_text(img, cand, solid, badges, T):
    """Text outside the ribbons, as lines: [{text, box:[x,y,w,h], height, conf, color}].
    Also returns the leftover ink blobs that are not text (icon candidates)."""
    h, w = solid.shape
    ink = (cand > 0) & (cv2.dilate(solid, np.ones((int(T * 0.12) | 1,) * 2, np.uint8)) == 0)
    ink = ink.astype(np.uint8)
    for (x, y, r) in badges:
        cv2.circle(ink, (int(x), int(y)), int(r * 1.25 + 4), 0, -1)
    # letters of one line merge when smeared sideways
    kx = max(3, int(T * 0.3))
    blobs = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((max(3, int(T * 0.06)), kx), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(blobs, 8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cands, others = [], []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bh < 7 or bw < 6 or area < 30:
            continue
        box = [int(x), int(y), int(bw), int(bh)]
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue                      # cut off by the picture's edge
        if bh > 1.3 * T or bw < bh * 0.8:
            others.append(box)            # too tall or too narrow for a line of text: maybe an icon
            continue
        pad = max(4, bh // 4)
        crop = gray[max(0, y - pad):y + bh + pad, max(0, x - pad):x + bw + pad]
        # black letters on pure white, like the ribbon strips
        k = 64.0 / max(1, crop.shape[0])
        crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        # two versions: Otsu (good for bold text) and adaptive (keeps thin, light text from merging)
        _, a = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        b = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        a, b = [255 - v if (v == 0).mean() > 0.5 else v for v in (a, b)]
        cands.append((box, a, b))
    reads = read_strips([v for _, a, b in cands for v in (a, b)]) if cands else []
    lines = []
    for k, (box, _, _) in enumerate(cands):
        rd = max(reads[2 * k], reads[2 * k + 1], key=lambda r: r['conf'] * min(1, len(r['text']) / 3))
        text, conf = rd['text'].strip(), rd['conf']
        letters = sum(ch.isalnum() for ch in text)
        x, y, bw, bh = box
        # a word of 3+ letters is clearly wider than tall; a square blob read as "bis" is an icon
        if letters < 2 or conf < 45 or letters < 0.5 * len(text.replace(' ', '')) or (letters >= 3 and bw < bh * 1.2):
            others.append(box)
            continue
        x, y, bw, bh = box
        m = ink[y:y + bh, x:x + bw] > 0
        color = np.median(img[y:y + bh, x:x + bw][m], axis=0) if m.sum() > 5 else np.array([40, 35, 31])
        rows = np.nonzero(m.mean(1) > 0.03)[0]
        lines.append({'text': text, 'box': box, 'height': float(rows[-1] - rows[0] + 1) if len(rows) else float(bh),
                      'conf': conf, 'color': '#%02X%02X%02X' % tuple(int(v) for v in color[::-1])})
    return lines, others


def read_digits(img, badges):
    """The number inside each badge, with one Tesseract call: [(text, conf)]."""
    crops = []
    for (x, y, r) in badges:
        k = max(4, int(r * 0.6))                  # inside the ring
        crop = img[max(0, int(y - k)):int(y + k), max(0, int(x - k)):int(x + k)]
        if crop.size == 0:
            crops.append(np.full((64, 64), 255, np.uint8))
            continue
        g = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 64), interpolation=cv2.INTER_CUBIC)
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if (b == 0).mean() > 0.5:
            b = 255 - b
        crops.append(cv2.copyMakeBorder(b, 0, 0, 40, 40, cv2.BORDER_CONSTANT, value=255))
    if not crops:
        return []
    return [(r['text'].replace(' ', ''), r['conf']) for r in read_strips(crops, '-c tessedit_char_whitelist=0123456789')]


def read_digit(img, x, y, r):
    """The number inside a badge."""
    k = int(r * 0.6)                    # inside the ring
    crop = img[max(0, int(y - k)):int(y + k), max(0, int(x - k)):int(x + k)]
    if crop.size == 0:
        return '', 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=100 / max(1, g.shape[0]), fy=100 / max(1, g.shape[0]), interpolation=cv2.INTER_CUBIC)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (b == 0).mean() > 0.5:
        b = 255 - b
    b = cv2.copyMakeBorder(b, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    data = pytesseract.image_to_data(b, config='--psm 10 -c tessedit_char_whitelist=0123456789',
                                     output_type=pytesseract.Output.DICT)
    best = ('', 0.0)
    for t, c in zip(data['text'], data['conf']):
        t = t.strip()
        if t and float(c) > best[1]:
            best = (t, float(c))
    return best

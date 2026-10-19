"""Things around the ribbons: numbered badges, an icon + name inside each loop, the title, other text."""
import cv2
import numpy as np


# label words → the closest of the app's 16 built-in icons (App/index.html ICONS)
ICON_WORDS = [
    (('data', 'dataset', 'collect', 'storage', 'cloud', 'ingest'), 'cloud'),
    (('database', 'db', 'sql', 'warehouse', 'store'), 'database'),
    (('ml', 'ai', 'model', 'brain', 'neural', 'learn', 'train'), 'network'),
    (('chip', 'gpu', 'hardware', 'compute'), 'chip'),
    (('dev', 'code', 'develop', 'build', 'software', 'engineering'), 'code'),
    (('ops', 'operate', 'operations', 'config', 'settings', 'process'), 'gears'),
    (('analytics', 'report', 'chart', 'metrics', 'kpi', 'finance', 'sales'), 'chart'),
    (('monitor', 'observe', 'dashboard', 'watch'), 'monitor'),
    (('deploy', 'launch', 'release', 'ship', 'go-live'), 'rocket'),
    (('security', 'secure', 'risk', 'safety', 'governance', 'compliance'), 'shield'),
    (('idea', 'ideate', 'innovate', 'insight', 'vision', 'strategy'), 'bulb'),
    (('people', 'team', 'users', 'customer', 'customers', 'hr', 'stakeholders'), 'users'),
    (('explore', 'search', 'research', 'discover', 'discovery'), 'search'),
    (('doc', 'docs', 'document', 'documentation', 'plan', 'requirements', 'legal'), 'doc'),
    (('test', 'qa', 'quality', 'validate', 'check', 'verify'), 'check'),
    (('cycle', 'loop', 'iterate', 'feedback', 'repeat'), 'loop'),
]


def icon_for(label):
    words = [w.strip(':.,').lower() for w in label.split()]
    for keys, icon in ICON_WORDS:
        if any(w in keys for w in words):
            return icon, 0.7
    return 'cloud', 0.3


def loop_interiors(solid, T):
    """Areas enclosed by ribbons (the inside of each loop)."""
    inv = (solid == 0).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(inv, 4)
    h, w = solid.shape
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h or area < T * T * 1.5:
            continue
        out.append({'mask': lab == i, 'box': [int(x), int(y), int(bw), int(bh)], 'c': (float(cents[i][0]), float(cents[i][1]))})
    return out


def inside(loop, x, y):
    xi, yi = int(x), int(y)
    m = loop['mask']
    return 0 <= yi < m.shape[0] and 0 <= xi < m.shape[1] and bool(m[yi, xi])


def ink_color(img, ink, box, strength=None):
    """Color of the drawn pixels in a box. strength (the ink map) keeps only the solid middle of
    strokes, not their blurred, paper-tinted edges."""
    x, y, w, h = box
    m = ink[y:y + h, x:x + w] > 0
    if strength is not None and m.sum() > 20:
        s = strength[y:y + h, x:x + w]
        m &= s >= np.percentile(s[m], 60)
    if m.sum() < 5:
        return None
    return np.median(img[y:y + h, x:x + w][m], axis=0)


def font_px(line):
    """Font size from a line's ink height: with descenders (g, p, y…) the ink spans about the whole
    font size; without them it is the capital height, about 0.72 of it."""
    desc = any(c in 'gjpqy' for c in line['text'])
    return line['box'][3] / (0.95 if desc else 0.72)


def hexc(bgr):
    return '#%02X%02X%02X' % tuple(int(np.clip(v, 0, 255)) for v in bgr[::-1])


def find(img, seg, lines, blobs, fix_color):
    """→ (circles, items, titles, texts) in work pixels, each with a conf."""
    T = seg['T']
    loops = loop_interiors(seg['solid'], T)
    used = set()

    circles = []
    for (x, y, r, digit, conf) in seg.get('badge_reads', []):
        if not digit or conf < 50:
            continue
        if any(inside(lp, x, y) for lp in loops):
            continue                     # an "o" in a loop's icon or name, not a badge
        h, w = seg['solid'].shape
        yy, xx = np.ogrid[:h, :w]
        ring = (np.abs(np.hypot(xx - x, yy - y) - r) < 2.5) & (seg['ink'] > 0.3)
        if ring.sum() > 20:
            ring &= seg['ink'] >= np.percentile(seg['ink'][ring], 60)
        stroke = np.median(img[ring], axis=0) if ring.sum() > 5 else np.array([51, 51, 51])
        circles.append({'cx': x, 'cy': y, 'r': r, 'text': digit, 'stroke': fix_color(stroke),
                        'strokeWidth': max(1.5, r * 0.09), 'fontSize': r * 0.95, 'conf': round(conf / 100, 2)})

    items = []
    for lp in loops:
        names = [i for i, ln in enumerate(lines) if inside(lp, ln['box'][0] + ln['box'][2] / 2, ln['box'][1] + ln['box'][3] / 2)]
        if not names:
            continue
        k = max(names, key=lambda i: lines[i]['height'])
        ln = lines[k]
        used.add(k)
        lx, ly, lw, lh = ln['box']
        # the icon: the biggest non-text ink blob in the loop, above the name
        icons = [b for b in blobs if inside(lp, b[0] + b[2] / 2, b[1] + b[3] / 2) and b[1] + b[3] / 2 < ly + lh / 2
                 and max(b[2], b[3]) > T * 0.4]
        name_color = fix_color(ink_color(img, seg['ink_all'], ln['box'], seg['ink']) if ink_color(img, seg['ink_all'], ln['box'], seg['ink']) is not None
                               else np.array([40, 35, 31]))
        icon, iconf = icon_for(ln['text'])
        if icons:
            b = max(icons, key=lambda b: b[2] * b[3])
            ic = ink_color(img, seg['ink_all'], b, seg['ink'])
            size = float(max(b[2], b[3]))
            gap = max(2.0, ly - (b[1] + b[3]))
            top = b[1]
            icon_color = fix_color(ic) if ic is not None else name_color
        else:
            size, gap, top, icon, icon_color = 0.0, 0.0, ly, '', name_color
        font = font_px(ln)
        H = (size + gap if size else 0) + font * 1.2
        cx = (lx + lw / 2) if not icons else (min(lx, b[0]) + max(lx + lw, b[0] + b[2])) / 2
        items.append({'x': cx, 'y': top + H / 2, 'icon': icon, 'iconSize': size or 56, 'iconColor': icon_color,
                      'text': ln['text'], 'textColor': name_color, 'fontSize': font, 'gap': gap,
                      'conf': round(min(ln['conf'] / 100, 0.5 + iconf * 0.5 if icons else 0.6), 2)})

    # title: the biggest line in the top part of the picture that is not inside a loop; subtitle under it
    titles, texts = [], []
    rest = [i for i in range(len(lines)) if i not in used]
    H = seg['solid'].shape[0]
    top = [i for i in rest if lines[i]['box'][1] < H * 0.35]
    if top:
        # the biggest line; when lines are about the same size, the one on top is the title
        big = max(font_px(lines[i]) for i in top)
        t = min((i for i in top if font_px(lines[i]) >= 0.85 * big), key=lambda i: lines[i]['box'][1])
        tl = lines[t]
        used.add(t)
        tx, ty, tw, th = tl['box']
        sub = [i for i in rest if i != t and 0 < lines[i]['box'][1] - (ty + th) < th * 1.6
               and abs(lines[i]['box'][0] - tx) < th * 1.5]
        title = {'x': tx, 'y': ty - tl['height'] * 0.15, 'title': tl['text'], 'subtitle': '',
                 'titleColor': fix_color(ink_color(img, seg['ink_all'], tl['box'], seg['ink'])), 'titleSize': font_px(tl),
                 'subtitleColor': '#4B5563', 'subtitleSize': 16, 'align': 'left', 'conf': round(tl['conf'] / 100, 2)}
        if sub:
            s = lines[sub[0]]
            used.add(sub[0])
            title.update(subtitle=s['text'], subtitleColor=fix_color(ink_color(img, seg['ink_all'], s['box'], seg['ink'])),
                         subtitleSize=font_px(s))
            title['conf'] = round(min(title['conf'], s['conf'] / 100), 2)
        titles.append(title)
    for i in rest:
        if i in used:
            continue
        ln = lines[i]
        if ln['conf'] < 70 or ln['height'] < T * 0.2:
            continue
        x, y, w, h = ln['box']
        texts.append({'x': x + w / 2, 'y': y + h / 2, 'text': ln['text'], 'textColor': fix_color(ink_color(img, seg['ink_all'], ln['box'], seg['ink'])),
                      'fontSize': font_px(ln), 'conf': round(ln['conf'] / 100, 2)})
    return circles, items, titles, texts

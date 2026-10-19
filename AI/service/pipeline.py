"""Picture → diagram plan (the /trace response). See INFO/FLOW-CHART-AI-PLAN.md §4 and §6.1.
All coordinates in the response are pixels of the straightened picture ('image'); the app maps them
onto its canvas."""
import time

import cv2
import numpy as np

import colors
import fit
import graph
import labels
import memory
import objects
import segment
import straighten

VERSION = '1.0.0'
PALETTE = ['#7B1E2B', '#5BA829', '#1F3F73', '#E0A526', '#8A8F99', '#2B8C8C', '#C2410C', '#6D28D9', '#1F2328', '#FFFFFF']


def hex_to_bgr(h):
    return np.array([int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)], float)


PAL_LAB = [(p, colors.lab_of(hex_to_bgr(p))) for p in PALETTE]


def color_fixer(flat):
    """Photos of screens come out grey: scale so the paper is white again, then snap to an app
    swatch when one is very close."""
    g = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    white = max(120.0, float(np.percentile(g, 99)))

    def fix(bgr):
        if bgr is None:
            return '#1F2328'
        c = np.clip(np.asarray(bgr, float) * 252 / white, 0, 255)
        lab = colors.lab_of(c)
        best = min(PAL_LAB, key=lambda p: np.linalg.norm(p[1] - lab))
        if np.linalg.norm(best[1] - lab) < 10:
            return best[0]
        return colors.to_hex(c)
    return fix


def text_color(fix, bgr, fill_bgr):
    """Labels on ribbons are nearly always white or black: a label much lighter than its ribbon
    is white, much darker is black (photos blur letters into the ribbon's color)."""
    lt, lf = colors.lab_of(bgr)[0], colors.lab_of(fill_bgr)[0]
    if lt > lf + 15:
        return #FFFFFF
    if lt < lf - 25:
        return #1F2328
    # too close to the ribbon to read: pick whichever of white / near-black stands out more
    return '#FFFFFF' if lf < 185 else '#1F2328'


def r1(v):
    return round(float(v), 1)


def chain_thickness(solid, pts, T):
    dt = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    h, w = solid.shape
    xs = np.clip(pts[:, 0].astype(int), 0, w - 1)
    ys = np.clip(pts[:, 1].astype(int), 0, h - 1)
    v = dt[ys, xs] * 2
    v = v[(v > T * 0.5) & (v < T * 1.5)]
    return float(np.median(v)) if v.size > 5 else T


def overlay_png(work, seg, chains, analysis):
    vis = (work * 0.45 + 140).astype(np.uint8)
    vis[seg['solid'] > 0] = (vis[seg['solid'] > 0] * 0.6 + np.array([200, 120, 255]) * 0.4).astype(np.uint8)
    vis[seg['text'] > 0] = (0, 200, 255)
    for ch, an in zip(chains, analysis):
        for pc in an['pieces']:
            q = ch['pts'][pc['ids']].astype(np.int32)
            cv2.polylines(vis, [q], False, tuple(int(v) for v in pc['bgr']), 6)
        for c in an['cuts']:
            x, y = ch['pts'][c['i']].astype(int)
            cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255) if c['kind'] == 'gap' else (255, 0, 0), 2)
    for (x, y, r) in seg['badges']:
        cv2.circle(vis, (int(x), int(y)), int(r), (0, 140, 0), 2)
    return straighten.encode_data_url(vis, 80)


def trace(req):
    t0 = time.time()
    tm = {}
    detect = req.get('detect') or {}
    want = lambda k: detect.get(k, True)
    detail = float(req.get('detail', 0.5))

    img = straighten.decode_data_url(req['image'])
    corners = req.get('corners')
    if not corners or len(corners) != 4:
        corners = straighten.suggest_corners(img)[0]
    work, H = straighten.warp(img, corners)
    tm['straighten'] = time.time() - t0

    # a picture you already corrected: give back your corrected diagram
    if req.get('recall', True):
        t = time.time()
        hit = memory.find(work)
        tm['recall'] = time.time() - t
        if hit:
            e, Hm, inliers, score = hit
            tm['total'] = time.time() - t0
            print(f"recalled {e['name']} ({inliers} matching features)")
            shapes = memory.transform(e['info'], Hm)
            ths = [x['thickness'] for x in shapes if x['type'] in ('path', 'arc', 'bezier')]
            return {
                'thickness': r1(np.median(ths)) if ths else 40.0,
                'flowtraceVersion': VERSION, 'model': 'memory',
                'image': {'src': straighten.encode_data_url(work), 'width': int(work.shape[1]), 'height': int(work.shape[0])},
                'corners': [[r1(x), r1(y)] for x, y in straighten.order_corners(corners)],
                'figures': [], 'bands': [], 'items': [], 'circles': [], 'titles': [], 'texts': [], 'warnings': [],
                'recall': {'source': e['name'], 'inliers': inliers, 'score': round(score, 2),
                           'background': e['info']['diagram'].get('canvas', {}).get('background', '#FFFFFF'),
                           'shapes': shapes},
                'timings': {k: round(v, 2) for k, v in tm.items()},
            }

    t = time.time()
    seg = segment.segment(work)
    T = seg['T']
    fix = color_fixer(seg['flat'])
    tm['segment'] = time.time() - t

    t = time.time()
    chains, nodes, _, _ = graph.centerlines(seg['solid'], T) if want('ribbons') else ([], {}, None, 1)
    analysis = colors.analyse(seg['flat'], seg, chains, nodes) if chains else []
    tm['graph'] = time.time() - t

    t = time.time()
    warnings = []
    ocr = labels.tesseract_ok()
    if not ocr and (want('labels') or want('icons') or want('badges') or want('title')):
        warnings.append({'kind': 'ocr', 'msg': 'Tesseract was not found, so no text was read. Install it to read labels.'})
    if ocr and want('labels') and chains:
        labels.ribbon_labels(work, chains, analysis, T)
    # badges: keep the circles that really hold a number
    reads = labels.read_digits(work, seg['badges']) if ocr and want('badges') else []
    seg['badge_reads'] = [(*b, d, c) for b, (d, c) in zip(seg['badges'], reads) if d and c >= 50]
    seg['badges'] = [b[:3] for b in seg['badge_reads']]
    lines, blobs = labels.free_text(seg['flat'], seg['ink_all'], seg['solid'], seg['badges'], T) if ocr else ([], [])
    circles, items, titles, texts = objects.find(seg['flat'], seg, lines, blobs, fix)
    tm['text'] = time.time() - t

    # over / under at crossings: the pass whose color fills the crossing's middle is on top
    passes = {}
    for ci, an in enumerate(analysis):
        for c in an['crossings']:
            passes.setdefault(c['node'], []).append((c['dist'], ci, c['i']))
    top_marks = []
    for node, ps in passes.items():
        if len(ps) < 2:
            continue
        ps.sort()
        if ps[1][0] - ps[0][0] < 12:
            xy = nodes[node]
            warnings.append({'kind': 'crossing', 'at': [r1(xy[0]), r1(xy[1])], 'msg': 'Could not tell which ribbon is on top'})
            continue
        top_marks.append((ps[0][1], ps[0][2], [p[1:] for p in ps[1:]]))

    figures, bands = [], []
    for ci, (ch, an) in enumerate(zip(chains, analysis)):
        pts = ch['pts']
        th = chain_thickness(seg['solid'], pts, T)
        has_gap = any(c['kind'] == 'gap' for c in an['cuts'])
        style = 'gap' if has_gap else 'chevron'
        alen = an['arrowLength'] if has_gap and an['arrowLength'] > 2 else th * (0.35 if has_gap else 0.8)
        pieces = []
        for pi, pc in enumerate(an['pieces']):
            layer = 0
            for tci, ti, _ in top_marks:
                if tci == ci and ti in set(pc['ids'].tolist()):
                    layer = 1
            mid = pts[pc['mid']]
            pieces.append({
                'at': [r1(mid[0]), r1(mid[1])], 'fill': fix(pc['bgr']),
                'text': pc.get('text', ''), 'textColor': text_color(fix, labels_color(pc), pc['bgr']),
                'fontSize': r1(pc['fontSize']) if pc.get('fontSize') else None,
                'layer': layer,
                'conf': {'fill': pc['conf'], 'text': pc.get('textConf', 1.0 if not pc.get('text') else 0.5)}
            })
        figures.append({
            'closed': bool(ch['closed']), 'thickness': r1(th),
            'points': [r1(v) for v in pts.reshape(-1)],
            'reverse': an['reverse'], 'arrowStyle': style,
            'arrowWidth': r1(th + 4 if has_gap else th * 1.6), 'arrowLength': r1(alen),
            'cuts': [{'at': [r1(pts[c['i']][0]), r1(pts[c['i']][1])], 'conf': 0.85 if c['kind'] == 'gap' else 0.6}
                     for c in an['cuts']],
            'pieces': pieces, 'z': sum(1 for tci, _, _ in top_marks if tci == ci),
            'conf': 0.8
        })
        # editable bands for "Stop after bands"
        tol = max(1.5, T * (0.03 + 0.1 * (1 - detail)))
        Ltot = colors.arclen(pts, ch['closed'])
        for b in fit.fit_chain(pts, ch['closed'], tol):
            if b['type'] == 'arc':
                a = np.radians((b['startAngle'] + b['endAngle']) / 2)
                mid = np.array([b['cx'] + b['r'] * np.cos(a), b['cy'] + b['r'] * np.sin(a)])
                band = {'type': 'arc', 'cx': r1(b['cx']), 'cy': r1(b['cy']), 'r': r1(b['r']),
                        'startAngle': r1(b['startAngle']), 'endAngle': r1(b['endAngle'])}
            else:
                mid = fit._bez(np.array([b['p0'], b['p1'], b['p2'], b['p3']]), np.array([0.5]))[0]
                band = {'type': 'bezier', **{k: [r1(b[k][0]), r1(b[k][1])] for k in ('p0', 'p1', 'p2', 'p3')}}
            i = int(np.argmin(np.linalg.norm(pts - mid, axis=1)))
            pc = next((p for p in an['pieces'] if i in set(p['ids'].tolist())), an['pieces'][0] if an['pieces'] else None)
            band.update(thickness=r1(th), fill=fix(pc['bgr']) if pc else '#5BA829', conf=0.7)
            bands.append(band)

    if not figures and want('ribbons'):
        warnings.append({'kind': 'ribbons', 'msg': 'No ribbons were found. Check the corners, or try a clearer picture.'})

    out = {
        'flowtraceVersion': VERSION, 'model': seg['backend'],
        'image': {'src': straighten.encode_data_url(work), 'width': int(work.shape[1]), 'height': int(work.shape[0])},
        'corners': [[r1(x), r1(y)] for x, y in straighten.order_corners(corners)],
        'thickness': r1(T),
        'figures': figures, 'bands': bands,
        'items': [rnd(i) for i in items] if want('icons') else [],
        'circles': [rnd(c) for c in circles] if want('badges') else [],
        'titles': [rnd(t) for t in titles] if want('title') else [],
        'texts': [rnd(t) for t in texts] if want('title') else [],
        'warnings': warnings,
    }
    if req.get('debug'):
        out['debug'] = {'overlay': overlay_png(work, seg, chains, analysis)}
    tm['total'] = time.time() - t0
    out['timings'] = {k: round(v, 2) for k, v in tm.items()}
    return out


def labels_color(pc):
    c = pc.get('textColor')
    return hex_to_bgr(c) if c else np.array([255, 255, 255])


def rnd(d):
    return {k: (r1(v) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}


def corners(req):
    img = straighten.decode_data_url(req['image'])
    c, conf = straighten.suggest_corners(img)
    return {'corners': c, 'conf': conf, 'width': int(img.shape[1]), 'height': int(img.shape[0])}

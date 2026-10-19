"""Make a training/test pair from a picture, its AI trace, and a correct "answer" diagram.

The trace fits the picture exactly but has some wrong labels and extra cuts; the answer has the right
labels, pieces, colors, icons and badges but was drawn separately (its own coordinates). This keeps
the trace's geometry and takes everything else from the answer, then writes a pair zip in the same
format as the app's "Save training pair".

    python answer_pair.py --picture ../../Sample-flow-diag.jpg ^
        --traced ../../JSON-FILES/flowchart-ai-generated-20261018-2135.json ^
        --answer ../../JSON-FILES/sample-json.json --out ../data/real/test/pair-sample-photo.zip

The two diagrams are lined up with the icon + name items they share (e.g. Data, ML, Dev, Ops):
at least 3 are needed. --corners is the corner quad used for tracing (default: the suggestion).
"""
import argparse
import json
import os
import sys
import time
import zipfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'service'))
import straighten   # noqa: E402


# ---------- snapped-figure geometry (same as the app's pathGeo / nearestOnPath / pieceRanges) ----------
def geo(s):
    p = np.array(s['points'], float).reshape(-1, 2)
    if s['closed']:
        p = np.vstack([p, p[:1]])
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    return p, cum


def at(s, t):
    p, cum = geo(s)
    d = t * cum[-1]
    return np.array([np.interp(d, cum, p[:, 0]), np.interp(d, cum, p[:, 1])])


def nearest_t(s, q):
    p, cum = geo(s)
    a, b = p[:-1], p[1:]
    ab = b - a
    u = np.clip(((q - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
    proj = a + ab * u[:, None]
    i = int(np.argmin(np.linalg.norm(proj - q, axis=1)))
    t = (cum[i] + (cum[i + 1] - cum[i]) * u[i]) / cum[-1]
    return 0.0 if s['closed'] and t >= 1 else float(t), float(np.linalg.norm(proj[i] - q))


def piece_ranges(s):
    c = s['cuts']
    if not s['closed']:
        b = [0] + c + [1]
        return [(b[i], b[i + 1]) for i in range(len(b) - 1)]
    if not c:
        return [(0, 1)]
    return [(c[i], c[i + 1] if i < len(c) - 1 else c[0] + 1) for i in range(len(c))]


def piece_at(s, t):
    for i, (a, b) in enumerate(piece_ranges(s)):
        if a <= t < b or a <= t + 1 < b:
            return i
    return 0


def tangent(s, t):
    L = geo(s)[1][-1]
    h = 3 / L
    return at(s, (t + h) % 1 if s['closed'] else min(1, t + h)) - at(s, (t - h) % 1 if s['closed'] else max(0, t - h))


# ---------- line the answer up with the trace ----------
def fit_affine(src, dst):
    A = np.hstack([np.array(src, float), np.ones((len(src), 1))])
    M, *_ = np.linalg.lstsq(A, np.array(dst, float), rcond=None)
    return lambda q: np.hstack([np.asarray(q, float), 1.0]) @ M


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--picture', required=True)
    ap.add_argument('--traced', required=True, help='the trace, saved with Save JSON (not corrected)')
    ap.add_argument('--answer', required=True, help='the correct diagram (any coordinates)')
    ap.add_argument('--corners', help='JSON list of 4 [x, y] picture corners used for tracing')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    traced = json.load(open(a.traced, encoding='utf-8'))
    answer = json.load(open(a.answer, encoding='utf-8'))
    pic_bytes = open(a.picture, 'rb').read()
    pic = cv2.imdecode(np.frombuffer(pic_bytes, np.uint8), cv2.IMREAD_COLOR)
    corners = json.loads(a.corners) if a.corners else straighten.suggest_corners(pic)[0]
    work, _ = straighten.warp(pic, corners)
    sh, sw = work.shape[:2]
    W, H = traced['canvas']['width'], traced['canvas']['height']
    k = min(W / sw, H / sh)
    shown = {'width': sw, 'height': sh, 'scale': k, 'offset': [(W - sw * k) / 2, (H - sh * k) / 2]}
    if abs(sh * k - H) > 2 and abs(sw * k - W) > 2:
        print('Warning: the canvas does not match these corners; pass the corners used for tracing with --corners')

    # answer → trace coordinates, from the shared icon items
    ti = {s['text'].strip().lower(): s for s in traced['shapes'] if s['type'] == 'item'}
    ai = {s['text'].strip().lower(): s for s in answer['shapes'] if s['type'] == 'item'}
    common = sorted(set(ti) & set(ai))
    if len(common) < 3:
        print('Need at least 3 icon items with the same names in both diagrams; found', common)
        return 1
    A2T = fit_affine([[ai[n]['x'], ai[n]['y']] for n in common], [[ti[n]['x'], ti[n]['y']] for n in common])
    err = max(np.linalg.norm(A2T([ai[n]['x'], ai[n]['y']]) - [ti[n]['x'], ti[n]['y']]) for n in common)
    scale = float(np.sqrt(abs(np.linalg.det(np.array([A2T([1, 0]) - A2T([0, 0]), A2T([0, 1]) - A2T([0, 0])])))))
    print(f'lined up with {common}; largest mismatch {err:.1f} px; answer is scaled x{scale:.2f}')
    o, ex, ey = A2T([0, 0]), A2T([1, 0]) - A2T([0, 0]), A2T([0, 1]) - A2T([0, 0])
    inv = np.linalg.inv(np.column_stack([ex, ey]))
    T2A = lambda q: inv @ (np.asarray(q, float) - o)          # trace → answer coordinates

    out = json.loads(json.dumps(traced))
    t_paths = [s for s in out['shapes'] if s['type'] == 'path']
    a_paths = [s for s in answer['shapes'] if s['type'] == 'path']
    for tp in t_paths:
        th = tp['thickness']
        L = geo(tp)[1][-1]
        # which answer figure is this? the one whose center line passes closest to ours
        def fit_of(ap_):
            ts = np.linspace(0, 1, 40, endpoint=False)
            return np.mean([nearest_t(tp, A2T(at(ap_, t)))[1] for t in ts])
        apath = min(a_paths, key=fit_of)
        # Match the two center lines along their whole length: same order of pieces, best start
        # point and direction. (Matching point by point jumps to the wrong ribbon at crossings.)
        U = np.linspace(0, 1, 400, endpoint=False)
        target = np.array([A2T(at(apath, u)) for u in U])
        V = np.linspace(0, 1, 4000, endpoint=False)
        ours = np.array([at(tp, v) for v in V])
        best = (1e18, 0.0, 1)
        for direction in (1, -1):
            for shift in np.linspace(0, 1, 800, endpoint=False):
                idx = (((shift + direction * U) % 1) * len(V)).astype(int) % len(V)
                cost = float(np.mean(np.linalg.norm(ours[idx] - target, axis=1)))
                if cost < best[0]:
                    best = (cost, shift, direction)
        cost, shift, direction = best
        print(f'center lines lined up: average gap {cost:.1f} px ({cost / th:.2f} of the ribbon thickness)')

        def to_ours(u):                       # answer position → ours, refined locally (never jumps far)
            v0 = (shift + direction * u) % 1
            vs = (v0 + np.linspace(-0.04, 0.04, 81)) % 1
            q = A2T(at(apath, u))
            return float(vs[int(np.argmin([np.linalg.norm(at(tp, v) - q) for v in vs]))])

        to_answer = lambda v: (direction * (v - shift)) % 1
        want = [to_ours(c) for c in apath['cuts']]
        keep = []
        for c in tp['cuts']:                      # our cuts sit exactly on the picture's gaps: keep the matching ones
            d = [min(abs(c - w), 1 - abs(c - w)) * L for w in want]
            j = int(np.argmin(d))
            if d[j] < th * 1.2 and all(j != k for _, k in keep):
                keep.append((c, j))
        used = {j for _, j in keep}
        cuts = [c for c, _ in keep] + [w for j, w in enumerate(want) if j not in used]
        tp['cuts'] = sorted(round(c, 5) for c in cuts)
        # pieces: the answer's piece at the middle of each of ours
        new = []
        for (s0, s1) in piece_ranges(tp):
            src = apath['pieces'][piece_at(apath, to_answer(((s0 + s1) / 2) % 1))]
            dst = dict(src)
            dst['fontSize'] = round(src['fontSize'] * scale, 2)
            new.append(dst)
        tp['pieces'] = new
        tp['arrowStyle'] = apath['arrowStyle']
        tp['arrowWidth'] = round(apath['arrowWidth'] * scale, 2)
        tp['arrowLength'] = round(apath['arrowLength'] * scale, 2)
        tp['reverse'] = apath['reverse'] if direction > 0 else not apath['reverse']
        print(f"figure: {len(tp['cuts'])} cuts ({len(keep)} kept from the trace), labels: {', '.join(p['text'] for p in new if p['text'])}")

    # icons, badges, title / text: take the answer's, placed where the trace found them (or mapped)
    rest = [s for s in out['shapes'] if s['type'] not in ('item', 'circle', 'title', 'text')]
    z = max([s['z'] for s in rest] + [0]) + 1
    for s in answer['shapes']:
        if s['type'] not in ('item', 'circle', 'title', 'text'):
            continue
        c = dict(s)
        if s['type'] == 'item' and s['text'].strip().lower() in ti:
            m = ti[s['text'].strip().lower()]
            c.update(x=m['x'], y=m['y'], iconSize=m['iconSize'], fontSize=m['fontSize'], gap=m['gap'])
        elif s['type'] == 'circle':
            p = A2T([s['cx'], s['cy']])
            near = [m for m in traced['shapes'] if m['type'] == 'circle' and m.get('text') == s.get('text')]
            if near:
                p = [near[0]['cx'], near[0]['cy']]
                c['r'] = near[0]['r']
            else:
                c['r'] = s['r'] * scale
            c.update(cx=round(float(p[0]), 2), cy=round(float(p[1]), 2), fontSize=round(c['r'] * 0.9, 2), strokeWidth=s['strokeWidth'] * scale)
        else:
            tt = [m for m in traced['shapes'] if m['type'] == 'title']
            if s['type'] == 'title' and tt:
                c.update(x=tt[0]['x'], y=tt[0]['y'], titleSize=tt[0]['titleSize'], subtitleSize=tt[0]['subtitleSize'])
            elif s['type'] == 'text' and tt and '\n' in s['text'] and s['y'] < answer['canvas']['height'] * 0.3:
                # the answer writes title + subtitle as one text box; keep it as a title where the trace found it
                lines = s['text'].split('\n')
                m = tt[0]
                c = {'id': s['id'], 'type': 'title', 'x': m['x'], 'y': m['y'], 'title': lines[0], 'subtitle': '\n'.join(lines[1:]),
                     'titleColor': s['textColor'], 'subtitleColor': s['textColor'], 'titleSize': m['titleSize'],
                     'subtitleSize': m['subtitleSize'], 'bold': True, 'align': 'left', 'opacity': 1}
            else:
                p = A2T([s['x'], s['y']])
                c.update(x=round(float(p[0]), 2), y=round(float(p[1]), 2))
                if c['type'] == 'text':
                    c['fontSize'] = round(s['fontSize'] * scale, 2)
        c['z'] = z
        z += 1
        rest.append(c)
    out['shapes'] = rest

    info = {
        'app': 'FLOW-CHART-APP', 'kind': 'flowtrace-training-pair', 'version': 1, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'source': {'traced': os.path.basename(a.traced), 'answer': os.path.basename(a.answer)},
        'picture': {'file': 'picture' + os.path.splitext(a.picture)[1].lower(), 'name': os.path.basename(a.picture),
                    'width': int(pic.shape[1]), 'height': int(pic.shape[0])},
        'corners': [[float(x), float(y)] for x, y in corners],
        'shown': dict(shown, file='straightened.jpg'),
        'diagram': out,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with zipfile.ZipFile(a.out, 'w', zipfile.ZIP_DEFLATED) as z_:
        z_.writestr(info['picture']['file'], pic_bytes)
        z_.writestr('straightened.jpg', cv2.imencode('.jpg', work, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
        z_.writestr('pair.json', json.dumps(info, indent=1))
    with open(os.path.splitext(a.out)[0] + '-diagram.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('wrote', os.path.abspath(a.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())

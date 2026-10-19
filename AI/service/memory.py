"""Remembered answers: when a picture being traced is one you already corrected (a pair zip in
AI/data/real, including its test/ folder), return your corrected diagram instead of a fresh guess.

Pictures are matched with ORB features + a RANSAC homography, so a slightly different corner choice,
crop or re-photo of the same slide still matches. The homography also places the remembered
diagram exactly onto the new picture.
"""
import glob
import json
import os
import zipfile

import cv2
import numpy as np

REAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'real')
MIN_INLIERS = 40            # matched features that agree on one placement
MIN_RATIO = 0.3             # ...as a share of all good matches
_orb = cv2.ORB_create(nfeatures=3000)
_cache = {}                 # path → (mtime, entry)


def _features(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    k = 1000 / max(g.shape)                       # match at a fixed size: faster, and scale-independent
    g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else g
    k = min(k, 1.0)
    kp, des = _orb.detectAndCompute(g, None)
    pts = np.float32([p.pt for p in kp]) / k if kp else np.zeros((0, 2), np.float32)
    return pts, des


def _load(path):
    with zipfile.ZipFile(path) as z:
        info = json.loads(z.read('pair.json'))
        shown = cv2.imdecode(np.frombuffer(z.read(info['shown']['file']), np.uint8), cv2.IMREAD_COLOR)
    pts, des = _features(shown)
    return {'name': os.path.basename(path), 'info': info, 'pts': pts, 'des': des}


def entries():
    """All remembered pairs, reloaded when a zip is added or changed."""
    paths = glob.glob(os.path.join(REAL, '*.zip')) + glob.glob(os.path.join(REAL, '*', '*.zip'))
    out = []
    for p in sorted(paths):
        try:
            m = os.path.getmtime(p)
            if p not in _cache or _cache[p][0] != m:
                _cache[p] = (m, _load(p))
            out.append(_cache[p][1])
        except Exception as e:                     # a broken zip must not stop tracing
            print('memory: skipping', os.path.basename(p), '-', e)
    for p in list(_cache):
        if p not in paths:
            del _cache[p]
    return out


def find(work):
    """Best remembered pair for this straightened picture: (entry, H from its shown picture to work px, score)."""
    pts, des = _features(work)
    if des is None or len(pts) < MIN_INLIERS:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    best = None
    for e in entries():
        if e['des'] is None or len(e['pts']) < MIN_INLIERS:
            continue
        good = [m for m, n in (x for x in bf.knnMatch(e['des'], des, k=2) if len(x) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < MIN_INLIERS:
            continue
        src = e['pts'][[m.queryIdx for m in good]]
        dst = pts[[m.trainIdx for m in good]]
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
        if H is None:
            continue
        inl = int(mask.sum())
        if inl < MIN_INLIERS or inl / len(good) < MIN_RATIO:
            continue
        # the remembered picture must land as a sensible (convex, not tiny) shape on this one
        w, h = e['info']['shown']['width'], e['info']['shown']['height']
        quad = cv2.perspectiveTransform(np.float32([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]]), H).reshape(4, 2)
        area = cv2.contourArea(quad)
        if not cv2.isContourConvex(quad.astype(np.float32)) or area < 0.1 * work.shape[0] * work.shape[1]:
            continue
        score = inl / len(good)
        if best is None or inl > best[2]:
            best = (e, H, inl, score)
    return best


# ---------- move a diagram (app JSON) through a homography ----------
def _mapper(info, H):
    """canvas px of the remembered diagram → work px of the new picture, and the local size factor."""
    sh = info['shown']
    k, (ox, oy) = sh['scale'], sh['offset']
    A = np.array([[1 / k, 0, -ox / k], [0, 1 / k, -oy / k], [0, 0, 1]])      # canvas → shown px
    M = H @ A

    def pt(x, y):
        v = M @ np.array([x, y, 1.0])
        return [round(float(v[0] / v[2]), 2), round(float(v[1] / v[2]), 2)]

    def size(x, y):
        p0, px, py = np.array(pt(x, y)), np.array(pt(x + 1, y)), np.array(pt(x, y + 1))
        return float(np.sqrt(abs(np.cross(px - p0, py - p0))))
    return pt, size


def transform(info, H):
    """The remembered diagram's shapes, placed in work px (same coordinate system as a /trace reply)."""
    pt, size = _mapper(info, H)
    out = []
    for s in info['diagram']['shapes']:
        s = json.loads(json.dumps(s))
        t = s['type']
        if t in ('arc', 'circle'):
            k = size(s['cx'], s['cy'])
            s['cx'], s['cy'] = pt(s['cx'], s['cy'])
            s['r'] = round(s['r'] * k, 2)
            for f in ('thickness', 'strokeWidth', 'fontSize'):
                if f in s:
                    s[f] = round(s[f] * k, 2)
        elif t == 'bezier':
            k = size(*s['p0'])
            for f in ('p0', 'p1', 'p2', 'p3'):
                s[f] = pt(*s[f])
            s['thickness'] = round(s['thickness'] * k, 2)
            s['fontSize'] = round(s['fontSize'] * k, 2)
        elif t == 'path':
            p = np.array(s['points'], float).reshape(-1, 2)
            c = p.mean(0)
            k = size(*c)
            s['points'] = [v for q in p for v in pt(*q)]
            for f in ('thickness', 'arrowWidth', 'arrowLength'):
                s[f] = round(s[f] * k, 2)
            for pc in s['pieces']:
                pc['fontSize'] = round(pc['fontSize'] * k, 2)
        elif t in ('text', 'item', 'title', 'arrow'):
            k = size(s.get('x', 0), s.get('y', 0))
            if t != 'arrow' or not s.get('parentId'):
                s['x'], s['y'] = pt(s['x'], s['y'])
            for f in ('fontSize', 'iconSize', 'gap', 'titleSize', 'subtitleSize', 'width', 'length'):
                if f in s:
                    s[f] = round(s[f] * k, 2)
        out.append(s)
    return out

"""Find the ribbons: which pixels are ribbon, which are text on a ribbon, and how thick the ribbons are.

Two back ends give the same result dictionary:
  * "model":   the trained FlowTrace-Seg network (AI/models/flowtrace-seg.onnx), when it exists
  * "classic": hand-written image processing, used until a model has been trained
"""
import os

import cv2
import numpy as np

from straighten import flatten_light, ink_map

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'flowtrace-seg.onnx')
_session = None


def disc(d):
    d = max(1, int(round(d)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def fill_small_holes(mask, max_area):
    """Fill holes (background areas enclosed by the mask) smaller than max_area."""
    inv = (mask == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    out = mask.copy()
    h, w = mask.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area <= max_area and x > 0 and y > 0 and x + bw < w and y + bh < h:
            out[lab == i] = 1
    return out


def estimate_thickness(mask, near=None):
    """Typical ribbon thickness: twice the distance-to-edge found along the ribbons' middles.
    near: a previous estimate; only middles within ±40 % of it count (ignores text strokes and crossings)."""
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt > 2)
    vals = dt[ridge]
    if near:
        vals = vals[(vals > near * 0.3) & (vals < near * 0.7)]
        return float(2 * np.median(vals)) if vals.size >= 20 else float(near)
    if vals.size < 20:
        return 20.0
    hist, edges = np.histogram(vals, bins=60, weights=vals)       # long ribbons outweigh small strokes
    i = int(np.argmax(hist))
    return float(2 * (edges[i] + edges[i + 1]) / 2)


SEG_LONG = 900        # segment at this size (faster); masks are scaled back up


def classic(img):
    h, w = img.shape[:2]
    f = min(1.0, SEG_LONG / max(h, w))
    if f >= 1.0:
        return _classic(img)
    res = _classic(cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA))
    for k in ('raw', 'solid', 'text', 'cand', 'ink_all'):
        res[k] = (cv2.resize(res[k].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    res['T'] /= f
    res['flat'] = flatten_light(img)
    res['ink'] = ink_map(res['flat'])
    res['badges'] = find_badges(res['flat'], res['ink'], res['T'])     # full size: exact centers for reading numbers
    return res


def _classic(img):
    flat = flatten_light(img)
    ink = ink_map(flat)
    hsv = cv2.cvtColor(flat, cv2.COLOR_BGR2HSV)
    cand = (ink > 0.32).astype(np.uint8)

    # first guess of the thickness from everything colored; labels punch holes in the ribbons,
    # so close those before keeping only shapes that thick
    def bodies(T):
        filled = fill_small_holes(cv2.morphologyEx(cand, cv2.MORPH_CLOSE, disc(max(3, T * 0.08))), T * T * 2.5)
        b = cv2.morphologyEx(filled, cv2.MORPH_OPEN, disc(T * 0.42))
        return cv2.dilate(b, disc(T * 0.2)) & filled      # opening shaves corners: grow back
    # labels punch holes in sharp ribbons: fill small holes before measuring, or the slivers between
    # the letters look like thin ribbons
    big = (max(cand.shape) / 12) ** 2
    T = estimate_thickness(cv2.morphologyEx(fill_small_holes(cand, big), cv2.MORPH_OPEN, disc(5)))
    # numbered badges often touch a ribbon: find them first and take them out
    badges = find_badges(flat, ink, T)
    ink_all = cand.copy()            # everything drawn, for reading free text later
    for (x, y, r) in badges:
        cv2.circle(cand, (int(x), int(y)), int(r * 1.25 + 4), 0, -1)
    T = estimate_thickness(bodies(T), near=T)
    body = bodies(T)

    # drop compact blobs (icons, badges) and big solid rectangles (bars, panels): ribbons are long and thin
    n, lab, stats, _ = cv2.connectedComponentsWithStats(body, 8)
    keep = np.zeros(n, bool)
    H, W = body.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        length = area / T
        rect_fill = area / float(bw * bh)
        if length < 2.6 * T:
            continue
        if rect_fill > 0.85 and min(bw, bh) > 1.6 * T:
            continue
        # a straight bar along the top or bottom edge (a slide's title bar), not a ribbon
        if (y <= 2 or y + bh >= H - 2) and bw > 0.5 * W and bh < 2.2 * T and rect_fill > 0.6:
            continue
        keep[i] = True
    kept = keep[lab].astype(np.uint8)

    # solid ribbons: bridge the thin cuts between pieces and fill the holes left by labels
    solid = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, disc(T * 0.3))
    solid = fill_small_holes(solid, T * T * 2.5)
    # raw: only the ribbon's own colored pixels (no labels, cuts or outlines)
    raw = (solid & cand).astype(np.uint8)

    # text written on the ribbons: light, low-saturation pixels inside the solid ribbon
    light = (hsv[..., 2] > 170) & (hsv[..., 1] < 90)
    on_text = (solid > 0) & light & (cv2.erode(solid, disc(T * 0.15)) > 0)
    # thin bright lines across the ribbon (cuts, outlines at crossings) are not text
    return {
        'T': T, 'raw': raw, 'solid': solid, 'text': on_text.astype(np.uint8),
        'cand': cand, 'ink_all': ink_all, 'ink': ink, 'flat': flat, 'badges': badges, 'backend': 'classic'
    }


def find_badges(flat, ink, T):
    """Numbered badges: small circles (radius about ¼–⅔ of a ribbon's thickness), either outlined with a
    light middle, or filled with a color and standing alone on the paper."""
    gray = cv2.GaussianBlur(cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY), (5, 5), 1.2)
    rmin, rmax = max(6, int(T * 0.22)), max(10, int(T * 0.7))
    found = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=rmin * 2, param1=100, param2=16,
                             minRadius=rmin, maxRadius=rmax)
    out = []
    if found is None:
        return out
    h, w = ink.shape
    for x, y, r in found[0]:
        R = r * 1.3 + 4
        x0, x1, y0, y1 = int(max(0, x - R)), int(min(w, x + R + 1)), int(max(0, y - R)), int(min(h, y + R + 1))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(xx - x, yy - y)
        ink_c = ink[y0:y1, x0:x1] > 0.3
        # Hough radii are rough: try a few and keep the one whose ring is most drawn
        best = (0, r)
        for rr in np.linspace(r * 0.8, r * 1.25, 10):
            ring = ink_c[(d > rr - 2.5) & (d < rr + 2)]
            if ring.size:
                best = max(best, (ring.mean(), rr))
        cover, rr = best
        inner = ink_c[d < rr * 0.7]
        # an outline: most of the ring is drawn, the inside is mostly paper (only the number)
        ok = cover > 0.6 and inner.size and inner.mean() < 0.45
        if not ok:
            # a filled disc: find its edge (ink inside, paper just outside), and it must stand alone
            edge = max(((ink_c[(d > e - 4) & (d <= e)].mean() - ink_c[(d > e + 1) & (d <= e + 5)].mean(), e)
                        for e in np.linspace(r * 0.8, r * 1.2, 9) if ((d > e + 1) & (d <= e + 5)).any()), default=(0, r))
            rr = edge[1]
            body = ink_c[d < rr * 0.9]
            around = ink_c[(d > rr * 1.15) & (d < rr * 1.3 + 3)]
            ok = edge[0] > 0.6 and body.size and body.mean() > 0.6 and around.size and around.mean() < 0.25
        if ok and all(np.hypot(x - o[0], y - o[1]) > o[2] for o in out):
            out.append((float(x), float(y), float(rr)))
    return out


def model(img):
    """FlowTrace-Seg: an ONNX network with ribbon / center / cut / text heads (see INFO/FLOW-CHART-AI-PLAN.md §6.1)."""
    global _session
    import onnxruntime as ort
    if _session is None:
        _session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    shape = _session.get_inputs()[0].shape
    size = int(shape[2]) if isinstance(shape[2], int) else 384
    res = classic(img)                  # keeps flat / cand and the fallbacks for heads the model lacks
    h, w = img.shape[:2]
    x = cv2.resize(res['flat'], (size, size), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255
    x = np.transpose(x, (2, 0, 1))[None]
    outs = _session.run(None, {_session.get_inputs()[0].name: x})
    names = [o.name for o in _session.get_outputs()]
    heads = {n: cv2.resize(o[0, 0], (w, h), interpolation=cv2.INTER_LINEAR) for n, o in zip(names, outs)}
    if 'ribbon' in heads:
        raw = (heads['ribbon'] > 0.5).astype(np.uint8)
        solid = fill_small_holes(cv2.morphologyEx(raw, cv2.MORPH_CLOSE, disc(res['T'] * 0.3)), res['T'] ** 2 * 2.5)
        # a model that disagrees wildly with plain image processing is not trusted (e.g. undertrained)
        a_model, a_classic = float(solid.sum()), float(res['solid'].sum()) or 1.0
        T = estimate_thickness(solid, near=res['T'])
        if not (0.4 < a_model / a_classic < 2.5) or not (0.5 < T / res['T'] < 2):
            print(f'FlowTrace-Seg result looks wrong (area x{a_model / a_classic:.2f}, thickness x{T / res["T"]:.2f}); using classic')
            return res
        res['raw'] = raw & (res['text'] == 0) & (res['cand'] > 0)
        res['solid'], res['T'] = solid, T
    if 'text' in heads:
        res['text'] = ((heads['text'] > 0.5) & (res['solid'] > 0)).astype(np.uint8)
    res['heads'] = heads
    res['backend'] = 'model'
    return res


def segment(img):
    if os.path.exists(MODEL_PATH):
        try:
            return model(img)
        except Exception as e:          # a broken or old model must not stop tracing
            print('FlowTrace-Seg model failed, using classic segmentation:', e)
    return classic(img)

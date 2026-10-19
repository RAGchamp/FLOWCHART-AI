"""Corner step: suggest the diagram's corners and warp the picture to a straight rectangle."""
import base64

import cv2
import numpy as np

WORK_LONG = 1400      # the long side of the straightened picture, in pixels


def decode_data_url(url):
    """data:image/...;base64,... → BGR image."""
    b64 = url.split(',', 1)[1] if ',' in url else url
    buf = np.frombuffer(base64.b64decode(b64), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('The picture could not be decoded.')
    return img


def encode_data_url(img, quality=88):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError('Could not encode the straightened picture.')
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')


def order_corners(pts):
    """Any 4 points → top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def warp(img, corners):
    """Warp the quad to a rectangle whose long side is WORK_LONG. Returns (image, 3x3 homography)."""
    q = order_corners(corners)
    w = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
    h = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
    k = WORK_LONG / max(w, h)
    W, H = int(round(w * k)), int(round(h * k))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    out = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, M


def flatten_light(img):
    """Divide out uneven lighting (photos of screens): estimate the white background and normalise to it."""
    h, w = img.shape[:2]
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)   # the light changes slowly
    k = max(5, (max(small.shape[:2]) // 14) | 1)
    bg = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    out = img.astype(np.float32) / np.maximum(bg.astype(np.float32), 1) * 245
    return np.clip(out, 0, 255).astype(np.uint8)


def ink_map(img):
    """0..1: how much a pixel looks like drawn content (colored or dark) rather than white paper."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) / 255
    s, v = hsv[..., 1], hsv[..., 2]
    return np.clip(np.maximum(s * np.clip(v * 1.6, 0, 1), (1 - v) * 1.4), 0, 1)


def suggest_corners(img):
    """Guess where the diagram is: the box around the big colored shapes, with a margin.
    Long thin bars (title bars, rules) and huge rectangles are ignored."""
    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    flat = flatten_light(small)
    ink = (ink_map(flat) > 0.35).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    sh, sw = small.shape[:2]
    boxes, bars = [], []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < sw * sh * 0.002:
            continue
        rect_fill = area / float(bw * bh)
        if rect_fill > 0.8 and (bw > sw * 0.6 or bh > sh * 0.6):   # a header bar or panel
            bars.append((x, y, x + bw, y + bh))
            continue
        if bw > sw * 0.8 and bh < sh * 0.25:           # a full-width bar
            bars.append((x, y, x + bw, y + bh))
            continue
        boxes.append((x, y, x + bw, y + bh, area))
    if not boxes:
        return [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], 0.2
    # keep the cluster around the largest shapes
    boxes.sort(key=lambda b: -b[4])
    x0, y0, x1, y1, _ = boxes[0]
    total = sum(b[4] for b in boxes)
    for b in boxes[1:]:
        gx = max(0, max(b[0], x0) - min(b[2], x1))
        gy = max(0, max(b[1], y0) - min(b[3], y1))
        if max(gx, gy) < 0.08 * max(sw, sh):
            x0, y0, x1, y1 = min(x0, b[0]), min(y0, b[1]), max(x1, b[2]), max(y1, b[3])
    m = 0.06 * max(x1 - x0, y1 - y0)
    top, bottom = y0, y1
    x0, y0 = max(0, x0 - m), max(0, y0 - m)
    x1, y1 = min(sw - 1, x1 + m), min(sh - 1, y1 + m)
    # the margin must not reach into a title bar above or below the diagram
    for bx0, by0, bx1, by1 in bars:
        if by1 <= top and by1 > y0:
            y0 = by1 + 2
        if by0 >= bottom and by0 < y1:
            y1 = by0 - 2
    c = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    conf = min(0.9, sum(b[4] for b in boxes) / max(total, 1))
    return [[round(px / scale, 1), round(py / scale, 1)] for px, py in c], round(conf, 2)

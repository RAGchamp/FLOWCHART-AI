"""Turn a chain's center line into editable app bands: Arc where it follows a circle, else cubic Bézier."""
import numpy as np


def _bez(P, t):
    u = 1 - t
    return (u ** 3)[:, None] * P[0] + (3 * u * u * t)[:, None] * P[1] + (3 * u * t * t)[:, None] * P[2] + (t ** 3)[:, None] * P[3]


def _chord_t(pts):
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return d / (d[-1] or 1)


def fit_cubic(pts, t0, t1):
    """Least-squares cubic with fixed end points and end directions (Schneider)."""
    p0, p3 = pts[0], pts[-1]
    t = _chord_t(pts)
    u = 1 - t
    A1 = (3 * u * u * t)[:, None] * t0
    A2 = (3 * u * t * t)[:, None] * t1
    rhs = pts - ((u ** 3 + 3 * u * u * t)[:, None] * p0 + (3 * u * t * t + t ** 3)[:, None] * p3)
    C = np.array([[np.sum(A1 * A1), np.sum(A1 * A2)], [np.sum(A1 * A2), np.sum(A2 * A2)]])
    X = np.array([np.sum(A1 * rhs), np.sum(A2 * rhs)])
    L = np.linalg.norm(p3 - p0)
    try:
        a1, a2 = np.linalg.solve(C, X)
    except np.linalg.LinAlgError:
        a1 = a2 = L / 3
    if a1 < L * 1e-3 or a2 < L * 1e-3:
        a1 = a2 = L / 3
    P = np.array([p0, p0 + t0 * a1, p3 + t1 * a2, p3])
    err = np.linalg.norm(_bez(P, t) - pts, axis=1)
    return P, err


def _dir(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def fit_circle(pts):
    """Kåsa circle fit → (cx, cy, r, rms error)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = np.sqrt(max(sol[2] + cx * cx + cy * cy, 1e-9))
    err = np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2))
    return cx, cy, r, err


def arc_of(pts):
    """An Arc band if the points follow a circle closely, else None."""
    if len(pts) < 8:
        return None
    cx, cy, r, err = fit_circle(pts)
    if err > max(1.5, 0.025 * r) or r > 4000:
        return None
    ang = np.degrees(np.unwrap(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)))
    if np.any(np.diff(ang) * np.sign(ang[-1] - ang[0]) < -1):      # must sweep one way
        return None
    sweep = ang[-1] - ang[0]
    if abs(sweep) < 15 or abs(sweep) > 359:
        return None
    return {'type': 'arc', 'cx': cx, 'cy': cy, 'r': r, 'startAngle': ang[0], 'endAngle': ang[-1]}


def fit_chain(pts, closed, tol):
    """Split a center line into Arc / Bézier bands, each within tol pixels of it."""
    if closed:
        pts = np.vstack([pts, pts[:1]])
    out = []

    def tan_at(i, fwd):
        j = min(len(pts) - 1, i + 4) if fwd else max(0, i - 4)
        return _dir(pts[j] - pts[i]) if fwd else _dir(pts[j] - pts[i])

    def rec(a, b, t0, t1, depth=0):
        seg = pts[a:b + 1]
        arc = arc_of(seg)
        if arc and b - a > 12:
            out.append((a, b, arc))
            return
        P, err = fit_cubic(seg, t0, t1)
        if err.max() <= tol or b - a < 8 or depth > 12:
            out.append((a, b, {'type': 'bezier', 'p0': P[0], 'p1': P[1], 'p2': P[2], 'p3': P[3]}))
            return
        m = a + int(np.argmax(err))
        m = min(max(m, a + 4), b - 4)
        tm = _dir(pts[min(len(pts) - 1, m + 2)] - pts[max(0, m - 2)])
        rec(a, m, t0, -tm, depth + 1)
        rec(m, b, tm, t1, depth + 1)

    n = len(pts) - 1
    # first, long smooth runs as arcs where possible (rings are mostly arcs)
    rec(0, n, tan_at(0, True), tan_at(n, False))
    return [s for _, _, s in sorted(out, key=lambda o: o[0])]

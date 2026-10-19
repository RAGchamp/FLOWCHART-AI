"""Pieces along each chain: where the cuts are (thin white chevron gaps or color changes), each piece's
color, which way the arrows point, and which ribbon is on top at each crossing."""
import cv2
import numpy as np


def tangents(pts, closed):
    n = len(pts)
    idx = np.arange(n)
    a = pts[(idx + 2) % n] if closed else pts[np.minimum(idx + 2, n - 1)]
    b = pts[(idx - 2) % n] if closed else pts[np.maximum(idx - 2, 0)]
    d = a - b
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
    return d


def arclen(pts, closed):
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]) if closed else pts, axis=0), axis=1)
    return np.concatenate([[0], np.cumsum(seg)])


def to_hex(bgr):
    b, g, r = [int(np.clip(round(v), 0, 255)) for v in bgr]
    return '#%02X%02X%02X' % (r, g, b)


def lab_of(bgr):
    return cv2.cvtColor(np.uint8([[np.clip(bgr, 0, 255)]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)


def sample_colors(img, raw, pts, closed, T):
    """Median ribbon color (BGR) at each chain point, from ribbon pixels across the band (labels excluded)."""
    h, w = raw.shape
    tg = tangents(pts, closed)
    nr = np.column_stack([-tg[:, 1], tg[:, 0]])
    offs = np.linspace(-0.36 * T, 0.36 * T, 9)
    xs = np.clip(np.round(pts[:, None, 0] + nr[:, None, 0] * offs[None]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[:, None, 1] + nr[:, None, 1] * offs[None]).astype(int), 0, h - 1)
    cols = img[ys, xs].astype(float)                    # N × 9 × 3
    ok = raw[ys, xs] > 0
    out = np.full((len(pts), 3), np.nan)
    for i in range(len(pts)):
        if ok[i].sum() >= 2:
            out[i] = np.median(cols[i][ok[i]], axis=0)
    return out, tg, nr


def fill_nan(a, closed):
    out = a.copy()
    n = len(a)
    good = ~np.isnan(a[:, 0])
    if not good.any():
        return np.full_like(a, 128)
    idx = np.arange(n)
    for c in range(a.shape[1]):
        if closed:
            gi = np.concatenate([idx[good] - n, idx[good], idx[good] + n])
            gv = np.tile(a[good, c], 3)
        else:
            gi, gv = idx[good], a[good, c]
        out[:, c] = np.interp(idx, gi, gv)
    return out


def find_gap_cuts(solid, cand, text, chains, nodes, T):
    """Thin light lines that cut across a whole ribbon = the 'thin chevron cut' arrows between pieces.
    Returns {chain index: [{'i': point index, 'dir': ±1, 'len': arrow length px}]}."""
    light = ((solid > 0) & (cand == 0)).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(light, 8)
    allp = [(ci, ch) for ci, ch in enumerate(chains)]
    node_xy = np.array(list(nodes.values())) if nodes else np.zeros((0, 2))
    out = {}
    for k in range(1, n):
        area = stats[k, 4]
        if area < T * 0.5 or area > T * T * 0.9:
            continue
        cx, cy = cents[k]
        if len(node_xy) and np.min(np.linalg.norm(node_xy - [cx, cy], axis=1)) < 0.9 * T:
            continue                     # outlines around a crossing, not a cut
        ys, xs = np.nonzero(lab[stats[k, 1]:stats[k, 1] + stats[k, 3], stats[k, 0]:stats[k, 0] + stats[k, 2]] == k)
        P = np.column_stack([xs + stats[k, 0], ys + stats[k, 1]]).astype(float)
        best = None
        for ci, ch in allp:
            d = np.linalg.norm(ch['pts'] - [cx, cy], axis=1)
            i = int(np.argmin(d))
            if d[i] < 0.6 * T and (best is None or d[i] < best[2]):
                best = (ci, i, d[i])
        if best is None:
            continue
        ci, i, _ = best
        ch = chains[ci]
        tg = tangents(ch['pts'], ch['closed'])[i]
        nr = np.array([-tg[1], tg[0]])
        rel = P - ch['pts'][i]
        along, across = rel @ tg, rel @ nr
        if across.max() - across.min() < 0.55 * T:     # does not span the ribbon: a label, not a cut
            continue
        if along.max() - along.min() > 0.9 * T:        # too long along the ribbon
            continue
        arms = along[np.abs(across) > 0.28 * T]
        apex = along[np.abs(across) < 0.12 * T]
        if len(arms) < 3 or len(apex) < 2:
            continue
        lead = float(apex.mean() - arms.mean())
        L = arclen(ch['pts'], ch['closed'])
        s = L[i] + float(arms.mean())
        j = int(np.clip(np.searchsorted(L, s), 0, len(ch['pts']) - 1))
        out.setdefault(ci, []).append({'i': j, 'dir': 1 if lead > 0 else -1, 'len': abs(lead), 'kind': 'gap'})
    return out


def color_cuts(lab, closed, L, T, blocked):
    """Point indices where the ribbon color changes for good (ΔE jump), away from crossings and other cuts."""
    n = len(lab)
    w_px = 0.55 * T
    res = []
    score = np.zeros(n)
    for i in range(n):
        if blocked[i]:
            continue
        s = L[i]
        if closed:
            Ltot = L[-1]
            db = (s - L[:n]) % Ltot
            da = (L[:n] - s) % Ltot
            before = (db > 0.12 * T) & (db < w_px)
            after = (da > 0.12 * T) & (da < w_px)
        else:
            before = (L[:n] < s - 0.12 * T) & (L[:n] > s - w_px)
            after = (L[:n] > s + 0.12 * T) & (L[:n] < s + w_px)
        before &= ~blocked
        after &= ~blocked
        if before.sum() < 3 or after.sum() < 3:
            continue
        score[i] = np.linalg.norm(np.median(lab[before], 0) - np.median(lab[after], 0))
    thr = 22.0
    order = np.argsort(-score)
    for i in order:
        if score[i] < thr:
            break
        if any(abs(L[i] - L[j]) < T * 1.2 for j in res):
            continue
        res.append(int(i))
    return res, score


def analyse(img, seg, chains, nodes):
    T, raw, solid, cand, text = seg['T'], seg['raw'], seg['solid'], seg['cand'], seg['text']
    gaps = find_gap_cuts(solid, cand, text, chains, nodes, T)
    node_xy = {k: np.array(v) for k, v in nodes.items()}
    out = []
    for ci, ch in enumerate(chains):
        pts, closed = ch['pts'], ch['closed']
        L = arclen(pts, closed)
        bgr, tg, nr = sample_colors(img, raw, pts, closed, T)
        # near a crossing the ribbon on top hides this one: don't trust colors there
        blocked = np.zeros(len(pts), bool)
        for c in ch['crossings']:
            blocked |= np.linalg.norm(pts - np.array(c['xy']), axis=1) < 0.75 * T
        clean = bgr.copy()
        clean[blocked] = np.nan
        clean = fill_nan(clean, closed)
        labc = np.array([lab_of(c) for c in clean])

        cuts = [dict(g) for g in gaps.get(ci, [])]
        near_gap = np.zeros(len(pts), bool)
        for g in cuts:
            near_gap |= np.abs(L[:len(pts)] - L[g['i']]) < 0.7 * T
        ccuts, _ = color_cuts(labc, closed, L, T, blocked | near_gap)
        cuts += [{'i': i, 'dir': 0, 'len': 0, 'kind': 'color'} for i in ccuts]
        cuts.sort(key=lambda c: c['i'])

        # flow direction: the thin chevron cuts point the way the ribbon flows
        votes = sum(c['dir'] for c in cuts if c['kind'] == 'gap')
        reverse = votes < 0
        arrow_len = float(np.median([c['len'] for c in cuts if c['kind'] == 'gap'])) if any(c['kind'] == 'gap' for c in cuts) else 0

        # pieces between cuts, each with its median color
        idx = [c['i'] for c in cuts]
        if closed and idx:
            bounds = [(idx[k], idx[(k + 1) % len(idx)]) for k in range(len(idx))]
        else:
            b = [0] + idx + [len(pts) - 1]
            bounds = [(b[k], b[k + 1]) for k in range(len(b) - 1)]
        pieces = []
        for a, b in bounds:
            ids = np.arange(a, b + 1) if b > a else np.concatenate([np.arange(a, len(pts)), np.arange(0, b + 1)])
            if len(ids) < 2:
                continue
            use = ids[~blocked[ids]] if (~blocked[ids]).sum() > 3 else ids
            col = np.nanmedian(bgr[use], axis=0) if not np.all(np.isnan(bgr[use])) else clean[ids].mean(0)
            mid = ids[len(ids) // 2]
            spread = float(np.nanstd(bgr[use], axis=0).mean()) if len(use) > 1 else 50
            pieces.append({'ids': ids, 'mid': int(mid), 'bgr': col, 'fill': to_hex(col),
                           'conf': round(float(np.clip(1 - spread / 60, 0.3, 0.98)), 2)})

        # over / under: the color in the middle of the crossing belongs to the ribbon on top
        crossings = []
        for node in sorted({c['node'] for c in ch['crossings']}):
            xy = np.array(nodes[node]) if node in nodes else None
            if xy is None:
                continue
            x0, y0 = int(xy[0]), int(xy[1])
            r = max(2, int(T * 0.22))
            patch = img[max(0, y0 - r):y0 + r + 1, max(0, x0 - r):x0 + r + 1].reshape(-1, 3).astype(float)
            pm = raw[max(0, y0 - r):y0 + r + 1, max(0, x0 - r):x0 + r + 1].reshape(-1) > 0
            if pm.sum() < 3:
                continue
            centre = lab_of(np.median(patch[pm], 0))
            # the chain may pass this crossing twice (a figure crossing itself): look at each pass
            d = np.linalg.norm(pts - xy, axis=1)
            close = d < 0.5 * T
            runs, cur = [], []
            for k in range(len(pts)):
                if close[k]:
                    cur.append(k)
                elif cur:
                    runs.append(cur); cur = []
            if cur:
                runs.append(cur)
            if closed and len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == len(pts) - 1:
                runs[0] = runs.pop() + runs[0]
            for run in runs:
                i = run[int(np.argmin(d[run]))]
                near = np.abs(L[:len(pts)] - L[i])
                if closed:
                    near = np.minimum(near, L[-1] - near)
                ring = (near > 0.8 * T) & (near < 1.6 * T) & ~blocked
                if ring.sum() < 2:
                    continue
                own = np.median(labc[ring], 0)
                crossings.append({'node': node, 'i': int(i), 'dist': float(np.linalg.norm(centre - own))})
        out.append({'cuts': cuts, 'pieces': pieces, 'reverse': bool(reverse), 'arrowLength': arrow_len,
                    'crossings': crossings, 'L': L})
    return out

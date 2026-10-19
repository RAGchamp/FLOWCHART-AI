"""Center lines: thin the ribbon mask to one-pixel lines, build a graph of branches and crossings,
and walk it into continuous chains (one per ribbon), passing straight through crossings."""
import cv2
import numpy as np

NB = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def thin(mask):
    """Zhang–Suen thinning (vectorised)."""
    img = np.pad((mask > 0).astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            P = img
            p2, p3, p4, p5 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:], P[2:, 2:]
            p6, p7, p8, p9 = P[2:, 1:-1], P[2:, :-2], P[1:-1, :-2], P[:-2, :-2]
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            c = P[1:-1, 1:-1] == 1
            if step == 0:
                m = c & (B >= 2) & (B <= 6) & (A == 1) & (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                m = c & (B >= 2) & (B <= 6) & (A == 1) & (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if m.any():
                img[1:-1, 1:-1][m] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1]


def neighbours(sk, y, x):
    h, w = sk.shape
    return [(y + dy, x + dx) for dy, dx in NB if 0 <= y + dy < h and 0 <= x + dx < w and sk[y + dy, x + dx]]


def bfs(start, pset):
    """Distances and parents over an 8-connected pixel set."""
    from collections import deque
    par, dist, q = {start: None}, {start: 0}, deque([start])
    while q:
        cur = q.popleft()
        y, x = cur
        for dy, dx in NB:
            p = (y + dy, x + dx)
            if p in pset and p not in dist:
                dist[p] = dist[cur] + 1
                par[p] = cur
                q.append(p)
    return dist, par


def order_path(pixels, start, sk_set):
    """The longest path through a thin pixel set (ignores little side twigs): farthest pixel from start,
    then the farthest from that one, and the shortest route between them."""
    d0, _ = bfs(start, sk_set)
    u = max(d0, key=d0.get)
    d1, par = bfs(u, sk_set)
    v = max(d1, key=d1.get)
    path = [v]
    while par[path[-1]] is not None:
        path.append(par[path[-1]])
    return path


def build_graph(sk, T):
    """Nodes = crossings / joins (clusters of branch pixels); edges = pixel paths between them."""
    k = np.ones((3, 3), np.uint8)
    # a branch point has 3+ separate runs of neighbours around it (a staircase corner has 3 neighbours
    # but only 2 runs, so counting neighbours alone would mark every diagonal step)
    P = np.pad(sk, 1)
    ring = [P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:], P[2:, 2:], P[2:, 1:-1], P[2:, :-2], P[1:-1, :-2], P[:-2, :-2]]
    runs = sum(((ring[i] == 0) & (ring[(i + 1) % 8] == 1)).astype(np.uint8) for i in range(8))
    junction = (sk > 0) & (runs >= 3)
    # one node per cluster of touching branch pixels; nearby nodes are merged later (contract)
    nodes_mask = cv2.dilate(junction.astype(np.uint8), k)
    n_nodes, node_lab = cv2.connectedComponents(nodes_mask, connectivity=8)
    node_center = {}
    for i in range(1, n_nodes):
        ys, xs = np.nonzero((node_lab == i) & (sk > 0))
        if len(ys) == 0:
            ys, xs = np.nonzero(node_lab == i)
        node_center[i] = (float(xs.mean()), float(ys.mean()))

    rest = ((sk > 0) & (nodes_mask == 0)).astype(np.uint8)
    n_e, e_lab = cv2.connectedComponents(rest, connectivity=8)
    edges = []
    big = cv2.dilate(nodes_mask, k)
    for i in range(1, n_e):
        ys, xs = np.nonzero(e_lab == i)
        px = list(zip(ys.tolist(), xs.tolist()))
        pset = set(px)
        ends = [p for p in px if sum(1 for q in neighbours(rest, *p) if q in pset) <= 1]
        start = ends[0] if ends else px[0]
        path = order_path(px, start, pset)
        if len(path) < 2:
            continue
        # which nodes this branch touches, and which of its two ends touches each
        comp = (e_lab == i).astype(np.uint8)
        touch = node_lab[(cv2.dilate(comp, k, iterations=2) > 0) & (node_lab > 0)]
        near = {}
        for nid in set(touch.tolist()):
            ny, nx = np.nonzero(node_lab == nid)
            d0 = np.min(np.hypot(ny - path[0][0], nx - path[0][1]))
            d1 = np.min(np.hypot(ny - path[-1][0], nx - path[-1][1]))
            near[nid] = (d0, d1)
        a = min(near, key=lambda n: near[n][0]) if near else None
        b = min(near, key=lambda n: near[n][1]) if near else None
        if a is not None and near[a][0] > 4:
            a = None
        if b is not None and near[b][1] > 4:
            b = None
        if a is not None and a == b and len(near) == 1 and min(near[a]) == near[a][0] and near[a][1] > 4:
            b = None
        closed = not near and not ends and len(path) > 8
        edges.append({'pts': [(x, y) for y, x in path], 'a': a, 'b': b, 'closed': closed})
    return node_center, edges


def edge_dir(e, side, span):
    """Where a branch leaves a node: an anchor point just outside the crossing (the pixels right at
    it are bent by thinning) and the unit direction away from the node there."""
    pts = np.array(e['pts'] if side == 'a' else e['pts'][::-1], float)
    n = len(pts) - 1
    i0, i1 = min(n, span // 2), min(n, span + span // 2)
    if i1 - i0 < 2:
        i0, i1 = 0, n
    d = pts[i1] - pts[i0]
    L = np.linalg.norm(d)
    return pts[i0], (d / L if L > 1e-6 else np.array([1.0, 0.0]))


def prune(nodes, edges, T):
    """Remove short spurs: branches that stick out from a crossing and end after less than a thickness."""
    changed = True
    while changed:
        changed = False
        deg = {}
        for e in edges:
            for s in ('a', 'b'):
                if e[s]:
                    deg[e[s]] = deg.get(e[s], 0) + 1
        for e in list(edges):
            free = [s for s in ('a', 'b') if e[s] is None]
            if e['closed'] or len(free) != 1:
                continue
            other = e['b'] if free[0] == 'a' else e['a']
            if len(e['pts']) < T * 0.9 and deg.get(other, 0) >= 3:
                edges.remove(e)
                changed = True
                break
    # tiny isolated bits
    return [e for e in edges if not (e['a'] is None and e['b'] is None and not e['closed'] and len(e['pts']) < T * 1.5)]


def contract(nodes, edges, T, dt=None):
    """Thinning turns an X crossing into two Y's joined by a stub: merge them into one node.
    The stub is short, or, where two ribbons cross at a shallow angle, longer but clearly wider
    than one ribbon (both ribbons overlap along it)."""
    def overlap(e):
        if dt is None or len(e['pts']) > T * 4:
            return False
        w = np.median([2 * dt[y, x] for x, y in e['pts']])
        return w > 1.3 * T
    changed = True
    while changed:
        changed = False
        for e in edges:
            a, b = e['a'], e['b']
            if a is None or b is None or a == b or (len(e['pts']) > T * 1.7 and not overlap(e)):
                continue
            edges.remove(e)
            for o in edges:
                if o['a'] == b:
                    o['a'] = a
                if o['b'] == b:
                    o['b'] = a
            (xa, ya), (xb, yb) = nodes[a], nodes[b]
            nodes[a] = ((xa + xb) / 2, (ya + yb) / 2)
            del nodes[b]
            changed = True
            break
    return nodes, edges


def pair_ends(nodes, edges, T):
    """At each node, pair the branches that continue each other most straightly: opposite
    directions, and each one's line passing through the other's anchor (no sideways jump)."""
    from itertools import combinations
    at = {}
    for ei, e in enumerate(edges):
        for s in ('a', 'b'):
            if e[s] is not None:
                at.setdefault(e[s], []).append((ei, s))
    link = {}
    span = max(3, int(T * 1.0))

    def cost(A, B):
        (pa, da), (pb, db) = A, B
        lat = (abs(np.cross(da, pb - pa)) + abs(np.cross(db, pa - pb))) / 2
        return float(np.dot(da, db)) + 0.6 * lat / T

    for nid, ends in at.items():
        info = {end: edge_dir(edges[end[0]], end[1], span) for end in ends}
        if len(ends) == 2:
            pairs = [tuple(ends)]
        elif len(ends) == 4:
            # try all three ways of making two pairs; keep the straightest overall
            a, b, c, d = ends
            options = [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]
            pairs = list(min(options, key=lambda o: sum(cost(info[x], info[y]) for x, y in o)))
        else:
            cands = sorted(((cost(info[x], info[y]), x, y) for x, y in combinations(ends, 2)), key=lambda c: c[0])
            pairs, used = [], set()
            for c, x, y in cands:
                if x in used or y in used or c > -0.35:       # would turn back on itself: leave as ends
                    continue
                pairs.append((x, y))
                used.update((x, y))
        for x, y in pairs:
            link[x] = y
            link[y] = x
    return link


def hermite(p0, d0, p1, d1, n):
    """Points of a smooth curve from p0 (heading d0) to p1 (heading d1), interior only."""
    L = np.linalg.norm(p1 - p0)
    m0, m1 = d0 * L, d1 * L
    out = []
    for i in range(1, n):
        t = i / n
        h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
        out.append(h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1)
    return out


def walk_chains(nodes, edges, link, T):
    """Follow edges through paired nodes into chains. Returns [{'pts': Nx2 array, 'closed': bool, 'crossings': [...]}]."""
    visited = set()
    trim = max(2, int(T * 0.35))
    chains = []

    def oriented(ei, entry):
        pts = edges[ei]['pts']
        return pts if entry == 'a' else pts[::-1]

    def run(ei, entry):
        seq, cross = [], []
        start = (ei, entry)
        cur, ent = ei, entry
        closed = False
        while True:
            visited.add(cur)
            seq.append((cur, ent))
            exit_side = 'b' if ent == 'a' else 'a'
            nxt = link.get((cur, exit_side))
            if nxt is None:
                break
            if nxt[0] in visited:
                closed = nxt == start
                break
            cur, ent = nxt
        # stitch: drop the pixels nearest each node and bridge with a smooth curve
        pts = []
        for k, (e, ent) in enumerate(seq):
            p = np.array(oriented(e, ent), float)
            a_node = edges[e]['a' if ent == 'a' else 'b']
            b_node = edges[e]['b' if ent == 'a' else 'a']
            lo = trim if (k > 0 or closed) and a_node is not None else 0
            hi = len(p) - (trim if (k < len(seq) - 1 or closed) and b_node is not None else 0)
            if hi - lo < 2:
                lo, hi = 0, len(p)
            seg = p[lo:hi]
            if pts:
                prev = pts[-1]
                prev_dir = prev[-1] - prev[max(0, len(prev) - 6)]
                next_dir = seg[min(len(seg) - 1, 5)] - seg[0]
                nd = lambda v: v / (np.linalg.norm(v) or 1)
                gap = np.linalg.norm(seg[0] - prev[-1])
                node = edges[e]['a' if ent == 'a' else 'b']
                cross.append({'node': node, 'at': len(np.concatenate(pts)) if pts else 0})
                pts.append(np.array(hermite(prev[-1], nd(prev_dir), seg[0], nd(next_dir), max(2, int(gap / 2)))).reshape(-1, 2))
            pts.append(seg)
        if closed and len(pts) > 1:
            first, last = pts[0], pts[-1]
            d0 = last[-1] - last[max(0, len(last) - 6)]
            d1 = first[min(len(first) - 1, 5)] - first[0]
            nd = lambda v: v / (np.linalg.norm(v) or 1)
            gap = np.linalg.norm(first[0] - last[-1])
            pts.append(np.array(hermite(last[-1], nd(d0), first[0], nd(d1), max(2, int(gap / 2)))).reshape(-1, 2))
        allp = np.concatenate([q for q in pts if len(q)])
        return {'pts': allp, 'closed': closed, 'crossings': cross}

    # open chains first: start at an end that is free or unpaired
    for ei, e in enumerate(edges):
        if ei in visited or e['closed']:
            continue
        for s in ('a', 'b'):
            if (ei, s) not in link:
                chains.append(run(ei, s))
                break
    for ei, e in enumerate(edges):
        if ei in visited:
            continue
        if e['closed']:
            visited.add(ei)
            chains.append({'pts': np.array(e['pts'], float), 'closed': True, 'crossings': []})
        else:
            chains.append(run(ei, 'a'))
    return chains


def extend_ends(pts, solid, T, scale):
    """Thinning stops about half a thickness short of a ribbon's end: extend along the end direction."""
    h, w = solid.shape
    out = pts
    for side in (0, 1):
        p = out if side == 1 else out[::-1]
        d = p[-1] - p[max(0, len(p) - 6)]
        n = np.linalg.norm(d)
        if n < 1e-6:
            continue
        d = d / n
        extra = []
        q = p[-1].copy()
        for _ in range(int(T / scale * 0.7)):
            q = q + d
            x, y = int(round(q[0] * scale)), int(round(q[1] * scale))
            if not (0 <= x < w and 0 <= y < h) or not solid[y, x]:
                break
            extra.append(q.copy())
        if extra:
            p = np.vstack([p, np.array(extra)])
        out = p if side == 1 else p[::-1]
    return out


def smooth(pts, win, closed):
    if len(pts) < 5 or win < 2:
        return pts
    k = np.ones(win) / win
    if closed:
        ext = np.vstack([pts[-win:], pts, pts[:win]])
        sm = np.column_stack([np.convolve(ext[:, i], k, 'same') for i in range(2)])
        return sm[win:-win]
    pad = win // 2
    ext = np.vstack([np.repeat(pts[:1], pad, 0), pts, np.repeat(pts[-1:], pad, 0)])
    sm = np.column_stack([np.convolve(ext[:, i], k, 'same') for i in range(2)])
    sm = sm[pad:pad + len(pts)]
    sm[0], sm[-1] = pts[0], pts[-1]
    return sm


def resample(pts, step, closed):
    src = np.vstack([pts, pts[:1]]) if closed else pts
    seg = np.linalg.norm(np.diff(src, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    L = cum[-1]
    if L < step:
        return pts
    n = max(2, int(L / step))
    t = np.linspace(0, L, n + (0 if closed else 1), endpoint=not closed)
    return np.column_stack([np.interp(t, cum, src[:, i]) for i in range(2)])


def centerlines(solid, T):
    """solid ribbon mask (work resolution) → list of chains with points in work pixels."""
    scale = min(1.0, 22.0 / T)              # thin at a size where ribbons are ~22 px thick (fast)
    small = cv2.resize(solid, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # smooth the outline first: every bump on it would become a spur of the center line
    small = (cv2.GaussianBlur(small.astype(np.float32), (0, 0), 1.5) > 0.5).astype(np.uint8)
    Ts = T * scale
    sk = thin(small)
    nodes, edges = build_graph(sk, Ts)
    edges = prune(nodes, edges, Ts)
    nodes, edges = contract(nodes, edges, Ts, cv2.distanceTransform(small, cv2.DIST_L2, 5))
    edges = prune(nodes, edges, Ts)
    link = pair_ends(nodes, edges, Ts)
    chains = walk_chains(nodes, edges, link, Ts)
    # a real crossing has two ribbons passing through it (4+ branches); the rest are bends,
    # small spurs left by cuts, or a ribbon end touching another
    deg = {}
    for e in edges:
        for s in ('a', 'b'):
            if e[s] is not None:
                deg[e[s]] = deg.get(e[s], 0) + 1
    nodes = {k: v for k, v in nodes.items() if deg.get(k, 0) >= 4}
    out = []
    for ch in chains:
        p = ch['pts']
        if not ch['closed']:
            p = extend_ends(p, solid, T, scale)
        p = smooth(p, max(3, int(Ts * 1.1)), ch["closed"]) / scale
        L = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        if L < T * 1.5:
            continue
        cross = [{'node': c['node'], 'xy': (nodes[c['node']][0] / scale, nodes[c['node']][1] / scale)}
                 for c in ch['crossings'] if c['node'] in nodes]
        out.append({'pts': resample(p, 4.0, ch['closed']), 'closed': ch['closed'], 'crossings': cross, 'length': L})
    node_xy = {k: (v[0] / scale, v[1] / scale) for k, v in nodes.items()}
    return out, node_xy, sk, scale

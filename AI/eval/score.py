"""Score FlowTrace on the frozen real test set (INFO/FLOW-CHART-AI-PLAN.md §9).

    python score.py                         # AI/data/real/test/*.zip → table + AI/eval/report.html
    python score.py --recall                # the same, but remembered answers allowed (should be all 1.00)
    python score.py --pairs some/folder     # any folder of "Save training pair" zips

For each pair it traces the original picture with the same corners, rebuilds the result in the app,
and compares it with the corrected diagram:
    ribbon IoU     overlap of traced and true ribbon area
    center dist    average distance from traced center lines to the true ones, in ribbon thicknesses
    labels P / R   piece labels read exactly right (precision / recall)
    badges         badge numbers found
    title          title text right (yes / no)
    seconds        tracing time
"""
import argparse
import base64
import glob
import html
import os
import sys
import time

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'service'))
sys.path.insert(0, os.path.join(HERE, '..', 'datagen'))
import pipeline                          # noqa: E402
from apppage import mask_for, open_app   # noqa: E402
from real_to_masks import load_pair      # noqa: E402

APPLY_JS = """(res) => {
  FlowAPI.loadDoc({ canvas: { width: 1600, height: 900, background: '#FFFFFF' }, shapes: [] });
  FlowAPI.applyTrace(res, { mode: 'replace' });
  const R = traceRect(res.image.width, res.image.height);
  return { doc: FlowAPI.getDoc(), area: { x: R.x, y: R.y, w: res.image.width * R.k, h: res.image.height * R.k } };
}"""


def b64img(img, q=80):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()


def labels(doc):
    out = []
    for s in doc['shapes']:
        if s['type'] == 'path':
            out += [p['text'].strip().upper() for p in s['pieces'] if p.get('text', '').strip()]
        elif s['type'] in ('arc', 'bezier') and s.get('text', '').strip():
            out.append(s['text'].strip().upper())
    return out


def multiset_match(pred, truth):
    t = list(truth)
    hit = 0
    for p in pred:
        if p in t:
            t.remove(p)
            hit += 1
    return hit


def center_dist(pred_c, true_c, T):
    if not true_c.any() or not pred_c.any():
        return float('nan')
    dt = cv2.distanceTransform((~true_c).astype(np.uint8), cv2.DIST_L2, 5)
    return float(dt[pred_c].mean() / max(T, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pairs', default=os.path.join(HERE, '..', 'data', 'real', 'test'))
    ap.add_argument('--report', default=os.path.join(HERE, 'report.html'))
    ap.add_argument('--recall', action='store_true',
                    help='allow remembered answers (by default the AI is tested on its own, or every test pair would be recalled)')
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.pairs, '*.zip')))
    if not paths:
        print('No test pairs in', os.path.abspath(a.pairs))
        return 1
    rows, cards = [], []
    with sync_playwright() as p:
        browser, page = open_app(p)
        for path in paths:
            info, pic, shown = load_pair(path)
            ok, buf = cv2.imencode('.png', pic)
            t = time.time()
            res = pipeline.trace({'image': 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode(),
                                  'corners': info['corners'], 'recall': a.recall})
            secs = time.time() - t
            W, H = res['image']['width'], res['image']['height']
            built = page.evaluate(APPLY_JS, res)
            sh = info['shown']
            true_area = {'x': sh['offset'][0], 'y': sh['offset'][1], 'w': sh['width'] * sh['scale'], 'h': sh['height'] * sh['scale']}
            m_true = mask_for(page, info['diagram'], true_area, (W, H))
            m_pred = mask_for(page, built['doc'], built['area'], (W, H))
            rt, rp = m_true[..., 2] > 0, m_pred[..., 2] > 0
            iou = float((rt & rp).sum() / max((rt | rp).sum(), 1))
            cd = center_dist(m_pred[..., 0] > 0, m_true[..., 0] > 0, res['thickness'])
            lt, lp = labels(info['diagram']), labels(built['doc'])
            hit = multiset_match(lp, lt)
            bt = sorted(str(s.get('text', '')) for s in info['diagram']['shapes'] if s['type'] == 'circle')
            bp = sorted(str(s.get('text', '')) for s in built['doc']['shapes'] if s['type'] == 'circle')
            norm = lambda t: ''.join(ch for ch in t.lower() if ch.isalnum())       # ignore spaces, capitals, punctuation
            # the title: a Title shape, or else the first line of a text box near the top
            def title_of(doc):
                top = doc['canvas']['height'] * 0.3
                return [norm(s.get('title', '')) for s in doc['shapes'] if s['type'] == 'title'] or \
                       [norm(s['text'].split('\n')[0]) for s in doc['shapes'] if s['type'] == 'text' and s['y'] < top]
            tt, tp = title_of(info['diagram']), title_of(built['doc'])
            row = {'pair': os.path.basename(path), 'iou': iou, 'center': cd,
                   'P': hit / max(len(lp), 1), 'R': hit / max(len(lt), 1),
                   'badges': f'{multiset_match(bp, bt)}/{len(bt)}', 'title': bool(tt) and tt[:1] == tp[:1], 'sec': secs}
            rows.append(row)
            # side by side: picture | traced ribbons over it | true ribbons over it
            straight = cv2.imdecode(np.frombuffer(base64.b64decode(res['image']['src'].split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
            def over(mask):
                v = straight.copy()
                v[mask[..., 2] > 0] = (v[mask[..., 2] > 0] * 0.4 + np.array([60, 60, 230]) * 0.6).astype(np.uint8)
                v[mask[..., 0] > 0] = (255, 255, 0)
                return v
            cards.append((row, b64img(straight), b64img(over(m_pred)), b64img(over(m_true)), lp, lt))
        browser.close()

    fmt = lambda v: '-' if v != v else f'{v:.2f}'
    print(f"{'pair':34} {'ribbonIoU':>9} {'center':>7} {'labelP':>7} {'labelR':>7} {'badges':>7} {'title':>6} {'sec':>5}")
    for r in rows:
        print(f"{r['pair'][:34]:34} {fmt(r['iou']):>9} {fmt(r['center']):>7} {fmt(r['P']):>7} {fmt(r['R']):>7} {r['badges']:>7} {('yes' if r['title'] else 'no'):>6} {r['sec']:>5.1f}")
    mean = lambda k: float(np.nanmean([r[k] for r in rows]))
    print(f"{'MEAN':34} {fmt(mean('iou')):>9} {fmt(mean('center')):>7} {fmt(mean('P')):>7} {fmt(mean('R')):>7} {'':>7} {'':>6} {mean('sec'):>5.1f}")
    print('targets (plan section 9): ribbon IoU >= 0.85, center <= 0.25, labels about 0.9, <= 20 s')

    with open(a.report, 'w', encoding='utf-8') as f:
        f.write('<!doctype html><meta charset="utf-8"><title>FlowTrace scores</title>'
                '<style>body{font-family:system-ui,sans-serif;margin:16px;background:#f6f7f9;color:#1f2328}'
                '.card{background:#fff;border:1px solid #d7dae0;border-radius:8px;padding:12px;margin:12px 0}'
                '.imgs{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.imgs img{width:100%}'
                'td,th{padding:3px 8px;text-align:right}th:first-child,td:first-child{text-align:left}</style>'
                f'<h1>FlowTrace scores</h1><p>{len(rows)} test pairs · model: {html.escape(pipeline.segment.MODEL_PATH if os.path.exists(pipeline.segment.MODEL_PATH) else "classic")}</p>'
                '<table><tr><th>pair</th><th>ribbon IoU</th><th>center</th><th>labels P</th><th>labels R</th><th>badges</th><th>title</th><th>s</th></tr>')
        for r in rows:
            f.write(f"<tr><td>{html.escape(r['pair'])}</td><td>{fmt(r['iou'])}</td><td>{fmt(r['center'])}</td><td>{fmt(r['P'])}</td>"
                    f"<td>{fmt(r['R'])}</td><td>{r['badges']}</td><td>{'yes' if r['title'] else 'no'}</td><td>{r['sec']:.1f}</td></tr>")
        f.write('</table>')
        for row, a_, b_, c_, lp, lt in cards:
            f.write(f"<div class='card'><b>{html.escape(row['pair'])}</b><div class='imgs'><figure><img src='{a_}'><figcaption>picture</figcaption></figure>"
                    f"<figure><img src='{b_}'><figcaption>traced</figcaption></figure><figure><img src='{c_}'><figcaption>truth</figcaption></figure></div>"
                    f"<p>labels read: {html.escape(', '.join(lp))}<br>true labels: {html.escape(', '.join(lt))}</p></div>")
    print('report:', os.path.abspath(a.report))
    return 0


if __name__ == '__main__':
    sys.exit(main())

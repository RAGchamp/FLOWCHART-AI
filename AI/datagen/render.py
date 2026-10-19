"""Make synthetic training data with the app's own renderer (INFO/FLOW-CHART-AI-PLAN.md §5.1).

    python render.py --count 20000 --val 1000          (overnight; ~2–4 pictures a second)
    python render.py --count 50 --out ../data/synthetic_test   (a quick look)

Opens App/index.html in headless Chromium, builds random diagrams with random_doc.js, and saves
for each one (in <out>/train or <out>/val):
    00042.jpg        the picture
    00042_mask.png   R = ribbon, G = ribbon labels, B = center lines (white = yes)
    00042.json       the diagram (app JSON) — for the Phase 2 model
Re-running continues where it stopped.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apppage import decode, open_app   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

RENDER_JS = """async ([seed, words, scale]) => {
  const { doc, parts } = makeSample(seed, words);
  const out = { doc };
  let area = null;             // the picture's crop around the diagram; the masks use the same one
  for (const k of ['image', 'ribbon', 'text', 'center']) {
    loadDoc(normalizeDoc(parts[k]));
    if (!area) area = exportArea('fit');
    out[k] = await FlowAPI.renderPNG(scale, area);
  }
  out.area = area;
  return out;
}"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--count', type=int, default=20000, help='training pictures')
    ap.add_argument('--val', type=int, default=1000, help='validation pictures')
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'data', 'synthetic'))
    ap.add_argument('--long', type=int, default=640, help='long side of the saved pictures, px')
    ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()

    words = [w.strip() for w in open(os.path.join(HERE, 'words.txt'), encoding='utf-8') if w.strip()]
    jobs = [('train', i) for i in range(a.count)] + [('val', i) for i in range(a.val)]
    for split in ('train', 'val'):
        os.makedirs(os.path.join(a.out, split), exist_ok=True)

    with sync_playwright() as p:
        browser, page = open_app(p)
        t0, done = time.time(), 0
        for split, i in jobs:
            base = os.path.join(a.out, split, f'{i:05d}')
            if os.path.exists(base + '_mask.png'):
                continue
            seed = a.seed * 1_000_003 + (0 if split == 'train' else 500_000_000) + i
            # canvas px → saved px: the long side becomes a.long
            r = page.evaluate(RENDER_JS, [seed, words, 1.0])
            img = decode(r['image'])
            k = a.long / max(img.shape[:2])
            size = (int(round(img.shape[1] * k)), int(round(img.shape[0] * k)))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            chans = []
            for part in ('ribbon', 'text', 'center'):
                m = cv2.resize(decode(r[part], cv2.IMREAD_GRAYSCALE), size, interpolation=cv2.INTER_AREA)
                if part == 'center':
                    m = cv2.dilate(m, np.ones((2, 2), np.uint8))
                chans.append(np.where(m > 110, 255, 0).astype(np.uint8))
            cv2.imwrite(base + '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(base + '_mask.png', cv2.merge(chans[::-1]))      # OpenCV is BGR: B=center, G=text, R=ribbon
            with open(base + '.json', 'w', encoding='utf-8') as f:
                json.dump(r['doc'], f, separators=(',', ':'))
            done += 1
            if done % 25 == 0:
                rate = done / (time.time() - t0)
                left = sum(1 for s, j in jobs if not os.path.exists(os.path.join(a.out, s, f'{j:05d}_mask.png')))
                print(f'{done} made, {rate:.1f}/s, about {left / max(rate, 1e-6) / 60:.0f} min left', flush=True)
        browser.close()
    print('done:', done, 'new pictures in', os.path.abspath(a.out))


if __name__ == '__main__':
    sys.exit(main())

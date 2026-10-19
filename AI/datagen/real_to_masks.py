"""Turn real training pairs (the app's "Save training pair" zips) into training masks.

    1. Put the pair-*.zip files in AI/data/real/
    2. python real_to_masks.py

About 70 % of the pairs become training examples (AI/data/real/train_masks/), the rest are the frozen
test set (AI/data/real/test/) that eval/score.py measures and training never sees. Which pair goes
where is fixed by its file name, so adding pairs later never moves old ones between the two.
"""
import glob
import hashlib
import json
import os
import shutil
import sys
import zipfile

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apppage import mask_for, open_app   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REAL = os.path.join(HERE, '..', 'data', 'real')


def is_test(name):
    return int(hashlib.md5(os.path.basename(name).encode()).hexdigest(), 16) % 10 < 3


def load_pair(path):
    z = zipfile.ZipFile(path)
    info = json.loads(z.read('pair.json'))
    pic = cv2.imdecode(np.frombuffer(z.read(info['picture']['file']), np.uint8), cv2.IMREAD_COLOR)
    shown = cv2.imdecode(np.frombuffer(z.read(info['shown']['file']), np.uint8), cv2.IMREAD_COLOR)
    return info, pic, shown


def main():
    real = sys.argv[1] if len(sys.argv) > 1 else REAL
    pairs = sorted(glob.glob(os.path.join(real, '*.zip')))
    if not pairs:
        print('No pairs in', os.path.abspath(real), '- save some with the app\'s "Save training pair" button.')
        return 1
    tr, te = os.path.join(real, 'train_masks'), os.path.join(real, 'test')
    os.makedirs(tr, exist_ok=True)
    os.makedirs(te, exist_ok=True)
    n_tr = n_te = 0
    with sync_playwright() as p:
        browser, page = open_app(p)
        for path in pairs:
            name = os.path.splitext(os.path.basename(path))[0]
            if is_test(path):
                shutil.copy(path, os.path.join(te, os.path.basename(path)))
                n_te += 1
                continue
            info, pic, shown = load_pair(path)
            sh = info['shown']
            # the canvas area covered by the shown (straightened) picture
            area = {'x': sh['offset'][0], 'y': sh['offset'][1], 'w': sh['width'] * sh['scale'], 'h': sh['height'] * sh['scale']}
            mask = mask_for(page, info['diagram'], area, (shown.shape[1], shown.shape[0]))
            cv2.imwrite(os.path.join(tr, name + '.jpg'), shown, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(os.path.join(tr, name + '_mask.png'), mask)
            n_tr += 1
        browser.close()
    print(f'{n_tr} training pairs -> {os.path.abspath(tr)}; {n_te} test pairs -> {os.path.abspath(te)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

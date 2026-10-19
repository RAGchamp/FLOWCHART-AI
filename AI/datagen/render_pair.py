"""Make a test/training pair from a diagram you know is correct: the app draws the picture, and the
diagram itself is the answer. Good for clean digital pictures (screenshots, slides).

    python render_pair.py ../../JSON-FILES/sample-json.json ../data/real/test/pair-sample-render.zip
"""
import json
import os
import sys
import time
import zipfile

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apppage import open_app   # noqa: E402

import base64   # noqa: E402


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    src, out = sys.argv[1], sys.argv[2]
    doc = json.load(open(src, encoding='utf-8'))
    with sync_playwright() as p:
        browser, page = open_app(p)
        page.evaluate('d => FlowAPI.loadDoc(d)', doc)
        doc = page.evaluate('FlowAPI.getDoc()')                     # normalized, as the app keeps it
        png = page.evaluate("FlowAPI.renderPNG(1, 'canvas')")
        browser.close()
    W, H = doc['canvas']['width'], doc['canvas']['height']
    info = {
        'app': 'FLOW-CHART-APP', 'kind': 'flowtrace-training-pair', 'version': 1, 'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'source': {'rendered': os.path.basename(src)},
        'picture': {'file': 'picture.png', 'name': os.path.basename(src), 'width': W, 'height': H},
        'corners': [[0, 0], [W, 0], [W, H], [0, H]],
        'shown': {'file': 'picture.png', 'width': W, 'height': H, 'scale': 1, 'offset': [0, 0]},
        'diagram': doc,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('picture.png', base64.b64decode(png.split(',', 1)[1]))
        z.writestr('pair.json', json.dumps(info, indent=1))
    print('wrote', os.path.abspath(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())

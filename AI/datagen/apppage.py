"""Drive App/index.html headlessly (Playwright) to draw diagrams and their label masks."""
import base64
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = 'file:///' + os.path.abspath(os.path.join(HERE, '..', '..', 'App', 'index.html')).replace('\\', '/')

PARTS_JS = """async ([doc, area, scale]) => {
  const out = {};
  const parts = labelDocs(normalizeDoc(doc));
  for (const k of ['ribbon', 'text', 'center']) {
    loadDoc(normalizeDoc(parts[k]));
    out[k] = await FlowAPI.renderPNG(scale, area);
  }
  return out;
}"""


def open_app(playwright):
    """A page with the app loaded, the welcome box closed and random_doc.js available."""
    browser = playwright.chromium.launch()
    page = browser.new_page(viewport={'width': 1400, 'height': 900})
    page.goto(APP)
    page.evaluate('localStorage.clear()')
    page.keyboard.press('Enter')
    page.add_script_tag(content=open(os.path.join(HERE, 'random_doc.js'), encoding='utf-8').read())
    return browser, page


def decode(url, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.frombuffer(base64.b64decode(url.split(',', 1)[1]), np.uint8), flags)


def mask_for(page, doc, area, size):
    """Render the ribbon / text / center label documents of `doc` over canvas `area`
    ({x, y, w, h}), resized to size (w, h). Returns an OpenCV BGR mask (B=center, G=text, R=ribbon)."""
    r = page.evaluate(PARTS_JS, [doc, area, 1.0])
    chans = []
    for part in ('ribbon', 'text', 'center'):
        m = cv2.resize(decode(r[part], cv2.IMREAD_GRAYSCALE), size, interpolation=cv2.INTER_AREA)
        if part == 'center':
            m = cv2.dilate(m, np.ones((2, 2), np.uint8))
        chans.append(np.where(m > 110, 255, 0).astype(np.uint8))
    return cv2.merge(chans[::-1])

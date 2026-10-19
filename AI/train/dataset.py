"""Training pictures: synthetic renders (AI/data/synthetic) and real pairs (AI/data/real), turned into
"photo of a slide on a screen" look-alikes on the fly (INFO/FLOW-CHART-AI-PLAN.md §5.1 augment)."""
import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

WORDS = ['Lorem ipsum dolor', 'Click to add text', 'Quarterly review', 'Data is a shared asset',
         'Phase 1', 'Notes', 'Summary', '1. Plan the work', '2. Work the plan']


def photo(img, mask, rng):
    """Make a clean render look like a phone photo of a slide: perspective, light, blur, noise, JPEG."""
    h, w = img.shape[:2]
    # slide context: sometimes a title bar, side text or a monitor bezel around the diagram
    if rng.random() < 0.35:
        pad = [int(h * rng.uniform(0.05, 0.25)), int(h * rng.uniform(0, 0.1)), int(w * rng.uniform(0, 0.08)), int(w * rng.uniform(0, 0.25))]
        img = cv2.copyMakeBorder(img, *pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        mask = cv2.copyMakeBorder(mask, *pad, cv2.BORDER_CONSTANT, value=0)
        h, w = img.shape[:2]
        if pad[0] > 10 and rng.random() < 0.6:
            col = tuple(int(c) for c in rng.integers(20, 120, 3))
            cv2.rectangle(img, (0, 0), (w, int(pad[0] * 0.7)), col, -1)
            cv2.putText(img, rng.choice(WORDS), (10, int(pad[0] * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, pad[0] / 90, (255, 255, 255), 1, cv2.LINE_AA)
        for _ in range(int(rng.integers(0, 6))):
            y = int(rng.uniform(pad[0], h - 10))
            cv2.putText(img, rng.choice(WORDS), (int(w - pad[3] + 4), y), cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.3, 0.6), (60, 60, 60), 1, cv2.LINE_AA)
    # perspective
    if rng.random() < 0.8:
        d = rng.uniform(0, 0.12)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32(rng.uniform(-d, d, (4, 2)) * [w, h])
        M = cv2.getPerspectiveTransform(src, dst)
        bg = tuple(int(c) for c in rng.integers(180, 256, 3))
        img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=bg)
        mask = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    f = img.astype(np.float32)
    # lighting: a soft gradient and a dimmer, greyer screen
    if rng.random() < 0.8:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        a = rng.uniform(0, 2 * np.pi)
        g = (np.cos(a) * xx / w + np.sin(a) * yy / h)
        f *= (rng.uniform(0.7, 1.0) + rng.uniform(0, 0.3) * (g - g.min()) / (np.ptp(g) + 1e-6))[..., None]
        f = f * rng.uniform(0.75, 1.0) + rng.uniform(0, 40)
    # screen pattern (moiré-like stripes)
    if rng.random() < 0.3:
        period = rng.uniform(3, 9)
        stripes = np.sin(np.arange(w) * 2 * np.pi / period)[None, :, None] * rng.uniform(2, 8)
        f += stripes
    # color shift
    f *= rng.uniform(0.88, 1.12, 3)[None, None, :]
    f = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        k = rng.choice([3, 5])
        f = cv2.GaussianBlur(f, (int(k), int(k)), rng.uniform(0.4, 1.5))
    if rng.random() < 0.6:
        f = np.clip(f.astype(np.float32) + rng.normal(0, rng.uniform(2, 9), f.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        ok, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 92))])
        f = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return f, mask


class FlowData(Dataset):
    def __init__(self, folders, size=256, train=True, seed=0):
        self.items = []
        for fo in folders:
            self.items += sorted(glob.glob(os.path.join(fo, '*_mask.png')))
        self.size, self.train, self.seed = size, train, seed
        self.epoch = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        mpath = self.items[i]
        img = cv2.imread(mpath.replace('_mask.png', '.jpg'))
        mask = cv2.imread(mpath)                             # B=center, G=text, R=ribbon
        rng = np.random.default_rng((self.seed, self.epoch, i) if not self.train else None)
        if self.train:
            img, mask = photo(img, mask, rng)
        # crop around the diagram (sometimes zoomed in), then square-pad and resize
        ys, xs = np.nonzero(mask[..., 2] > 0)
        h, w = img.shape[:2]
        if len(ys) and self.train and rng.random() < 0.5:
            cx, cy = rng.choice(xs), rng.choice(ys)
            s = int(min(h, w) * rng.uniform(0.45, 0.9))
            x0, y0 = int(np.clip(cx - s / 2, 0, max(0, w - s))), int(np.clip(cy - s / 2, 0, max(0, h - s)))
            img, mask = img[y0:y0 + s, x0:x0 + s], mask[y0:y0 + s, x0:x0 + s]
        h, w = img.shape[:2]
        side = max(h, w)
        pad = ((side - h) // 2, side - h - (side - h) // 2, (side - w) // 2, side - w - (side - w) // 2)
        fill = tuple(int(v) for v in np.median(img.reshape(-1, 3), 0))
        img = cv2.copyMakeBorder(img, *pad, cv2.BORDER_CONSTANT, value=fill)
        mask = cv2.copyMakeBorder(mask, *pad, cv2.BORDER_CONSTANT, value=0)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        # masks: area-resize then threshold keeps thin center lines
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.train and rng.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(img[..., ::-1]).astype(np.float32) / 255).permute(2, 0, 1)
        y = np.stack([mask[..., 2] > 100, mask[..., 1] > 100, mask[..., 0] > 40]).astype(np.float32)
        return x, torch.from_numpy(y)

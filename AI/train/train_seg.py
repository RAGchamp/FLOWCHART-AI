"""Train FlowTrace-Seg on this computer's CPU (INFO/FLOW-CHART-AI-PLAN.md §6.1, milestone P3).

    python train_seg.py                      # train; stops after --hours, resumes next time
    python train_seg.py --benchmark          # P0: how long one step takes here
    python export_onnx.py                    # then publish the best model to AI/models/

Checkpoints go to AI/train/runs/<name>/ (last.pt every few minutes, best.pt on the best validation
score). Ctrl+C is safe: the next run continues from last.pt.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import FlowData        # noqa: E402
from model_seg import HEADS, FlowTraceSeg    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
LOG = None          # runs/<name>/train-log.txt: everything printed while training, with the time


def log(*parts):
    msg = ' '.join(str(p) for p in parts)
    print(msg, flush=True)
    if LOG:
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S  ') + msg + '\n')


def loss_fn(logits, y):
    """BCE + Dice per head; the thin center lines get extra weight on their few pixels."""
    total, parts = 0, {}
    for i, name in enumerate(HEADS):
        lg, t = logits[:, i], y[:, i]
        pos = t.mean().clamp(min=1e-4)
        w = 1 + t * (min(20.0, float(0.5 / pos)) if name != 'ribbon' else 1)
        bce = F.binary_cross_entropy_with_logits(lg, t, weight=w)
        p = torch.sigmoid(lg)
        dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
        parts[name] = float(bce + dice)
        total = total + bce + dice * (1.0 if name == 'ribbon' else 0.5)
    return total, parts


@torch.no_grad()
def evaluate(net, loader, max_batches=40):
    net.eval()
    inter = np.zeros(len(HEADS)); union = np.zeros(len(HEADS))
    for k, (x, y) in enumerate(loader):
        if k >= max_batches:
            break
        p = torch.sigmoid(net(x)) > 0.5
        t = y > 0.5
        inter += (p & t).sum((0, 2, 3)).numpy()
        union += (p | t).sum((0, 2, 3)).numpy()
    net.train()
    return {h: float(inter[i] / max(union[i], 1)) for i, h in enumerate(HEADS)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', default='seg-v1')
    ap.add_argument('--size', type=int, default=256, help='training picture size (256 first, 384 to finish)')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--hours', type=float, default=10, help='stop after this many hours (resume later)')
    ap.add_argument('--threads', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--real-weight', type=int, default=20, help='how many times each real pair is repeated')
    ap.add_argument('--benchmark', action='store_true', help='time a few steps and exit (milestone P0)')
    ap.add_argument('--data', default=DATA, help='folder with synthetic/{train,val} and real/train_masks')
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    run = os.path.join(HERE, 'runs', a.name)
    net = FlowTraceSeg()
    print(f'FlowTrace-Seg: {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M parameters, input {a.size}px, {a.threads} threads')

    if a.benchmark:
        x, y = torch.rand(a.batch, 3, a.size, a.size), (torch.rand(a.batch, 3, a.size, a.size) > 0.9).float()
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
        for i in range(6):
            t = time.time()
            loss, _ = loss_fn(net(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            if i >= 1:
                dt = time.time() - t
                print(f'step {i}: {dt:.2f} s for {a.batch} pictures -> {dt / a.batch * 1000:.0f} ms each')
        per = dt / a.batch
        for n, ep in ((20000, 10), (20000, 30)):
            print(f'{n} pictures x {ep} epochs = about {n * ep * per / 3600:.1f} hours (plus data loading)')
        return

    os.makedirs(run, exist_ok=True)
    global LOG
    LOG = os.path.join(run, 'train-log.txt')
    train_dirs = [os.path.join(a.data, 'synthetic', 'train')]
    real = os.path.join(a.data, 'real', 'train_masks')
    ds = FlowData(train_dirs, a.size, train=True)
    if os.path.isdir(real):
        rd = FlowData([real], a.size, train=True)
        ds.items += rd.items * a.real_weight
    val = FlowData([os.path.join(a.data, 'synthetic', 'val')], a.size, train=False)
    if not len(ds):
        print('No training pictures. Make some first:  python ../datagen/render.py'); return
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    vl = DataLoader(val, batch_size=a.batch, shuffle=False, num_workers=0) if len(val) else None

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    state = {'epoch': 0, 'step': 0, 'best': -1.0, 'hours': 0.0}
    last = os.path.join(run, 'last.pt')
    if os.path.exists(last):
        ck = torch.load(last, map_location='cpu', weights_only=False)   # our own checkpoint
        net.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); state = ck['state']
        log(f"resuming: epoch {state['epoch']}, step {state['step']}, best {state['best']:.3f}, {state['hours']:.1f} h so far")

    def save(path):
        torch.save({'model': net.state_dict(), 'opt': opt.state_dict(), 'state': state, 'size': a.size}, path)

    log(f'training {a.name}: {len(ds)} pictures per epoch ({len(dl)} steps of {a.batch}), {a.size}px, stops after {a.hours} h')
    t_start, t_ck = time.time(), time.time()
    hours0 = state['hours']
    net.train()
    try:
        while True:
            ds.epoch = state['epoch']
            t_epoch = time.time()
            for k, (x, y) in enumerate(dl, 1):
                lr = a.lr * 0.5 * (1 + np.cos(np.pi * min(1.0, state['step'] / 200000)))
                for g in opt.param_groups:
                    g['lr'] = lr
                loss, parts = loss_fn(net(x), y)
                opt.zero_grad(); loss.backward(); opt.step()
                state['step'] += 1
                state['hours'] = hours0 + (time.time() - t_start) / 3600
                if state['step'] % 20 == 0:
                    log(f"epoch {state['epoch']} ({100 * k // len(dl)}%) step {state['step']} loss {float(loss):.3f} " +
                          ' '.join(f'{h} {v:.3f}' for h, v in parts.items()) + f" lr {lr:.5f} {state['hours']:.2f} h")
                if time.time() - t_ck > 600:                     # every 10 minutes
                    save(last); t_ck = time.time()
                if state['hours'] - hours0 >= a.hours:
                    raise KeyboardInterrupt
            state['epoch'] += 1
            log(f"epoch {state['epoch'] - 1} finished in {(time.time() - t_epoch) / 60:.0f} min")
            if vl:
                iou = evaluate(net, vl)
                score = iou['ribbon'] * 0.5 + iou['text'] * 0.25 + iou['center'] * 0.25
                log('validation IoU', {k: round(v, 3) for k, v in iou.items()}, 'score', round(score, 3))
                if score > state['best']:
                    state['best'] = float(score)
                    save(os.path.join(run, 'best.pt'))
                    log('  new best -> best.pt')
            save(last)
    except KeyboardInterrupt:
        save(last)
        log(f"stopped; saved {last}. Run again to continue. Best validation score {state['best']:.3f}.")


if __name__ == '__main__':
    main()

"""Publish a trained FlowTrace-Seg checkpoint as AI/models/flowtrace-seg.onnx, where the FlowTrace
service picks it up automatically (restart start-flowtrace.bat). The old model is kept as .bak.

    python export_onnx.py                              # runs/seg-v1/best.pt → models/flowtrace-seg.onnx
    python export_onnx.py --ckpt runs/x/last.pt --out somewhere.onnx
"""
import argparse
import json
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_seg import HEADS, Exported, FlowTraceSeg   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', default=os.path.join(HERE, 'runs', 'seg-v1', 'best.pt'))
    ap.add_argument('--out', default=os.path.join(HERE, '..', 'models', 'flowtrace-seg.onnx'))
    ap.add_argument('--size', type=int, default=384, help='input size the service will use')
    ap.add_argument('--min-score', type=float, default=0.6, help='refuse to publish a model below this validation score')
    ap.add_argument('--force', action='store_true', help='publish even below --min-score')
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location='cpu', weights_only=False)   # our own checkpoint
    best = (ck.get('state') or {}).get('best', 0) or 0
    if best < a.min_score and not a.force:
        print(f'Not published: validation score {best:.3f} is below {a.min_score}. Train longer, or use --force.')
        print('(The FlowTrace service keeps using classic vision until a good model is published.)')
        return 1
    net = FlowTraceSeg()
    net.load_state_dict(ck['model'])
    net.eval()
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        shutil.copy(out, out + '.bak')
    torch.onnx.export(Exported(net), torch.zeros(1, 3, a.size, a.size), out, input_names=['image'],
                      output_names=list(HEADS), opset_version=17, dynamo=False)
    st = ck.get('state', {})
    card = {'model': 'FlowTrace-Seg', 'file': os.path.basename(out), 'exported': time.strftime('%Y-%m-%d %H:%M'),
            'checkpoint': os.path.relpath(a.ckpt, HERE), 'input_size': a.size, 'heads': list(HEADS),
            'trained_steps': st.get('step'), 'epochs': st.get('epoch'), 'train_hours': round(st.get('hours', 0), 2),
            'best_val_score': st.get('best')}
    with open(os.path.splitext(out)[0] + '.json', 'w', encoding='utf-8') as f:
        json.dump(card, f, indent=1)
    print('exported', out)
    print(json.dumps(card, indent=1))


if __name__ == '__main__':
    sys.exit(main())

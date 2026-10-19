"""FlowTrace-Seg: a small U-Net that marks, for every pixel, ribbon / ribbon label / ribbon center line
(INFO/FLOW-CHART-AI-PLAN.md §6.1). About 0.25 M parameters, so it trains on a laptop CPU."""
import torch
import torch.nn as nn
import torch.nn.functional as F

HEADS = ('ribbon', 'text', 'center')


class SepConv(nn.Module):
    """Depthwise-separable 3×3 conv + BN + ReLU (MobileNet style: cheap on a CPU)."""
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.a = SepConv(cin, cout, stride)
        self.b = SepConv(cout, cout)
        self.skip = None if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = self.b(self.a(x))
        return y + (x if self.skip is None else self.skip(x))


class FlowTraceSeg(nn.Module):
    def __init__(self, ch=(16, 32, 64, 96, 160)):
        super().__init__()
        # the stem halves the size at once: full-size layers are what make a CPU slow
        self.stem = nn.Sequential(nn.Conv2d(3, ch[0], 3, 2, 1, bias=False), nn.BatchNorm2d(ch[0]), nn.ReLU(inplace=True))
        self.e1 = Block(ch[0], ch[0])              # 1/2
        self.e2 = Block(ch[0], ch[1], 2)          # 1/4
        self.e3 = Block(ch[1], ch[2], 2)          # 1/8
        self.e4 = Block(ch[2], ch[3], 2)          # 1/16
        self.e5 = nn.Sequential(Block(ch[3], ch[4], 2), Block(ch[4], ch[4]))   # 1/32: sees whole loops
        self.d4 = Block(ch[4] + ch[3], ch[3])
        self.d3 = Block(ch[3] + ch[2], ch[2])
        self.d2 = Block(ch[2] + ch[1], ch[1])
        self.d1 = Block(ch[1] + ch[0], ch[0])
        self.out = nn.Conv2d(ch[0], len(HEADS), 1)

    def forward(self, x):
        s0 = self.e1(self.stem(x))
        s1 = self.e2(s0)
        s2 = self.e3(s1)
        s3 = self.e4(s2)
        y = self.e5(s3)
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        y = self.d4(torch.cat([up(y, s3), s3], 1))
        y = self.d3(torch.cat([up(y, s2), s2], 1))
        y = self.d2(torch.cat([up(y, s1), s1], 1))
        y = self.d1(torch.cat([up(y, s0), s0], 1))
        return up(self.out(y), x)             # logits at the input size, one channel per head


class Exported(nn.Module):
    """For ONNX: probabilities, one named output per head."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        p = torch.sigmoid(self.net(x))
        return tuple(p[:, i:i + 1] for i in range(p.shape[1]))


if __name__ == '__main__':
    m = FlowTraceSeg()
    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')
    print(m(torch.randn(1, 3, 256, 256)).shape)

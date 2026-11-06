"""DenoiseSRNet: residual restoration network (2,751,747 parameters at 96 channels / 16 blocks)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Scaled residual block; the 0.1 scale stabilizes training with many blocks."""

    def __init__(self, n_feat):
        super().__init__()
        self.c1 = nn.Conv2d(n_feat, n_feat, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_feat)
        self.c2 = nn.Conv2d(n_feat, n_feat, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_feat)
        self.scale = 0.1

    def forward(self, x):
        r = F.relu(self.bn1(self.c1(x)), inplace=True)
        r = self.bn2(self.c2(r))
        return x + self.scale * r


class DenoiseSRNet(nn.Module):
    """Input and output: [B, 3, 256, 256]. Global residual: output = clamp(input + correction)."""

    def __init__(self, n_feat=96, n_blocks=16):
        super().__init__()
        self.head = nn.Conv2d(3, n_feat, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(n_feat) for _ in range(n_blocks)])
        self.fuse = nn.Conv2d(n_feat, n_feat, 3, padding=1)
        self.tail = nn.Conv2d(n_feat, 3, 3, padding=1)

    def forward(self, x):
        feat = F.relu(self.head(x), inplace=True)
        out = self.fuse(self.body(feat)) + feat
        out = self.tail(out)
        return torch.clamp(x + out, 0.0, 1.0)


if __name__ == "__main__":
    m = DenoiseSRNet()
    print(sum(p.numel() for p in m.parameters()))  # 2751747

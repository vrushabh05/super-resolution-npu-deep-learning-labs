"""LiteSRNet: lightweight super-resolution network designed for NPU deployment.

Input and output are [B, 3, 256, 256] images in [0, 1]. Only standard Conv,
ConvTranspose and ReLU are used (no concatenation skips, no exotic activations)
so the graph compiles cleanly to the Mobilint NPU.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Two-conv residual block with ReLU only and no BatchNorm in the residual path."""

    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=True)
        # small init for stable residual learning
        nn.init.normal_(self.conv2.weight, std=0.01)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class LiteSRNet(nn.Module):
    """Stride-2 encoder, deep residual body at half resolution, transposed-conv decoder."""

    def __init__(self, ch=56, num_blocks=9):
        super().__init__()
        # Encoder: 256x256 -> 128x128
        self.encoder = nn.Sequential(
            nn.Conv2d(3, ch, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.feat_norm = ConvBlock(ch, ch, kernel_size=3, stride=1, padding=1)
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(num_blocks)])
        self.body_proj = nn.Conv2d(ch, ch, 3, padding=1, bias=True)
        # Decoder: 128x128 -> 256x256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2, bias=True),
            nn.ReLU(inplace=True),
        )
        self.tail = nn.Conv2d(ch, 3, kernel_size=3, padding=1, bias=True)
        # start as a near-identity mapping (global residual)
        nn.init.normal_(self.tail.weight, std=0.001)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        e = self.encoder(x)
        f = self.feat_norm(e)
        b = self.body_proj(self.body(f)) + f
        d = self.decoder(b)
        return torch.clamp(x + self.tail(d), 0.0, 1.0)


if __name__ == "__main__":
    model = LiteSRNet(ch=56, num_blocks=9).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 256, 256))
    print("output shape:", list(out.shape))
    print("parameters:", f"{sum(p.numel() for p in model.parameters()):,}")

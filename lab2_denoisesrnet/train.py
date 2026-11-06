"""Train DenoiseSRNet on paired LR/HR images (random 128px crops, flips, 90-degree rotations).

Usage: python train.py --lr-dirs LR_train LR_train1 ... --hr-dirs HR_train HR_train1 ... \
                       --val-lr LR_val --val-hr HR_val
"""
import argparse
import glob
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from model import DenoiseSRNet

CFG = dict(seed=42, patch=128, batch=16, epochs=150, lr=2e-4, wd=1e-5, grad_clip=1.0, channels=96, blocks=16)
EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")


def gather(folder):
    paths = []
    for e in EXTS:
        paths += glob.glob(os.path.join(folder, "**", e), recursive=True)
    return sorted(paths)


class Train(Dataset):
    def __init__(self, lr, hr, crop):
        self.lr, self.hr, self.cs = lr, hr, crop

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, i):
        lr = TF.to_tensor(Image.open(self.lr[i]).convert("RGB"))
        hr = TF.to_tensor(Image.open(self.hr[i]).convert("RGB"))
        _, h, w = lr.shape
        cs = self.cs
        if h < cs or w < cs:
            ph, pw = max(0, cs - h), max(0, cs - w)
            lr = F.pad(lr, (0, pw, 0, ph), mode="reflect")
            hr = F.pad(hr, (0, pw, 0, ph), mode="reflect")
            _, h, w = lr.shape
        t, l = random.randint(0, h - cs), random.randint(0, w - cs)
        lr, hr = lr[:, t:t + cs, l:l + cs], hr[:, t:t + cs, l:l + cs]
        if random.random() < 0.5:
            lr, hr = lr.flip(2), hr.flip(2)
        if random.random() < 0.5:
            lr, hr = lr.flip(1), hr.flip(1)
        k = random.randint(0, 3)
        if k:
            lr, hr = torch.rot90(lr, k, [1, 2]), torch.rot90(hr, k, [1, 2])
        return lr, hr


class Val(Dataset):
    def __init__(self, lr, hr):
        self.lr, self.hr = lr, hr

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, i):
        return (TF.to_tensor(Image.open(self.lr[i]).convert("RGB")),
                TF.to_tensor(Image.open(self.hr[i]).convert("RGB")))


def charbonnier(pred, target, eps=1e-6):
    d = pred - target
    return torch.mean(torch.sqrt(d * d + eps))


def psnr(pred, target):
    mse = torch.mean((pred.clamp(0, 1) - target.clamp(0, 1)) ** 2).item()
    return float("inf") if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    loss = ps = 0.0
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        p = model(lr)
        loss += charbonnier(p, hr).item()
        ps += psnr(p[0], hr[0])
    return loss / len(loader), ps / len(loader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr-dirs", nargs="+", required=True)
    ap.add_argument("--hr-dirs", nargs="+", required=True)
    ap.add_argument("--val-lr", required=True)
    ap.add_argument("--val-hr", required=True)
    ap.add_argument("--out", default="checkpoints")
    a = ap.parse_args()

    random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)

    lr_files = [f for d in a.lr_dirs for f in gather(d)]
    hr_files = [f for d in a.hr_dirs for f in gather(d)]
    assert len(lr_files) == len(hr_files), "LR/HR count mismatch"
    train_dl = DataLoader(Train(lr_files, hr_files, CFG["patch"]), batch_size=CFG["batch"], shuffle=True,
                          num_workers=2, drop_last=True)
    val_dl = DataLoader(Val(gather(a.val_lr), gather(a.val_hr)), batch_size=1)

    model = DenoiseSRNet(CFG["channels"], CFG["blocks"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"], eta_min=1e-6)

    best = -float("inf")
    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        total = 0.0
        for lr, hr in train_dl:
            lr, hr = lr.to(device), hr.to(device)
            opt.zero_grad(set_to_none=True)
            loss = charbonnier(model(lr), hr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            opt.step()
            total += loss.item()
        sched.step()
        v_loss, v_psnr = validate(model, val_dl, device)
        print(f"Ep {epoch:03d}/{CFG['epochs']} train={total / len(train_dl):.6f} val={v_loss:.6f} psnr={v_psnr:.3f}")
        if v_psnr > best:
            best = v_psnr
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_psnr": v_psnr},
                       os.path.join(a.out, "best.pth"))
    print(f"Best validation PSNR: {best:.3f} dB")


if __name__ == "__main__":
    main()

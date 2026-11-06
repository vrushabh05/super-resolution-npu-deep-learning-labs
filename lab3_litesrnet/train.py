"""Train LiteSRNet for DATA 255 Lab 3 (paired LR/HR images, 256x256, values in [0, 1]).

Recipe from the original Colab notebook: Adam (lr 3e-4), 10-epoch warmup then cosine
decay, L1 loss, mixed precision, gradient clipping 1.0, EMA of weights (decay 0.999),
350 epochs, batch size 16. Validation PSNR is measured with the EMA weights.

Example:
    python train.py --lr-dirs data/LR_train data/LR_train1 --hr-dirs data/HR_train data/HR_train1 \
                    --lr-val data/LR_val --hr-val data/HR_val
"""
import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import LiteSRNet

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def find_images(root):
    paths = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                paths.append(os.path.join(dirpath, f))
    return sorted(paths)


class SRDataset(Dataset):
    """Paired LR/HR dataset. Input and output are 256x256x3 normalized to [0, 1]."""

    def __init__(self, lr_paths, hr_paths, augment=False):
        assert len(lr_paths) == len(hr_paths)
        self.lr_paths, self.hr_paths, self.augment, self.size = lr_paths, hr_paths, augment, 256

    def __len__(self):
        return len(self.lr_paths)

    def _load(self, path):
        img = Image.open(path).convert('RGB')
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BILINEAR)
        return img

    def __getitem__(self, idx):
        lr, hr = self._load(self.lr_paths[idx]), self._load(self.hr_paths[idx])
        if self.augment:
            if random.random() > 0.5:
                lr, hr = TF.hflip(lr), TF.hflip(hr)
            if random.random() > 0.5:
                lr, hr = TF.vflip(lr), TF.vflip(hr)
            if random.random() > 0.3:
                k = random.choice([1, 2, 3])
                lr, hr = TF.rotate(lr, k * 90), TF.rotate(hr, k * 90)
            if random.random() > 0.5:
                f = random.uniform(0.85, 1.15)
                lr, hr = TF.adjust_brightness(lr, f), TF.adjust_brightness(hr, f)
        to_t = lambda im: torch.from_numpy(np.array(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return to_t(lr), to_t(hr)


class EMA:
    """Exponential moving average of model weights."""

    def __init__(self, model, decay=0.999):
        self.model, self.decay = model, decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data = self.shadow[n]

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]


def calc_psnr(pred, target):
    with torch.no_grad():
        mse = torch.mean((pred.clamp(0, 1) - target.clamp(0, 1)) ** 2)
        return 100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lr-dirs', nargs='+', required=True)
    ap.add_argument('--hr-dirs', nargs='+', required=True)
    ap.add_argument('--lr-val', required=True)
    ap.add_argument('--hr-val', required=True)
    ap.add_argument('--epochs', type=int, default=350)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--out', default='checkpoints')
    args = ap.parse_args()

    SEED, LR_INIT, LR_MIN, GRAD_CLIP = 42, 3e-4, 6e-9, 1.0
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)

    all_lr, all_hr = [], []
    for lr_root, hr_root in zip(args.lr_dirs, args.hr_dirs):
        lrs, hrs = find_images(lr_root), find_images(hr_root)
        n = min(len(lrs), len(hrs))
        all_lr += lrs[:n]; all_hr += hrs[:n]
    train_ds = SRDataset(all_lr, all_hr, augment=True)
    val_ds = SRDataset(find_images(args.lr_val), find_images(args.hr_val), augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    n_batches = len(train_loader)
    print(f'train pairs {len(train_ds)}, val pairs {len(val_ds)}, train batches {n_batches}')

    model = LiteSRNet(ch=56, num_blocks=9).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR_INIT, betas=(0.9, 0.999), weight_decay=0)

    total_steps, warmup_steps = args.epochs * n_batches, 10 * n_batches

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return LR_MIN / LR_INIT + 0.5 * (1 - LR_MIN / LR_INIT) * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMA(model, decay=0.999)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda', init_scale=1024)
    best_psnr = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        for lr_imgs, hr_imgs in train_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device, non_blocking=True), hr_imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == 'cuda'):
                preds = model(lr_imgs)
            loss = torch.mean(torch.abs(preds.float().clamp(0, 1) - hr_imgs.float()))  # L1
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update()
            ep_loss += loss.item()

        # validate every 10 epochs, plus the first 5 and the last 20
        if epoch <= 5 or epoch % 10 == 0 or epoch > args.epochs - 20:
            ema.apply_shadow()
            model.eval()
            total, count = 0.0, 0
            with torch.no_grad():
                for lr_b, hr_b in val_loader:
                    pred = model(lr_b.to(device))
                    hr_b = hr_b.to(device)
                    for i in range(pred.size(0)):
                        total += calc_psnr(pred[i], hr_b[i]); count += 1
            mean_psnr = total / max(count, 1)
            print(f'Ep [{epoch:3d}/{args.epochs}]  loss={ep_loss / max(n_batches, 1):.5f}  '
                  f'PSNR={mean_psnr:.2f} dB  lr={optimizer.param_groups[0]["lr"]:.2e}')
            if mean_psnr > best_psnr:
                best_psnr = mean_psnr
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'ema_shadow': ema.shadow, 'best_psnr': best_psnr},
                           os.path.join(args.out, 'best_model.pth'))
            ema.restore()
            model.train()

    print(f'Training complete. Best PSNR: {best_psnr:.2f} dB')


if __name__ == '__main__':
    main()

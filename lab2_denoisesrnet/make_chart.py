"""Draw assets/lab2_training_curve.png from results/training_log.csv (every 10th epoch plus first and best)."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / "results" / "training_log.csv").open()))
ep = [int(r["epoch"]) for r in rows]
ps = [float(r["val_psnr_db"]) for r in rows]
tl = [float(r["train_loss"]) for r in rows]

BG, INK, MUT, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=200, facecolor=BG)
for ax in (a1, a2):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUT)
    ax.set_xlabel("Epoch", color=MUT)
a1.plot(ep, tl, color=ORANGE, lw=2, marker="o", ms=3)
a1.set_ylabel("Train loss (Charbonnier)", color=MUT)
a1.set_title("Training loss", loc="left", fontsize=11, fontweight="bold", color=INK)
a2.plot(ep, ps, color=BLUE, lw=2, marker="o", ms=3)
a2.annotate("22.423 dB (CPU)\nNPU: 22.400 dB", xy=(148, 22.423), xytext=(70, 21.6), color=INK, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=MUT))
a2.set_ylabel("Validation PSNR (dB)", color=MUT)
a2.set_title("Validation PSNR, 100 images", loc="left", fontsize=11, fontweight="bold", color=INK)
fig.tight_layout()
out = root.parent / "assets" / "lab2_training_curve.png"
fig.savefig(out, facecolor=BG)
print("wrote", out)

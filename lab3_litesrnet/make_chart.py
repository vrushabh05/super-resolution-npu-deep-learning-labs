"""Draw assets/lab3_training_curve.png from results/training_log_psnr.csv."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / "results" / "training_log_psnr.csv").open()))
ep = [int(r["epoch"]) for r in rows]
ps = [float(r["val_psnr_db_ema"]) for r in rows]

BG, INK, MUT, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
fig, ax = plt.subplots(figsize=(8.5, 4), dpi=200, facecolor=BG)
ax.set_facecolor(BG)
ax.plot(ep, ps, color=BLUE, lw=2, marker="o", ms=3)
ax.axhline(24.3, color=ORANGE, lw=1.5, ls="--")
ax.text(4, 24.315, "target 24.3 dB", color=MUT, fontsize=9)
ax.axhline(23.1, color="#8a8984", lw=1, ls=":")
ax.text(200, 23.115, "baseline 23.1 dB", color=MUT, fontsize=9)
ax.annotate("NPU after quantization: 24.283 dB", xy=(350, 24.283), xytext=(150, 23.55), color=INK, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=MUT))
ax.set_xlabel("Epoch", color=MUT)
ax.set_ylabel("Validation PSNR (dB, EMA weights)", color=MUT)
ax.set_title("LiteSRNet reaches 24.33 dB on CPU; quantized NPU build lands at 24.283 dB", loc="left",
             fontsize=11, fontweight="bold", color=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=MUT)
fig.tight_layout()
out = root.parent / "assets" / "lab3_training_curve.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, facecolor=BG)
print("wrote", out)

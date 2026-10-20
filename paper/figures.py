"""Figures for the paper: pipeline diagram (Fig. 1) and rotation gains (Fig. 3). 300 dpi PNG."""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(sys.argv[1])
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MINUS = "−"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": BASE,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2})

# ---------------- Fig. 1: pipeline -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 2.4))
ax.set_xlim(-0.5, 101)
ax.set_ylim(0, 36)
ax.axis("off")


def box(x, y, w, h, title, body, edge=BASE, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8",
                                fc="#fcfcfb", ec=edge, lw=lw))
    ax.text(x + w / 2, y + h - 1.0, title, ha="center", va="top", fontsize=6.2, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 3.2, "\n".join(body), ha="center", va="top", fontsize=5.3, color=INK2,
            linespacing=1.3)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=INK2))


inputs = [
    ("Parcel metadata (8)", ["area, latitude, NDVI peak and timing,", "observation count"]),
    ("Phenology (15)", ["clear-date NDVI, NDRE, NDMI,", "mowing drops, seasonal contrast"]),
    ("Pretrained L1C networks (19)", ["MSResNet + Transformer logits,", "8-view test-time augmentation"]),
    ("MSResNet trained on L2A (9)", ["clear dates, mean of 3 seeds,", "8-view test-time augmentation"]),
]
ax.text(13.5, 35.6, "per-parcel Sentinel-2 time series", ha="center", va="top", fontsize=6, color=MUTED)
mid = 19.5
for i, (t, b) in enumerate(inputs):
    y = 26.6 - i * 7.7
    box(0.3, y, 26.5, 6.6, t, b)
    arrow(27.0, y + 3.3, 31.0, mid)
box(31.5, 14, 22.5, 11, "Meta-classifier",
    ["XGBoost + CatBoost,", "51 features, orchard weight m,", "blend + orchard / meadow", "log-biases (294 settings)"])
arrow(54.5, mid, 57.5, mid)
box(58, 14, 19, 11, "Candidates",
    ["meta orchard score above", "a threshold keeping 90%", "of out-of-fold training", "orchards"])
arrow(77.5, mid, 80.0, mid)
box(80.5, 14, 19.3, 11, "Orchard verifier",
    ["binary XGBoost on meta", "scores, phenology,", "monthly means, geometry,", "neighbour predictions"],
    edge=ORANGE, lw=1.4)
ax.text(42.75, 12.2, "Output 1: 9-class label\nat a chosen operating point", ha="center", va="top", fontsize=5.8,
        color=INK2, linespacing=1.3)
ax.text(90.15, 12.2, "Output 2: orchard if the verifier\nscore passes t, otherwise the\nmeta label without orchards",
        ha="center", va="top", fontsize=5.8, color=INK2, linespacing=1.3)
fig.savefig(OUT / "fig1_pipeline.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------- Fig. 3: rotation gains (seed 42, paired block bootstrap) ----------------------
regions = ["frh01", "frh02", "frh03", "frh04"]
checks = [  # (label, colour, [(estimate, lo, hi) per region])
    (f"L2A network (B {MINUS} A)", BLUE,
     [(13.09, 9.61, 16.85), (5.04, 2.23, 8.19), (14.87, 12.45, 17.58), (11.35, 8.19, 14.62)]),
    (f"Phenology (B {MINUS} D)", AQUA,
     [(3.68, 2.64, 4.61), (2.95, 1.58, 4.51), (5.63, 4.20, 7.01), (3.73, 2.60, 4.94)]),
    (f"Verifier (C {MINUS} B)", ORANGE,
     [(4.03, 2.19, 5.86), (4.23, 0.65, 8.18), (-1.14, -3.44, 1.25), (4.01, 2.33, 5.71)]),
]
fig, ax = plt.subplots(figsize=(3.45, 2.75))
y, ticks, labels = 0.0, [], []
for name, col, rows in checks:
    ax.text(-4.6, y + 0.15, name, fontsize=7, fontweight="bold", color=INK, va="bottom")
    y -= 0.75
    for reg, (est, lo, hi) in zip(regions, rows):
        ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="round", zorder=2)
        ax.plot(est, y, "o", ms=4.5, color=col, mec="white", mew=0.8, zorder=3)
        ax.text(hi + 0.35, y, f"{est:+.1f}".replace("-", MINUS), va="center", fontsize=6, color=INK2)
        ticks.append(y)
        labels.append(reg)
        y -= 0.62
    y -= 0.35
ax.axvline(0, color=INK2, lw=0.8, zorder=1)
ax.set_yticks(ticks, labels)
ax.tick_params(axis="y", length=0, labelsize=6.5)
ax.tick_params(axis="x", labelsize=6.5, length=2)
ax.set_xlim(-4.6, 20.5)
ax.set_xlabel("Gain in orchard AP (points), 95% block-bootstrap interval", fontsize=6.5)
ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
fig.savefig(OUT / "fig3_rotation.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("ok")

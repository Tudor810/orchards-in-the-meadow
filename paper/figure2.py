"""Fig. 2: (a, b) monthly clear-date NDVI / NDMI for orchards vs meadows (training regions frh01-frh03),
(c) orchard precision-recall curves on frh04 (seed 42, the 122,557 parcels with an L1C series).

Run from the repo root: python paper/figure2.py   (writes paper/fig2_evidence.png)
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

from agricultureai.verifier import candidate_threshold, monthly_means, verifier_scores  # noqa: E402

ORCHARD, PERM, TEMP = 5, 7, 8
NDVI, NDMI, NCH = 10, 12, 13
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CLASS_COL = {ORCHARD: "#eb6834", PERM: "#2a78d6", TEMP: "#1baf7a"}
CLASS_NAME = {ORCHARD: "Orchards", PERM: "Permanent meadows", TEMP: "Temporary meadows"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": BASE,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})

# ---- (a, b) seasonal trajectories ---------------------------------------------------------------
curves = {c: {NDVI: [], NDMI: []} for c in CLASS_COL}
for r in ("frh01", "frh02", "frh03"):
    store = np.load(ROOT / f"artifacts/series_store/{r}.npz")
    mm = monthly_means(store).reshape(-1, 12, NCH)
    for c in CLASS_COL:
        rows = mm[store["labels"] == c]
        for ch in (NDVI, NDMI):
            curves[c][ch].append(rows[:, :, ch])
stats = {c: {ch: np.nanpercentile(np.vstack(v), [25, 50, 75], axis=0) for ch, v in d.items()} for c, d in curves.items()}

# ---- (c) precision-recall on frh04 --------------------------------------------------------------
seed = 42
ev = torch.load(ROOT / "artifacts/cached_logits_eval.pt", map_location="cpu", weights_only=True)
y_ev = ev["labels"].numpy().astype(np.int64)
matched = ev["msresnet_l1c_matched"].numpy().astype(bool)
logits = ev["msresnet_logits"].numpy().astype(np.float64)
p_base = np.exp(logits - logits.max(1, keepdims=True))
p_base = (p_base / p_base.sum(1, keepdims=True))[:, ORCHARD]
y_tr = torch.load(ROOT / "artifacts/cached_logits_train.pt", map_location="cpu", weights_only=True)["labels"].numpy()
feats = np.load(ROOT / f"artifacts/orchard_verifier_final/features_seed{seed}.npz")
pheno = {s: np.load(ROOT / f"artifacts/phenology_{s}.npz")["features"] for s in ("train", "eval")}
groups = lambda s: np.hstack([feats[f"{s}_meta"], pheno[s], feats[f"{s}_monthly"], feats[f"{s}_geometry"], feats[f"{s}_neighbours"]])  # noqa: E731
p_tr, p_ev = feats["train_meta"], feats["eval_meta"]
tau = candidate_threshold(p_tr, y_tr)
cand_tr, cand_ev = p_tr[:, ORCHARD] >= tau, p_ev[:, ORCHARD] >= tau
p_meta = p_ev[:, ORCHARD].astype(np.float64)
p_ver, _ = verifier_scores(groups("train"), y_tr, cand_tr, groups("eval"), p_ev, cand_ev, seed)
yb = (y_ev[matched] == ORCHARD).astype(int)
systems = [("Pretrained", p_base, "#eda100", "--"), ("Meta-classifier", p_meta, "#6250d6", "-"),
           ("Meta + verifier", p_ver, "#008300", "-")]

# ---- plot ------------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), gridspec_kw={"width_ratios": [1, 1, 1.05]})
months = np.arange(1, 13)
for ax, ch, title in ((axes[0], NDVI, "(a) NDVI, clear dates"), (axes[1], NDMI, "(b) NDMI, clear dates")):
    for c in (PERM, TEMP, ORCHARD):
        q25, q50, q75 = stats[c][ch]
        ax.fill_between(months, q25, q75, color=CLASS_COL[c], alpha=0.13, lw=0)
        ax.plot(months, q50, color=CLASS_COL[c], lw=1.6 if c == ORCHARD else 1.2, label=CLASS_NAME[c])
    ax.set_xticks([1, 3, 5, 7, 9, 11], ["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
    ax.set_title(title, fontsize=7.5, color=INK, loc="left")
    ax.grid(color=GRID, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("median, IQR band", fontsize=6.5)
axes[0].legend(frameon=False, fontsize=6, loc="lower center", handlelength=1.5)
ax = axes[2]
for name, s, col, ls in systems:
    prec, rec, _ = precision_recall_curve(yb, s[matched])
    apv = 100 * average_precision_score(yb, s[matched])
    ax.plot(100 * rec, 100 * prec, color=col, ls=ls, lw=1.4, label=f"{name} (AP {apv:.1f})")
    print(f"{name}: AP {apv:.2f}")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel("Orchard recall (%)", fontsize=6.5)
ax.set_ylabel("Orchard precision (%)", fontsize=6.5)
ax.set_title("(c) Orchards on frh04", fontsize=7.5, color=INK, loc="left")
ax.grid(color=GRID, lw=0.5)
ax.legend(frameon=False, fontsize=5.8, loc="upper right", handlelength=2, borderaxespad=0.2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(w_pad=1.2)
fig.savefig(Path(__file__).with_name("fig2_evidence.png"), dpi=300, facecolor="white")
print("ok")

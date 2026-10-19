"""Paired block-bootstrap CIs for orchard AP differences B-A and B-D in the region rotation.

Usage: python scripts/rotation_ap_ci.py > artifacts/rotation/rotation_ap_ci.log  (after run_rotation.sh)

Same resampling as region_rotation.stage_verify: 10 km blocks from folds_<R>.npz, 1000 resamples,
default_rng(0). Scores = geo_mean blend of the m=1 XGBoost/CatBoost probabilities (the AP the
protocol logs report). A resample is a block-multiplicity weight vector, which gives exactly the
AP of the concatenated resample (duplicates share scores, so ties group the same way).
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agricultureai.meta_features import geometric_mean_probs  # noqa: E402

ORCHARD = 5
WORK = ROOT / "artifacts" / "rotation"


def sfx(s):
    return "" if s == 42 else f"_seed{s}"


def prep(score, y):
    order = np.argsort(-score, kind="stable")
    s = score[order]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]  # last index of each distinct score
    return order, last


def wap(order, last, y, w):
    yw = (w * y)[order]
    tp = np.cumsum(yw)[last]
    fp = np.cumsum(w[order] - yw)[last]
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / tp[-1]
    return 100.0 * np.sum(np.diff(np.r_[0.0, rec]) * prec)


for region in ("frh01", "frh02", "frh03", "frh04"):
    y = (np.load(WORK / "nets" / f"l2a_test{region}" / f"logits_{region}.npz")["labels"] == ORCHARD).astype(float)
    blocks = np.load(WORK / f"folds_{region}.npz")["blocks"]
    _, gidx = np.unique(blocks, return_inverse=True)
    rng = np.random.default_rng(0)
    weights = [np.bincount(rng.integers(0, gidx.max() + 1, gidx.max() + 1), minlength=gidx.max() + 1)[gidx].astype(float)
               for _ in range(1000)]
    for seed in (42, 1, 2):
        aps, prepped = {}, {}
        for arm in "BAD":
            z = np.load(WORK / region / f"proba_{arm}{sfx(seed)}" / "proba_pair_compact_m1.npz")
            sc = geometric_mean_probs(z["p_xgb"], z["p_cat"])[:, ORCHARD].astype(np.float64)
            prepped[arm] = prep(sc, y)
            aps[arm] = wap(*prepped[arm], y, np.ones_like(y))
            if arm == "B" and seed == 42:
                assert abs(aps[arm] - 100 * average_precision_score(y, sc)) < 1e-6
        boot = {arm: np.array([wap(*prepped[arm], y, w) for w in weights]) for arm in "BAD"}
        parts = []
        for other in "AD":
            d = boot["B"] - boot[other]
            lo, hi = np.percentile(d, [2.5, 97.5])
            parts.append(f"B-{other} {aps['B'] - aps[other]:+6.2f} [{lo:+.2f},{hi:+.2f}]")
        print(f"{region} seed {seed:2d}: AP B={aps['B']:.2f} A={aps['A']:.2f} D={aps['D']:.2f} | " + " | ".join(parts), flush=True)

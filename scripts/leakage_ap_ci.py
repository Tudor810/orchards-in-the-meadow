"""Paired block-bootstrap CI for orchard AP in the stacking-leakage test: frh04-oof-trained minus frh01-03-trained
final design (geo_mean blend, m=1), frh04 10 km blocks, 1000 resamples, rng 0 (same as rotation_ap_ci.py).

Usage (from the repo root, after the noexpens and noexpensoof probability caches exist):
    python scripts/leakage_ap_ci.py > artifacts/leakage/leakage_ap_ci.log
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agricultureai.meta_features import geometric_mean_probs

ORCHARD = 5
folds = np.load(ROOT / "artifacts/frh04_spatial_folds.npz")
ens = np.load(ROOT / "artifacts/checkpoints/msresnet_l2a_clear_ens3/logits_eval.npz")
assert np.array_equal(folds["parcel_ids"], ens["parcel_ids"])
y = (ens["labels"] == ORCHARD).astype(float)
_, gidx = np.unique(folds["blocks"], return_inverse=True)
G = gidx.max() + 1
rng = np.random.default_rng(0)
weights = [np.bincount(rng.integers(0, G, G), minlength=G)[gidx].astype(float) for _ in range(1000)]

def prep(sc):
    o = np.argsort(-sc, kind="stable"); s = sc[o]
    return o, np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]

def wap(o, last, w):
    yw = (w * y)[o]; tp = np.cumsum(yw)[last]; fp = np.cumsum(w[o] - yw)[last]
    return 100 * np.sum(np.diff(np.r_[0.0, tp / tp[-1]]) * tp / np.maximum(tp + fp, 1e-12))

for s, sfx in ((42, ""), (1, "_seed1"), (2, "_seed2")):
    pr = {}
    for arm, d in (("final", ROOT / "artifacts/pruned_proba_cache_noexpens"), ("oof", ROOT / "artifacts/pruned_proba_cache_noexpensoof")):
        z = np.load(f"{d}{sfx}/proba_pair_compact_m1.npz")
        pr[arm] = prep(geometric_mean_probs(z["p_xgb"], z["p_cat"])[:, ORCHARD].astype(np.float64))
    one = np.ones_like(y)
    ap_f, ap_o = wap(*pr["final"], one), wap(*pr["oof"], one)
    d = np.array([wap(*pr["oof"], w) - wap(*pr["final"], w) for w in weights])
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"seed {s:2d}: AP final={ap_f:.2f} oof={ap_o:.2f} diff={ap_o - ap_f:+.2f} [{lo:+.2f},{hi:+.2f}]", flush=True)

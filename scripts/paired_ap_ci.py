"""Paired block-bootstrap CI for the orchard-AP difference between two frh04 arms (NEW - OLD), per seed.

Arms are probability-cache bases as used by ``evaluate_meta_protocol.py`` (``<base>[_seed1|_seed2]/
proba_pair_compact_m1.npz``); scores = geo_mean blend at m = 1; 10 km blocks of ``frh04_spatial_folds.npz``,
1000 resamples, rng 0 (same as rotation_ap_ci.py / leakage_ap_ci.py).

Example::

    python scripts/paired_ap_ci.py artifacts/pruned_proba_cache_noexpens artifacts/pruned_proba_cache_orchw10dial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agricultureai.meta_features import geometric_mean_probs  # noqa: E402

ORCHARD = 5


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("new", type=Path)
    p.add_argument("old", type=Path)
    p.add_argument("--seeds", default="42,1,2")
    args = p.parse_args()

    folds = np.load(ROOT / "artifacts/frh04_spatial_folds.npz")
    ens = np.load(ROOT / "artifacts/checkpoints/msresnet_l2a_clear_ens3/logits_eval.npz")
    assert np.array_equal(folds["parcel_ids"], ens["parcel_ids"])
    y = (ens["labels"] == ORCHARD).astype(float)
    _, gidx = np.unique(folds["blocks"], return_inverse=True)
    g = gidx.max() + 1
    rng = np.random.default_rng(0)
    weights = [np.bincount(rng.integers(0, g, g), minlength=g)[gidx].astype(float) for _ in range(1000)]

    def prep(sc):
        o = np.argsort(-sc, kind="stable")
        s = sc[o]
        return o, np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]

    def wap(o, last, w):
        yw = (w * y)[o]
        tp, fp = np.cumsum(yw)[last], np.cumsum(w[o] - yw)[last]
        return 100 * np.sum(np.diff(np.r_[0.0, tp / tp[-1]]) * tp / np.maximum(tp + fp, 1e-12))

    for seed in (int(s) for s in args.seeds.split(",")):
        sfx = "" if seed == 42 else f"_seed{seed}"
        pr = {}
        for name, base in (("new", args.new), ("old", args.old)):
            z = np.load(base.with_name(base.name + sfx) / "proba_pair_compact_m1.npz")
            pr[name] = prep(geometric_mean_probs(z["p_xgb"], z["p_cat"])[:, ORCHARD].astype(np.float64))
        one = np.ones_like(y)
        a_new, a_old = wap(*pr["new"], one), wap(*pr["old"], one)
        d = np.array([wap(*pr["new"], w) - wap(*pr["old"], w) for w in weights])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"seed {seed:2d}: AP {args.new.name}={a_new:.2f} {args.old.name}={a_old:.2f} "
              f"diff={a_new - a_old:+.2f} [{lo:+.2f},{hi:+.2f}]", flush=True)


if __name__ == "__main__":
    main()

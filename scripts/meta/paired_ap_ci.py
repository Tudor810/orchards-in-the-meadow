"""Paired block-bootstrap CI for the orchard-AP difference between two arms (NEW - OLD), per seed.

Arms are probability-cache bases as used by ``evaluate_meta_protocol.py``: ``<base>[_seed1|_seed2]/
proba_pair_compact_m1.npz``. Scores are the geo_mean blend at m = 1; resampling is over the 10 km blocks of
``--folds`` (1000 resamples, rng 0; see ``agricultureai.evaluation.bootstrap``). Labels come from ``--labels``
(an eval cache ``.pt`` or any ``.npz`` with ``parcel_ids`` / ``labels``), checked against the folds' parcel ids.

Examples (from the repo root)::

    # frh04: final design vs the end-to-end network with the same decision dial
    python scripts/meta/paired_ap_ci.py artifacts/pruned_proba_cache_noexpens artifacts/pruned_proba_cache_orchw10dial
    # stacking leakage: meta-classifier trained on frh04's own folds vs on frh01-frh03
    python scripts/meta/paired_ap_ci.py artifacts/pruned_proba_cache_noexpensoof artifacts/pruned_proba_cache_noexpens
    # region rotation, test region frh03: arm B vs arm A (no L2A network)
    python scripts/meta/paired_ap_ci.py artifacts/rotation/frh03/proba_B artifacts/rotation/frh03/proba_A \\
        --folds artifacts/rotation/folds_frh03.npz --labels artifacts/rotation/ids_frh03.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from agricultureai.evaluation.bootstrap import paired_ap_ci
from agricultureai.meta_features import geometric_mean_probs
from agricultureai.pipeline import GLOBAL_ORCHARDS


def load_labels(path: Path):
    if path.suffix == ".pt":
        d = torch.load(path, map_location="cpu", weights_only=True)
        return d["parcel_ids"].numpy(), d["labels"].numpy()
    d = np.load(path)
    return d["parcel_ids"], d["labels"]


def orchard_score(base: Path, seed: int) -> np.ndarray:
    sfx = "" if seed == 42 else f"_seed{seed}"
    z = np.load(base.with_name(base.name + sfx) / "proba_pair_compact_m1.npz")
    return geometric_mean_probs(z["p_xgb"], z["p_cat"])[:, GLOBAL_ORCHARDS].astype(np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("new", type=Path)
    p.add_argument("old", type=Path)
    p.add_argument("--folds", type=Path, default=Path("artifacts/frh04_spatial_folds.npz"))
    p.add_argument("--labels", type=Path, default=Path("artifacts/cached_logits_eval.pt"))
    p.add_argument("--seeds", default="42,1,2")
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()

    folds = np.load(args.folds)
    ids, labels = load_labels(args.labels)
    if not np.array_equal(folds["parcel_ids"], ids):
        raise RuntimeError(f"{args.labels} rows do not match {args.folds}")
    y = (labels == GLOBAL_ORCHARDS).astype(np.int8)
    for seed in (int(s) for s in args.seeds.split(",")):
        ap_new, ap_old, lo, hi = paired_ap_ci(y, orchard_score(args.new, seed), orchard_score(args.old, seed),
                                              folds["blocks"], n_boot=args.n_boot)
        print(f"seed {seed:2d}: AP {args.new.name}={ap_new:.2f} {args.old.name}={ap_old:.2f} "
              f"diff={ap_new - ap_old:+.2f} [{lo:+.2f},{hi:+.2f}]", flush=True)


if __name__ == "__main__":
    main()

"""Building blocks of the out-of-sample meta-classifier protocol (``scripts/meta/evaluate_meta_protocol.py``).

* :func:`load_spatial_folds` - 10 km spatial blocks grouped into folds, computed from parcel centroids once
  and cached as ``*_folds.npz``;
* :func:`config_grid` - every (orchard weight m, blend, orchard bias, meadow bias) decode of one seed's
  cached XGBoost / CatBoost probabilities, the grid the selection rules choose from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from agricultureai.evaluation.selection_protocol import spatial_block_folds
from agricultureai.meta_features import apply_orchard_meadow_log_bias, decode_probs, make_blend
from agricultureai.pipeline import GLOBAL_ORCHARDS

CLASS_NAMES = (
    "barley", "wheat", "rapeseed", "corn", "sunflower",
    "orchards", "nuts", "perm_meadow", "temp_meadow",
)
METRIC_CRS = "EPSG:2154"  # Lambert-93, metres


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def load_spatial_folds(
    path: Path,
    parcel_ids: np.ndarray,
    *,
    region: str,
    level: str,
    block_m: float,
    n_folds: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Fold and block ids per cache row; computed from parcel centroids once, then cached."""
    params = {"region": region, "block_m": block_m, "n_folds": n_folds, "seed": seed}
    if path.is_file():
        saved = np.load(path)
        same = json.loads(str(saved["params"])) == params
        if same and np.array_equal(saved["parcel_ids"], parcel_ids):
            return saved["folds"], saved["blocks"], params
        print(f"{path} does not match this cache / parameters; recomputing.")

    import breizhcrops as bzh

    print(f"Loading {region} parcel geometry for spatial folds ...")
    geo = bzh.BreizhCrops(region, level=level, load_timeseries=False).geodataframe()
    geo = geo.loc[~geo.geometry.isna()].set_index("id")
    geo = geo.loc[~geo.index.duplicated()].to_crs(METRIC_CRS)
    cent = geo.geometry.centroid
    lookup_x = dict(zip(geo.index.astype(np.int64), cent.x.to_numpy()))
    lookup_y = dict(zip(geo.index.astype(np.int64), cent.y.to_numpy()))
    x = np.array([lookup_x.get(int(i), np.nan) for i in parcel_ids])
    y = np.array([lookup_y.get(int(i), np.nan) for i in parcel_ids])
    folds, blocks = spatial_block_folds(x, y, block_size=block_m, n_folds=n_folds, seed=seed)
    np.savez(path, parcel_ids=parcel_ids, x=x, y=y, folds=folds, blocks=blocks, params=json.dumps(params))
    print(f"Saved spatial folds -> {path} ({int(np.isnan(x).sum())} parcels without geometry)")
    return folds, blocks, params


def config_grid(proba_dir: Path, args) -> Tuple[List[dict], np.ndarray, Dict[str, np.ndarray]]:
    """Every (m, blend, b_orchard, b_meadow) decode of one seed, in sweep order.

    ``args`` needs ``multipliers``, ``layout``, ``blends``, ``orchard_biases`` and ``meadow_biases`` (strings).
    """
    configs, preds, orchard_scores = [], [], {}
    for m in parse_float_list(args.multipliers):
        pair = np.load(proba_dir / f"proba_pair_{args.layout}_m{m:g}.npz")
        for blend in [b.strip() for b in args.blends.split(",") if b.strip()]:
            proba = make_blend(blend, pair["p_xgb"], pair["p_cat"])
            orchard_scores[f"{blend} m={m:g}"] = proba[:, GLOBAL_ORCHARDS]
            for b_o in parse_float_list(args.orchard_biases):
                for b_m in parse_float_list(args.meadow_biases):
                    biased = apply_orchard_meadow_log_bias(proba, b_orchard=b_o, b_meadow=b_m)
                    preds.append(decode_probs(biased).astype(np.int8))
                    configs.append({"blend": blend, "m": m, "b_orchard": b_o, "b_meadow": b_m})
    return configs, np.stack(preds), orchard_scores


def describe(cfg: dict) -> str:
    return f"{cfg['blend']} m={cfg['m']:g} b_o={cfg['b_orchard']:g} b_m={cfg['b_meadow']:g}"

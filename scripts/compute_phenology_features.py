"""Per-parcel phenology features aimed at telling orchards from meadows.

96% of false orchards are meadows, so these target what differs between grassed
orchards and grassland: winter greenness (tree dormancy vs grass), seasonal
contrast, mowing / grazing (sharp NDVI drops in the growing season), red-edge and
moisture indices, and parcel shape. Only clear observations are used: the L2A
``CLD`` flag is 1 on cloudy dates (about 60% of all observations).

Rows follow the cache's parcel order (BreizhCrops index order per region) and
are written to a separate ``.npz`` (``parcel_ids``, ``labels``, ``features``,
``columns``); the cache itself is not modified. Missing values (e.g. no clear
winter date) are NaN, which XGBoost and CatBoost handle natively.

Example::

    python scripts/compute_phenology_features.py artifacts/cached_logits_eval.pt \\
        --output artifacts/phenology_eval.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import breizhcrops as bzh
import h5py
import numpy as np
import torch
from tqdm import tqdm

from agricultureai.cache_alignment import parcel_ids_and_labels
from agricultureai.data.breizhcrops_series import doa_values_to_doy

METRIC_CRS = "EPSG:2154"
# L2A HDF5 columns (breizhcrops SELECTED_BANDS["L2A"]).
DOA, B4, B5, B8, B8A, B11, CLD = 0, 3, 4, 7, 8, 9, 11
SEASONS = {  # day-of-year windows
    "winter": lambda d: (d >= 335) | (d < 60),
    "spring": lambda d: (d >= 60) & (d < 152),
    "summer": lambda d: (d >= 152) & (d < 244),
    "autumn": lambda d: (d >= 244) & (d < 335),
}
GROWING_SEASON = (91, 274)  # April - September
MOW_DROP = 0.15  # NDVI drop between consecutive clear dates counted as a cut
MOW_MAX_GAP_DAYS = 30
COLUMNS = (
    "ndvi_winter_mean", "ndvi_spring_mean", "ndvi_summer_mean", "ndvi_autumn_mean",
    "ndvi_winter_min", "ndvi_std", "ndvi_summer_minus_winter",
    "mow_drops", "max_ndvi_drop",
    "ndre_spring_mean", "ndre_summer_mean", "ndmi_spring_mean", "ndmi_summer_mean",
    "n_clear_obs", "compactness",
)


def _index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / np.maximum(a + b, 1e-6)


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else np.nan


def parcel_features(raw: np.ndarray) -> np.ndarray:
    """Phenology features of one raw L2A parcel row block ``(T, 14)``; compactness left NaN."""
    out = np.full(len(COLUMNS), np.nan, dtype=np.float32)
    raw = raw[~np.isnan(raw[:, 1:11]).any(axis=1)]
    clear = raw[raw[:, CLD] == 0]
    out[COLUMNS.index("n_clear_obs")] = clear.shape[0]
    if clear.shape[0] == 0:
        return out
    doy = doa_values_to_doy(clear[:, DOA])
    order = np.argsort(doy, kind="stable")
    clear, doy = clear[order], doy[order]
    ndvi = _index(clear[:, B8], clear[:, B4])
    ndre = _index(clear[:, B8A], clear[:, B5])
    ndmi = _index(clear[:, B8], clear[:, B11])

    for name, in_season in SEASONS.items():
        out[COLUMNS.index(f"ndvi_{name}_mean")] = _mean(ndvi[in_season(doy)])
    winter = ndvi[SEASONS["winter"](doy)]
    out[COLUMNS.index("ndvi_winter_min")] = float(winter.min()) if winter.size else np.nan
    out[COLUMNS.index("ndvi_std")] = float(ndvi.std()) if ndvi.size > 1 else np.nan
    out[COLUMNS.index("ndvi_summer_minus_winter")] = (
        out[COLUMNS.index("ndvi_summer_mean")] - out[COLUMNS.index("ndvi_winter_mean")]
    )
    for name in ("spring", "summer"):
        sel = SEASONS[name](doy)
        out[COLUMNS.index(f"ndre_{name}_mean")] = _mean(ndre[sel])
        out[COLUMNS.index(f"ndmi_{name}_mean")] = _mean(ndmi[sel])

    in_growing = (doy >= GROWING_SEASON[0]) & (doy < GROWING_SEASON[1])
    g_ndvi, g_doy = ndvi[in_growing], doy[in_growing]
    if g_ndvi.size > 1:
        drops = g_ndvi[:-1] - g_ndvi[1:]
        close = np.diff(g_doy) <= MOW_MAX_GAP_DAYS
        out[COLUMNS.index("mow_drops")] = float(((drops > MOW_DROP) & close).sum())
        out[COLUMNS.index("max_ndvi_drop")] = float(drops.max())
    else:
        out[COLUMNS.index("mow_drops")] = 0.0
    return out


def region_compactness(region: str, level: str) -> Dict[int, float]:
    """``4 * pi * area / perimeter**2`` per parcel id (1 = circle, small = elongated / ragged)."""
    geo = bzh.BreizhCrops(region, level=level, load_timeseries=False).geodataframe()
    geo = geo.loc[~geo.geometry.isna()].set_index("id")
    geo = geo.loc[~geo.index.duplicated()].to_crs(METRIC_CRS)
    area, perim = geo.geometry.area.to_numpy(), geo.geometry.length.to_numpy()
    value = np.where(perim > 0, 4.0 * np.pi * area / np.maximum(perim, 1e-9) ** 2, np.nan)
    return dict(zip(geo.index.astype(np.int64), value))


def compute_for_cache(cache_path: Path, max_samples: int = 0) -> dict:
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    meta = payload["meta"]
    regions, level = list(meta["regions"]), str(meta.get("level", "L2A"))
    ids, labels = parcel_ids_and_labels(regions, level=level, max_samples=max_samples)
    if not max_samples:
        if not torch.equal(labels, payload["labels"].long()):
            raise RuntimeError("Cache labels do not match BreizhCrops index order.")
        if "parcel_ids" in payload and not torch.equal(ids, payload["parcel_ids"].long()):
            raise RuntimeError("Cache parcel_ids do not match BreizhCrops index order.")

    rows: List[np.ndarray] = []
    compact = COLUMNS.index("compactness")
    for region in regions:
        shape = region_compactness(region, level)
        ds = bzh.BreizhCrops(region, level=level)
        index = ds.index if not max_samples else ds.index.iloc[:max_samples]
        with h5py.File(ds.h5path, "r") as h5:
            for path, pid in tqdm(zip(index["path"], index["id"]), total=len(index),
                                  desc=f"phenology {region}", mininterval=10):
                feats = parcel_features(np.array(h5[path]))
                feats[compact] = shape.get(int(pid), np.nan)
                rows.append(feats)
    return {
        "parcel_ids": ids.numpy(),
        "labels": labels.numpy(),
        "features": np.stack(rows),
        "columns": np.array(COLUMNS),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cache", type=Path, help="Cache whose parcel order to follow.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=0, help="Smoke test: first N parcels per region.")
    p.add_argument("--force", action="store_true", help="Overwrite an existing output.")
    args = p.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} exists; pass --force to overwrite.")
    result = compute_for_cache(args.cache, max_samples=args.max_samples)
    np.savez(args.output, **result)
    feats, y = result["features"], result["labels"]
    print(f"Saved {args.output}: {feats.shape}")
    for c, name in enumerate(COLUMNS):
        col = feats[:, c]
        orch, mead = col[y == 5], col[np.isin(y, (7, 8))]
        print(f"  {name:26s} missing={np.isnan(col).mean():6.1%}  "
              f"orchard={np.nanmean(orch):8.3f}  meadow={np.nanmean(mead):8.3f}")


if __name__ == "__main__":
    main()

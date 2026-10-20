"""Append parcel metadata features to cached DMOE logits (row-aligned).

8 columns::

    0: log1p(area_ha)
    1: max NDVI
    2: NDVI amplitude
    3: centroid latitude (WGS84)
    4: mean DOY / 366
    5: std DOY / 366
    6: DOY at max NDVI / 366
    7: log1p(number of valid, cloud-free observations)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple

import breizhcrops as bzh
import geopandas as gpd
import h5py
import numpy as np
import torch
from tqdm import tqdm

from agricultureai.data.breizhcrops_series import (
    _L1C_ALL_BANDS,
    _L1C_SPECTRAL_BANDS,
    _L2A_ALL_BANDS,
    _L2A_SPECTRAL_BANDS,
    doa_values_to_doy,
)
from agricultureai.meta_features import METADATA_COLS, METADATA_FEATURE_DIM

_REFLECTANCE_SCALE = 1e-4
_METRIC_CRS = "EPSG:2154"
_DOY_SCALE = 366.0


def _band_indices(level: str):
    """Return (spectral_idxs, doa_idx, b4_local, b8_local) for the given level."""
    level = level.upper()
    if level == "L1C":
        all_bands, spectral = _L1C_ALL_BANDS, _L1C_SPECTRAL_BANDS
    else:
        all_bands, spectral = _L2A_ALL_BANDS, _L2A_SPECTRAL_BANDS
    spec_idxs = np.array([all_bands.index(b) for b in spectral], dtype=np.int64)
    doa_idx = all_bands.index("doa")
    b4_local = spectral.index("B4")
    b8_local = spectral.index("B8")
    return spec_idxs, doa_idx, b4_local, b8_local


def _valid_spectral_and_doy(raw: np.ndarray, *, level: str) -> Tuple[np.ndarray, np.ndarray]:
    spec_idxs, doa_idx, _, _ = _band_indices(level)
    spectral = raw[:, spec_idxs].astype(np.float32) * _REFLECTANCE_SCALE
    doa = raw[:, doa_idx]
    valid = ~np.isnan(spectral).any(axis=1) & ~np.isnan(doa)
    return spectral[valid], doa[valid]


def ndvi_and_doy_features(
    raw: np.ndarray, *, level: str = "L2A"
) -> Tuple[float, float, float, float, float, float]:
    """Return (max_ndvi, ndvi_amplitude, mean_doy, std_doy, doy_at_max_ndvi, n_valid).

    ``n_valid`` is the raw count of cloud-free observations; the caller applies
    ``log1p`` before storing it so the feature scale matches ``log1p_area_ha``.
    """
    _, _, b4_local, b8_local = _band_indices(level)
    spectral, doa = _valid_spectral_and_doy(raw, level=level)
    if spectral.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    red = spectral[:, b4_local]
    nir = spectral[:, b8_local]
    ndvi = (nir - red) / (nir + red + 1e-6)
    doy = doa_values_to_doy(doa).astype(np.float32)

    max_ndvi = float(np.max(ndvi))
    min_ndvi = float(np.min(ndvi))
    ndvi_amplitude = max_ndvi - min_ndvi
    mean_doy = float(doy.mean() / _DOY_SCALE)
    std_doy = float(doy.std() / _DOY_SCALE) if doy.size > 1 else 0.0
    doy_at_peak = float(doy[int(np.argmax(ndvi))] / _DOY_SCALE)
    n_valid = float(spectral.shape[0])
    return max_ndvi, ndvi_amplitude, mean_doy, std_doy, doy_at_peak, n_valid


def _area_lookup_for_region(region: str, level: str) -> dict[int, tuple[float, float]]:
    ds = bzh.BreizhCrops(region, level=level, load_timeseries=False, verbose=True)
    print(f"Loading shapefile geometry for {region} (downloads if missing) ...")
    geo = ds.geodataframe()
    if geo.crs is None:
        raise ValueError(f"No CRS on shapefile for region {region}")
    geo = geo.set_index("id")
    geo = geo.loc[~geo.geometry.isna()]
    projected = geo.to_crs(_METRIC_CRS)
    lookup: dict[int, tuple[float, float]] = {}
    for pid, row in projected.iterrows():
        area_ha = float(row.geometry.area) / 10_000.0
        cent = row.geometry.centroid
        cent_wgs = gpd.GeoSeries([cent], crs=_METRIC_CRS).to_crs("EPSG:4326").iloc[0]
        lookup[int(pid)] = (float(np.log1p(max(area_ha, 0.0))), float(cent_wgs.y))
    print(f"  {region}: {len(lookup):,} parcel geometries indexed.")
    return lookup


def extract_metadata_for_region(
    region: str,
    *,
    level: str,
    max_samples: int,
    use_area: bool,
    use_doy: bool,
) -> np.ndarray:
    if not use_area:
        raise ValueError("Full metadata pipeline requires --with-area.")

    ds = bzh.BreizhCrops(region, level=level)
    indices = list(range(len(ds)))
    if max_samples > 0:
        indices = indices[: min(max_samples, len(indices))]

    area_lookup = _area_lookup_for_region(region, level)
    rows = np.zeros((len(indices), METADATA_FEATURE_DIM), dtype=np.float32)
    with h5py.File(ds.h5path, "r") as h5:
        for out_i, base_idx in enumerate(tqdm(indices, desc=f"Metadata {region}")):
            row = ds.index.iloc[int(base_idx)]
            pid = int(row.id)
            log_area, lat = area_lookup.get(pid, (0.0, 0.0))
            rows[out_i, 0] = log_area
            rows[out_i, 3] = lat

            raw = np.array(h5[row.path])
            max_ndvi, ndvi_amp, mean_doy, std_doy, doy_peak, n_valid = ndvi_and_doy_features(
                raw, level=level
            )
            rows[out_i, 1] = max_ndvi
            rows[out_i, 2] = ndvi_amp
            rows[out_i, 7] = float(np.log1p(max(n_valid, 0.0)))
            if use_doy:
                rows[out_i, 4] = mean_doy
                rows[out_i, 5] = std_doy
                rows[out_i, 6] = doy_peak
    return rows


def patch_cache(
    cache_path: Path,
    *,
    level: str,
    max_samples: int,
    use_area: bool,
    use_doy: bool,
) -> None:
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    meta = payload.get("meta")
    if not isinstance(meta, dict) or "regions" not in meta:
        raise KeyError(f"Cache {cache_path} missing meta['regions'].")

    regions = list(meta["regions"])
    expected_n = int(payload["labels"].numel())
    chunks = [
        extract_metadata_for_region(
            region,
            level=str(meta.get("level", level)),
            max_samples=max_samples,
            use_area=use_area,
            use_doy=use_doy,
        )
        for region in regions
    ]
    matrix = np.concatenate(chunks, axis=0)
    if matrix.shape != (expected_n, METADATA_FEATURE_DIM):
        raise RuntimeError(
            f"Metadata shape {matrix.shape} != expected ({expected_n}, {METADATA_FEATURE_DIM})"
        )

    payload["parcel_metadata"] = torch.from_numpy(matrix)
    meta = dict(meta)
    meta["parcel_metadata"] = datetime.now().isoformat(timespec="seconds")
    meta["parcel_metadata_cols"] = list(METADATA_COLS)
    payload["meta"] = meta
    torch.save(payload, cache_path)
    print(
        f"Patched parcel_metadata {matrix.shape} -> {cache_path}  "
        f"(area mean={matrix[:, 0].mean():.3f}, ndvi mean={matrix[:, 1].mean():.3f})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append parcel metadata to logit caches.")
    parser.add_argument("caches", nargs="*", type=Path)
    parser.add_argument("--level", default="L2A")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--with-area",
        action="store_true",
        help="Include shapefile area + centroid lat.",
    )
    parser.add_argument(
        "--with-doy",
        action="store_true",
        help="Include DOY calendar features (requires --with-area).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.with_doy and not args.with_area:
        raise SystemExit("--with-doy requires --with-area.")
    caches = args.caches or [
        Path("artifacts/cached_logits_train.pt"),
        Path("artifacts/cached_logits_eval.pt"),
    ]
    for cache in caches:
        patch_cache(
            cache.resolve(),
            level=args.level,
            max_samples=int(args.max_samples),
            use_area=bool(args.with_area),
            use_doy=bool(args.with_doy),
        )


if __name__ == "__main__":
    main()

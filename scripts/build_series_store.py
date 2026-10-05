"""Pack every parcel's L2A series and a per-date clear-sky flag into one ``.npz``.

Training on the full series from HDF5 reads ~900 parcels/s (about 9 minutes per
epoch on frh01-frh03); this store holds the same rows in memory-friendly ragged
arrays, in BreizhCrops index order (the cache row order):

    offsets   (N + 1,) int64    parcel i owns rows offsets[i]:offsets[i+1]
    spectral  (R, 10)  float32  reflectance (x 1e-4), NaN rows dropped
    doy       (R,)     int16    day of year, sorted within each parcel
    clear     (R,)     bool     ``clear_observation_mask`` (cloud fraction,
                                brightness and NDVI-spike tests)
    labels, parcel_ids (N,)

Example::

    python scripts/build_series_store.py frh01 --output artifacts/series_store/frh01.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import breizhcrops as bzh
import h5py
import numpy as np
from tqdm import tqdm

from agricultureai.cache_alignment import parcel_ids_and_labels
from agricultureai.data.breizhcrops_series import doa_values_to_doy
from agricultureai.data.cloud_mask import L2A_DOA, L2A_SPECTRAL, clear_observation_mask

MASK = dict(max_cloud=0.1, max_blue=1200.0, max_edge=0.5, spike=0.15, spike_gap_days=20)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("region")
    p.add_argument("--level", default="L2A")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; delete it first.")

    ids, labels = parcel_ids_and_labels([args.region], level=args.level, max_samples=args.max_samples)
    ds = bzh.BreizhCrops(args.region, level=args.level)
    paths = ds.index["path"].tolist()[: len(ids)]
    offsets, spectral, doy, clear = [0], [], [], []
    with h5py.File(ds.h5path, "r") as h5:
        for path in tqdm(paths, desc=f"store {args.region}", mininterval=10):
            raw = np.array(h5[path])
            raw = raw[~np.isnan(raw[:, L2A_SPECTRAL]).any(axis=1)]
            mask = clear_observation_mask(raw, **MASK)
            d = doa_values_to_doy(raw[:, L2A_DOA]) if len(raw) else np.zeros(0, np.int64)
            order = np.argsort(d, kind="stable")
            spectral.append((raw[order, L2A_SPECTRAL] * 1e-4).astype(np.float32))
            doy.append(d[order].astype(np.int16))
            clear.append(mask[order])
            offsets.append(offsets[-1] + len(raw))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        offsets=np.asarray(offsets, dtype=np.int64),
        spectral=np.concatenate(spectral),
        doy=np.concatenate(doy),
        clear=np.concatenate(clear),
        labels=labels.numpy(),
        parcel_ids=ids.numpy(),
        mask_params=str(MASK),
    )
    n_clear = np.add.reduceat(np.concatenate(clear).astype(np.int64), np.asarray(offsets[:-1])) if len(ids) else []
    print(f"Saved {args.output}: {len(ids):,} parcels, {offsets[-1]:,} dates, "
          f"{np.concatenate(clear).mean():.1%} clear, median clear dates/parcel {np.median(n_clear):.0f}")


if __name__ == "__main__":
    main()

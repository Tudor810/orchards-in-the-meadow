"""Summary of the BreizhCrops data used in the paper (Section 3.1, Table 1): parcels and classes per region,
series lengths, clear dates, parcels without an L1C series, orchards per rotation fold and, with ``--raw``, the
share of L2A observations with a non-zero cloud fraction (reads the raw HDF5 files; slow).

Run from the repo root after ``build_series_store.py`` (L2A for all regions; L1C optional)::

    python scripts/data/dataset_summary.py --raw --output artifacts/dataset_summary.log
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REGIONS = ("frh01", "frh02", "frh03", "frh04")
TRAIN, TEST = REGIONS[:3], REGIONS[3:]
CLASSES = ("barley", "wheat", "rapeseed", "corn", "sunflower", "orchards", "nuts",
           "permanent meadows", "temporary meadows")
ORCHARD = 5
MAX_DATES = 45


def cloudy_share(region: str) -> tuple[int, int]:
    """(observations with CLD > 0, valid observations) over the raw L2A rows of ``region``."""
    import breizhcrops as bzh
    import h5py

    from agricultureai.data.cloud_mask import L2A_CLD

    ds = bzh.BreizhCrops(region, level="L2A")
    cloudy = total = 0
    with h5py.File(ds.h5path, "r") as h5:
        for path in ds.index["path"]:
            raw = np.array(h5[path])
            raw = raw[~np.isnan(raw[:, 1:11]).any(axis=1)]  # dates with a missing band are dropped, as in BreizhCrops
            cloudy += int((raw[:, L2A_CLD] > 0).sum())
            total += len(raw)
    return cloudy, total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store-dir", type=Path, default=Path("artifacts/series_store"))
    p.add_argument("--l1c-store-dir", type=Path, default=Path("artifacts/series_store_l1c"))
    p.add_argument("--raw", action="store_true", help="Also read the raw L2A files for the cloud statistics.")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    lines = []

    def out(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    stores = {r: np.load(args.store_dir / f"{r}.npz") for r in REGIONS}
    labels = {r: stores[r]["labels"] for r in REGIONS}
    lengths = {r: np.diff(stores[r]["offsets"]) for r in REGIONS}
    clear = {r: stores[r]["clear"] for r in REGIONS}

    out(f"# BreizhCrops L2A, from {args.store_dir}/<region>.npz (scripts/data/build_series_store.py)")
    out("\n## Parcels, orchards and series per region")
    out(f"{'region':8s} {'parcels':>9s} {'orchards':>9s} {'orch.%':>7s} {'dates median':>13s} {'<45 dates':>10s} {'clear dates':>12s}")
    for r in REGIONS:
        n, o = len(labels[r]), int((labels[r] == ORCHARD).sum())
        out(f"{r:8s} {n:9,d} {o:9,d} {100 * o / n:6.2f}% {np.median(lengths[r]):13.0f} "
            f"{100 * (lengths[r] < MAX_DATES).mean():9.1f}% {100 * clear[r].mean():11.1f}%")
    for name, regs in (("train frh01-frh03", TRAIN), ("test frh04", TEST), ("total", REGIONS)):
        out(f"{name}: {sum(len(labels[r]) for r in regs):,} parcels")

    out("\n## Classes (training = frh01-frh03, test = frh04)")
    out(f"{'class':20s} {'train':>9s} {'train%':>7s} {'test':>8s} {'test%':>7s}")
    y_tr = np.concatenate([labels[r] for r in TRAIN])
    y_te = np.concatenate([labels[r] for r in TEST])
    for c, name in enumerate(CLASSES):
        a, b = int((y_tr == c).sum()), int((y_te == c).sum())
        out(f"{name:20s} {a:9,d} {100 * a / len(y_tr):6.2f}% {b:8,d} {100 * b / len(y_te):6.2f}%")
    out(f"{'total':20s} {len(y_tr):9,d} {100.0:6.2f}% {len(y_te):8,d} {100.0:6.2f}%")

    out("\n## Orchards per region and per rotation fold (training orchards when that region is held out)")
    orch = {r: int((labels[r] == ORCHARD).sum()) for r in REGIONS}
    tot = sum(orch.values())
    for r in REGIONS:
        out(f"{r}: {orch[r]:,} orchards ({orch[r] / tot:.1%} of {tot:,}); held out -> {tot - orch[r]:,} training orchards")

    if args.l1c_store_dir.is_dir():
        out("\n## Parcels without an L1C series (L1C-network features set to missing)")
        for name, regs in (("train", TRAIN), ("test", TEST)):
            missing = 0
            for r in regs:
                l1c = np.load(args.l1c_store_dir / f"{r}.npz")
                has = np.isin(stores[r]["parcel_ids"], l1c["parcel_ids"][np.diff(l1c["offsets"]) > 0])
                missing += int((~has).sum())
            out(f"{name}: {missing:,}")

    if args.raw:
        out("\n## Raw L2A observations with a non-zero cloud fraction (CLD > 0)")
        for r in REGIONS:
            cloudy, total = cloudy_share(r)
            out(f"{r}: {cloudy:,} of {total:,} ({100 * cloudy / total:.1f}%)")

    if args.output:
        args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

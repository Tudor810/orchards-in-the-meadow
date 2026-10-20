"""Recompute the pretrained MSResNet / Transformer logits in DMOE logit caches.

Why: ``msresnet_logits`` and ``transformer_logits`` come from the pretrained
BreizhCrops models, trained on L1C through the stock transform (every parcel
resampled to exactly 45 steps, never zero-padded). They were extracted with
``BreizhCropsParcelDataset`` (first 45 steps, then zeros) and copied into the
L2A caches by parcel id. This script re-extracts them on L1C from seeded
resampled views (mean of the model outputs over ``--draws`` views), aligns them
to the cache rows by parcel id (rows without an L1C parcel get zero logits, as
before) and replaces only the keys of ``--models``.

Per-region logits are stored in ``--logits-dir`` and reused, so cache variants
with different ``--models`` cost no extra inference. Output goes to
``<stem><suffix>.pt`` with ``--output-suffix`` (never overwriting), or in place
after a verified backup (``<name>.pre_global_patch.pt``, kept if present).
``--compare-only`` reports old vs new accuracy and writes no cache.

Example::

    python scripts/cascade/patch_global_logits.py --models msresnet --output-suffix _msfix \\
        artifacts/cached_logits_train.pt artifacts/cached_logits_eval.pt
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import zlib
from datetime import datetime
from pathlib import Path
from typing import Dict

import breizhcrops as bzh
import h5py
import numpy as np
import torch
from tqdm import tqdm

from agricultureai.cache_alignment import align_logits_by_parcel_id, parcel_ids_and_labels
from agricultureai.data.breizhcrops_series import resample_series_views, split_raw_series
from agricultureai.evaluation.transformer import MAX_SEQ_LEN

MODELS = {"msresnet": "MSResNet", "transformer": "Transformer"}
BACKUP_SUFFIX = ".pre_global_patch.pt"
GLOBAL_ORCHARDS, GLOBAL_WHEAT = 5, 1


@torch.inference_mode()
def region_logits(
    models: Dict[str, torch.nn.Module],
    region: str,
    *,
    level: str,
    method: str,
    draws: int,
    seed: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    max_samples: int = 0,
) -> Dict[str, np.ndarray]:
    """Mean model output over resampled views for every parcel of ``region``, in index order."""
    ds = bzh.BreizhCrops(region, level=level)
    paths = ds.index["path"].tolist()
    if max_samples > 0:
        paths = paths[:max_samples]
    rng = np.random.default_rng([seed, zlib.crc32(region.encode())])
    out = {name: [] for name in models}
    batch = []

    def flush() -> None:
        views = torch.from_numpy(np.stack(batch)).to(device)  # (B, V, T, C)
        b, v = views.shape[:2]
        flat = views.reshape(b * v, *views.shape[2:])
        for name, model in models.items():
            out[name].append(model(flat).float().view(b, v, -1).mean(dim=1).cpu().numpy())
        batch.clear()

    with h5py.File(ds.h5path, "r") as h5:
        for path in tqdm(paths, desc=f"{'+'.join(models)} {level} {region}", mininterval=10):
            spectral, _ = split_raw_series(np.array(h5[path]), level)
            batch.append(resample_series_views(spectral, seq_len, method=method, draws=draws, rng=rng))
            if len(batch) >= batch_size:
                flush()
    if batch:
        flush()
    return {name: np.concatenate(chunks) for name, chunks in out.items()}


def load_or_compute_region(region: str, args, models_loader) -> Dict[str, np.ndarray]:
    """Per-region ids, labels and logits of every model; cached in ``--logits-dir``."""
    tag = f"{args.source_level}_{region}_{args.method}{args.draws}_seed{args.seed}"
    if args.max_samples:
        tag += f"_first{args.max_samples}"
    path = args.logits_dir / f"{tag}.npz"
    ids, labels = parcel_ids_and_labels([region], level=args.source_level, max_samples=args.max_samples)
    if path.is_file():
        saved = np.load(path)
        if np.array_equal(saved["ids"], ids.numpy()) and all(m in saved for m in MODELS):
            print(f"Loaded {path}")
            return {k: saved[k] for k in saved.files}
        print(f"{path} does not match the current index; recomputing.")
    logits = region_logits(
        models_loader(),
        region,
        level=args.source_level,
        method=args.method,
        draws=args.draws,
        seed=args.seed,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=args.device,
        max_samples=args.max_samples,
    )
    result = {"ids": ids.numpy(), "labels": labels.numpy(), **logits}
    args.logits_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, **result)
    print(f"Saved {path}")
    return result


def summarize(logits: torch.Tensor, labels: torch.Tensor, matched: torch.Tensor) -> str:
    pred, y = logits[matched].argmax(dim=1), labels[matched]
    recall = lambda c: 100.0 * float((pred[y == c] == c).float().mean())  # noqa: E731
    return (f"acc={100.0 * float((pred == y).float().mean()):6.2f}%  "
            f"wheat={recall(GLOBAL_WHEAT):6.2f}%  orchard={recall(GLOBAL_ORCHARDS):6.2f}%")


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def backup_once(cache_path: Path) -> Path:
    backup = cache_path.with_name(cache_path.stem + BACKUP_SUFFIX)
    if backup.exists():
        print(f"Backup already exists, keeping it: {backup}")
        return backup
    shutil.copy2(cache_path, backup)
    if sha256_file(backup) != sha256_file(cache_path):
        raise RuntimeError(f"Backup verification failed for {backup}")
    print(f"Backup (sha256 verified): {backup}")
    return backup


def output_path(cache_path: Path, suffix: str) -> Path:
    return cache_path.with_name(f"{cache_path.stem}{suffix}{cache_path.suffix}") if suffix else cache_path


def patch_cache(cache_path: Path, args, models_loader) -> None:
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    meta = dict(payload["meta"])
    regions, level = list(meta["regions"]), str(meta.get("level", "L2A"))
    target_ids, target_labels = parcel_ids_and_labels(regions, level=level, max_samples=args.max_samples)
    if not torch.equal(target_labels, payload["labels"].long()):
        raise RuntimeError("Cache labels do not match BreizhCrops index order.")
    if "parcel_ids" in payload and not torch.equal(target_ids, payload["parcel_ids"].long()):
        raise RuntimeError("Cache parcel_ids do not match BreizhCrops index order.")

    per_region = [load_or_compute_region(r, args, models_loader) for r in regions]
    source_ids = torch.from_numpy(np.concatenate([r["ids"] for r in per_region]))
    source_labels = torch.from_numpy(np.concatenate([r["labels"] for r in per_region]))

    print(f"\n{cache_path.name}: {target_labels.numel():,} rows")
    for name in args.models:
        source = torch.from_numpy(np.concatenate([r[name] for r in per_region]))
        aligned, matched = align_logits_by_parcel_id(source, source_labels, source_ids, target_ids, target_labels)
        old_matched = payload.get(f"{name}_l1c_matched")
        if old_matched is not None and not torch.equal(old_matched.bool(), matched):
            raise RuntimeError(f"{name}: parcel-id match mask differs from the cached one.")
        print(f"  {name:11s} cached     {summarize(payload[f'{name}_logits'].float(), target_labels, matched)}")
        print(f"  {name:11s} {args.method:10s} {summarize(aligned, target_labels, matched)}")
        payload[f"{name}_logits"] = aligned
        payload[f"{name}_l1c_matched"] = matched
        meta[name] = f"pretrained_{args.source_level}_{args.method}{args.draws}_aligned"
        meta[f"{name}_sampling"] = {"method": args.method, "draws": args.draws, "seed": args.seed}
        meta[f"{name}_patched"] = datetime.now().isoformat(timespec="seconds")
    if args.compare_only:
        return

    target = output_path(cache_path, args.output_suffix)
    if target == cache_path:
        meta["global_logits_backup"] = str(backup_once(cache_path))
    else:
        meta["global_logits_patch_source"] = str(cache_path)
    payload["meta"] = meta
    tmp = target.with_name(target.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)
    print(f"Patched {', '.join(args.models)} -> {target}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("caches", nargs="+", type=Path, help="L2A cached_logits_*.pt to patch.")
    p.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    p.add_argument("--source-level", default="L1C", help="Level the pretrained models expect.")
    p.add_argument("--method", choices=("tta", "random", "linspace", "zero_pad"), default="tta")
    p.add_argument("--draws", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--batch-size", type=int, default=256, help="Parcels per forward pass.")
    p.add_argument("--max-samples", type=int, default=0, help="Must match the original extraction.")
    p.add_argument("--logits-dir", type=Path, default=Path("artifacts/global_logits_tta"))
    p.add_argument("--output-suffix", default="")
    p.add_argument("--compare-only", action="store_true")
    args = p.parse_args()
    if args.draws < 1:
        p.error("--draws must be >= 1.")
    if any(sep in args.output_suffix for sep in ("/", "\\")):
        p.error("--output-suffix must not contain path separators.")
    return args


def main() -> None:
    args = parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {args.device}")
    caches = [c.resolve() for c in args.caches]
    if args.output_suffix and not args.compare_only:
        existing = [str(output_path(c, args.output_suffix)) for c in caches if output_path(c, args.output_suffix).exists()]
        if existing:
            raise SystemExit(f"Refusing to overwrite existing outputs: {existing}")

    loaded: Dict[str, torch.nn.Module] = {}

    def models_loader() -> Dict[str, torch.nn.Module]:
        # Always compute every model so the stored region logits serve any --models choice.
        if not loaded:
            for name, bzh_name in MODELS.items():
                model = bzh.models.pretrained(bzh_name, device=args.device)
                model.eval()
                loaded[name] = model
        return loaded

    for cache in caches:
        patch_cache(cache, args, models_loader)


if __name__ == "__main__":
    main()

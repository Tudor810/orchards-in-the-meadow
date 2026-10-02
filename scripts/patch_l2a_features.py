"""Recompute ``l2a_logits`` in DMOE logit caches with training-consistent sampling.

Why: the L2a cereal LTAE (``train_expert_cereals.py``) is trained on
``Subset(bzh.BreizhCrops)``, i.e. the stock BreizhCrops transform, which
resamples every parcel to exactly 45 time steps and never zero-pads. Cache
extraction (``sweep_dmoe_parameters.py`` via ``BreizhCropsParcelDataset``)
instead truncates to the first 45 steps and zero-pads shorter series, which the
unmasked LTAE never saw. Only ``l2a_logits`` is recomputed; every other key is
left untouched.

Sampling (``--method``):
    tta       mean logits over ``--draws`` seeded random resamples (default)
    random    one seeded random resample, as in training/validation
    linspace  deterministic evenly spaced resample

Caches are rewritten in place after a verified byte-identical backup
(``<name>.pre_l2a_patch.pt``; an existing backup is kept, never overwritten),
or written to ``--output``. ``--compare-only`` reports L2a accuracy for every
method and writes nothing.

Example::

    python scripts/patch_l2a_features.py --compare-only artifacts/cached_logits_eval.pt
    python scripts/patch_l2a_features.py artifacts/cached_logits_eval.pt artifacts/cached_logits_train.pt
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import breizhcrops as bzh
import h5py
import numpy as np
import torch
from tqdm import tqdm

from agricultureai.cache_alignment import parcel_ids_and_labels
from agricultureai.data.breizhcrops_series import resample_indices, split_raw_series
from agricultureai.evaluation.transformer import MAX_SEQ_LEN

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from evaluate_dmoe_pipeline import load_l2a_cereal  # noqa: E402

DEFAULT_WEIGHTS = Path("artifacts/checkpoints/expert_cereals/best.pt")
CEREAL_LABELS = (0, 1)  # barley, wheat == L2a local ids


@torch.inference_mode()
def compute_l2a_logits(
    model: torch.nn.Module,
    regions: list[str],
    level: str,
    *,
    with_linspace: bool,
    draws: int,
    seed: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    max_samples: int = 0,
) -> Dict[str, Optional[np.ndarray]]:
    """Run L2a on resampled series in cache extraction order.

    Returns ``linspace`` ``(N, 2)``, ``draws`` ``(N, draws, 2)`` and ``n_steps`` ``(N,)``.
    """
    rng = np.random.default_rng(seed)
    variants = int(with_linspace) + draws
    lin_out, draw_out, n_steps = [], [], []
    batch: list[np.ndarray] = []
    empty = 0

    def flush() -> None:
        x = torch.from_numpy(np.stack(batch)).to(device)
        logits = model(x).float().cpu().numpy().reshape(-1, variants, 2)
        if with_linspace:
            lin_out.append(logits[:, 0])
        if draws:
            draw_out.append(logits[:, int(with_linspace):])
        batch.clear()

    for region in regions:
        ds = bzh.BreizhCrops(region, level=level)
        paths = ds.index["path"].tolist()
        if max_samples > 0:
            paths = paths[:max_samples]
        with h5py.File(ds.h5path, "r") as h5:
            for path in tqdm(paths, desc=f"L2a {region}"):
                spectral, _ = split_raw_series(np.array(h5[path]), level)
                t = spectral.shape[0]
                n_steps.append(t)
                if t == 0:
                    empty += 1
                    batch.extend([np.zeros((seq_len, spectral.shape[1]), np.float32)] * variants)
                else:
                    if with_linspace:
                        batch.append(spectral[resample_indices(t, seq_len, method="linspace")])
                    for _ in range(draws):
                        batch.append(spectral[resample_indices(t, seq_len, method="random", rng=rng)])
                if len(batch) >= batch_size * variants:
                    flush()
    if batch:
        flush()
    if empty:
        print(f"  {empty} parcels had no valid observations; fed all-zero input.")
    return {
        "linspace": np.concatenate(lin_out) if with_linspace else None,
        "draws": np.concatenate(draw_out) if draws else None,
        "n_steps": np.asarray(n_steps),
    }


def cereal_metrics(logits: np.ndarray, labels: np.ndarray, n_steps: np.ndarray, seq_len: int):
    mask = np.isin(labels, CEREAL_LABELS)
    pred, y, t = logits[mask].argmax(1), labels[mask], n_steps[mask]
    short = t < seq_len
    pct = lambda m: 100.0 * float((pred[m] == y[m]).mean()) if m.any() else float("nan")
    return {
        "cereal_acc": pct(np.ones_like(y, dtype=bool)),
        "short_acc": pct(short),
        "long_acc": pct(~short),
        "barley_recall": pct(y == 0),
        "wheat_recall": pct(y == 1),
        "n_short": int(short.sum()),
        "n_long": int((~short).sum()),
    }


def print_comparison(rows: Dict[str, dict], seq_len: int) -> None:
    first = next(iter(rows.values()))
    print(
        f"\nL2a accuracy on cereal parcels (T<{seq_len}: {first['n_short']:,}, "
        f"T>={seq_len}: {first['n_long']:,}):"
    )
    print(f"  {'method':28s} {'cereal':>7s} {'T<' + str(seq_len):>7s} {'T>=' + str(seq_len):>7s} {'barley':>7s} {'wheat':>7s}")
    for name, r in rows.items():
        print(
            f"  {name:28s} {r['cereal_acc']:6.2f}% {r['short_acc']:6.2f}% {r['long_acc']:6.2f}% "
            f"{r['barley_recall']:6.2f}% {r['wheat_recall']:6.2f}%"
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def backup_once(cache_path: Path) -> Path:
    backup = cache_path.with_name(cache_path.stem + ".pre_l2a_patch.pt")
    if backup.exists():
        print(f"Backup already exists, keeping it: {backup}")
        return backup
    shutil.copy2(cache_path, backup)
    if sha256_file(backup) != sha256_file(cache_path):
        raise RuntimeError(f"Backup verification failed for {backup}")
    print(f"Backup (sha256 verified): {backup}")
    return backup


def patch_cache(cache_path: Path, args: argparse.Namespace, model, device) -> None:
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    meta = payload.get("meta")
    if not isinstance(meta, dict) or "regions" not in meta:
        raise KeyError(f"Cache {cache_path} is missing meta['regions'].")
    regions = list(meta["regions"])
    level = str(meta.get("level", "L2A"))
    labels = payload["labels"].long()

    ids, order_labels = parcel_ids_and_labels(regions, level=level, max_samples=args.max_samples)
    if not torch.equal(order_labels, labels):
        raise RuntimeError("Cache labels do not match BreizhCrops index order.")
    if "parcel_ids" in payload and not torch.equal(ids, payload["parcel_ids"].long()):
        raise RuntimeError("Cache parcel_ids do not match BreizhCrops index order.")

    compare = bool(args.compare_only)
    out = compute_l2a_logits(
        model,
        regions,
        level,
        with_linspace=compare or args.method == "linspace",
        draws=max(args.draws, 1) if compare or args.method == "tta" else int(args.method == "random"),
        seed=args.seed,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
        max_samples=args.max_samples,
    )
    n_steps, y = out["n_steps"], labels.numpy()
    if n_steps.size != y.size:
        raise RuntimeError(f"Row count mismatch: cache {y.size}, recomputed {n_steps.size}.")

    candidates = {"zero_pad (current cache)": payload["l2a_logits"].float().numpy()}
    if out["linspace"] is not None:
        candidates["linspace"] = out["linspace"]
    if out["draws"] is not None:
        candidates["random (1 draw)"] = out["draws"][:, 0]
        if out["draws"].shape[1] > 1:
            candidates[f"tta (mean of {out['draws'].shape[1]} draws)"] = out["draws"].mean(axis=1)
    print_comparison(
        {k: cereal_metrics(v, y, n_steps, args.seq_len) for k, v in candidates.items()},
        args.seq_len,
    )
    if compare:
        return

    new_logits = {"tta": out["draws"].mean(axis=1) if out["draws"] is not None else None,
                  "random": out["draws"][:, 0] if out["draws"] is not None else None,
                  "linspace": out["linspace"]}[args.method]
    payload["l2a_logits"] = torch.from_numpy(np.ascontiguousarray(new_logits, dtype=np.float32))
    meta = dict(meta)
    meta.update(
        {
            "l2a_sampling": args.method,
            "l2a_draws": int(args.draws) if args.method == "tta" else 1,
            "l2a_seed": int(args.seed),
            "l2a_weights": str(Path(args.l2a_weights).resolve()),
            "l2a_patched": datetime.now().isoformat(timespec="seconds"),
        }
    )

    target = Path(args.output).resolve() if args.output else cache_path
    if target == cache_path:
        meta["l2a_backup"] = str(backup_once(cache_path))
    payload["meta"] = meta
    tmp = target.with_name(target.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)
    print(f"Patched l2a_logits ({args.method}) for {y.size:,} parcels -> {target}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("caches", nargs="+", type=Path, help="cached_logits_*.pt files to patch.")
    p.add_argument("--l2a-weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--method", choices=("tta", "random", "linspace"), default="tta")
    p.add_argument("--draws", type=int, default=8, help="Random resamples averaged by --method tta.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--batch-size", type=int, default=512, help="Parcels per forward pass.")
    p.add_argument("--max-samples", type=int, default=0, help="Must match the original extraction.")
    p.add_argument("--output", type=Path, default=None, help="Write here instead of in place (single cache).")
    p.add_argument("--compare-only", action="store_true", help="Report accuracies; write nothing.")
    args = p.parse_args()
    if args.output and len(args.caches) != 1:
        p.error("--output requires exactly one cache.")
    if args.draws < 1:
        p.error("--draws must be >= 1.")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = load_l2a_cereal(args.l2a_weights.resolve(), device, max_seq_len=args.seq_len)
    for cache_path in args.caches:
        print(f"\n=== {cache_path.resolve()} ===")
        patch_cache(cache_path.resolve(), args, model, device)


if __name__ == "__main__":
    main()

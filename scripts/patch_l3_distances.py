"""Recompute ``l3_distances`` in an existing DMOE logit cache.

Preserves MSResNet / Transformer keys and only updates L3-related tensors.
Row order matches ``sweep_dmoe_parameters.py`` extraction (verified against the
BreizhCrops index before anything is written).

Sampling (``--sampling``): the L3 ProtoNet and linear head are trained through
``Level3FewShotDataset``, i.e. the stock BreizhCrops transform, which resamples
every parcel to exactly 45 time steps and never zero-pads.

    tta       mean distance over ``--draws`` seeded random resamples (default);
              each support parcel contributes the same number of draws
    random    one seeded random resample (support and queries)
    linspace  deterministic evenly spaced resample (support and queries)
    legacy    previous behaviour: support through the stock transform
              (unseeded, so prototypes change between runs), queries truncated
              to the first 45 steps and zero-padded

Prototypes are locked once per run and shared by every cache in it, so train and
eval features are measured against the same prototypes.

Caches are rewritten in place after a verified byte-identical backup
(``<name>.pre_l3_patch.pt``; an existing backup is kept, never overwritten), or
written next to the source as ``<stem><suffix>.pt`` with ``--output-suffix``
(an existing output is never overwritten). ``--compare-only`` reports L3
accuracy for the cached and recomputed distances and writes nothing.

Example::

    python scripts/patch_l3_distances.py --output-suffix _l3fix \\
        --l3-weights artifacts/checkpoints/level3_protonet_l2a_episodic/best.pt \\
        artifacts/cached_logits_train.pt artifacts/cached_logits_eval.pt
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Tuple

import breizhcrops as bzh
import h5py
import numpy as np
import torch
from tqdm import tqdm

from agricultureai.cache_alignment import parcel_ids_and_labels
from agricultureai.data.breizhcrops_series import resample_series_views, split_raw_series
from agricultureai.data.episodic_sampler import (
    LEVEL3_ID_TO_NAME,
    NUM_LEVEL3_CLASSES,
    Level3FewShotDataset,
    build_reference_support_pool,
    build_resampled_support_pool,
)
from agricultureai.evaluation.transformer import MAX_SEQ_LEN
from agricultureai.models.proto_net import (
    Level3LinearExpert,
    PrototypicalNetwork,
    squared_mahalanobis_distance,
)
from agricultureai.pipeline import L3_LOCAL_TO_GLOBAL
from agricultureai.training.checkpoints import (
    architecture_config_from_checkpoint,
    load_checkpoint,
    model_state_dict_from_checkpoint,
)

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from evaluate_dmoe_pipeline import load_l3_protonet  # noqa: E402

SAMPLINGS = ("tta", "random", "linspace", "legacy")
BACKUP_SUFFIX = ".pre_l3_patch.pt"
# Rewritten on every patch so keys from an earlier mode/sampling never linger.
L3_META_KEYS = (
    "l3_weights",
    "l3_linear_weights",
    "l3_sampling",
    "l3_draws",
    "l3_sampling_seed",
    "l3_backup",
    "l3_source",
)

DistanceFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class L3Scorer:
    """``(B, T, C) -> (B, 5)`` distances plus the cache entries describing them."""

    distance_fn: DistanceFn
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
    drop_keys: Tuple[str, ...] = ()
    meta: dict = field(default_factory=dict)


def view_method(sampling: str) -> str:
    """Per-parcel view method for ``--sampling`` (legacy queries are zero-padded)."""
    return "zero_pad" if sampling == "legacy" else sampling


@torch.inference_mode()
def score_views(
    distance_fn: DistanceFn,
    views: list[np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    """Distances for per-parcel views ``[(V, T, C), ...]``, averaged over the V views."""
    x = torch.from_numpy(np.concatenate(views)).to(device)
    distances = distance_fn(x).float().cpu()
    return distances.view(len(views), views[0].shape[0], -1).mean(dim=1)


def extract_l3_distances(
    distance_fn: DistanceFn,
    *,
    regions: list[str],
    level: str,
    sampling: str,
    draws: int,
    sampling_seed: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    max_samples: int = 0,
) -> torch.Tensor:
    """L3 distances for every parcel, in cache extraction order."""
    method = view_method(sampling)
    rng = np.random.default_rng(sampling_seed)
    chunks: list[torch.Tensor] = []
    batch: list[np.ndarray] = []
    for region in regions:
        ds = bzh.BreizhCrops(region, level=level)
        paths = ds.index["path"].tolist()
        if max_samples > 0:
            paths = paths[:max_samples]
        with h5py.File(ds.h5path, "r") as h5:
            for path in tqdm(paths, desc=f"L3 {sampling} {region}"):
                spectral, _ = split_raw_series(np.array(h5[path]), level)
                batch.append(
                    resample_series_views(spectral, seq_len, method=method, draws=draws, rng=rng)
                )
                if len(batch) >= batch_size:
                    chunks.append(score_views(distance_fn, batch, device))
                    batch.clear()
    if batch:
        chunks.append(score_views(distance_fn, batch, device))
    return torch.cat(chunks, dim=0)


def build_support_set(
    dataset: Level3FewShotDataset,
    *,
    sampling: str,
    k_shot: int,
    seed: int,
    draws: int,
    sampling_seed: int,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference support pool; with ``tta`` every draw is its own support sample."""
    if sampling == "legacy":
        return build_reference_support_pool(
            dataset, k_shot=k_shot, n_way=NUM_LEVEL3_CLASSES, seed=seed
        )
    return build_resampled_support_pool(
        dataset,
        k_shot=k_shot,
        method=sampling,
        draws=draws,
        sampling_seed=sampling_seed,
        seq_len=seq_len,
        seed=seed,
    )


def load_linear_head(path: Path, device: torch.device, *, channels: int) -> Level3LinearExpert:
    checkpoint = load_checkpoint(path, map_location=device)
    cfg = architecture_config_from_checkpoint(checkpoint)
    encoder = PrototypicalNetwork(
        in_channels=int(cfg.get("in_channels", channels)),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        encoder=str(cfg.get("encoder", "cnn")).lower(),
        device=device,
    )
    model = Level3LinearExpert(encoder, num_classes=NUM_LEVEL3_CLASSES)
    model.load_state_dict(model_state_dict_from_checkpoint(checkpoint), strict=True)
    model.eval()
    print(f"L3 linear head loaded from {path}")
    return model


def build_scorer(args: argparse.Namespace, *, level: str, device: torch.device) -> L3Scorer:
    channels = 13 if level == "L1C" else 10
    sampling_meta = {
        "l3_sampling": args.sampling,
        "l3_draws": int(args.draws) if args.sampling == "tta" else 1,
    }
    if args.sampling in ("tta", "random"):
        sampling_meta["l3_sampling_seed"] = int(args.sampling_seed)

    if args.mode == "linear":
        if args.linear_weights is None:
            raise ValueError("--linear-weights is required when --mode linear")
        model = load_linear_head(args.linear_weights.resolve(), device, channels=channels)
        return L3Scorer(
            distance_fn=model.pseudo_distances,
            drop_keys=("prototypes", "inv_variances"),
            meta={
                **sampling_meta,
                "l3_metric": "linear_head",
                "l3_linear_weights": str(args.linear_weights.resolve()),
            },
        )

    l3, _hidden = load_l3_protonet(args.l3_weights.resolve(), device, channels=channels)
    print(
        f"Locking Mahalanobis prototypes from {args.train_regions} "
        f"(ref_k_shot={args.ref_k_shot}, shrinkage={args.shrinkage}, seed={args.seed}, "
        f"sampling={args.sampling}) ..."
    )
    proto_dataset = Level3FewShotDataset(
        regions=args.train_regions,
        level=level,
        max_samples=args.max_samples,
        max_seq_len=args.max_seq_len,
    )
    support_x, support_y = build_support_set(
        proto_dataset,
        sampling=args.sampling,
        k_shot=args.ref_k_shot,
        seed=args.seed,
        draws=args.draws,
        sampling_seed=args.sampling_seed,
        seq_len=args.max_seq_len,
    )
    prototypes, inv_variances = l3.prototypes_and_inv_variances_from_support(
        support_x.to(device),
        support_y.to(device),
        NUM_LEVEL3_CLASSES,
        shrinkage=args.shrinkage,
    )
    print(
        f"Locked {prototypes.size(0)} prototypes of dim {prototypes.size(1)} "
        f"from {support_x.size(0)} support series."
    )
    return L3Scorer(
        distance_fn=lambda x: squared_mahalanobis_distance(l3.embed(x), prototypes, inv_variances),
        tensors={"prototypes": prototypes.cpu(), "inv_variances": inv_variances.cpu()},
        meta={
            **sampling_meta,
            "l3_metric": "mahalanobis",
            "l3_weights": str(args.l3_weights.resolve()),
            "train_regions": list(args.train_regions),
            "ref_k_shot": int(args.ref_k_shot),
            "shrinkage": float(args.shrinkage),
        },
    )


def l3_metrics(distances: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """5-way argmin accuracy and per-class recall (%) on rows with an L3 label."""
    global_ids = torch.tensor([L3_LOCAL_TO_GLOBAL[c] for c in range(NUM_LEVEL3_CLASSES)])
    mask = torch.isin(labels, global_ids)
    y = labels[mask]
    pred = global_ids[distances[mask].argmin(dim=1)]
    out = {"n": float(mask.sum()), "acc": 100.0 * (pred == y).float().mean().item()}
    for local_id, global_id in enumerate(global_ids.tolist()):
        hit = y == global_id
        out[LEVEL3_ID_TO_NAME[local_id]] = (
            100.0 * (pred[hit] == global_id).float().mean().item() if hit.any() else float("nan")
        )
    return out


def format_metrics(name: str, metrics: Dict[str, float]) -> str:
    recalls = "  ".join(
        f"{LEVEL3_ID_TO_NAME[c]}={metrics[LEVEL3_ID_TO_NAME[c]]:.2f}"
        for c in range(NUM_LEVEL3_CLASSES)
    )
    return f"  {name:24s} acc={metrics['acc']:.2f}  {recalls}"


def output_path(cache_path: Path, suffix: str) -> Path:
    return cache_path.with_name(f"{cache_path.stem}{suffix}{cache_path.suffix}") if suffix else cache_path


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


def patch_cache(
    cache_path: Path,
    args: argparse.Namespace,
    scorer_for: Callable[[str], L3Scorer],
    device: torch.device,
) -> None:
    if not cache_path.is_file():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    meta = payload.get("meta")
    if not isinstance(meta, dict) or "regions" not in meta:
        raise KeyError(f"Cache {cache_path} is missing meta['regions'].")
    regions = list(meta["regions"])
    level = str(meta.get("level", args.level))
    labels = payload["labels"].long()

    ids, order_labels = parcel_ids_and_labels(regions, level=level, max_samples=args.max_samples)
    if not torch.equal(order_labels, labels):
        raise RuntimeError("Cache labels do not match BreizhCrops index order.")
    if "parcel_ids" in payload and not torch.equal(ids, payload["parcel_ids"].long()):
        raise RuntimeError("Cache parcel_ids do not match BreizhCrops index order.")

    scorer = scorer_for(level)
    l3_distances = extract_l3_distances(
        scorer.distance_fn,
        regions=regions,
        level=level,
        sampling=args.sampling,
        draws=args.draws,
        sampling_seed=args.sampling_seed,
        seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=device,
        max_samples=args.max_samples,
    )
    expected = (labels.numel(), NUM_LEVEL3_CLASSES)
    if tuple(l3_distances.shape) != expected:
        raise RuntimeError(f"Expected L3 distances of shape {expected}; got {tuple(l3_distances.shape)}.")

    print(f"\nL3 5-way argmin on {cache_path.name}:")
    cached = payload.get("l3_distances")
    if cached is not None and tuple(cached.shape) == expected:
        print(format_metrics("cached", l3_metrics(cached.float(), labels)))
    print(format_metrics(f"{args.sampling} ({args.mode})", l3_metrics(l3_distances, labels)))
    if args.compare_only:
        return

    payload["l3_distances"] = l3_distances
    payload.update(scorer.tensors)
    for key in scorer.drop_keys:
        payload.pop(key, None)
    meta = {k: v for k, v in meta.items() if k not in L3_META_KEYS}
    meta.update(scorer.meta)
    meta["l3_patched"] = datetime.now().isoformat(timespec="seconds")

    target = output_path(cache_path, args.output_suffix)
    if target == cache_path:
        meta["l3_backup"] = str(backup_once(cache_path))
    else:
        meta["l3_source"] = str(cache_path)
    payload["meta"] = meta
    tmp = target.with_name(target.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)
    print(f"Patched l3_distances ({args.sampling}, {args.mode}) for {labels.numel():,} parcels -> {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "caches",
        nargs="*",
        type=Path,
        help="Paths to cached_logits_*.pt (default: train + eval caches).",
    )
    parser.add_argument(
        "--cache",
        dest="extra_caches",
        action="append",
        type=Path,
        default=[],
        help="Additional cache paths to patch (may be repeated).",
    )
    parser.add_argument(
        "--mode",
        choices=("mahalanobis", "linear"),
        default="mahalanobis",
        help="L3 distance source: locked ProtoNet (default) or linear head.",
    )
    parser.add_argument(
        "--l3-weights",
        type=Path,
        default=Path("artifacts/checkpoints/level3_protonet/best.pt"),
        help="Production Level-3 ProtoNet checkpoint (mahalanobis mode).",
    )
    parser.add_argument(
        "--linear-weights",
        type=Path,
        default=None,
        help="Level3LinearExpert checkpoint (linear mode).",
    )
    parser.add_argument(
        "--sampling",
        choices=SAMPLINGS,
        default="tta",
        help="Series resampling for support and queries (see module docstring).",
    )
    parser.add_argument("--draws", type=int, default=8, help="Random resamples per parcel for --sampling tta.")
    parser.add_argument("--sampling-seed", type=int, default=0, help="Seed for tta/random resampling.")
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Write <stem><suffix>.pt next to each cache instead of patching in place.",
    )
    parser.add_argument("--compare-only", action="store_true", help="Report accuracies; write nothing.")
    parser.add_argument("--train-regions", nargs="+", default=["frh01", "frh02", "frh03"])
    parser.add_argument("--ref-k-shot", type=int, default=50)
    parser.add_argument("--shrinkage", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0, help="Reference support pool selection seed.")
    parser.add_argument("--level", default="L2A")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=64, help="Parcels per forward pass.")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("--draws must be >= 1.")
    if any(sep in args.output_suffix for sep in ("/", "\\")):
        parser.error("--output-suffix must not contain path separators.")
    return args


def main() -> None:
    args = parse_args()
    paths: list[Path] = [p.resolve() for p in (*args.caches, *args.extra_caches)]
    if not paths:
        paths = [
            Path("artifacts/cached_logits_train.pt").resolve(),
            Path("artifacts/cached_logits_eval.pt").resolve(),
        ]
    if args.output_suffix and not args.compare_only:
        existing = [t for t in (output_path(p, args.output_suffix) for p in paths) if t.exists()]
        if existing:
            raise SystemExit(f"Refusing to overwrite existing outputs: {[str(t) for t in existing]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    scorers: Dict[str, L3Scorer] = {}

    def scorer_for(level: str) -> L3Scorer:
        # Locked once per level, then shared by every cache in this run.
        if level not in scorers:
            scorers[level] = build_scorer(args, level=level, device=device)
        return scorers[level]

    for cache_path in paths:
        print(f"\n=== Patching {cache_path} ({args.mode}, sampling={args.sampling}) ===")
        patch_cache(cache_path, args, scorer_for, device)


if __name__ == "__main__":
    main()

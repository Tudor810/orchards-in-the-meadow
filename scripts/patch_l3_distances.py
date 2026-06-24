"""Recompute ``l3_distances`` in an existing DMOE logit cache.

Preserves MSResNet / Transformer keys and only updates L3-related tensors.
Row order matches ``sweep_dmoe_parameters.py`` extraction.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import breizhcrops as bzh
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from agricultureai.data.breizhcrops_series import (
    BreizhCropsParcelDataset,
    pad_collate_breizhcrops_series,
)
from agricultureai.data.episodic_sampler import (
    NUM_LEVEL3_CLASSES,
    Level3FewShotDataset,
    build_reference_support_pool,
)
from agricultureai.evaluation.transformer import MAX_SEQ_LEN
from agricultureai.models.proto_net import (
    Level3LinearExpert,
    PrototypicalNetwork,
    squared_mahalanobis_distance,
)
from agricultureai.training.checkpoints import (
    architecture_config_from_checkpoint,
    load_checkpoint,
    model_state_dict_from_checkpoint,
)

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from evaluate_dmoe_pipeline import load_l3_protonet  # noqa: E402


def build_eval_loader(
    *,
    regions: list[str],
    level: str,
    max_samples: int,
    batch_size: int,
    max_seq_len: int,
) -> DataLoader:
    parts = []
    for region in regions:
        print(f"Loading region={region} level={level} (all classes) ...")
        ds = bzh.BreizhCrops(region, level=level)
        indices = list(range(len(ds)))
        if max_samples > 0:
            indices = indices[: min(max_samples, len(indices))]
        parts.append(
            BreizhCropsParcelDataset(
                ds, indices, max_seq_len=max_seq_len, train=False, seed=0
            )
        )
    return DataLoader(
        ConcatDataset(parts),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(pad_collate_breizhcrops_series, max_seq_len=max_seq_len),
        pin_memory=torch.cuda.is_available(),
    )


@torch.inference_mode()
def extract_protonet_distances(
    l3: PrototypicalNetwork,
    prototypes: torch.Tensor,
    inv_variances: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for x, _doy, _padding_mask, _targets, _ in tqdm(loader, desc="Mahalanobis L3 distances"):
        x = x.to(device, non_blocking=True)
        embeddings = l3.embed(x)
        distances = squared_mahalanobis_distance(embeddings, prototypes, inv_variances)
        chunks.append(distances.float().cpu())
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def extract_linear_pseudo_distances(
    model: Level3LinearExpert,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for x, _doy, _padding_mask, _targets, _ in tqdm(loader, desc="Linear-head L3 pseudo-distances"):
        x = x.to(device, non_blocking=True)
        chunks.append(model.pseudo_distances(x).float().cpu())
    return torch.cat(chunks, dim=0)


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


def patch_cache(
    cache_path: Path,
    *,
    mode: str,
    l3_weights: Path,
    linear_weights: Path | None,
    train_regions: list[str],
    ref_k_shot: int,
    shrinkage: float,
    seed: int,
    level: str,
    max_samples: int,
    batch_size: int,
    max_seq_len: int,
) -> None:
    if not cache_path.is_file():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    meta = payload.get("meta")
    if not isinstance(meta, dict) or "regions" not in meta:
        raise KeyError(f"Cache {cache_path} is missing meta['regions'].")

    regions = list(meta["regions"])
    level = str(meta.get("level", level))
    expected_n = int(payload["labels"].numel())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    channels = 13 if level == "L1C" else 10

    loader = build_eval_loader(
        regions=regions,
        level=level,
        max_samples=max_samples,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )

    meta = dict(meta)
    meta["l3_patched"] = datetime.now().isoformat(timespec="seconds")

    if mode == "linear":
        if linear_weights is None:
            raise ValueError("--linear-weights is required when --mode linear")
        model = load_linear_head(linear_weights.resolve(), device, channels=channels)
        l3_distances = extract_linear_pseudo_distances(model, loader, device)
        meta["l3_metric"] = "linear_head"
        meta["l3_linear_weights"] = str(linear_weights.resolve())
        payload.pop("prototypes", None)
        payload.pop("inv_variances", None)
    else:
        l3, _hidden = load_l3_protonet(l3_weights.resolve(), device, channels=channels)
        print(
            f"Locking Mahalanobis prototypes from {train_regions} "
            f"(ref_k_shot={ref_k_shot}, shrinkage={shrinkage}, seed={seed}) ..."
        )
        proto_dataset = Level3FewShotDataset(
            regions=train_regions,
            level=level,
            max_samples=max_samples,
            max_seq_len=max_seq_len,
        )
        support_x, support_y = build_reference_support_pool(
            proto_dataset, k_shot=ref_k_shot, n_way=NUM_LEVEL3_CLASSES, seed=seed
        )
        prototypes, inv_variances = l3.prototypes_and_inv_variances_from_support(
            support_x.to(device),
            support_y.to(device),
            NUM_LEVEL3_CLASSES,
            shrinkage=shrinkage,
        )
        print(f"Locked {prototypes.size(0)} prototypes of dim {prototypes.size(1)}.")
        l3_distances = extract_protonet_distances(
            l3, prototypes, inv_variances, loader, device
        )
        payload["prototypes"] = prototypes.cpu()
        payload["inv_variances"] = inv_variances.cpu()
        meta["train_regions"] = list(train_regions)
        meta["ref_k_shot"] = int(ref_k_shot)
        meta["shrinkage"] = float(shrinkage)
        meta["l3_metric"] = "mahalanobis"

    if l3_distances.size(0) != expected_n:
        raise RuntimeError(
            f"Row count mismatch: cache has {expected_n}, L3 extracted {l3_distances.size(0)}."
        )
    if l3_distances.size(1) != NUM_LEVEL3_CLASSES:
        raise RuntimeError(
            f"Expected {NUM_LEVEL3_CLASSES} L3 classes; got {l3_distances.size(1)}."
        )

    payload["l3_distances"] = l3_distances
    payload["meta"] = meta
    torch.save(payload, cache_path)
    print(f"Patched l3_distances for {expected_n:,} samples -> {cache_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch l3_distances in cached DMOE logits."
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
    parser.add_argument("--train-regions", nargs="+", default=["frh01", "frh02", "frh03"])
    parser.add_argument("--ref-k-shot", type=int, default=50)
    parser.add_argument("--shrinkage", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", default="L1C")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths: list[Path] = list(args.caches)
    paths.extend(args.extra_caches)
    if not paths:
        paths = [
            Path("artifacts/cached_logits_train.pt"),
            Path("artifacts/cached_logits_eval.pt"),
        ]

    for cache_path in paths:
        print(f"\n=== Patching {cache_path.resolve()} ({args.mode}) ===")
        patch_cache(
            cache_path.resolve(),
            mode=str(args.mode),
            l3_weights=args.l3_weights,
            linear_weights=args.linear_weights,
            train_regions=list(args.train_regions),
            ref_k_shot=int(args.ref_k_shot),
            shrinkage=float(args.shrinkage),
            seed=int(args.seed),
            level=str(args.level),
            max_samples=int(args.max_samples),
            batch_size=int(args.batch_size),
            max_seq_len=int(args.max_seq_len),
        )


if __name__ == "__main__":
    main()

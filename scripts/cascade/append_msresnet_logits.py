"""Append pretrained MSResNet 9-class logits to an existing DMOE logit cache.

Uses the same ``BreizhCropsParcelDataset`` loader and row order as
``sweep_dmoe_parameters.py`` extraction so ``msresnet_logits`` aligns with
``l1_logits``, ``l2a_logits``, etc.
"""

from __future__ import annotations

import argparse
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
from agricultureai.evaluation.transformer import MAX_SEQ_LEN


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
def extract_msresnet_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    logit_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []

    for x, _doy, _padding_mask, targets, _ in tqdm(loader, desc="MSResNet 9-class logits"):
        x = x.to(device, non_blocking=True)
        logit_chunks.append(model(x).float().cpu())
        label_chunks.append(targets.long().cpu())

    return torch.cat(logit_chunks, dim=0), torch.cat(label_chunks, dim=0)


def append_msresnet_to_cache(
    cache_path: Path,
    *,
    batch_size: int,
    max_samples: int,
    max_seq_len: int,
    force: bool,
) -> None:
    if not cache_path.is_file():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if "msresnet_logits" in payload and not force:
        print(f"{cache_path} already contains msresnet_logits; use --force to overwrite.")
        return

    meta = payload.get("meta")
    if not isinstance(meta, dict) or "regions" not in meta:
        raise KeyError(f"Cache {cache_path} is missing meta['regions'].")

    regions = list(meta["regions"])
    level = str(meta.get("level", "L1C"))
    expected_n = int(payload["labels"].numel())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading pretrained MSResNet ...")
    model = bzh.models.pretrained("MSResNet", device=device)
    model.eval()

    loader = build_eval_loader(
        regions=regions,
        level=level,
        max_samples=max_samples,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )
    ms_logits, labels = extract_msresnet_logits(model, loader, device)

    if ms_logits.size(0) != expected_n:
        raise RuntimeError(
            f"Row count mismatch: cache has {expected_n}, MSResNet extracted {ms_logits.size(0)}."
        )
    if ms_logits.size(1) != 9:
        raise RuntimeError(f"Expected MSResNet logits with 9 classes; got {ms_logits.size(1)}.")
    if not torch.equal(labels, payload["labels"].long()):
        raise RuntimeError("Label order mismatch between cache and MSResNet extraction.")

    payload["msresnet_logits"] = ms_logits
    meta = dict(meta)
    meta["msresnet"] = "pretrained"
    meta["msresnet_appended"] = datetime.now().isoformat(timespec="seconds")
    payload["meta"] = meta

    torch.save(payload, cache_path)
    print(f"Appended msresnet_logits {tuple(ms_logits.shape)} -> {cache_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append pretrained MSResNet 9-class logits to a DMOE logit cache."
    )
    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="Path to cached_logits_*.pt produced by sweep_dmoe_parameters.py.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit samples per region (0 = all). Must match original extraction.",
    )
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing msresnet_logits in the cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    append_msresnet_to_cache(
        args.cache.resolve(),
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_seq_len=args.max_seq_len,
        force=args.force,
    )


if __name__ == "__main__":
    main()

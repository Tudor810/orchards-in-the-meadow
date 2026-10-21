"""Append pretrained BreizhCrops Transformer 9-class logits to a DMOE logit cache."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import breizhcrops as bzh
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from agricultureai.data.breizhcrops_loaders import build_cache_eval_loader
from agricultureai.evaluation.transformer import MAX_SEQ_LEN


@torch.inference_mode()
def extract_transformer_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    logit_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []

    for x, _doy, _padding_mask, targets, _ in tqdm(loader, desc="Transformer 9-class logits"):
        x = x.to(device, non_blocking=True)
        logit_chunks.append(model(x).float().cpu())
        label_chunks.append(targets.long().cpu())

    return torch.cat(logit_chunks, dim=0), torch.cat(label_chunks, dim=0)


def append_transformer_to_cache(
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
    if "transformer_logits" in payload and not force:
        print(f"{cache_path} already contains transformer_logits; use --force to overwrite.")
        return

    meta = payload.get("meta")
    if not isinstance(meta, dict) or "regions" not in meta:
        raise KeyError(f"Cache {cache_path} is missing meta['regions'].")

    regions = list(meta["regions"])
    level = str(meta.get("level", "L1C"))
    expected_n = int(payload["labels"].numel())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading pretrained Transformer (9-class) ...")
    model = bzh.models.pretrained("Transformer", device=device)
    model.eval()

    loader = build_cache_eval_loader(
        regions=regions,
        level=level,
        max_samples=max_samples,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )
    logits, labels = extract_transformer_logits(model, loader, device)

    if logits.size(0) != expected_n:
        raise RuntimeError(
            f"Row count mismatch: cache has {expected_n}, Transformer extracted {logits.size(0)}."
        )
    if logits.size(1) != 9:
        raise RuntimeError(f"Expected Transformer logits with 9 classes; got {logits.size(1)}.")
    if not torch.equal(labels, payload["labels"].long()):
        raise RuntimeError("Label order mismatch between cache and Transformer extraction.")

    payload["transformer_logits"] = logits
    meta = dict(meta)
    meta["transformer"] = "pretrained"
    meta["transformer_appended"] = datetime.now().isoformat(timespec="seconds")
    payload["meta"] = meta

    torch.save(payload, cache_path)
    print(f"Appended transformer_logits {tuple(logits.shape)} -> {cache_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append pretrained Transformer 9-class logits to a DMOE logit cache."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    append_transformer_to_cache(
        args.cache.resolve(),
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_seq_len=args.max_seq_len,
        force=args.force,
    )


if __name__ == "__main__":
    main()

"""Align cached L1C global-model logits into an L2A DMOE cache."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from agricultureai.cache_alignment import (
    align_logits_by_parcel_id,
    parcel_ids_and_labels,
)


def append_cached_global_logits(
    target_cache: Path,
    source_cache: Path,
    *,
    max_samples: int = 0,
) -> None:
    target = torch.load(target_cache, map_location="cpu", weights_only=True)
    source = torch.load(source_cache, map_location="cpu", weights_only=True)

    for key in ("msresnet_logits", "transformer_logits", "labels"):
        if key not in source:
            raise KeyError(f"Source cache {source_cache} is missing {key!r}.")

    target_meta = target.get("meta")
    source_meta = source.get("meta")
    if not isinstance(target_meta, dict) or "regions" not in target_meta:
        raise KeyError(f"Target cache {target_cache} has no region metadata.")
    if not isinstance(source_meta, dict) or "regions" not in source_meta:
        raise KeyError(f"Source cache {source_cache} has no region metadata.")
    if list(target_meta["regions"]) != list(source_meta["regions"]):
        raise ValueError(
            f"Region mismatch: target={target_meta['regions']} "
            f"source={source_meta['regions']}"
        )

    regions = list(target_meta["regions"])
    target_level = str(target_meta.get("level", "L2A"))
    source_level = str(source_meta.get("level", "L1C"))
    target_ids, target_labels = parcel_ids_and_labels(
        regions, level=target_level, max_samples=max_samples
    )
    source_ids, source_labels = parcel_ids_and_labels(
        regions, level=source_level, max_samples=max_samples
    )
    if not torch.equal(target_labels, target["labels"].long()):
        raise ValueError("Target cache labels do not match reconstructed parcel order.")
    if not torch.equal(source_labels, source["labels"].long()):
        raise ValueError("Source cache labels do not match reconstructed parcel order.")

    matched_mask = None
    for key in ("msresnet_logits", "transformer_logits"):
        aligned, matched = align_logits_by_parcel_id(
            source[key],
            source_labels,
            source_ids,
            target_ids,
            target_labels,
        )
        if matched_mask is not None and not torch.equal(matched_mask, matched):
            raise RuntimeError("Global-model match masks unexpectedly differ.")
        matched_mask = matched
        target[key] = aligned
        target[f"{key.removesuffix('_logits')}_l1c_matched"] = matched

    assert matched_mask is not None
    target["parcel_ids"] = target_ids
    target_meta = dict(target_meta)
    target_meta.update(
        {
            "global_logits_source": str(source_cache.resolve()),
            "global_logits_source_level": source_level,
            "global_logits_alignment": "parcel_id",
            "global_logits_matched": int(matched_mask.sum()),
            "global_logits_missing": int((~matched_mask).sum()),
            "global_logits_appended": datetime.now().isoformat(timespec="seconds"),
            "msresnet": f"pretrained_{source_level}_cached_aligned",
            "transformer": f"pretrained_{source_level}_cached_aligned",
        }
    )
    target["meta"] = target_meta
    torch.save(target, target_cache)
    print(
        f"Appended cached global logits -> {target_cache} "
        f"(rows={target_ids.numel():,}, matched={int(matched_mask.sum()):,}, "
        f"neutral_missing={int((~matched_mask).sum()):,})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    append_cached_global_logits(
        args.target_cache.resolve(),
        args.source_cache.resolve(),
        max_samples=int(args.max_samples),
    )


if __name__ == "__main__":
    main()

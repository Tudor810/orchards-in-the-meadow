"""BreizhCrops DataLoaders for the package Transformer."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from functools import partial

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

import breizhcrops as bzh

from agricultureai.data.breizhcrops_series import (
    BreizhCropsParcelDataset,
    pad_collate_breizhcrops_series,
)
from agricultureai.data.breizhcrops_super_classes import (
    map_raw_to_super_class,
    raw_class_ids_for_super_targets,
)
from agricultureai.evaluation.transformer import MAX_SEQ_LEN as DEFAULT_MAX_SEQ_LEN
from agricultureai.evaluation.transformer import pad_collate


def filtered_indices(ds, raw_class_ids: Sequence[int], max_samples: int) -> List[int]:
    idx_df = ds.index
    if max_samples > 0:
        idx_df = idx_df.iloc[:max_samples]
    keep = idx_df["classid"].isin(set(int(x) for x in raw_class_ids)).to_numpy()
    return [int(i) for i in range(len(keep)) if bool(keep[i])]


def build_breizhcrops_loaders(
    *,
    train_regions: Sequence[str],
    val_regions: Sequence[str],
    level: str,
    target_ids: Sequence[int],
    max_samples: int,
    batch_size: int,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    use_train_region_holdout: bool = False,
    use_doy: bool = False,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    train_parts: List[Dataset] = []
    val_parts: List[Dataset] = []
    raw_filter_ids = raw_class_ids_for_super_targets(int(x) for x in target_ids)
    max_n = int(max_samples)

    def wrap_region(base_ds, kept: List[int], *, train: bool) -> Dataset:
        """Return a DOY-aware parcel dataset or a plain index subset."""
        if use_doy:
            return BreizhCropsParcelDataset(
                base_ds, kept, max_seq_len=max_seq_len, train=train, seed=seed
            )
        return Subset(base_ds, kept)

    for region in train_regions:
        print(f"Loading train region={region} level={level} ...")
        ds = bzh.BreizhCrops(region, level=level)
        kept = filtered_indices(ds, raw_filter_ids, max_n)
        if len(kept) < 2:
            raise RuntimeError(f"Not enough train samples after filtering for region={region}.")
        if use_train_region_holdout:
            split = max(1, int(0.8 * len(kept)))
            train_parts.append(wrap_region(ds, kept[:split], train=True))
            val_parts.append(wrap_region(ds, kept[split:], train=False))
        else:
            train_parts.append(wrap_region(ds, kept, train=True))

    if val_regions:
        overlap = set(train_regions) & set(val_regions)
        if overlap:
            raise ValueError(f"Train and val regions must not overlap: {sorted(overlap)}")
        for region in val_regions:
            print(f"Loading val region={region} level={level} (full region, for best.pt) ...")
            ds = bzh.BreizhCrops(region, level=level)
            kept = filtered_indices(ds, raw_filter_ids, max_n)
            if len(kept) < 1:
                raise RuntimeError(f"No val samples after filtering for region={region}.")
            val_parts.append(wrap_region(ds, kept, train=False))

    if not val_parts:
        raise RuntimeError("No validation data: set val_regions or use train_region_holdout.")

    if use_doy:
        collate = lambda batch: pad_collate_breizhcrops_series(
            batch, max_seq_len=max_seq_len, map_label=map_raw_to_super_class
        )
    else:
        collate = lambda batch: pad_collate(batch, map_to_super_class=True, max_seq_len=max_seq_len)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        ConcatDataset(train_parts),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        ConcatDataset(val_parts),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=pin,
    )
    return train_loader, val_loader


def build_breizhcrops_region_eval_loader(
    *,
    regions: Sequence[str],
    level: str,
    target_ids: Sequence[int],
    max_samples: int,
    batch_size: int,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    full_region: bool = False,
) -> DataLoader:
    parts: List[Dataset] = []
    raw_filter_ids = raw_class_ids_for_super_targets(int(x) for x in target_ids)
    for region in regions:
        ds = bzh.BreizhCrops(region, level=level)
        kept = filtered_indices(ds, raw_filter_ids, int(max_samples))
        if len(kept) < 1:
            raise RuntimeError(f"No samples after filtering for region={region}.")
        if full_region:
            use = kept
        else:
            if len(kept) < 2:
                raise RuntimeError(f"Not enough samples after filtering for region={region}.")
            split = max(1, int(0.8 * len(kept)))
            use = kept[split:]
        parts.append(Subset(ds, use))

    collate = lambda batch: pad_collate(batch, map_to_super_class=True, max_seq_len=max_seq_len)
    return DataLoader(
        ConcatDataset(parts),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )


def build_cache_eval_loader(
    *,
    regions: Sequence[str],
    level: str,
    max_samples: int,
    batch_size: int,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> DataLoader:
    """Every parcel of each region (all classes) in dataset order: the row order of the logit caches."""
    parts: List[Dataset] = []
    for region in regions:
        print(f"Loading region={region} level={level} (all classes) ...")
        ds = bzh.BreizhCrops(region, level=level)
        indices = list(range(len(ds)))
        if max_samples > 0:
            indices = indices[: min(max_samples, len(indices))]
        parts.append(BreizhCropsParcelDataset(ds, indices, max_seq_len=max_seq_len, train=False, seed=0))
    return DataLoader(
        ConcatDataset(parts),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(pad_collate_breizhcrops_series, max_seq_len=max_seq_len),
        pin_memory=torch.cuda.is_available(),
    )

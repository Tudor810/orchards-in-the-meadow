"""Data helpers for the Stage-2 Winter Cereal (barley vs wheat) LTAE expert."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

import breizhcrops as bzh

from agricultureai.data.breizhcrops_loaders import filtered_indices
from agricultureai.evaluation.transformer import MAX_SEQ_LEN as DEFAULT_MAX_SEQ_LEN

CEREAL_EXPERT_RAW_IDS: List[int] = [0, 1]

RAW_TO_CEREAL_EXPERT: Dict[int, int] = {
    0: 0,  # barley
    1: 1,  # wheat
}

CEREAL_EXPERT_ID_TO_NAME: Dict[int, str] = {
    0: "Barley",
    1: "Wheat",
}

NUM_CEREAL_EXPERT_CLASSES = len(CEREAL_EXPERT_ID_TO_NAME)
CEREAL_EXPERT_TARGET_IDS = list(CEREAL_EXPERT_ID_TO_NAME.keys())


def map_raw_to_cereal_expert(label: int) -> int:
    try:
        return RAW_TO_CEREAL_EXPERT[int(label)]
    except KeyError as exc:
        raise ValueError(
            f"Raw label {label} is outside cereal expert ids {CEREAL_EXPERT_RAW_IDS}"
        ) from exc


def pad_collate_cereal_expert(batch, max_seq_len: int = DEFAULT_MAX_SEQ_LEN):
    xs, ys, ids = zip(*batch)
    padded_xs = []
    for x in xs:
        t, c = x.shape
        if t >= max_seq_len:
            padded_x = x[:max_seq_len, :]
        else:
            padding = torch.zeros(max_seq_len - t, c)
            padded_x = torch.cat([x, padding], dim=0)
        padded_xs.append(padded_x)
    mapped_ys = [map_raw_to_cereal_expert(y) for y in ys]
    return torch.stack(padded_xs), torch.tensor(mapped_ys).long(), ids


def build_cereal_expert_loaders(
    *,
    train_regions: Sequence[str],
    val_regions: Sequence[str],
    level: str,
    raw_target_ids: Sequence[int],
    max_samples: int,
    batch_size: int,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    use_train_region_holdout: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    raw_ids = [int(x) for x in raw_target_ids]
    unknown = set(raw_ids) - set(CEREAL_EXPERT_RAW_IDS)
    if unknown:
        raise ValueError(
            f"Cereal expert only supports raw ids {CEREAL_EXPERT_RAW_IDS}; got extra {sorted(unknown)}"
        )

    train_parts: List[Dataset] = []
    val_parts: List[Dataset] = []
    max_n = int(max_samples)

    for region in train_regions:
        print(f"Loading train region={region} level={level} (cereal expert filter) ...")
        ds = bzh.BreizhCrops(region, level=level)
        kept = filtered_indices(ds, raw_ids, max_n)
        if len(kept) < 2:
            raise RuntimeError(f"Not enough train samples after filtering for region={region}.")
        if use_train_region_holdout:
            split = max(1, int(0.8 * len(kept)))
            train_parts.append(Subset(ds, kept[:split]))
            val_parts.append(Subset(ds, kept[split:]))
        else:
            train_parts.append(Subset(ds, kept))

    if val_regions:
        overlap = set(train_regions) & set(val_regions)
        if overlap:
            raise ValueError(f"Train and val regions must not overlap: {sorted(overlap)}")
        for region in val_regions:
            print(f"Loading val region={region} level={level} (cereal expert, for best.pt) ...")
            ds = bzh.BreizhCrops(region, level=level)
            kept = filtered_indices(ds, raw_ids, max_n)
            if len(kept) < 1:
                raise RuntimeError(f"No val samples after filtering for region={region}.")
            val_parts.append(Subset(ds, kept))

    if not val_parts:
        raise RuntimeError("No validation data: set val_regions or use train_region_holdout.")

    collate = lambda batch: pad_collate_cereal_expert(batch, max_seq_len=max_seq_len)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        ConcatDataset(train_parts),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        ConcatDataset(val_parts),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=pin,
    )
    return train_loader, val_loader

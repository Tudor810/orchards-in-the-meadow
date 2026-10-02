"""Level-2c meadow expert: binary Permanent (7) vs Temporary (8) meadows."""

from __future__ import annotations

from collections import Counter
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import breizhcrops as bzh

from agricultureai.data.breizhcrops_loaders import filtered_indices
from agricultureai.data.breizhcrops_series import (
    BreizhCropsParcelDataset,
    pad_collate_breizhcrops_series,
)
from agricultureai.evaluation.transformer import MAX_SEQ_LEN as DEFAULT_MAX_SEQ_LEN

L2C_RAW_IDS: List[int] = [7, 8]

RAW_TO_L2C: dict[int, int] = {
    7: 0,  # permanent meadows
    8: 1,  # temporary meadows
}

L2C_ID_TO_NAME: dict[int, str] = {
    0: "Permanent Meadows",
    1: "Temporary Meadows",
}

MEADOW_LOCAL_TO_GLOBAL: dict[int, int] = {
    0: 7,
    1: 8,
}

NUM_L2C_CLASSES = 2


def map_raw_to_l2c(label: int) -> int:
    try:
        return RAW_TO_L2C[int(label)]
    except KeyError as exc:
        raise ValueError(
            f"Raw label {label} is outside L2c ids {L2C_RAW_IDS}"
        ) from exc


def _mapped_labels_for_train_parts(train_parts: Sequence[Dataset]) -> List[int]:
    labels: List[int] = []
    for part in train_parts:
        if not isinstance(part, BreizhCropsParcelDataset):
            raise TypeError("Expected BreizhCropsParcelDataset parts")
        base_ds = part.base_ds
        for base_idx in part.indices:
            row = base_ds.index.iloc[int(base_idx)]
            raw = int(base_ds.mapping.loc[row["CODE_CULTU"]].id)
            labels.append(map_raw_to_l2c(raw))
    return labels


def pad_collate_l2c_meadow(batch, *, max_seq_len: int = DEFAULT_MAX_SEQ_LEN):
    return pad_collate_breizhcrops_series(
        batch,
        max_seq_len=max_seq_len,
        map_label=map_raw_to_l2c,
    )


def build_l2c_meadow_loaders(
    *,
    train_regions: Sequence[str],
    val_regions: Sequence[str],
    level: str,
    max_samples: int,
    batch_size: int,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    use_train_region_holdout: bool = False,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    raw_ids = list(L2C_RAW_IDS)
    train_parts: List[Dataset] = []
    val_parts: List[Dataset] = []
    max_n = int(max_samples)

    for region in train_regions:
        print(f"Loading L2c train region={region} level={level} ...")
        ds = bzh.BreizhCrops(region, level=level)
        kept = filtered_indices(ds, raw_ids, max_n)
        if len(kept) < 2:
            raise RuntimeError(f"Not enough L2c train samples for region={region}.")
        if use_train_region_holdout:
            split = max(1, int(0.8 * len(kept)))
            train_parts.append(
                BreizhCropsParcelDataset(
                    ds, kept[:split], max_seq_len=max_seq_len, train=True, seed=seed
                )
            )
            val_parts.append(
                BreizhCropsParcelDataset(
                    ds, kept[split:], max_seq_len=max_seq_len, train=False, seed=seed
                )
            )
        else:
            train_parts.append(
                BreizhCropsParcelDataset(
                    ds, kept, max_seq_len=max_seq_len, train=True, seed=seed
                )
            )

    if val_regions:
        overlap = set(train_regions) & set(val_regions)
        if overlap:
            raise ValueError(f"Train and val regions must not overlap: {sorted(overlap)}")
        for region in val_regions:
            print(f"Loading L2c val region={region} level={level} ...")
            ds = bzh.BreizhCrops(region, level=level)
            kept = filtered_indices(ds, raw_ids, max_n)
            if len(kept) < 1:
                raise RuntimeError(f"No L2c val samples for region={region}.")
            val_parts.append(
                BreizhCropsParcelDataset(
                    ds, kept, max_seq_len=max_seq_len, train=False, seed=seed
                )
            )

    if not val_parts:
        raise RuntimeError("No L2c validation data: set val_regions or use holdout.")

    collate = lambda batch: pad_collate_l2c_meadow(batch, max_seq_len=max_seq_len)
    pin = torch.cuda.is_available()
    train_dataset = ConcatDataset(train_parts)

    mapped_labels = _mapped_labels_for_train_parts(train_parts)
    counts = Counter(mapped_labels)
    counts_str = ", ".join(f"{L2C_ID_TO_NAME[c]}={counts.get(c, 0)}" for c in range(NUM_L2C_CLASSES))
    print(f"L2c train class counts: {counts_str}")
    print("L2c training: uniform shuffle (balanced meadows).")

    train_loader = DataLoader(
        train_dataset,
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

"""Level-2b anomaly gate: binary Grass (7/8) vs Anomaly (4/5/6)."""

from __future__ import annotations

from collections import Counter
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

import breizhcrops as bzh

from agricultureai.data.breizhcrops_loaders import filtered_indices
from agricultureai.data.breizhcrops_series import (
    BreizhCropsParcelDataset,
    pad_collate_breizhcrops_series,
)
from agricultureai.evaluation.transformer import MAX_SEQ_LEN as DEFAULT_MAX_SEQ_LEN

L2B_RAW_IDS: List[int] = [4, 5, 6, 7, 8]

RAW_TO_L2B: dict[int, int] = {
    7: 0,  # grass (permanent meadow)
    8: 0,  # grass (temporary meadow)
    4: 1,  # anomaly
    5: 1,
    6: 1,
}

L2B_ID_TO_NAME: dict[int, str] = {
    0: "Grass",
    1: "Anomaly",
}

L2B_GRASS_LOCAL = 0
L2B_ANOMALY_LOCAL = 1
NUM_L2B_CLASSES = 2


def map_raw_to_l2b(label: int) -> int:
    try:
        return RAW_TO_L2B[int(label)]
    except KeyError as exc:
        raise ValueError(
            f"Raw label {label} is outside L2b ids {L2B_RAW_IDS}"
        ) from exc


def _raw_label_from_base_index(base_ds: bzh.BreizhCrops, base_idx: int) -> int:
    row = base_ds.index.iloc[int(base_idx)]
    return int(base_ds.mapping.loc[row["CODE_CULTU"]].id)


def _mapped_labels_for_train_parts(train_parts: Sequence[Dataset]) -> List[int]:
    labels: List[int] = []
    for part in train_parts:
        if not isinstance(part, BreizhCropsParcelDataset):
            raise TypeError("Expected BreizhCropsParcelDataset parts")
        base_ds = part.base_ds
        for base_idx in part.indices:
            labels.append(map_raw_to_l2b(_raw_label_from_base_index(base_ds, base_idx)))
    return labels


def inverse_frequency_sample_weights(mapped_labels: Sequence[int]) -> List[float]:
    """Heavy anomaly oversampling via plain inverse class frequency."""
    counts = Counter(int(label) for label in mapped_labels)
    class_count = {c: max(int(counts.get(c, 0)), 1) for c in range(NUM_L2B_CLASSES)}
    class_weight = {c: 1.0 / class_count[c] for c in range(NUM_L2B_CLASSES)}
    return [class_weight[int(label)] for label in mapped_labels]


def pad_collate_l2b_anomaly_gate(
    batch,
    *,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
):
    return pad_collate_breizhcrops_series(
        batch,
        max_seq_len=max_seq_len,
        map_label=map_raw_to_l2b,
    )


def build_l2b_anomaly_gate_loaders(
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
    raw_ids = list(L2B_RAW_IDS)
    train_parts: List[Dataset] = []
    val_parts: List[Dataset] = []
    max_n = int(max_samples)

    for region in train_regions:
        print(f"Loading L2b train region={region} level={level} ...")
        ds = bzh.BreizhCrops(region, level=level)
        kept = filtered_indices(ds, raw_ids, max_n)
        if len(kept) < 2:
            raise RuntimeError(f"Not enough L2b train samples for region={region}.")
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
            print(f"Loading L2b val region={region} level={level} ...")
            ds = bzh.BreizhCrops(region, level=level)
            kept = filtered_indices(ds, raw_ids, max_n)
            if len(kept) < 1:
                raise RuntimeError(f"No L2b val samples for region={region}.")
            val_parts.append(
                BreizhCropsParcelDataset(
                    ds, kept, max_seq_len=max_seq_len, train=False, seed=seed
                )
            )

    if not val_parts:
        raise RuntimeError("No L2b validation data: set val_regions or use holdout.")

    collate = lambda batch: pad_collate_l2b_anomaly_gate(batch, max_seq_len=max_seq_len)
    pin = torch.cuda.is_available()
    train_dataset = ConcatDataset(train_parts)

    mapped_labels = _mapped_labels_for_train_parts(train_parts)
    counts = Counter(mapped_labels)
    counts_str = ", ".join(f"{L2B_ID_TO_NAME[c]}={counts.get(c, 0)}" for c in range(NUM_L2B_CLASSES))
    print(f"L2b train class counts: {counts_str}")
    print("L2b training: WeightedRandomSampler (inverse-frequency anomaly oversampling).")

    sample_weights = inverse_frequency_sample_weights(mapped_labels)
    train_sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_dataset),
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
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

"""Align auxiliary model outputs across BreizhCrops processing levels."""

from __future__ import annotations

from typing import Sequence

import breizhcrops as bzh
import torch


def parcel_ids_and_labels(
    regions: Sequence[str],
    *,
    level: str,
    max_samples: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return parcel IDs and mapped labels in cache extraction order."""
    ids: list[int] = []
    labels: list[int] = []
    for region in regions:
        dataset = bzh.BreizhCrops(region, level=level)
        count = len(dataset) if max_samples <= 0 else min(max_samples, len(dataset))
        rows = dataset.index.iloc[:count]
        ids.extend(int(value) for value in rows.id)
        labels.extend(
            int(dataset.mapping.loc[code].id) for code in rows["CODE_CULTU"]
        )
    return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def align_logits_by_parcel_id(
    source_logits: torch.Tensor,
    source_labels: torch.Tensor,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    target_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align source rows to target parcel IDs.

    Missing target parcels receive all-zero logits (neutral after softmax).
    The returned boolean mask identifies rows backed by a real source observation.
    """
    if source_logits.ndim != 2:
        raise ValueError(f"Expected 2-d logits; got {tuple(source_logits.shape)}")
    if source_logits.size(0) != source_ids.numel():
        raise ValueError("Source logits and parcel ID counts differ.")
    if source_labels.numel() != source_ids.numel():
        raise ValueError("Source labels and parcel ID counts differ.")
    if target_labels.numel() != target_ids.numel():
        raise ValueError("Target labels and parcel ID counts differ.")

    source_id_list = source_ids.tolist()
    if len(source_id_list) != len(set(source_id_list)):
        raise ValueError("Source parcel IDs are not unique.")
    source_rows = {parcel_id: row for row, parcel_id in enumerate(source_id_list)}

    aligned = torch.zeros(
        target_ids.numel(),
        source_logits.size(1),
        dtype=source_logits.dtype,
    )
    matched = torch.zeros(target_ids.numel(), dtype=torch.bool)
    for target_row, parcel_id in enumerate(target_ids.tolist()):
        source_row = source_rows.get(parcel_id)
        if source_row is None:
            continue
        if int(source_labels[source_row]) != int(target_labels[target_row]):
            raise ValueError(
                f"Label mismatch for parcel {parcel_id}: "
                f"source={int(source_labels[source_row])}, "
                f"target={int(target_labels[target_row])}"
            )
        aligned[target_row] = source_logits[source_row]
        matched[target_row] = True
    return aligned, matched

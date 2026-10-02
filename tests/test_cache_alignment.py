"""Tests for parcel-ID logit alignment across processing levels."""

from __future__ import annotations

import torch

from agricultureai.cache_alignment import align_logits_by_parcel_id


def test_align_logits_by_parcel_id_matches_and_fills_missing():
    source_ids = torch.tensor([10, 20, 30])
    source_labels = torch.tensor([0, 1, 2])
    source_logits = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float32
    )
    target_ids = torch.tensor([20, 40, 10])
    target_labels = torch.tensor([1, 3, 0])

    aligned, matched = align_logits_by_parcel_id(
        source_logits, source_labels, source_ids, target_ids, target_labels
    )

    assert aligned.shape == (3, 2)
    assert torch.equal(aligned[0], source_logits[1])
    assert torch.equal(aligned[2], source_logits[0])
    assert torch.equal(aligned[1], torch.zeros(2))
    assert matched.tolist() == [True, False, True]

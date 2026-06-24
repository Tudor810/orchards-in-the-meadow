"""Unit tests for meta-classifier feature layout (64-d with metadata)."""

from __future__ import annotations

import torch

from agricultureai.meta_features import (
    FULL_FEATURE_DIM,
    METADATA_FEATURE_DIM,
    NEURAL_FEATURE_DIM,
    build_neural_feature_matrix,
    load_feature_matrix,
)


def _synthetic_payload(*, with_metadata: bool = False) -> dict:
    n = 4
    payload = {
        "l1_logits": torch.randn(n, 4),
        "l2a_logits": torch.randn(n, 2),
        "l2b_logits": torch.randn(n, 2),
        "l2c_logits": torch.randn(n, 2),
        "l3_distances": torch.rand(n, 5),
        "msresnet_logits": torch.randn(n, 9),
        "transformer_logits": torch.randn(n, 9),
        "labels": torch.tensor([0, 1, 5, 8]),
    }
    if with_metadata:
        payload["parcel_metadata"] = torch.randn(n, METADATA_FEATURE_DIM)
    return payload


def test_neural_dim():
    x = build_neural_feature_matrix(_synthetic_payload())
    assert x.shape == (4, NEURAL_FEATURE_DIM)


def test_load_feature_matrix_with_metadata(tmp_path):
    cache = tmp_path / "cache.pt"
    torch.save(_synthetic_payload(with_metadata=True), cache)
    x, y = load_feature_matrix(cache, use_metadata=True)
    assert x.shape == (4, FULL_FEATURE_DIM)
    assert y.tolist() == [0, 1, 5, 8]


def test_load_feature_matrix_without_metadata(tmp_path):
    cache = tmp_path / "cache.pt"
    torch.save(_synthetic_payload(with_metadata=False), cache)
    x, y = load_feature_matrix(cache, use_metadata=False)
    assert x.shape == (4, NEURAL_FEATURE_DIM)
    assert y.tolist() == [0, 1, 5, 8]

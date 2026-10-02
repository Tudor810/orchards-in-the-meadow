"""Unit tests for meta-classifier feature layout (66-d with metadata)."""

from __future__ import annotations

import numpy as np
import torch

from agricultureai.meta_features import (
    COMPACT_FEATURE_DIM,
    DMOE_ONLY_FULL_FEATURE_DIM,
    DMOE_ONLY_NEURAL_DIM,
    FULL_FEATURE_DIM,
    METADATA_FEATURE_DIM,
    NEURAL_FEATURE_DIM,
    apply_orchard_meadow_log_bias,
    build_neural_feature_matrix,
    feature_column_indices,
    load_feature_matrix,
    select_feature_layout,
)
from agricultureai.pipeline import (
    GLOBAL_ORCHARDS,
    GLOBAL_PERMANENT_MEADOWS,
    GLOBAL_TEMPORARY_MEADOWS,
    NUM_GLOBAL_CLASSES,
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


def test_unmatched_l1c_global_logits_become_nan():
    payload = _synthetic_payload()
    matched = torch.tensor([True, False, True, True])
    payload["msresnet_logits"][1] = 0.0
    payload["transformer_logits"][1] = 0.0
    payload["msresnet_l1c_matched"] = matched
    payload["transformer_l1c_matched"] = matched

    x = build_neural_feature_matrix(payload)
    global_cols = feature_column_indices(("msresnet", "global_eng", "transformer"))

    assert x.shape == (4, NEURAL_FEATURE_DIM)
    assert global_cols.size == 29
    assert np.isnan(x[1, global_cols]).all()
    assert not np.isnan(np.delete(x, global_cols, axis=1)).any()
    assert not np.isnan(x[[0, 2, 3]]).any()
    assert not torch.isnan(payload["msresnet_logits"]).any()


def _dmoe_only_payload(*, with_metadata: bool = False) -> dict:
    payload = _synthetic_payload(with_metadata=False)
    payload.pop("msresnet_logits")
    payload.pop("transformer_logits")
    if with_metadata:
        payload["parcel_metadata"] = torch.randn(4, METADATA_FEATURE_DIM)
    return payload


def test_dmoe_only_neural_dim():
    x = build_neural_feature_matrix(_dmoe_only_payload())
    assert x.shape == (4, DMOE_ONLY_NEURAL_DIM)


def test_load_feature_matrix_dmoe_only_with_metadata(tmp_path):
    cache = tmp_path / "cache.pt"
    torch.save(_dmoe_only_payload(with_metadata=True), cache)
    x, y = load_feature_matrix(cache, use_metadata=True)
    assert x.shape == (4, DMOE_ONLY_FULL_FEATURE_DIM)
    assert y.tolist() == [0, 1, 5, 8]


def test_select_feature_layout_compact(tmp_path):
    cache = tmp_path / "cache.pt"
    torch.save(_synthetic_payload(with_metadata=True), cache)
    x, _ = load_feature_matrix(cache, use_metadata=True)
    compact = select_feature_layout(x, "compact")
    assert compact.shape == (4, COMPACT_FEATURE_DIM)
    assert select_feature_layout(x, "full").shape[1] == FULL_FEATURE_DIM
    assert select_feature_layout(x, "minus_l2b_l2c").shape[1] == FULL_FEATURE_DIM - 2 - 2 - 2


def test_apply_orchard_meadow_log_bias_shifts_mass():
    probs = np.full((1, NUM_GLOBAL_CLASSES), 1.0 / NUM_GLOBAL_CLASSES, dtype=np.float32)
    out = apply_orchard_meadow_log_bias(probs, b_orchard=2.0, b_meadow=1.0)
    assert out.shape == probs.shape
    assert abs(float(out.sum()) - 1.0) < 1e-5
    assert out[0, GLOBAL_ORCHARDS] > probs[0, GLOBAL_ORCHARDS]
    assert out[0, GLOBAL_PERMANENT_MEADOWS] < probs[0, GLOBAL_PERMANENT_MEADOWS]
    assert out[0, GLOBAL_TEMPORARY_MEADOWS] < probs[0, GLOBAL_TEMPORARY_MEADOWS]

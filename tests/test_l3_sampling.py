"""Tests for training-consistent L3 resampling (``scripts/cascade/patch_l3_distances.py``)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

from agricultureai.data.breizhcrops_series import (
    SAMPLING_METHODS,
    pad_series_with_doy,
    resample_series_views,
    subsample_series,
)
from agricultureai.data.episodic_sampler import (
    build_reference_support_pool,
    reference_support_indices,
)

import patch_l3_distances as p3


def _series(t: int, c: int = 3) -> np.ndarray:
    """Row ``i`` holds ``i + 1`` in every channel, so views reveal their time indices."""
    return np.repeat(np.arange(1, t + 1, dtype=np.float32)[:, None], c, axis=1)


@pytest.mark.parametrize("t", [5, 27, 45, 63])
def test_zero_pad_view_matches_cache_extraction(t):
    x = _series(t)
    view = resample_series_views(x, 45, method="zero_pad")
    sub_x, sub_doy = subsample_series(x, np.arange(t), 45, train=False)
    expected, _, _ = pad_series_with_doy(
        torch.from_numpy(sub_x), torch.from_numpy(sub_doy), max_seq_len=45
    )
    assert view.shape == (1, 45, 3)
    assert np.array_equal(view[0], expected.numpy())


@pytest.mark.parametrize("t", [5, 27, 45, 63])
def test_tta_views_resample_like_training_and_never_pad(t):
    views = resample_series_views(_series(t), 45, method="tta", draws=8, rng=np.random.default_rng(0))
    assert views.shape == (8, 45, 3)
    idx = views[:, :, 0].astype(int) - 1  # (draws, 45) time indices
    assert idx.min() >= 0 and idx.max() < t  # every step is a real observation
    assert np.all(np.diff(idx, axis=1) >= 0)  # time order kept
    if t >= 45:
        assert all(len(set(row.tolist())) == 45 for row in idx)  # no replacement
    # T == 45 is the identity draw (as in the stock transform); otherwise draws differ.
    assert len({tuple(row) for row in idx}) == (1 if t == 45 else 8)


def test_seeded_views_are_reproducible_and_random_is_first_tta_draw():
    x = _series(27)
    a = resample_series_views(x, 45, method="tta", draws=4, rng=np.random.default_rng(5))
    b = resample_series_views(x, 45, method="tta", draws=4, rng=np.random.default_rng(5))
    one = resample_series_views(x, 45, method="random", rng=np.random.default_rng(5))
    assert np.array_equal(a, b)
    assert np.array_equal(one[0], a[0])
    lin = resample_series_views(x, 45, method="linspace")
    assert lin.shape == (1, 45, 3) and lin[0, 0, 0] == 1 and lin[0, -1, 0] == 27


def test_empty_series_gives_zero_views_without_consuming_rng():
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    for method in SAMPLING_METHODS:
        views = resample_series_views(np.zeros((0, 3), np.float32), 45, method=method, draws=3, rng=rng)
        assert views.shape == ((3 if method == "tta" else 1), 45, 3)
        assert not views.any()
    assert rng.bit_generator.state == state


def test_resample_series_views_rejects_bad_arguments():
    with pytest.raises(ValueError):
        resample_series_views(_series(10), 45, method="stock")
    with pytest.raises(ValueError):
        resample_series_views(_series(10), 45, method="tta", draws=0)


class _FakeLevel3Dataset:
    """Stands in for ``Level3FewShotDataset``: labels, stock-style items, raw series."""

    def __init__(self, labels, lengths):
        self.labels = list(labels)
        self._lengths = list(lengths)

    def __getitem__(self, idx):
        return torch.full((45, 3), float(idx)), self.labels[idx]

    def raw_series(self, idx):
        return _series(self._lengths[idx]) + 100.0 * idx


# Class sizes: 0 -> 4, 1 -> 3, 2 -> 3, 3 -> 1, 4 -> 1.
LABELS = [0, 1, 2, 0, 1, 2, 3, 0, 1, 4, 0, 2]


def test_reference_support_indices_match_build_reference_support_pool():
    ds = _FakeLevel3Dataset(LABELS, [10] * len(LABELS))
    picks = reference_support_indices(ds.labels, k_shot=2, n_way=5, seed=0)
    support_x, support_y = build_reference_support_pool(ds, k_shot=2, n_way=5, seed=0)
    assert support_y.tolist() == [c for _, c in picks] == [0, 0, 1, 1, 2, 2, 3, 4]
    assert support_x[:, 0, 0].long().tolist() == [i for i, _ in picks]
    assert all(ds.labels[i] == c for i, c in picks)
    assert picks == reference_support_indices(ds.labels, k_shot=2, n_way=5, seed=0)
    assert picks != reference_support_indices(ds.labels, k_shot=2, n_way=5, seed=1)


def test_tta_support_set_expands_each_parcel_into_seeded_draws():
    ds = _FakeLevel3Dataset(LABELS, [7, 30, 45, 60, 12, 20, 33, 8, 50, 16, 40, 27])
    kwargs = dict(k_shot=2, seed=0, draws=3, sampling_seed=0, seq_len=45)
    x, y = p3.build_support_set(ds, sampling="tta", **kwargs)
    picks = reference_support_indices(ds.labels, k_shot=2, n_way=5, seed=0)
    assert x.shape == (len(picks) * 3, 45, 3)
    assert y.tolist() == [c for _, c in picks for _ in range(3)]
    assert (x[:, :, 0] // 100).long().tolist() == [[i] * 45 for i, _ in picks for _ in range(3)]
    assert (x > 0).all()  # resampled, never zero-padded
    x2, _ = p3.build_support_set(ds, sampling="tta", **kwargs)
    assert torch.equal(x, x2)


def test_legacy_support_set_uses_stock_items():
    ds = _FakeLevel3Dataset(LABELS, [10] * len(LABELS))
    x, y = p3.build_support_set(
        ds, sampling="legacy", k_shot=2, seed=0, draws=8, sampling_seed=0, seq_len=45
    )
    ref_x, ref_y = build_reference_support_pool(ds, k_shot=2, n_way=5, seed=0)
    assert torch.equal(x, ref_x) and torch.equal(y, ref_y)


def test_score_views_averages_distances_over_draws():
    views = [np.full((4, 45, 2), v, dtype=np.float32) for v in (1.0, 2.0)]
    views[0][1:] = 3.0  # parcel 0 draws: 1, 3, 3, 3 -> mean 2.5
    scale = torch.arange(1.0, 6.0)
    out = p3.score_views(
        lambda x: x[:, :, 0].mean(dim=1, keepdim=True) * scale, views, torch.device("cpu")
    )
    assert out.shape == (2, 5)
    assert torch.allclose(out, torch.tensor([[2.5], [2.0]]) * scale)


def test_view_method_maps_legacy_queries_to_zero_pad():
    assert p3.view_method("legacy") == "zero_pad"
    assert all(p3.view_method(s) == s for s in ("tta", "random", "linspace"))


def test_l3_metrics_maps_global_labels_to_l3_locals():
    # PM(7), TM(8), orchard(5), nut(6), sunflower(4), wheat(1, ignored)
    labels = torch.tensor([7, 8, 5, 6, 4, 1])
    local_pred = torch.tensor([0, 0, 2, 3, 2, 4])  # TM -> PM and sunflower -> orchard wrong
    distances = torch.ones(6, 5)
    distances[torch.arange(6), local_pred] = 0.0
    m = p3.l3_metrics(distances, labels)
    assert m["n"] == 5
    assert m["acc"] == pytest.approx(60.0)
    assert m["Orchards"] == 100.0 and m["Nut Trees"] == 100.0
    assert m["Temporary Meadows"] == 0.0 and m["Sunflowers"] == 0.0


def test_backup_once_verifies_and_never_overwrites(tmp_path):
    cache = tmp_path / "cached_logits_eval.pt"
    cache.write_bytes(b"original")
    backup = p3.backup_once(cache)
    assert backup.name == "cached_logits_eval.pre_l3_patch.pt"
    assert backup.read_bytes() == b"original"
    cache.write_bytes(b"patched")
    assert p3.backup_once(cache) == backup
    assert backup.read_bytes() == b"original"


def test_output_path_suffix():
    cache = Path("artifacts/cached_logits_train.pt")
    assert p3.output_path(cache, "_l3fix") == Path("artifacts/cached_logits_train_l3fix.pt")
    assert p3.output_path(cache, "") == cache


def _run_patch(tmp_path, monkeypatch, **overrides):
    labels = torch.tensor([7, 5, 1])
    src = tmp_path / "cached_logits_eval.pt"
    torch.save(
        {
            "labels": labels,
            "parcel_ids": torch.tensor([10, 11, 12]),
            "l3_distances": torch.zeros(3, 5),
            "meta": {"regions": ["frh04"], "level": "L2A", "l3_linear_weights": "stale.pt"},
        },
        src,
    )
    new = torch.rand(3, 5)
    monkeypatch.setattr(
        p3, "parcel_ids_and_labels", lambda regions, level, max_samples: (torch.tensor([10, 11, 12]), labels)
    )
    monkeypatch.setattr(p3, "extract_l3_distances", lambda fn, **kwargs: new)
    scorer = p3.L3Scorer(
        distance_fn=lambda x: x,
        tensors={"prototypes": torch.ones(5, 4)},
        meta={"l3_sampling": "tta", "l3_metric": "mahalanobis"},
    )
    args = argparse.Namespace(
        level="L2A", max_samples=0, sampling="tta", draws=8, sampling_seed=0, max_seq_len=45,
        batch_size=4, mode="mahalanobis", compare_only=False, output_suffix="_l3fix",
    )
    vars(args).update(overrides)
    before = src.read_bytes()
    p3.patch_cache(src, args, lambda level: scorer, torch.device("cpu"))
    return src, before, new


def test_patch_cache_writes_copy_and_leaves_source_untouched(tmp_path, monkeypatch):
    src, before, new = _run_patch(tmp_path, monkeypatch)
    assert src.read_bytes() == before
    out = torch.load(tmp_path / "cached_logits_eval_l3fix.pt", weights_only=True)
    assert torch.equal(out["l3_distances"], new)
    assert torch.equal(out["prototypes"], torch.ones(5, 4))
    assert out["meta"]["l3_sampling"] == "tta"
    assert out["meta"]["l3_source"] == str(src)
    assert "l3_linear_weights" not in out["meta"]  # stale key from an earlier linear patch
    assert not (tmp_path / "cached_logits_eval.pre_l3_patch.pt").exists()


def test_patch_cache_in_place_keeps_verified_backup(tmp_path, monkeypatch):
    src, before, new = _run_patch(tmp_path, monkeypatch, output_suffix="")
    backup = tmp_path / "cached_logits_eval.pre_l3_patch.pt"
    assert backup.read_bytes() == before
    out = torch.load(src, weights_only=True)
    assert torch.equal(out["l3_distances"], new)
    assert out["meta"]["l3_backup"] == str(backup)


def test_patch_cache_compare_only_writes_nothing(tmp_path, monkeypatch):
    src, before, _ = _run_patch(tmp_path, monkeypatch, compare_only=True, output_suffix="")
    assert src.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cached_logits_eval.pt"]

"""Tests for the meta-classifier variant trainer and phenology features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compute_phenology_features as cpf  # noqa: E402
import train_meta_variants as tmv  # noqa: E402

from agricultureai.meta_features import COMPACT_FEATURE_DIM  # noqa: E402


def test_compact_column_names_cover_the_compact_layout():
    names = tmv.compact_column_names()
    assert len(names) == COMPACT_FEATURE_DIM == len(set(names))
    assert names[0] == "l3_raw_0" and names[-1] == "metadata_7"


def test_missing_classes_get_one_negligible_row_each():
    x = np.arange(12, dtype=np.float32).reshape(6, 2)
    y = np.array([0, 1, 2, 3, 5, 7])
    w = np.ones(6, dtype=np.float32)
    x2, y2, w2 = tmv.ensure_all_classes(x, y, w, np.random.default_rng(0))
    assert sorted(np.unique(y2).tolist()) == list(range(9))
    assert len(y2) == 9 and np.all(w2[6:] == tmv.MISSING_CLASS_WEIGHT)
    assert all(any(np.array_equal(r, row) for row in x) for r in x2[6:])  # copies of real rows
    same = tmv.ensure_all_classes(x2, y2, w2, np.random.default_rng(0))
    assert same[0] is x2  # nothing missing -> unchanged


def test_extra_features_must_match_parcel_ids(tmp_path, monkeypatch):
    import torch

    cache = tmp_path / "c.pt"
    torch.save({"parcel_ids": torch.tensor([1, 2, 3])}, cache)
    monkeypatch.setattr(tmv, "load_feature_matrix", lambda path, use_metadata: (np.zeros((3, 66), np.float32), np.zeros(3)))
    good = tmp_path / "good.npz"
    np.savez(good, parcel_ids=np.array([1, 2, 3]), features=np.ones((3, 2)), columns=np.array(["a", "b"]))
    x, _, names = tmv.load_features(cache, [("pheno", good)])
    assert x.shape == (3, COMPACT_FEATURE_DIM + 2) and names[-2:] == ["pheno:a", "pheno:b"]
    bad = tmp_path / "bad.npz"
    np.savez(bad, parcel_ids=np.array([1, 3, 2]), features=np.ones((3, 2)), columns=np.array(["a", "b"]))
    with pytest.raises(RuntimeError, match="parcel ids"):
        tmv.load_features(cache, [("pheno", bad)])


def _raw(doys, ndvi_like, cld):
    """L2A raw rows: doa as ns timestamps, B4/B8 set so NDVI follows ``ndvi_like``."""
    t = len(doys)
    raw = np.full((t, 14), 1000.0)
    # Nanoseconds since the epoch, as the BreizhCrops HDF5 stores ``doa``.
    dates = np.datetime64("2017-01-01", "ns") + (np.array(doys) - 1).astype("timedelta64[D]")
    raw[:, cpf.DOA] = dates.astype(np.int64)
    raw[:, cpf.B8] = 1000.0 * (1 + np.array(ndvi_like))
    raw[:, cpf.B4] = 1000.0 * (1 - np.array(ndvi_like))
    raw[:, cpf.CLD] = cld
    return raw


def test_parcel_features_use_clear_dates_and_count_cuts():
    doys = [20, 100, 110, 120, 200, 300]
    ndvi = [0.5, 0.8, 0.5, 0.8, 0.7, 0.6]  # a 0.3 drop between day 100 and 110
    feats = dict(zip(cpf.COLUMNS, cpf.parcel_features(_raw(doys, ndvi, [0, 0, 0, 0, 0, 1]))))
    assert feats["n_clear_obs"] == 5  # the cloudy autumn date is dropped
    assert np.isnan(feats["ndvi_autumn_mean"])
    assert feats["ndvi_winter_mean"] == pytest.approx(0.5, abs=1e-6)
    assert feats["mow_drops"] == 1 and feats["max_ndvi_drop"] == pytest.approx(0.3, abs=1e-6)
    assert feats["ndvi_summer_minus_winter"] == pytest.approx(0.2, abs=1e-6)
    assert np.isnan(feats["compactness"])  # filled per region, not per series


def test_parcel_features_all_cloudy_is_all_nan_except_count():
    feats = cpf.parcel_features(_raw([10, 50], [0.5, 0.6], [1, 1]))
    assert feats[cpf.COLUMNS.index("n_clear_obs")] == 0
    assert np.isnan(np.delete(feats, cpf.COLUMNS.index("n_clear_obs"))).all()

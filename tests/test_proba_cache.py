"""Tests for fingerprinted meta-classifier probability caches."""

from __future__ import annotations

import numpy as np
import pytest

from agricultureai.proba_cache import array_digest, load_npz, meta_proba_fingerprint, save_npz


def test_array_digest_tracks_values_and_canonicalizes_nan():
    a = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    other_nan = a.copy()
    other_nan.view(np.uint32)[0, 1] = 0x7FC00001  # NaN with a different payload
    changed = a.copy()
    changed[1, 1] = 5.0

    assert array_digest(a) == array_digest(a.copy())
    assert array_digest(a) == array_digest(other_nan)
    assert array_digest(a) != array_digest(changed)
    assert array_digest(a) != array_digest(a.astype(np.float64))


def test_load_npz_hit_miss_stale_and_force(tmp_path, capsys):
    path = tmp_path / "proba.npz"
    proba = np.random.rand(3, 9).astype(np.float32)

    assert load_npz(path, "fp-a") is None  # missing file

    save_npz(path, "fp-a", proba=proba)
    hit = load_npz(path, "fp-a")
    assert hit is not None and set(hit) == {"proba"}
    np.testing.assert_array_equal(hit["proba"], proba)

    assert load_npz(path, "fp-b") is None
    assert "fingerprint mismatch" in capsys.readouterr().out
    assert load_npz(path, "fp-a", force=True) is None

    np.savez_compressed(path, proba=proba)  # legacy file written before fingerprints
    assert load_npz(path, "fp-a") is None
    assert "no fingerprint" in capsys.readouterr().out


def test_meta_proba_fingerprint_changes_with_data_and_spec():
    pytest.importorskip("xgboost")
    pytest.importorskip("catboost")
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(20, 5)).astype(np.float32)
    y_train = rng.integers(0, 9, size=20)
    x_eval = rng.normal(size=(6, 5)).astype(np.float32)

    base = meta_proba_fingerprint(x_train, y_train, x_eval, seed=42, kind="pair", m=15.0)
    assert base == meta_proba_fingerprint(x_train, y_train, x_eval, seed=42, kind="pair", m=15.0)

    x_eval_nan = x_eval.copy()
    x_eval_nan[0, 0] = np.nan
    assert base != meta_proba_fingerprint(x_train, y_train, x_eval_nan, seed=42, kind="pair", m=15.0)
    assert base != meta_proba_fingerprint(x_train, y_train, x_eval, seed=42, kind="pair", m=3.0)
    assert base != meta_proba_fingerprint(x_train, y_train, x_eval, seed=7, kind="pair", m=15.0)

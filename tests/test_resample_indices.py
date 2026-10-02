"""Tests for training-consistent time-series resampling (L2a cache patch)."""

from __future__ import annotations

import numpy as np
import pytest

from agricultureai.data.breizhcrops_series import resample_indices


@pytest.mark.parametrize("n_steps", [1, 27, 44, 45, 63])
def test_random_matches_breizhcrops_default_transform(n_steps):
    idx = resample_indices(n_steps, 45, method="random", rng=np.random.default_rng(0))
    assert idx.shape == (45,)
    assert np.all(np.diff(idx) >= 0)
    assert idx.min() >= 0 and idx.max() < n_steps
    if n_steps >= 45:
        assert len(set(idx.tolist())) == 45  # no replacement, like the stock transform


@pytest.mark.parametrize("n_steps", [1, 27, 44, 45, 63])
def test_linspace_is_deterministic_and_covers_series(n_steps):
    idx = resample_indices(n_steps, 45, method="linspace")
    assert idx.shape == (45,)
    assert np.array_equal(idx, resample_indices(n_steps, 45, method="linspace"))
    assert idx[0] == 0 and idx[-1] == n_steps - 1
    assert len(set(idx.tolist())) == min(n_steps, 45)


def test_seeded_random_is_reproducible_and_rejects_bad_input():
    a = resample_indices(27, 45, method="random", rng=np.random.default_rng(3))
    b = resample_indices(27, 45, method="random", rng=np.random.default_rng(3))
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        resample_indices(0, 45)
    with pytest.raises(ValueError):
        resample_indices(10, 45, method="zero_pad")

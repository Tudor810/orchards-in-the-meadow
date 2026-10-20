"""Tests for the spatial block bootstrap of threshold-free orchard metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from agricultureai.evaluation.bootstrap import (
    block_resample_indices,
    block_resample_weights,
    paired_ap_ci,
    ranking,
    weighted_ap,
)


def _data(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(np.int8)
    score = np.round(rng.random(n) + 0.5 * y, 2)  # rounded: many ties
    blocks = rng.integers(0, 25, size=n)
    return y, score, blocks


def test_weighted_ap_with_unit_weights_is_sklearn_ap():
    y, score, _ = _data()
    assert np.isclose(weighted_ap(ranking(score), y.astype(float), np.ones(len(y))),
                      100 * average_precision_score(y, score))


def test_weights_and_indices_are_the_same_resamples():
    y, score, blocks = _data()
    idx = block_resample_indices(blocks, n_boot=20, seed=3)
    wts = block_resample_weights(blocks, n_boot=20, seed=3)
    rank = ranking(score)
    for i, w in zip(idx, wts):
        assert np.array_equal(np.bincount(i, minlength=len(y)), w.astype(int))
        assert np.isclose(weighted_ap(rank, y.astype(float), w), 100 * average_precision_score(y[i], score[i]))


def test_paired_ap_ci_of_a_score_against_itself_is_zero():
    y, score, blocks = _data()
    ap_new, ap_old, lo, hi = paired_ap_ci(y, score, score, blocks, n_boot=50)
    assert ap_new == ap_old and lo == hi == 0.0

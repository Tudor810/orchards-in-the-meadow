"""Spatial block bootstrap for threshold-free orchard metrics.

Parcels are resampled by spatial block (the 10 km blocks of a ``*_folds.npz`` file): each resample draws as
many blocks as there are, with replacement. Two equivalent forms of the same resamples (same ``seed``, same
draws):

* :func:`block_resample_indices` - one index array per resample (concatenated block members);
* :func:`block_resample_weights` - one multiplicity weight per parcel and resample, used with
  :func:`weighted_ap`, which gives exactly the AP of the concatenated resample (duplicates share scores, so
  ties group the same way) and is much faster.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _block_index(blocks: np.ndarray) -> np.ndarray:
    _, gidx = np.unique(blocks, return_inverse=True)
    return gidx


def block_resample_indices(blocks: np.ndarray, n_boot: int = 1000, seed: int = 0) -> List[np.ndarray]:
    gidx = _block_index(blocks)
    members = np.split(np.argsort(gidx, kind="stable"), np.cumsum(np.bincount(gidx))[:-1])
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))]) for _ in range(n_boot)]


def block_resample_weights(blocks: np.ndarray, n_boot: int = 1000, seed: int = 0) -> List[np.ndarray]:
    gidx = _block_index(blocks)
    g = gidx.max() + 1
    rng = np.random.default_rng(seed)
    return [np.bincount(rng.integers(0, g, g), minlength=g)[gidx].astype(float) for _ in range(n_boot)]


def ranking(score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Descending order of ``score`` and the last position of each distinct score (for :func:`weighted_ap`)."""
    order = np.argsort(-score, kind="stable")
    s = score[order]
    return order, np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]


def weighted_ap(rank: Tuple[np.ndarray, np.ndarray], y_bin: np.ndarray, w: np.ndarray) -> float:
    """Average precision (x100) of a ranking with per-parcel weights (sklearn's step-wise definition)."""
    order, last = rank
    yw = (w * y_bin)[order]
    tp = np.cumsum(yw)[last]
    fp = np.cumsum(w[order] - yw)[last]
    return 100.0 * float(np.sum(np.diff(np.r_[0.0, tp / tp[-1]]) * tp / np.maximum(tp + fp, 1e-12)))


def paired_ap_ci(y_bin: np.ndarray, score_new: np.ndarray, score_old: np.ndarray, blocks: np.ndarray, *,
                 n_boot: int = 1000, seed: int = 0) -> Tuple[float, float, float, float]:
    """(AP new, AP old, 2.5%, 97.5%) of the paired block-bootstrap distribution of AP new - AP old."""
    y = y_bin.astype(float)
    r_new, r_old = ranking(score_new.astype(np.float64)), ranking(score_old.astype(np.float64))
    one = np.ones_like(y)
    d = np.array([weighted_ap(r_new, y, w) - weighted_ap(r_old, y, w)
                  for w in block_resample_weights(blocks, n_boot, seed)])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return weighted_ap(r_new, y, one), weighted_ap(r_old, y, one), float(lo), float(hi)

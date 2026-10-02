"""Out-of-sample operating-point selection for the frh04 meta-classifier.

The meta-classifier's decode hyper-parameters (probability blend, orchard
multiplier ``m``, orchard / meadow log-biases) used to be chosen on frh04 and
scored on the same parcels, which is optimistic. Here every parcel is scored
with the configuration a fixed rule selects on the *other* spatial folds of
frh04 (spatially blocked cross-fitting); the meta-classifier itself is still
trained on frh01-frh03 only.

Everything is plain NumPy over predicted labels so it can be unit-tested
without models or caches.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from agricultureai.meta_features import (
    BASELINE_ACCURACY,
    BASELINE_ORCHARD_RECALL,
    STRETCH_ACCURACY,
    STRETCH_ORCHARD_RECALL,
)
from agricultureai.pipeline import GLOBAL_ORCHARDS, GLOBAL_WHEAT, NUM_GLOBAL_CLASSES

Metrics = Dict[str, float]
SelectionRule = Callable[[Sequence[Metrics]], int]

HEADLINE_KEYS: Tuple[str, ...] = (
    "accuracy",
    "orchard_recall",
    "orchard_precision",
    "orchard_f1",
    "wheat_recall",
    "macro_f1",
)


def spatial_block_folds(
    x: np.ndarray,
    y: np.ndarray,
    *,
    block_size: float,
    n_folds: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign parcels to folds by square spatial blocks.

    Blocks of side ``block_size`` (units of ``x`` / ``y``) are visited in a
    seeded random order and each goes to the fold with the fewest parcels so
    far, which balances fold sizes. Parcels with non-finite coordinates get a
    seeded random fold and a block of their own.

    Returns:
        folds: ``(N,)`` fold id in ``[0, n_folds)``.
        blocks: ``(N,)`` block id (for block bootstrap).
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2; got {n_folds}")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)
    ok = np.isfinite(x) & np.isfinite(y)

    cells = np.stack(
        [np.floor(x[ok] / block_size), np.floor(y[ok] / block_size)], axis=1
    ).astype(np.int64)
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()

    sizes = np.zeros(n_folds, dtype=np.int64)
    block_fold = np.empty(counts.size, dtype=np.int64)
    for block in rng.permutation(counts.size):
        fold = int(np.argmin(sizes))
        block_fold[block] = fold
        sizes[fold] += counts[block]

    folds = np.empty(x.size, dtype=np.int64)
    blocks = np.empty(x.size, dtype=np.int64)
    folds[ok] = block_fold[inverse]
    blocks[ok] = inverse
    n_missing = int((~ok).sum())
    folds[~ok] = rng.integers(0, n_folds, size=n_missing)
    blocks[~ok] = counts.size + np.arange(n_missing)
    return folds, blocks


def confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = NUM_GLOBAL_CLASSES) -> np.ndarray:
    """``(K, K)`` confusion matrix, rows = true class."""
    flat = y_true.astype(np.int64) * num_classes + y_pred.astype(np.int64)
    return np.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den > 0)


def per_class_from_confusion(cm: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-class support, precision, recall and F1 (fractions)."""
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    precision = _ratio(tp, predicted)
    recall = _ratio(tp, support)
    f1 = _ratio(2.0 * precision * recall, precision + recall)
    return {"support": support, "precision": precision, "recall": recall, "f1": f1}


def metrics_from_confusion(cm: np.ndarray) -> Metrics:
    """Headline metrics in percent; macro averages over classes with support."""
    pc = per_class_from_confusion(cm)
    present = pc["support"] > 0
    return {
        "accuracy": 100.0 * float(np.trace(cm)) / max(float(cm.sum()), 1.0),
        "orchard_recall": 100.0 * float(pc["recall"][GLOBAL_ORCHARDS]),
        "orchard_precision": 100.0 * float(pc["precision"][GLOBAL_ORCHARDS]),
        "orchard_f1": 100.0 * float(pc["f1"][GLOBAL_ORCHARDS]),
        "wheat_recall": 100.0 * float(pc["recall"][GLOBAL_WHEAT]),
        "macro_recall": 100.0 * float(pc["recall"][present].mean()),
        "macro_f1": 100.0 * float(pc["f1"][present].mean()),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    return metrics_from_confusion(confusion(y_true, y_pred))


# --------------------------------------------------------------------------- #
# Selection rules (fixed before looking at held-out folds)                     #
# --------------------------------------------------------------------------- #
def _argbest(rows: Sequence[Metrics], key: Callable[[Metrics], tuple], allowed: Sequence[int]) -> int:
    # Ties go to the earliest config, so selection is deterministic.
    return max(allowed, key=lambda i: (key(rows[i]), -i))


def select_accuracy_first(
    rows: Sequence[Metrics], *, min_orchard_recall: float = BASELINE_ORCHARD_RECALL
) -> int:
    """Highest accuracy with orchard recall >= floor (else highest orchard recall)."""
    feasible = [i for i, r in enumerate(rows) if r["orchard_recall"] >= min_orchard_recall]
    if not feasible:
        return _argbest(rows, lambda r: (r["orchard_recall"], r["accuracy"]), range(len(rows)))
    return _argbest(rows, lambda r: (r["accuracy"], r["orchard_recall"]), feasible)


def select_orchard_first(rows: Sequence[Metrics], *, min_accuracy: float = BASELINE_ACCURACY) -> int:
    """Highest orchard recall with accuracy >= floor (else highest accuracy)."""
    feasible = [i for i, r in enumerate(rows) if r["accuracy"] >= min_accuracy]
    if not feasible:
        return _argbest(rows, lambda r: (r["accuracy"], r["orchard_recall"]), range(len(rows)))
    return _argbest(rows, lambda r: (r["orchard_recall"], r["accuracy"]), feasible)


def select_closest_stretch(rows: Sequence[Metrics]) -> int:
    """Smallest summed shortfall to the stretch targets (acc, orchard recall)."""

    def shortfall(r: Metrics) -> tuple:
        acc_gap = max(0.0, STRETCH_ACCURACY - r["accuracy"])
        orch_gap = max(0.0, STRETCH_ORCHARD_RECALL - r["orchard_recall"])
        return (-(acc_gap + orch_gap), r["accuracy"])

    return _argbest(rows, shortfall, range(len(rows)))


SELECTION_RULES: Dict[str, SelectionRule] = {
    "accuracy_first": select_accuracy_first,
    "orchard_first": select_orchard_first,
    "closest_stretch": select_closest_stretch,
}


def select_on(preds: np.ndarray, y_true: np.ndarray, mask: np.ndarray, rule: SelectionRule) -> int:
    """Index of the config (row of ``preds``) that ``rule`` picks on ``mask``."""
    rows = [classification_metrics(y_true[mask], p[mask]) for p in preds]
    return rule(rows)


def cross_fit(
    preds: np.ndarray,
    y_true: np.ndarray,
    folds: np.ndarray,
    rule: SelectionRule,
) -> Tuple[np.ndarray, List[int]]:
    """Score each fold with the config ``rule`` selects on the remaining folds.

    Args:
        preds: ``(n_configs, N)`` predicted labels, one row per configuration.
        y_true: ``(N,)`` labels.
        folds: ``(N,)`` fold ids.

    Returns:
        ``(N,)`` cross-fitted predictions and the chosen config index per fold
        (in ascending fold-id order).
    """
    out = np.empty(y_true.shape[0], dtype=preds.dtype)
    chosen: List[int] = []
    for fold in np.unique(folds):
        held_out = folds == fold
        k = select_on(preds, y_true, ~held_out, rule)
        chosen.append(k)
        out[held_out] = preds[k, held_out]
    return out, chosen


# --------------------------------------------------------------------------- #
# Bootstrap                                                                    #
# --------------------------------------------------------------------------- #
def bootstrap_metric_samples(
    y_true: np.ndarray,
    preds: Mapping[str, np.ndarray],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    groups: Optional[np.ndarray] = None,
    keys: Sequence[str] = HEADLINE_KEYS,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Bootstrap replicates of each metric for every prediction vector.

    All prediction vectors share the same resamples, so differences between
    them are paired. With ``groups`` (e.g. spatial blocks) whole groups are
    resampled, which respects spatial correlation between nearby parcels.

    Returns ``{name: {metric: (n_boot,) samples}}``.
    """
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    if groups is not None:
        _, group_idx = np.unique(groups, return_inverse=True)
        members = np.split(np.argsort(group_idx, kind="stable"), np.cumsum(np.bincount(group_idx))[:-1])
    out = {name: {k: np.empty(n_boot) for k in keys} for name in preds}
    for b in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, size=n)
        else:
            picked = rng.integers(0, len(members), size=len(members))
            idx = np.concatenate([members[g] for g in picked])
        y_b = y_true[idx]
        for name, pred in preds.items():
            m = classification_metrics(y_b, pred[idx])
            for k in keys:
                out[name][k][b] = m[k]
    return out


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    lo, hi = np.percentile(samples, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(lo), float(hi)

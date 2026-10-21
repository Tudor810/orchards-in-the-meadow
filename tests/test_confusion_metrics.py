"""Tests for the confusion-matrix metrics shared by the evaluation scripts."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from agricultureai.evaluation.confusion_metrics import summarize_confusion, safe_div


def _confusion(y_true, y_pred, k):
    cm = torch.zeros((k, k), dtype=torch.long)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


@pytest.mark.parametrize("seed", range(5))
def test_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    k = 6
    y_true = rng.integers(0, k - 1, 300)  # class k-1 never occurs: zero support
    y_pred = np.where(rng.random(300) < 0.6, y_true, rng.integers(0, k, 300))
    y_pred[y_pred == 2] = 3  # class 2 is never predicted: zero precision
    metrics, per_class = summarize_confusion(_confusion(y_true, y_pred, k))

    labels = list(range(k))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    assert per_class["support"] == s.tolist()
    np.testing.assert_allclose(per_class["precision"], 100 * p)
    np.testing.assert_allclose(per_class["recall"], 100 * r)
    np.testing.assert_allclose(per_class["accuracy"], 100 * r)
    np.testing.assert_allclose(per_class["f1"], 100 * f)
    assert metrics["overall_accuracy"] == pytest.approx(100 * accuracy_score(y_true, y_pred))
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        assert metrics[f"{avg}_precision"] == pytest.approx(100 * p)
        assert metrics[f"{avg}_recall"] == pytest.approx(100 * r)
        assert metrics[f"{avg}_f1"] == pytest.approx(100 * f)


def test_empty_confusion_is_all_zero():
    metrics, per_class = summarize_confusion(torch.zeros((3, 3), dtype=torch.long))
    assert all(v == 0.0 for v in metrics.values())
    assert per_class["support"] == [0, 0, 0]


def test_safe_div():
    assert safe_div(1, 4) == 0.25
    assert safe_div(3, 0) == 0.0

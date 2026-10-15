"""Tests for the frh04 confusion-matrix / micro-macro metrics script."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compute_model_metrics as cmm  # noqa: E402

from agricultureai.evaluation.selection_protocol import confusion  # noqa: E402
from agricultureai.pipeline import GLOBAL_ORCHARDS, NUM_GLOBAL_CLASSES  # noqa: E402


def test_summarize_matches_sklearn_micro_and_macro():
    rng = np.random.default_rng(0)
    y = rng.integers(0, NUM_GLOBAL_CLASSES, size=500)
    pred = np.where(rng.random(500) < 0.7, y, rng.integers(0, NUM_GLOBAL_CLASSES, size=500))
    pred[pred == 4] = 3  # one class never predicted -> precision 0 (zero_division=0)
    s = cmm.summarize(confusion(y, pred))
    labels = list(range(NUM_GLOBAL_CLASSES))
    for avg in ("micro", "macro"):
        p, r, f, _ = precision_recall_fscore_support(y, pred, labels=labels, average=avg, zero_division=0)
        np.testing.assert_allclose(
            [s[f"{avg}_precision"], s[f"{avg}_recall"], s[f"{avg}_f1"]], [100 * p, 100 * r, 100 * f], atol=1e-9
        )
    assert s["micro_f1"] == s["accuracy"] == 100.0 * np.mean(y == pred)


def test_verifier_decode_orchard_only_above_threshold():
    p_meta = np.array([[0.1, 0.2, 0, 0, 0, 0.6, 0, 0.1, 0],   # meta says orchard, runner-up wheat
                       [0.0, 0.0, 0, 0, 0, 0.3, 0, 0.7, 0]])  # meta says meadow
    score = np.array([0.4, 1.8])
    rows = cmm.verifier_decode(p_meta, score, np.array([0.5, 1.5, 2.0]))
    assert rows.tolist() == [[1, GLOBAL_ORCHARDS], [1, GLOBAL_ORCHARDS], [1, 7]]

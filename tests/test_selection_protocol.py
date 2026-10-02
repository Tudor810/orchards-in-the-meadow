"""Tests for out-of-sample decode selection (spatial cross-fitting + bootstrap)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from agricultureai.evaluation.selection_protocol import (
    bootstrap_metric_samples,
    classification_metrics,
    cross_fit,
    percentile_ci,
    select_accuracy_first,
    select_closest_stretch,
    select_orchard_first,
    spatial_block_folds,
)
from agricultureai.pipeline import GLOBAL_ORCHARDS, GLOBAL_WHEAT


def test_spatial_block_folds_keep_blocks_together_and_balance_sizes():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 100_000, 5000), rng.uniform(0, 100_000, 5000)
    folds, blocks = spatial_block_folds(x, y, block_size=10_000, n_folds=5, seed=0)
    assert set(np.unique(folds)) == set(range(5))
    for b in np.unique(blocks):
        assert len(np.unique(folds[blocks == b])) == 1  # a block never straddles folds
    cell = np.floor(x / 10_000) * 1000 + np.floor(y / 10_000)
    assert len(np.unique(blocks)) == len(np.unique(cell))
    sizes = np.bincount(folds)
    assert sizes.max() - sizes.min() <= np.bincount(blocks).max()
    again, _ = spatial_block_folds(x, y, block_size=10_000, n_folds=5, seed=0)
    assert np.array_equal(folds, again)


def test_spatial_block_folds_handle_missing_coordinates():
    x = np.array([0.0, 1.0, np.nan, 50_000.0, np.nan])
    y = np.array([0.0, 1.0, 5.0, 50_000.0, np.nan])
    folds, blocks = spatial_block_folds(x, y, block_size=10_000, n_folds=2, seed=1)
    assert folds.min() >= 0 and folds.max() < 2
    assert blocks[0] == blocks[1]  # same cell
    assert len({blocks[2], blocks[4], blocks[0], blocks[3]}) == 4  # missing ones get own blocks
    with pytest.raises(ValueError):
        spatial_block_folds(x, y, block_size=10_000, n_folds=1)


def test_classification_metrics_match_sklearn():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 9, 4000)
    y_pred = np.where(rng.random(4000) < 0.7, y_true, rng.integers(0, 9, 4000))
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(100 * (y_true == y_pred).mean())
    assert m["orchard_recall"] == pytest.approx(
        100 * recall_score(y_true, y_pred, labels=[GLOBAL_ORCHARDS], average="macro")
    )
    assert m["orchard_precision"] == pytest.approx(
        100 * precision_score(y_true, y_pred, labels=[GLOBAL_ORCHARDS], average="macro")
    )
    assert m["orchard_f1"] == pytest.approx(100 * f1_score(y_true, y_pred, labels=[GLOBAL_ORCHARDS], average="macro"))
    assert m["wheat_recall"] == pytest.approx(
        100 * recall_score(y_true, y_pred, labels=[GLOBAL_WHEAT], average="macro")
    )
    assert m["macro_f1"] == pytest.approx(100 * f1_score(y_true, y_pred, average="macro"))


def test_macro_averages_skip_absent_classes():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 0])
    m = classification_metrics(y_true, y_pred)
    assert m["macro_recall"] == pytest.approx(100 * (1.0 + 0.5) / 2)


def _row(acc, orch):
    return {"accuracy": acc, "orchard_recall": orch}


def test_selection_rules_respect_floors_and_fall_back():
    rows = [_row(81.5, 30.0), _row(81.2, 40.0), _row(80.8, 50.0), _row(80.0, 62.0)]
    assert select_accuracy_first(rows) == 1  # 81.5 fails the 35% orchard floor
    assert select_orchard_first(rows) == 2  # 80.0 fails the 80.69% accuracy floor
    assert select_closest_stretch(rows) == 3  # shortfall 1.0 beats 10.2 / 20 / 30
    low = [_row(79.0, 20.0), _row(78.0, 25.0)]
    assert select_accuracy_first(low) == 1  # no config reaches 35% -> most orchard recall
    assert select_orchard_first(low) == 0  # no config reaches 80.69% -> most accuracy
    assert select_accuracy_first([_row(81.0, 40.0), _row(81.0, 40.0)]) == 0  # ties -> first


def test_cross_fit_never_selects_on_the_held_out_fold():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    wrong = 1 - y
    # Config 0 is perfect on fold 0 only, config 1 on fold 1 only.
    preds = np.stack([np.where(folds == 0, y, wrong), np.where(folds == 1, y, wrong)])
    rule = lambda rows: int(np.argmax([r["accuracy"] for r in rows]))  # noqa: E731
    out, chosen = cross_fit(preds, y, folds, rule)
    assert chosen == [1, 0]  # each fold uses the config chosen on the other fold
    assert np.array_equal(out, wrong)  # so neither fold gets its own perfect config


def test_bootstrap_is_paired_reproducible_and_brackets_the_estimate():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 9, 3000)
    a = np.where(rng.random(3000) < 0.8, y, rng.integers(0, 9, 3000))
    groups = np.repeat(np.arange(60), 50)
    s = bootstrap_metric_samples(y, {"a": a, "same": a.copy()}, n_boot=200, seed=0, groups=groups)
    assert np.array_equal(s["a"]["accuracy"], s["same"]["accuracy"])  # paired resamples
    lo, hi = percentile_ci(s["a"]["accuracy"])
    assert lo <= classification_metrics(y, a)["accuracy"] <= hi
    s2 = bootstrap_metric_samples(y, {"a": a}, n_boot=200, seed=0, groups=groups)
    assert np.array_equal(s["a"]["accuracy"], s2["a"]["accuracy"])
    plain = bootstrap_metric_samples(y, {"a": a}, n_boot=200, seed=0)
    assert plain["a"]["accuracy"].shape == (200,)

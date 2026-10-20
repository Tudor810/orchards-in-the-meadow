"""Meta-classifier training helpers shared by the training, verifier and rotation scripts.

Feature matrices use the compact layout of :mod:`agricultureai.meta_features` (optionally without whole
blocks) plus extra per-parcel blocks (phenology, network logits) checked against the cache's parcel ids.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from agricultureai.meta_features import (
    COMPACT_BLOCK_NAMES,
    build_catboost,
    build_xgb,
    fit_catboost,
    fit_xgb,
    full_feature_blocks,
    load_feature_matrix,
    orchard_sample_weights,
    predict_proba_catboost,
    predict_proba_xgb,
    select_feature_layout,
)
from agricultureai.pipeline import NUM_GLOBAL_CLASSES

MISSING_CLASS_WEIGHT = 1e-6


def compact_column_names() -> List[str]:
    blocks = full_feature_blocks()
    return [f"{b}_{i}" for b in COMPACT_BLOCK_NAMES for i in range(len(blocks[b]))]


def load_features(cache: Path, extras: List[Tuple[str, Path]], drop_blocks: Tuple[str, ...] = ()) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    x_full, y = load_feature_matrix(cache, use_metadata=True)
    x, names = select_feature_layout(x_full, "compact"), compact_column_names()
    if drop_blocks:  # ablation: remove whole compact blocks (e.g. the expert outputs)
        keep = [i for i, n in enumerate(names) if n.rsplit("_", 1)[0] not in drop_blocks]
        x, names = x[:, keep], [names[i] for i in keep]
    if extras:
        ids = torch.load(cache, map_location="cpu", weights_only=True)["parcel_ids"].numpy()
        for block, path in extras:
            data = np.load(path)
            if not np.array_equal(data["parcel_ids"], ids):
                raise RuntimeError(f"{path}: parcel ids do not match {cache}")
            x = np.hstack([x, data["features"].astype(np.float32)])
            names += [f"{block}:{c}" for c in data["columns"]]
    return x, y, names


def ensure_all_classes(x: np.ndarray, y: np.ndarray, w: np.ndarray, rng: np.random.Generator):
    """XGBoost needs every class: add one copy of a random row per missing class, with negligible weight."""
    missing = sorted(set(range(NUM_GLOBAL_CLASSES)) - set(np.unique(y).tolist()))
    if not missing:
        return x, y, w
    rows = rng.integers(0, len(y), size=len(missing))
    return (
        np.vstack([x, x[rows]]),
        np.concatenate([y, np.array(missing, dtype=y.dtype)]),
        np.concatenate([w, np.full(len(missing), MISSING_CLASS_WEIGHT, dtype=w.dtype)]),
    )


def fit_predict(x_tr, y_tr, x_ev, *, m: float, seed: int):
    """XGBoost + CatBoost with orchards up-weighted by ``m``; returns (p_xgb, p_cat, xgb model)."""
    w = orchard_sample_weights(y_tr, m)
    x_tr, y_tr, w = ensure_all_classes(x_tr, y_tr, w, np.random.default_rng(seed))
    xgb = build_xgb(seed)
    fit_xgb(xgb, x_tr, y_tr, w)
    cat = build_catboost(seed)
    fit_catboost(cat, x_tr, y_tr, w)
    return predict_proba_xgb(xgb, x_ev), predict_proba_catboost(cat, x_ev), xgb


def fit_xgb_scores(x_tr, y_tr, x_ev, *, seed: int) -> np.ndarray:
    """XGBoost only, m = 1 (the verifier's out-of-fold meta scores)."""
    w = orchard_sample_weights(y_tr, 1.0)
    x_tr, y_tr, w = ensure_all_classes(x_tr, y_tr, w, np.random.default_rng(seed))
    model = build_xgb(seed)
    fit_xgb(model, x_tr, y_tr, w)
    return predict_proba_xgb(model, x_ev).astype(np.float32)


def top_gain(model, names: List[str], k: int = 15) -> List[Tuple[str, float]]:
    gain = model.get_booster().get_score(importance_type="gain")
    scored = [(names[int(f[1:])], v) for f, v in gain.items()]
    total = sum(v for _, v in scored) or 1.0
    return [(n, v / total) for n, v in sorted(scored, key=lambda kv: -kv[1])[:k]]

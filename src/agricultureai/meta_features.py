"""Feature construction for the late-fusion meta-classifier.

Full neural block (58-d) when MSResNet + Transformer logits are present::

    [ raw DMOE 15 | cereal engineered 7 | MSResNet 9+10 | branch soft 7 | Transformer 10 ]

Reduced neural block (29-d) when only DMOE branch logits are cached::

    [ raw DMOE 15 | cereal engineered 7 | branch soft 7 ]

Metadata block (8-d)::

    log1p(area), max_ndvi, ndvi_amplitude, centroid_lat, mean_doy, std_doy,
    doy_at_max_ndvi, log1p(n_valid_obs)

Parcels whose MSResNet / Transformer logits were zero-filled during L1C parcel-ID
alignment (``*_l1c_matched == False``) get NaN in those 29 columns, so tree
models treat them as missing instead of as a uniform prediction.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from agricultureai.pipeline import (
    GLOBAL_ORCHARDS,
    GLOBAL_PERMANENT_MEADOWS,
    GLOBAL_TEMPORARY_MEADOWS,
    NUM_GLOBAL_CLASSES,
)

RAW_FEATURE_DIM = 15
CEREAL_ENGINEERED_DIM = 7
GLOBAL_LOGIT_DIM = 9
MSRESNET_ENGINEERED_DIM = 10
BRANCH_SOFTMAX_DIM = 7
TRANSFORMER_BLOCK_DIM = 10
DMOE_ONLY_NEURAL_DIM = RAW_FEATURE_DIM + CEREAL_ENGINEERED_DIM + BRANCH_SOFTMAX_DIM
FULL_NEURAL_FEATURE_DIM = (
    RAW_FEATURE_DIM
    + CEREAL_ENGINEERED_DIM
    + GLOBAL_LOGIT_DIM
    + MSRESNET_ENGINEERED_DIM
    + BRANCH_SOFTMAX_DIM
    + TRANSFORMER_BLOCK_DIM
)
# Backward-compatible alias for caches that include global model logits.
NEURAL_FEATURE_DIM = FULL_NEURAL_FEATURE_DIM
METADATA_FEATURE_DIM = 8
FULL_FEATURE_DIM = FULL_NEURAL_FEATURE_DIM + METADATA_FEATURE_DIM
DMOE_ONLY_FULL_FEATURE_DIM = DMOE_ONLY_NEURAL_DIM + METADATA_FEATURE_DIM

# Compact stretch layout: drop L2b/L2c + near-noise L1/L2a/global_eng/L3_soft.
# Columns follow ``build_neural_feature_matrix`` (+ metadata):
#   l1(4) l2a(2) l2b(2) l2c(2) l3(5) | cereal(7) | msresnet(9) global_eng(10)
#   | l2c_soft(2) l3_soft(5) | transformer(10) | metadata(8)
COMPACT_BLOCK_NAMES: Tuple[str, ...] = (
    "l3_raw",
    "cereal",
    "msresnet",
    "transformer",
    "metadata",
)
COMPACT_FEATURE_DIM = 5 + 7 + 9 + 10 + 8  # 39

BASELINE_ACCURACY = 80.69
BASELINE_ORCHARD_RECALL = 35.0
STRETCH_ACCURACY = 81.0
STRETCH_ORCHARD_RECALL = 60.0

METADATA_COLS = (
    "log1p_area_ha",
    "max_ndvi",
    "ndvi_amplitude",
    "centroid_lat",
    "mean_doy_norm",
    "std_doy_norm",
    "doy_at_max_ndvi_norm",
    "log1p_n_valid_obs",
)


def full_feature_blocks(full_dim: int = FULL_FEATURE_DIM) -> Dict[str, range]:
    """Name -> column range for the full 58-neural + 8-metadata layout."""
    if full_dim != FULL_FEATURE_DIM:
        raise ValueError(
            f"Compact layouts require the full {FULL_FEATURE_DIM}-d matrix; got {full_dim}."
        )
    meta_start = full_dim - METADATA_FEATURE_DIM
    return {
        "l1": range(0, 4),
        "l2a": range(4, 6),
        "l2b": range(6, 8),
        "l2c_raw": range(8, 10),
        "l3_raw": range(10, 15),
        "cereal": range(15, 22),
        "msresnet": range(22, 31),
        "global_eng": range(31, 41),
        "l2c_soft": range(41, 43),
        "l3_soft": range(43, 48),
        "transformer": range(48, 58),
        "metadata": range(meta_start, full_dim),
    }


def feature_column_indices(
    block_names: Tuple[str, ...] | list[str],
    *,
    full_dim: int = FULL_FEATURE_DIM,
) -> np.ndarray:
    blocks = full_feature_blocks(full_dim)
    keep: list[int] = []
    for name in block_names:
        if name not in blocks:
            raise KeyError(f"Unknown feature block {name!r}. Known: {sorted(blocks)}")
        keep.extend(blocks[name])
    return np.array(sorted(keep), dtype=np.int64)


def select_feature_layout(
    x: np.ndarray,
    layout: str = "full",
) -> np.ndarray:
    """Slice a full feature matrix to a named layout.

    Layouts
    -------
    full:
        All 66 columns (58 neural + 8 metadata).
    minus_l2b_l2c:
        Drop L2b and L2c (raw + soft).
    compact:
        Keep L3 raw, cereal, MSResNet logits, Transformer block, metadata (39-d).
    """
    if layout == "full":
        return x
    if x.shape[1] != FULL_FEATURE_DIM:
        raise ValueError(
            f"Layout '{layout}' expects {FULL_FEATURE_DIM}-d input; got {x.shape[1]}-d."
        )
    if layout == "minus_l2b_l2c":
        names = [
            n
            for n in full_feature_blocks()
            if n not in ("l2b", "l2c_raw", "l2c_soft")
        ]
        return x[:, feature_column_indices(names)]
    if layout == "compact":
        cols = feature_column_indices(COMPACT_BLOCK_NAMES)
        if cols.size != COMPACT_FEATURE_DIM:
            raise RuntimeError(
                f"Compact layout expected {COMPACT_FEATURE_DIM} cols; got {cols.size}."
            )
        return x[:, cols]
    raise ValueError(f"Unknown feature layout {layout!r}")



def cache_has_global_logits(payload: dict) -> bool:
    return "msresnet_logits" in payload and "transformer_logits" in payload


def feature_dims(*, use_metadata: bool, payload: dict | None = None) -> Tuple[int, int]:
    """Return ``(neural_dim, full_dim)`` for a cache payload or the full layout."""
    if payload is not None and not cache_has_global_logits(payload):
        neural = DMOE_ONLY_NEURAL_DIM
        full = DMOE_ONLY_FULL_FEATURE_DIM if use_metadata else neural
        return neural, full
    neural = FULL_NEURAL_FEATURE_DIM
    full = FULL_FEATURE_DIM if use_metadata else neural
    return neural, full


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32, copy=False)


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def top12_margin(probs: np.ndarray) -> np.ndarray:
    top2 = np.partition(probs, -2, axis=1)[:, -2:]
    return (top2[:, 1:2] - top2[:, 0:1]).astype(np.float32)


def wheat_barley_margin(l2a: np.ndarray) -> np.ndarray:
    """L2a local index 0=barley, 1=wheat."""
    p_l2a = softmax_rows(l2a)
    return (p_l2a[:, 1:2] - p_l2a[:, 0:1]).astype(np.float32)


def cereal_engineered_features(l1: np.ndarray, l2a: np.ndarray) -> np.ndarray:
    p_l1 = softmax_rows(l1)
    p_l2a = softmax_rows(l2a)
    l2a_margin = wheat_barley_margin(l2a)
    return np.concatenate([p_l1, p_l2a, l2a_margin], axis=1)


def global_model_engineered_features(logits: np.ndarray) -> np.ndarray:
    probs = softmax_rows(logits)
    return np.concatenate([probs, top12_margin(probs)], axis=1)


def branch_softmax_features(l2c: np.ndarray, l3_neg: np.ndarray) -> np.ndarray:
    p_l2c = softmax_rows(l2c)
    p_l3 = softmax_rows(l3_neg)
    return np.concatenate([p_l2c, p_l3], axis=1)


def transformer_block(logits: np.ndarray) -> np.ndarray:
    probs = softmax_rows(logits)
    return np.concatenate([logits, top12_margin(probs)], axis=1)


def raw_branch_features(payload: dict) -> np.ndarray:
    l1 = tensor_to_numpy(payload["l1_logits"])
    l2a = tensor_to_numpy(payload["l2a_logits"])
    l2b = tensor_to_numpy(payload["l2b_logits"])
    l2c = tensor_to_numpy(payload["l2c_logits"])
    l3_neg = -tensor_to_numpy(payload["l3_distances"])
    raw = np.concatenate([l1, l2a, l2b, l2c, l3_neg], axis=1)
    if raw.shape[1] != RAW_FEATURE_DIM:
        raise ValueError(f"Expected raw dim {RAW_FEATURE_DIM}; got {raw.shape[1]}")
    return raw


def global_logits_missing_mask(payload: dict, model: str) -> np.ndarray:
    """True for rows whose ``{model}_logits`` are zero-filled placeholders.

    ``append_l1c_cached_global_logits.py`` aligns L1C logits by parcel ID,
    zero-fills L2A parcels absent from the L1C cache, and records
    ``{model}_l1c_matched``. Caches without that key have no placeholders.
    """
    key = f"{model}_l1c_matched"
    if key not in payload:
        return np.zeros(payload[f"{model}_logits"].shape[0], dtype=bool)
    return ~payload[key].detach().cpu().numpy().astype(bool)


def build_neural_feature_matrix(payload: dict) -> np.ndarray:
    """Build the neural feature block (29-d or 58-d depending on cache contents)."""
    l1 = tensor_to_numpy(payload["l1_logits"])
    l2a = tensor_to_numpy(payload["l2a_logits"])
    l2c = tensor_to_numpy(payload["l2c_logits"])
    l3_neg = -tensor_to_numpy(payload["l3_distances"])
    raw = raw_branch_features(payload)
    blocks = [
        raw,
        cereal_engineered_features(l1, l2a),
        branch_softmax_features(l2c, l3_neg),
    ]
    if cache_has_global_logits(payload):
        msresnet = tensor_to_numpy(payload["msresnet_logits"])
        transformer = tensor_to_numpy(payload["transformer_logits"])
        msresnet_feats = np.concatenate(
            [msresnet, global_model_engineered_features(msresnet)], axis=1
        )
        msresnet_feats[global_logits_missing_mask(payload, "msresnet")] = np.nan
        transformer_feats = transformer_block(transformer)
        transformer_feats[global_logits_missing_mask(payload, "transformer")] = np.nan
        blocks[2:2] = [msresnet_feats]
        blocks.append(transformer_feats)
    return np.concatenate(blocks, axis=1)


def load_cache_payload(cache_path: Path) -> dict:
    if not cache_path.is_file():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    required = (
        "l1_logits",
        "l2a_logits",
        "l2b_logits",
        "l2c_logits",
        "l3_distances",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Cache missing keys: {missing}")
    if "labels" in payload:
        labels = payload["labels"]
    elif "targets" in payload:
        labels = payload["targets"]
    else:
        raise KeyError("Cache must contain 'labels' or 'targets'.")
    payload["_labels_np"] = labels.detach().cpu().numpy().astype(np.int64, copy=False)
    return payload


def load_feature_matrix(
    cache_path: Path,
    *,
    use_metadata: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load ``(N, D)`` features and labels from a logit cache."""
    payload = load_cache_payload(cache_path)
    x = build_neural_feature_matrix(payload)
    y = payload["_labels_np"]

    if use_metadata:
        if "parcel_metadata" not in payload:
            raise KeyError(
                f"{cache_path} has no parcel_metadata. "
                "Run scripts/cascade/append_parcel_metadata.py first."
            )
        meta = tensor_to_numpy(payload["parcel_metadata"])
        if meta.shape[0] != x.shape[0]:
            raise ValueError(
                f"Metadata rows {meta.shape[0]} != feature rows {x.shape[0]}"
            )
        if meta.shape[1] != METADATA_FEATURE_DIM:
            raise ValueError(
                f"Expected metadata dim {METADATA_FEATURE_DIM}; got {meta.shape[1]}. "
                "Re-run append_parcel_metadata.py --with-area --with-doy."
            )
        x = np.concatenate([x, meta.astype(np.float32)], axis=1)

    _, expected_full = feature_dims(use_metadata=use_metadata, payload=payload)
    if x.shape[1] != expected_full:
        raise ValueError(f"Expected feature dim {expected_full}; got {x.shape[1]}")
    if y.min() < 0 or y.max() >= NUM_GLOBAL_CLASSES:
        raise ValueError(f"Invalid label range: min={y.min()}, max={y.max()}")
    return x, y


def orchard_sample_weights(y_train: np.ndarray, orchard_multiplier: float) -> np.ndarray:
    from sklearn.utils.class_weight import compute_sample_weight

    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    weights[y_train == GLOBAL_ORCHARDS] *= orchard_multiplier
    return weights.astype(np.float64)


def geometric_mean_probs(p_a: np.ndarray, p_b: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Row-wise geometric mean of two probability matrices, renormalized."""
    g = np.sqrt(np.maximum(p_a, eps) * np.maximum(p_b, eps))
    g /= np.sum(g, axis=1, keepdims=True)
    return g.astype(np.float32)


def weighted_arithmetic_probs(
    p_a: np.ndarray,
    p_b: np.ndarray,
    *,
    weight_a: float = 0.5,
    eps: float = 1e-12,
) -> np.ndarray:
    """Row-wise weighted arithmetic mean of two probability matrices."""
    w = float(weight_a)
    mixed = w * p_a + (1.0 - w) * p_b
    mixed = np.clip(mixed, eps, None)
    mixed /= np.sum(mixed, axis=1, keepdims=True)
    return mixed.astype(np.float32)


def make_blend(name: str, p_xgb: np.ndarray, p_cat: np.ndarray) -> np.ndarray:
    """XGBoost / CatBoost probability blend by name (geo_mean, arith_mean, arith_xgb0.7, xgb_only, cat_only)."""
    if name == "geo_mean":
        return geometric_mean_probs(p_xgb, p_cat)
    if name == "arith_mean":
        return weighted_arithmetic_probs(p_xgb, p_cat, weight_a=0.5)
    if name == "arith_xgb0.7":
        return weighted_arithmetic_probs(p_xgb, p_cat, weight_a=0.7)
    if name == "xgb_only":
        return p_xgb
    if name == "cat_only":
        return p_cat
    raise ValueError(f"Unknown blend {name!r}")


def apply_class_log_bias(
    probs: np.ndarray,
    class_id: int,
    bias: float,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Add ``bias`` in log-space to one class column; return renormalized probs."""
    return apply_log_biases(probs, {int(class_id): float(bias)}, eps=eps)


def apply_log_biases(
    probs: np.ndarray,
    biases: Dict[int, float],
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Add per-class log-biases; return renormalized probabilities."""
    log_p = np.log(np.maximum(probs, eps))
    for class_id, bias in biases.items():
        if abs(bias) > 0.0:
            log_p[:, int(class_id)] += float(bias)
    log_p -= np.max(log_p, axis=1, keepdims=True)
    exp = np.exp(log_p)
    exp /= np.sum(exp, axis=1, keepdims=True)
    return exp.astype(np.float32)


def apply_orchard_meadow_log_bias(
    probs: np.ndarray,
    *,
    b_orchard: float = 0.0,
    b_meadow: float = 0.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Boost orchards and down-weight both meadow classes in log-space.

    ``b_meadow`` is subtracted from permanent and temporary meadow logits, which
    reduces the dominant orchard false-positive mode (meadow→orchard).
    """
    return apply_log_biases(
        probs,
        {
            GLOBAL_ORCHARDS: float(b_orchard),
            GLOBAL_PERMANENT_MEADOWS: -float(b_meadow),
            GLOBAL_TEMPORARY_MEADOWS: -float(b_meadow),
        },
        eps=eps,
    )


def decode_probs(probs: np.ndarray) -> np.ndarray:
    return np.argmax(probs, axis=1).astype(np.int64)


def orchard_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pred_mask = y_pred == GLOBAL_ORCHARDS
    if not pred_mask.any():
        return 0.0
    return float((y_true[pred_mask] == GLOBAL_ORCHARDS).mean() * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, recall_score

    acc = accuracy_score(y_true, y_pred) * 100.0
    orchard_mask = y_true == GLOBAL_ORCHARDS
    orchard_recall = (
        float((y_pred[orchard_mask] == GLOBAL_ORCHARDS).mean() * 100.0)
        if orchard_mask.any()
        else 0.0
    )
    wheat_mask = y_true == 1
    wheat_recall = (
        float((y_pred[wheat_mask] == 1).mean() * 100.0) if wheat_mask.any() else 0.0
    )
    macro_recall = (
        recall_score(
            y_true,
            y_pred,
            labels=list(range(NUM_GLOBAL_CLASSES)),
            average="macro",
            zero_division=0,
        )
        * 100.0
    )
    return {
        "accuracy": acc,
        "orchard_recall": orchard_recall,
        "orchard_precision": orchard_precision(y_true, y_pred),
        "wheat_recall": wheat_recall,
        "macro_recall": macro_recall,
        "acc_ok": acc > BASELINE_ACCURACY,
        "orchard_ok": orchard_recall > BASELINE_ORCHARD_RECALL,
    }


def format_metrics(name: str, metrics: Dict[str, float]) -> str:
    return (
        f"{name:40s}  acc={metrics['accuracy']:6.2f}%  "
        f"orchard={metrics['orchard_recall']:5.2f}%  "
        f"wheat={metrics['wheat_recall']:5.2f}%  "
        f"acc_ok={'Y' if metrics['acc_ok'] else 'N'}  "
        f"orch_ok={'Y' if metrics['orchard_ok'] else 'N'}"
    )


def _has_xgboost() -> bool:
    return importlib.util.find_spec("xgboost") is not None


def build_xgb(
    seed: int,
    *,
    max_depth: int = 8,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    n_estimators: int = 400,
    learning_rate: float = 0.05,
) -> Any:
    if not _has_xgboost():
        raise RuntimeError("XGBoost not installed.")
    import xgboost as xgb

    return xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=NUM_GLOBAL_CLASSES,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        eval_metric="mlogloss",
    )


def fit_xgb(
    clf: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
) -> Any:
    clf.fit(x_train, y_train, sample_weight=sample_weight, verbose=False)
    return clf


def predict_xgb(clf: Any, x: np.ndarray) -> np.ndarray:
    return clf.predict(x)


def predict_proba_xgb(clf: Any, x: np.ndarray) -> np.ndarray:
    return clf.predict_proba(x).astype(np.float32)


def _has_catboost() -> bool:
    return importlib.util.find_spec("catboost") is not None


def build_catboost(
    seed: int,
    *,
    depth: int = 8,
    iterations: int = 400,
    learning_rate: float = 0.05,
) -> Any:
    if not _has_catboost():
        raise RuntimeError("CatBoost not installed.")
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        loss_function="MultiClass",
        classes_count=NUM_GLOBAL_CLASSES,
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def fit_catboost(
    clf: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
) -> Any:
    clf.fit(x_train, y_train, sample_weight=sample_weight, verbose=False)
    return clf


def predict_proba_catboost(clf: Any, x: np.ndarray) -> np.ndarray:
    return clf.predict_proba(x).astype(np.float32)

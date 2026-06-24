"""Late-fusion meta-classifier on cached DMOE branch outputs.

Regional protocol (default)::

    Train meta-classifier on frh01–frh03 cached logits
    Evaluate on held-out frh04 cached logits

Production feature layout (64-d with metadata)::

    58-d neural block from cached DMOE / MSResNet / Transformer outputs
    + 6-d parcel metadata (area, NDVI, lat, DOY)
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

from agricultureai.meta_features import (
    BASELINE_ACCURACY,
    BASELINE_ORCHARD_RECALL,
    FULL_FEATURE_DIM,
    NEURAL_FEATURE_DIM,
    load_feature_matrix,
    orchard_sample_weights,
)
from agricultureai.pipeline import (
    GLOBAL_ORCHARDS,
    GLOBAL_TEMPORARY_MEADOWS,
    NUM_GLOBAL_CLASSES,
)

GLOBAL_ID_TO_NAME: Dict[int, str] = {
    0: "barley",
    1: "wheat",
    2: "rapeseed",
    3: "corn",
    4: "sunflower",
    5: "orchards",
    6: "nuts",
    7: "permanent meadows",
    8: "temporary meadows",
}

DEFAULT_TRAIN_CACHE = Path("artifacts/cached_logits_train.pt")
DEFAULT_EVAL_CACHE = Path("artifacts/cached_logits_eval.pt")


def _has_xgboost() -> bool:
    return importlib.util.find_spec("xgboost") is not None


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)


def _cache_regions(cache_path: Path) -> Optional[list]:
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    meta = payload.get("meta")
    if isinstance(meta, dict):
        regions = meta.get("regions")
        if regions is not None:
            return list(regions)
    return None


def stratified_split(
    x: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


def build_classifier(name: str, seed: int) -> Any:
    choice = name.lower()
    if choice == "auto":
        choice = "xgboost" if _has_xgboost() else "rf"

    if choice in ("xgboost", "xgb"):
        if not _has_xgboost():
            raise RuntimeError(
                "XGBoost is not installed. Use --classifier rf or install xgboost."
            )
        import xgboost as xgb

        return xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=NUM_GLOBAL_CLASSES,
            n_estimators=400,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
            eval_metric="mlogloss",
        )

    if choice in ("rf", "random_forest", "randomforest"):
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown classifier {name!r}; use auto, xgboost, or rf.")


def fit_classifier(
    clf: Any, x_train: np.ndarray, y_train: np.ndarray, orchard_multiplier: float
) -> None:
    sample_weight = orchard_sample_weights(y_train, orchard_multiplier)
    print(
        f"Applying {orchard_multiplier}x weight multiplier to "
        f"Orchards (Class {GLOBAL_ORCHARDS})..."
    )
    if clf.__class__.__name__ == "XGBClassifier":
        clf.fit(x_train, y_train, sample_weight=sample_weight, verbose=False)
    else:
        clf.fit(x_train, y_train, sample_weight=sample_weight)


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, class_id: int) -> float:
    mask = y_true == class_id
    if mask.sum() == 0:
        return 0.0
    return float((y_pred[mask] == class_id).mean() * 100.0)


def print_confusion_matrix(cm: np.ndarray) -> None:
    n = cm.shape[0]
    print("\nConfusion matrix (rows=true, cols=pred):")
    header = "true\\pred".ljust(14) + "".join(
        f"{GLOBAL_ID_TO_NAME[c][:6]:>8}" for c in range(n)
    )
    print(header)
    for row in range(n):
        name = GLOBAL_ID_TO_NAME[row][:12].ljust(14)
        counts = "".join(f"{int(cm[row, col]):>8}" for col in range(n))
        print(f"{name}{counts}")


def evaluate_split(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    split_label: str,
) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred) * 100.0
    orchard_recall = per_class_recall(y_true, y_pred, GLOBAL_ORCHARDS)
    wheat_recall = per_class_recall(y_true, y_pred, 1)
    tm_recall = per_class_recall(y_true, y_pred, GLOBAL_TEMPORARY_MEADOWS)
    macro_recall = recall_score(
        y_true, y_pred, labels=list(range(NUM_GLOBAL_CLASSES)), average="macro", zero_division=0
    ) * 100.0

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_GLOBAL_CLASSES)))

    print(f"\n=== {split_label} ===")
    print(f"Samples: {len(y_true):,}")
    print(f"Global accuracy: {acc:.2f}%")
    print(f"Wheat recall (class 1): {wheat_recall:.2f}%")
    print(f"Orchard recall (class {GLOBAL_ORCHARDS}): {orchard_recall:.2f}%")
    print(
        f"Temporary meadow recall (class {GLOBAL_TEMPORARY_MEADOWS}): "
        f"{tm_recall:.2f}%"
    )
    print(f"Macro recall (9 classes): {macro_recall:.2f}%")

    acc_ok = acc > BASELINE_ACCURACY
    orchard_ok = orchard_recall > BASELINE_ORCHARD_RECALL
    print(
        f"\nBaseline targets: accuracy > {BASELINE_ACCURACY:.2f}%, "
        f"orchard recall > {BASELINE_ORCHARD_RECALL:.2f}%"
    )
    print(f"  Accuracy target met: {'YES' if acc_ok else 'NO'}")
    print(f"  Orchard recall target met: {'YES' if orchard_ok else 'NO'}")

    print_confusion_matrix(cm)

    return {
        "accuracy": acc,
        "wheat_recall": wheat_recall,
        "orchard_recall": orchard_recall,
        "temporary_meadow_recall": tm_recall,
        "macro_recall": macro_recall,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a late-fusion meta-classifier on cached DMOE logits."
    )
    parser.add_argument("--train-cache", type=Path, default=DEFAULT_TRAIN_CACHE)
    parser.add_argument("--eval-cache", type=Path, default=DEFAULT_EVAL_CACHE)
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Legacy single-cache mode with random stratified split.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classifier",
        choices=("auto", "xgboost", "xgb", "rf", "random_forest"),
        default="auto",
    )
    parser.add_argument(
        "--use-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append 6-d parcel metadata (default: on).",
    )
    parser.add_argument(
        "--orchard-multiplier",
        type=float,
        default=25.0,
        help="Orchard sample-weight multiplier (production best: 25).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    clf = build_classifier(args.classifier, seed=args.seed)
    backend = clf.__class__.__name__
    print(f"Meta-classifier backend: {backend}")

    if args.cache is not None:
        cache_path = args.cache.resolve()
        print(f"\n[Legacy mode] Loading single cache from {cache_path} ...")
        x, y = load_feature_matrix(cache_path, use_metadata=args.use_metadata)
        print(f"Loaded {x.shape[0]:,} samples, {x.shape[1]} features.")
        x_train, x_val, y_train, y_val = stratified_split(
            x, y, test_size=args.test_size, seed=args.seed
        )
        print("WARNING: this is not the frh01–03 / frh04 regional protocol.")
        fit_classifier(clf, x_train, y_train, orchard_multiplier=args.orchard_multiplier)
        y_pred = clf.predict(x_val)
        evaluate_split(y_val, y_pred, split_label="Validation metrics (random split)")
        return

    train_path = args.train_cache.resolve()
    eval_path = args.eval_cache.resolve()
    train_regions = _cache_regions(train_path)
    eval_regions = _cache_regions(eval_path)

    print(f"\nLoading train cache: {train_path}")
    if train_regions:
        print(f"  regions: {train_regions}")
    x_train, y_train = load_feature_matrix(train_path, use_metadata=args.use_metadata)
    print(f"  samples: {len(y_train):,}, features: {x_train.shape[1]}")

    print(f"\nLoading eval cache: {eval_path}")
    if eval_regions:
        print(f"  regions: {eval_regions}")
    x_eval, y_eval = load_feature_matrix(eval_path, use_metadata=args.use_metadata)
    print(f"  samples: {len(y_eval):,}, features: {x_eval.shape[1]}")

    expected = FULL_FEATURE_DIM if args.use_metadata else NEURAL_FEATURE_DIM
    if x_train.shape[1] != expected:
        raise ValueError(f"Expected {expected} features; got {x_train.shape[1]}")

    print(
        "\nRegional protocol: train meta-classifier on train cache, "
        "evaluate on full eval cache (no random split)."
    )
    fit_classifier(clf, x_train, y_train, orchard_multiplier=args.orchard_multiplier)
    y_pred = clf.predict(x_eval)
    evaluate_split(
        y_eval,
        y_pred,
        split_label="Held-out evaluation (regional split)",
    )


if __name__ == "__main__":
    main()

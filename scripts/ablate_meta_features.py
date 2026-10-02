"""Ablate meta-classifier feature blocks to measure each expert's contribution.

Trains one fixed-m XGBoost per named feature subset (full, minus-L2c,
minus-L2b, global+metadata-only, DMOE-only) and reports global accuracy,
orchard recall, and the permanent/temporary meadow recalls together with the
mutual confusion counts between the two meadow classes.

Column offsets follow ``build_neural_feature_matrix`` (+ appended metadata):

    l1(4) l2a(2) l2b(2) l2c(2) l3(5) | cereal(7) | msresnet(9) global_eng(10)
    | l2c_soft(2) l3_soft(5) | transformer(10) | metadata(8)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agricultureai.meta_features import (
    METADATA_FEATURE_DIM,
    build_xgb,
    feature_dims,
    fit_xgb,
    load_feature_matrix,
    orchard_sample_weights,
    predict_xgb,
)
from agricultureai.pipeline import (
    GLOBAL_ORCHARDS,
    GLOBAL_PERMANENT_MEADOWS,
    GLOBAL_TEMPORARY_MEADOWS,
    NUM_GLOBAL_CLASSES,
)


def _column_blocks(full_dim: int) -> dict[str, range]:
    """Return name -> column range for the full 58-neural + 8-metadata layout."""
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


def _cols(blocks: dict[str, range], names: list[str]) -> np.ndarray:
    keep: list[int] = []
    for name in names:
        keep.extend(blocks[name])
    return np.array(sorted(keep), dtype=np.int64)


def build_subsets(blocks: dict[str, range]) -> dict[str, np.ndarray]:
    all_names = list(blocks.keys())
    return {
        "full": _cols(blocks, all_names),
        "minus_l2c": _cols(blocks, [n for n in all_names if n not in ("l2c_raw", "l2c_soft")]),
        "minus_l2b": _cols(blocks, [n for n in all_names if n != "l2b"]),
        "global_metadata_only": _cols(blocks, ["msresnet", "global_eng", "transformer", "metadata"]),
        "dmoe_only": _cols(
            blocks,
            ["l1", "l2a", "l2b", "l2c_raw", "l3_raw", "cereal", "l2c_soft", "l3_soft"],
        ),
    }


def evaluate_subset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    columns: np.ndarray,
    *,
    orchard_multiplier: float,
    seed: int,
) -> dict:
    clf = build_xgb(seed)
    weights = orchard_sample_weights(y_train, orchard_multiplier)
    fit_xgb(clf, x_train[:, columns], y_train, weights)
    y_pred = predict_xgb(clf, x_eval[:, columns]).astype(np.int64)

    def recall(class_id: int) -> float:
        mask = y_eval == class_id
        return float((y_pred[mask] == class_id).mean() * 100.0) if mask.any() else 0.0

    perm_as_temp = int(
        ((y_eval == GLOBAL_PERMANENT_MEADOWS) & (y_pred == GLOBAL_TEMPORARY_MEADOWS)).sum()
    )
    temp_as_perm = int(
        ((y_eval == GLOBAL_TEMPORARY_MEADOWS) & (y_pred == GLOBAL_PERMANENT_MEADOWS)).sum()
    )
    return {
        "n_features": int(columns.size),
        "accuracy": float((y_pred == y_eval).mean() * 100.0),
        "orchard_recall": recall(GLOBAL_ORCHARDS),
        "permanent_meadow_recall": recall(GLOBAL_PERMANENT_MEADOWS),
        "temporary_meadow_recall": recall(GLOBAL_TEMPORARY_MEADOWS),
        "permanent_predicted_as_temporary": perm_as_temp,
        "temporary_predicted_as_permanent": temp_as_perm,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache", type=Path, default=Path("artifacts/cached_logits_train.pt"))
    p.add_argument("--eval-cache", type=Path, default=Path("artifacts/cached_logits_eval.pt"))
    p.add_argument("--orchard-multiplier", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-log", type=Path, default=Path("artifacts/ablate_meta_features.log"))
    p.add_argument("--output-json", type=Path, default=Path("artifacts/ablate_meta_features.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _, full_dim = feature_dims(use_metadata=True)
    x_train, y_train = load_feature_matrix(args.train_cache, use_metadata=True)
    x_eval, y_eval = load_feature_matrix(args.eval_cache, use_metadata=True)
    if x_train.shape[1] != full_dim:
        raise SystemExit(
            f"Ablation expects the full {full_dim}-d cache; got {x_train.shape[1]}-d. "
            "Rebuild caches with global logits + metadata."
        )

    blocks = _column_blocks(full_dim)
    subsets = build_subsets(blocks)
    results = {
        name: evaluate_subset(
            x_train, y_train, x_eval, y_eval, columns,
            orchard_multiplier=args.orchard_multiplier, seed=args.seed,
        )
        for name, columns in subsets.items()
    }

    header = (
        f"{'subset':22}  {'feats':>5}  {'acc':>7}  {'orch_R':>7}  "
        f"{'permR':>7}  {'tempR':>7}  {'P->T':>6}  {'T->P':>6}"
    )
    lines = [
        "Meta-feature ablation (fixed-m XGBoost)",
        f"Train={len(y_train):,}  Eval={len(y_eval):,}  m={args.orchard_multiplier:g}",
        "",
        header,
        "-" * len(header),
    ]
    for name, r in results.items():
        lines.append(
            f"{name:22}  {r['n_features']:5d}  {r['accuracy']:6.2f}%  "
            f"{r['orchard_recall']:6.2f}%  {r['permanent_meadow_recall']:6.2f}%  "
            f"{r['temporary_meadow_recall']:6.2f}%  "
            f"{r['permanent_predicted_as_temporary']:6d}  "
            f"{r['temporary_predicted_as_permanent']:6d}"
        )
    report = "\n".join(lines)
    print("\n" + report)
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(report + "\n", encoding="utf-8")
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output_log.resolve()}\nSaved: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()

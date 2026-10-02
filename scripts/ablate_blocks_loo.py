"""Leave-one-block-out ablation on the best meta feature base (minus L2b+L2c).

Trains fixed-m XGBoost for the base subset and each variant with one remaining
block removed. Reports accuracy and orchard recall deltas vs base.
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
)


def column_blocks(full_dim: int) -> dict[str, range]:
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


def cols(blocks: dict[str, range], names: list[str]) -> np.ndarray:
    keep: list[int] = []
    for name in names:
        keep.extend(blocks[name])
    return np.array(sorted(keep), dtype=np.int64)


def evaluate(
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

    return {
        "n_features": int(columns.size),
        "accuracy": float((y_pred == y_eval).mean() * 100.0),
        "orchard_recall": recall(GLOBAL_ORCHARDS),
        "permanent_meadow_recall": recall(GLOBAL_PERMANENT_MEADOWS),
        "temporary_meadow_recall": recall(GLOBAL_TEMPORARY_MEADOWS),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache", type=Path, default=Path("artifacts/cached_logits_train.pt"))
    p.add_argument("--eval-cache", type=Path, default=Path("artifacts/cached_logits_eval.pt"))
    p.add_argument("--orchard-multiplier", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-log", type=Path, default=Path("artifacts/ablate_blocks_minus_l2b_l2c.log"))
    p.add_argument("--output-json", type=Path, default=Path("artifacts/ablate_blocks_minus_l2b_l2c.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _, full_dim = feature_dims(use_metadata=True)
    x_train, y_train = load_feature_matrix(args.train_cache, use_metadata=True)
    x_eval, y_eval = load_feature_matrix(args.eval_cache, use_metadata=True)
    blocks = column_blocks(full_dim)

    # Current best stretch-oriented base: drop L2b + L2c.
    base_names = [
        n
        for n in blocks
        if n not in ("l2b", "l2c_raw", "l2c_soft")
    ]
    # Also evaluate full / minus_l2b / minus_l2c for ranking context.
    context = {
        "full": list(blocks.keys()),
        "minus_l2c": [n for n in blocks if n not in ("l2c_raw", "l2c_soft")],
        "minus_l2b": [n for n in blocks if n != "l2b"],
        "minus_l2b_l2c": base_names,
    }

    results: dict[str, dict] = {}
    print(f"Train={len(y_train):,} Eval={len(y_eval):,} m={args.orchard_multiplier:g}")
    for name, names in context.items():
        print(f"Evaluating {name} ({len(cols(blocks, names))} feats) ...", flush=True)
        results[name] = evaluate(
            x_train, y_train, x_eval, y_eval, cols(blocks, names),
            orchard_multiplier=args.orchard_multiplier, seed=args.seed,
        )

    base = results["minus_l2b_l2c"]
    for block in base_names:
        kept = [n for n in base_names if n != block]
        tag = f"base_minus_{block}"
        print(f"Evaluating {tag} ...", flush=True)
        results[tag] = evaluate(
            x_train, y_train, x_eval, y_eval, cols(blocks, kept),
            orchard_multiplier=args.orchard_multiplier, seed=args.seed,
        )
        results[tag]["dropped_block"] = block
        results[tag]["delta_acc"] = results[tag]["accuracy"] - base["accuracy"]
        results[tag]["delta_orchard"] = (
            results[tag]["orchard_recall"] - base["orchard_recall"]
        )

    lines = [
        "Block ablation (XGBoost only)",
        f"m={args.orchard_multiplier:g}  seed={args.seed}",
        "",
        "=== Context subsets ===",
        f"{'subset':22}  {'feats':>5}  {'acc':>7}  {'orch_R':>7}",
        "-" * 50,
    ]
    for name in ("full", "minus_l2c", "minus_l2b", "minus_l2b_l2c"):
        r = results[name]
        lines.append(
            f"{name:22}  {r['n_features']:5d}  {r['accuracy']:6.2f}%  {r['orchard_recall']:6.2f}%"
        )

    lines.extend(
        [
            "",
            "=== Leave-one-block-out from minus_l2b_l2c ===",
            "Negative delta_acc => block HELPS accuracy when kept.",
            "Negative delta_orch => block HELPS orchard when kept.",
            "",
            f"{'dropped':22}  {'feats':>5}  {'acc':>7}  {'orch_R':>7}  {'d_acc':>7}  {'d_orch':>7}  verdict",
            "-" * 90,
        ]
    )
    loo = [
        (k, v)
        for k, v in results.items()
        if k.startswith("base_minus_")
    ]
    loo.sort(key=lambda kv: kv[1]["delta_acc"])
    for name, r in loo:
        d_acc = r["delta_acc"]
        d_orch = r["delta_orchard"]
        if d_acc < -0.15 and d_orch < -1.0:
            verdict = "HELPS both"
        elif d_acc < -0.15:
            verdict = "HELPS acc"
        elif d_orch < -1.0:
            verdict = "HELPS orchard"
        elif d_acc > 0.15 and d_orch > 1.0:
            verdict = "HINDERS both"
        elif d_acc > 0.15:
            verdict = "HINDERS acc"
        elif d_orch > 1.0:
            verdict = "HINDERS orchard"
        else:
            verdict = "neutral / small"
        lines.append(
            f"{r['dropped_block']:22}  {r['n_features']:5d}  {r['accuracy']:6.2f}%  "
            f"{r['orchard_recall']:6.2f}%  {d_acc:+6.2f}  {d_orch:+6.2f}  {verdict}"
        )

    report = "\n".join(lines)
    print("\n" + report)
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(report + "\n", encoding="utf-8")
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output_log.resolve()}\nSaved: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()

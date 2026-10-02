"""Sweep orchard_multiplier for production XGBoost meta-classifier (64-d)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from agricultureai.meta_features import (
    BASELINE_ACCURACY,
    BASELINE_ORCHARD_RECALL,
    build_xgb,
    compute_metrics,
    feature_dims,
    fit_xgb,
    format_metrics,
    load_feature_matrix,
    orchard_sample_weights,
    predict_xgb,
)

DEFAULT_TRAIN = Path("artifacts/cached_logits_train.pt")
DEFAULT_EVAL = Path("artifacts/cached_logits_eval.pt")


def parse_multipliers(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep orchard_multiplier on 64-d XGBoost (area + NDVI + lat + DOY)."
    )
    p.add_argument("--train-cache", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-cache", type=Path, default=DEFAULT_EVAL)
    p.add_argument(
        "--multipliers",
        default="25,30,35,38,40,42,45,48,50,55,60",
        help="Comma-separated orchard multipliers to try.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-log",
        type=Path,
        default=Path("artifacts/sweep_m_area_doy.log"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    multipliers = parse_multipliers(args.multipliers)
    np.random.seed(args.seed)

    _, full_dim = feature_dims(use_metadata=True)

    print(f"Loading {full_dim}-d features (area + NDVI + lat + DOY) ...")
    x_train, y_train = load_feature_matrix(args.train_cache, use_metadata=True)
    x_eval, y_eval = load_feature_matrix(args.eval_cache, use_metadata=True)
    print(f"  train={len(y_train):,}  eval={len(y_eval):,}  dim={x_train.shape[1]}")

    lines = [
        f"XGBoost orchard_multiplier sweep â€” {full_dim}-d area+DOY metadata",
        f"Targets: acc>{BASELINE_ACCURACY}%, orchard>{BASELINE_ORCHARD_RECALL}%",
        f"m values: {multipliers}",
        "",
        f"{'m':>6}  {'acc':>7}  {'orchard':>8}  {'wheat':>7}  {'acc_ok':>6}  {'orch_ok':>7}",
        "-" * 52,
    ]

    best_dual: tuple[float, dict] | None = None
    best_acc: tuple[float, dict] | None = None
    best_orchard: tuple[float, dict] | None = None

    for m in multipliers:
        print(f"Training XGB m={m:g} ...", flush=True)
        weights = orchard_sample_weights(y_train, m)
        clf = build_xgb(args.seed)
        fit_xgb(clf, x_train, y_train, weights)
        pred = predict_xgb(clf, x_eval)
        metrics = compute_metrics(y_eval, pred)
        line = (
            f"{m:6g}  {metrics['accuracy']:6.2f}%  "
            f"{metrics['orchard_recall']:7.2f}%  "
            f"{metrics['wheat_recall']:6.2f}%  "
            f"{'Y' if metrics['acc_ok'] else 'N':>6}  "
            f"{'Y' if metrics['orchard_ok'] else 'N':>7}"
        )
        lines.append(line)
        print(line, flush=True)

        if metrics["acc_ok"] and metrics["orchard_ok"]:
            if best_dual is None or metrics["accuracy"] > best_dual[1]["accuracy"]:
                best_dual = (m, metrics)
        if best_acc is None or metrics["accuracy"] > best_acc[1]["accuracy"]:
            best_acc = (m, metrics)
        if metrics["orchard_ok"] and (
            best_orchard is None
            or metrics["orchard_recall"] > best_orchard[1]["orchard_recall"]
            or (
                metrics["orchard_recall"] == best_orchard[1]["orchard_recall"]
                and metrics["accuracy"] > best_orchard[1]["accuracy"]
            )
        ):
            best_orchard = (m, metrics)

    lines.append("")
    if best_dual:
        lines.append(
            "Best dual-target: "
            + format_metrics(f"m={best_dual[0]:g}", best_dual[1])
        )
    else:
        lines.append("No m met BOTH accuracy and orchard targets.")
    if best_acc:
        lines.append(
            "Best accuracy: "
            + format_metrics(f"m={best_acc[0]:g}", best_acc[1])
        )
    if best_orchard:
        lines.append(
            "Best orchard (>=35%): "
            + format_metrics(f"m={best_orchard[0]:g}", best_orchard[1])
        )

    report = "\n".join(lines)
    print("\n" + report)
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    try:
        args.output_log.write_text(report + "\n", encoding="utf-8")
    except OSError:
        alt = args.output_log.with_name(args.output_log.stem + "_latest.log")
        alt.write_text(report + "\n", encoding="utf-8")
        print(f"Saved: {alt.resolve()}")
    else:
        print(f"Saved: {args.output_log.resolve()}")


if __name__ == "__main__":
    main()


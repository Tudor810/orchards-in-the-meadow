"""Frontier sweeps on pruned feature subsets (drop L2b/L2c, then bias-tune).

Trains geometric-mean(XGB, CatBoost) once per subset at a fixed orchard
multiplier ``m``, caches eval probabilities, then sweeps orchard log-bias ``b``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from agricultureai.meta_features import (
    BASELINE_ACCURACY,
    BASELINE_ORCHARD_RECALL,
    METADATA_FEATURE_DIM,
    apply_class_log_bias,
    build_catboost,
    build_xgb,
    compute_metrics,
    decode_probs,
    feature_dims,
    fit_catboost,
    fit_xgb,
    geometric_mean_probs,
    load_feature_matrix,
    orchard_sample_weights,
    predict_proba_catboost,
    predict_proba_xgb,
)
from agricultureai.pipeline import GLOBAL_ORCHARDS
from agricultureai.proba_cache import load_npz, meta_proba_fingerprint, save_npz

STRETCH_ACCURACY = 81.0
STRETCH_ORCHARD_RECALL = 60.0


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


def build_subsets(blocks: dict[str, range]) -> dict[str, np.ndarray]:
    all_names = list(blocks.keys())
    return {
        "full": cols(blocks, all_names),
        "minus_l2b": cols(blocks, [n for n in all_names if n != "l2b"]),
        "minus_l2c": cols(
            blocks, [n for n in all_names if n not in ("l2c_raw", "l2c_soft")]
        ),
        "minus_l2b_l2c": cols(
            blocks,
            [n for n in all_names if n not in ("l2b", "l2c_raw", "l2c_soft")],
        ),
        "global_metadata_only": cols(
            blocks, ["msresnet", "global_eng", "transformer", "metadata"]
        ),
    }


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def train_blended_probs(x_train, y_train, x_eval, *, m: float, seed: int) -> np.ndarray:
    weights = orchard_sample_weights(y_train, m)
    xgb = build_xgb(seed)
    fit_xgb(xgb, x_train, y_train, weights)
    cat = build_catboost(seed)
    fit_catboost(cat, x_train, y_train, weights)
    return geometric_mean_probs(
        predict_proba_xgb(xgb, x_eval), predict_proba_catboost(cat, x_eval)
    )


def format_row(r: dict) -> str:
    return (
        f"m={r['m']:4g}  b={r['b_orchard']:4g}  acc={r['accuracy']:6.2f}%  "
        f"orch_R={r['orchard_recall']:6.2f}%  orch_P={r['orchard_precision']:6.2f}%  "
        f"wheat={r['wheat_recall']:6.2f}%"
    )


def pick_best(rows: list[dict], *, acc_min: float, orchard_min: float):
    feasible = [
        r
        for r in rows
        if r["accuracy"] >= acc_min and r["orchard_recall"] >= orchard_min
    ]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda r: (r["orchard_recall"], r["accuracy"], r["orchard_precision"]),
    )


def pareto_frontier(rows: list[dict]) -> list[dict]:
    frontier, best_orchard = [], -1.0
    for r in sorted(rows, key=lambda x: (x["accuracy"], x["orchard_recall"]), reverse=True):
        if r["orchard_recall"] > best_orchard:
            frontier.append(r)
            best_orchard = r["orchard_recall"]
    return frontier


def load_or_train_proba(
    *,
    name: str,
    columns: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    m: float,
    seed: int,
    proba_dir: Path,
    force: bool = False,
) -> np.ndarray:
    proba_path = proba_dir / f"proba_{name}_m{m:g}.npz"
    x_train_sub, x_eval_sub = x_train[:, columns], x_eval[:, columns]
    fingerprint = meta_proba_fingerprint(
        x_train_sub, y_train, x_eval_sub, seed=seed, kind="geo_mean_xgb_cat", m=float(m)
    )
    cached = load_npz(proba_path, fingerprint, force=force)
    if cached is not None:
        print(f"Loaded cached probs: {proba_path}", flush=True)
        return cached["proba"].astype(np.float32)
    print(
        f"Training XGB+CatBoost  subset={name}  feats={columns.size}  m={m:g} ...",
        flush=True,
    )
    proba = train_blended_probs(x_train_sub, y_train, x_eval_sub, m=m, seed=seed)
    save_npz(proba_path, fingerprint, proba=proba)
    print(f"Saved probs: {proba_path.resolve()}", flush=True)
    return proba


def sweep_subset(
    *,
    name: str,
    columns: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    multipliers: list[float],
    biases: list[float],
    seed: int,
    proba_dir: Path,
    force: bool = False,
) -> dict:
    rows: list[dict] = []
    for m in multipliers:
        print(f"\n=== subset={name}  feats={columns.size}  m={m:g} ===", flush=True)
        proba = load_or_train_proba(
            name=name,
            columns=columns,
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            m=m,
            seed=seed,
            proba_dir=proba_dir,
            force=force,
        )
        for b in biases:
            pred = decode_probs(apply_class_log_bias(proba, GLOBAL_ORCHARDS, b))
            rows.append(
                {
                    "subset": name,
                    "n_features": int(columns.size),
                    "m": float(m),
                    "b_orchard": float(b),
                    **compute_metrics(y_eval, pred),
                }
            )

    best_prod = pick_best(
        rows, acc_min=BASELINE_ACCURACY, orchard_min=BASELINE_ORCHARD_RECALL
    )
    best_stretch = pick_best(
        rows, acc_min=STRETCH_ACCURACY, orchard_min=STRETCH_ORCHARD_RECALL
    )
    best_acc = max(rows, key=lambda r: (r["accuracy"], r["orchard_recall"]))
    best_orchard = max(rows, key=lambda r: (r["orchard_recall"], r["accuracy"]))
    # Closest to stretch: minimize positive deficits, then maximize surplus.
    def stretch_distance(r: dict) -> tuple:
        acc_gap = max(0.0, STRETCH_ACCURACY - r["accuracy"])
        orch_gap = max(0.0, STRETCH_ORCHARD_RECALL - r["orchard_recall"])
        return (
            acc_gap + orch_gap,
            acc_gap,
            orch_gap,
            -r["accuracy"],
            -r["orchard_recall"],
        )

    closest_stretch = min(rows, key=stretch_distance)
    return {
        "subset": name,
        "n_features": int(columns.size),
        "rows": rows,
        "best_production": best_prod,
        "best_stretch": best_stretch,
        "best_accuracy": best_acc,
        "best_orchard": best_orchard,
        "closest_stretch": closest_stretch,
        "pareto": pareto_frontier(rows),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache", type=Path, default=Path("artifacts/cached_logits_train.pt"))
    p.add_argument("--eval-cache", type=Path, default=Path("artifacts/cached_logits_eval.pt"))
    p.add_argument(
        "--multipliers",
        default="15",
        help="Comma-separated orchard multipliers (one XGB+CatBoost train per m).",
    )
    p.add_argument("--orchard-biases", default="0,0.5,1,1.5,2,2.5,3,4,5,6,8")
    p.add_argument(
        "--subsets",
        default="",
        help="Comma-separated subset names (default: all). "
        "Options: full,minus_l2b,minus_l2c,minus_l2b_l2c,global_metadata_only",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--proba-dir", type=Path, default=Path("artifacts/pruned_proba_cache"))
    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached probabilities even if their fingerprint matches.",
    )
    p.add_argument(
        "--output-log",
        type=Path,
        default=Path("artifacts/sweep_pruned_frontier.log"),
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/sweep_pruned_frontier.json"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    multipliers = parse_float_list(args.multipliers)
    biases = parse_float_list(args.orchard_biases)
    np.random.seed(args.seed)

    _, full_dim = feature_dims(use_metadata=True)
    print(f"Loading {full_dim}-d features ...")
    x_train, y_train = load_feature_matrix(args.train_cache, use_metadata=True)
    x_eval, y_eval = load_feature_matrix(args.eval_cache, use_metadata=True)
    if x_train.shape[1] != full_dim:
        raise SystemExit(
            f"Expected {full_dim}-d cache; got {x_train.shape[1]}-d."
        )
    print(f"  train={len(y_train):,}  eval={len(y_eval):,}")

    all_subsets = build_subsets(column_blocks(full_dim))
    if args.subsets.strip():
        wanted = [s.strip() for s in args.subsets.split(",") if s.strip()]
        missing = [s for s in wanted if s not in all_subsets]
        if missing:
            raise SystemExit(f"Unknown subsets: {missing}. Known: {list(all_subsets)}")
        subsets = {k: all_subsets[k] for k in wanted}
    else:
        subsets = all_subsets

    results = {}
    for name, columns in subsets.items():
        results[name] = sweep_subset(
            name=name,
            columns=columns,
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            y_eval=y_eval,
            multipliers=multipliers,
            biases=biases,
            seed=int(args.seed),
            proba_dir=args.proba_dir,
            force=bool(args.force_recompute),
        )

    lines = [
        "Pruned-feature frontier sweeps (geo-mean XGB+CatBoost, m x b grid)",
        f"m={multipliers}  biases={biases}",
        f"Production: acc>{BASELINE_ACCURACY}%, orchard>{BASELINE_ORCHARD_RECALL}%",
        f"Stretch: acc>={STRETCH_ACCURACY}%, orchard>={STRETCH_ORCHARD_RECALL}%",
        "",
        f"{'subset':18}  {'feats':>5}  {'best_acc':>30}  {'closest_stretch':>30}  "
        f"{'prod?':>5}  {'stretch?':>8}",
        "-" * 120,
    ]
    for name, payload in results.items():
        ba = payload["best_accuracy"]
        cs = payload["closest_stretch"]
        lines.append(
            f"{name:18}  {payload['n_features']:5d}  "
            f"acc={ba['accuracy']:5.2f}/R={ba['orchard_recall']:5.2f}"
            f"@m={ba['m']:g}/b={ba['b_orchard']:g}  "
            f"acc={cs['accuracy']:5.2f}/R={cs['orchard_recall']:5.2f}"
            f"@m={cs['m']:g}/b={cs['b_orchard']:g}  "
            f"{'Y' if payload['best_production'] else 'N':>5}  "
            f"{'Y' if payload['best_stretch'] else 'N':>8}"
        )

    lines.extend(["", "=== Per-subset details ==="])
    for name, payload in results.items():
        lines.append(f"\n## {name} ({payload['n_features']}-d)")
        lines.append(
            "Best production: "
            + (
                format_row(payload["best_production"])
                if payload["best_production"]
                else "none"
            )
        )
        lines.append(
            "Best stretch:    "
            + (
                format_row(payload["best_stretch"])
                if payload["best_stretch"]
                else "none"
            )
        )
        lines.append("Closest stretch: " + format_row(payload["closest_stretch"]))
        lines.append("Best accuracy:   " + format_row(payload["best_accuracy"]))
        lines.append("Best orchard:    " + format_row(payload["best_orchard"]))
        lines.append(f"Pareto ({len(payload['pareto'])} pts):")
        for r in payload["pareto"][:16]:
            lines.append("  " + format_row(r))
        # Acc-feasible band near stretch: acc>=80.5 with max orchard
        near = [
            r
            for r in rows_sorted_near_stretch(payload["rows"])
            if r["accuracy"] >= 80.0
        ][:12]
        if near:
            lines.append("High-acc band (acc>=80%), sorted by orchard then acc:")
            for r in near:
                lines.append("  " + format_row(r))

    report = "\n".join(lines)
    print("\n" + report)
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(report + "\n", encoding="utf-8")
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output_log.resolve()}\nSaved: {args.output_json.resolve()}")


def rows_sorted_near_stretch(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (-r["orchard_recall"], -r["accuracy"], r["m"], r["b_orchard"]),
    )


if __name__ == "__main__":
    main()

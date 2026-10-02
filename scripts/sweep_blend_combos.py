"""Sweep XGB/CatBoost blend combinations + orchard log-bias.

Trains each model once per ``m``, caches separate eval probabilities, then
evaluates blend recipes (solo / arithmetic / geometric / weighted).
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


SUBSET_DROP: dict[str, set[str]] = {
    "minus_l2b_l2c": {"l2b", "l2c_raw", "l2c_soft"},
    "minus_l2b": {"l2b"},
    "minus_l2c": {"l2c_raw", "l2c_soft"},
    "full": set(),
}


def subset_cols(blocks: dict[str, range], subset: str) -> np.ndarray:
    if subset not in SUBSET_DROP:
        raise SystemExit(f"Unknown subset {subset!r}. Known: {sorted(SUBSET_DROP)}")
    drop = SUBSET_DROP[subset]
    keep: list[int] = []
    for name, rng in blocks.items():
        if name not in drop:
            keep.extend(rng)
    return np.array(sorted(keep), dtype=np.int64)


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def normalize_rows(p: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, None)
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)


def arith_mean(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    return normalize_rows(0.5 * (p_a + p_b))


def weighted_arith(p_a: np.ndarray, p_b: np.ndarray, w_a: float) -> np.ndarray:
    return normalize_rows(float(w_a) * p_a + (1.0 - float(w_a)) * p_b)


def weighted_geo(p_a: np.ndarray, p_b: np.ndarray, w_a: float, *, eps: float = 1e-12) -> np.ndarray:
    a = np.clip(p_a, eps, None)
    b = np.clip(p_b, eps, None)
    return normalize_rows(np.power(a, float(w_a)) * np.power(b, 1.0 - float(w_a)))


def blend_recipes(p_xgb: np.ndarray, p_cat: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {
        "xgb_only": p_xgb,
        "cat_only": p_cat,
        "arith_mean": arith_mean(p_xgb, p_cat),
        "geo_mean": geometric_mean_probs(p_xgb, p_cat),
    }
    for w in (0.2, 0.3, 0.4, 0.6, 0.7, 0.8):
        out[f"arith_xgb{w:g}"] = weighted_arith(p_xgb, p_cat, w)
        out[f"geo_xgb{w:g}"] = weighted_geo(p_xgb, p_cat, w)
    return out


def train_model_probs(x_train, y_train, x_eval, *, m: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    weights = orchard_sample_weights(y_train, m)
    xgb = build_xgb(seed)
    fit_xgb(xgb, x_train, y_train, weights)
    cat = build_catboost(seed)
    fit_catboost(cat, x_train, y_train, weights)
    return (
        predict_proba_xgb(xgb, x_eval).astype(np.float32),
        predict_proba_catboost(cat, x_eval).astype(np.float32),
    )


def load_or_train_pair(
    *,
    cache_path: Path,
    x_train,
    y_train,
    x_eval,
    m: float,
    seed: int,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    fingerprint = meta_proba_fingerprint(
        x_train, y_train, x_eval, seed=seed, kind="xgb_cat_pair", m=float(m)
    )
    cached = load_npz(cache_path, fingerprint, force=force)
    if cached is not None:
        print(f"Loaded model probs: {cache_path}", flush=True)
        return cached["p_xgb"].astype(np.float32), cached["p_cat"].astype(np.float32)
    print(f"Training XGB+CatBoost @ m={m:g} ...", flush=True)
    p_xgb, p_cat = train_model_probs(x_train, y_train, x_eval, m=m, seed=seed)
    save_npz(cache_path, fingerprint, p_xgb=p_xgb, p_cat=p_cat)
    print(f"Saved model probs: {cache_path.resolve()}", flush=True)
    return p_xgb, p_cat


def pick_best(rows: list[dict], *, acc_min: float, orchard_min: float):
    feasible = [
        r for r in rows if r["accuracy"] >= acc_min and r["orchard_recall"] >= orchard_min
    ]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda r: (r["orchard_recall"], r["accuracy"], r["orchard_precision"]),
    )


def stretch_distance(r: dict) -> tuple:
    acc_gap = max(0.0, STRETCH_ACCURACY - r["accuracy"])
    orch_gap = max(0.0, STRETCH_ORCHARD_RECALL - r["orchard_recall"])
    return (acc_gap + orch_gap, acc_gap, orch_gap, -r["accuracy"], -r["orchard_recall"])


def pareto_frontier(rows: list[dict]) -> list[dict]:
    frontier, best_orchard = [], -1.0
    for r in sorted(rows, key=lambda x: (x["accuracy"], x["orchard_recall"]), reverse=True):
        if r["orchard_recall"] > best_orchard:
            frontier.append(r)
            best_orchard = r["orchard_recall"]
    return frontier


def format_row(r: dict) -> str:
    return (
        f"{r['blend']:14s}  m={r['m']:4g}  b={r['b_orchard']:4g}  "
        f"acc={r['accuracy']:6.2f}%  orch_R={r['orchard_recall']:6.2f}%  "
        f"orch_P={r['orchard_precision']:6.2f}%"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache", type=Path, default=Path("artifacts/cached_logits_train.pt"))
    p.add_argument("--eval-cache", type=Path, default=Path("artifacts/cached_logits_eval.pt"))
    p.add_argument(
        "--subset",
        default="minus_l2b_l2c",
        choices=sorted(SUBSET_DROP),
        help="Feature subset for the blend sweep.",
    )
    p.add_argument("--multipliers", default="1,3")
    p.add_argument("--orchard-biases", default="0,0.5,1,1.5,2,2.5,3,4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--proba-dir", type=Path, default=Path("artifacts/pruned_proba_cache"))
    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached probabilities even if their fingerprint matches.",
    )
    p.add_argument("--output-log", type=Path, default=Path("artifacts/sweep_blend_combos.log"))
    p.add_argument("--output-json", type=Path, default=Path("artifacts/sweep_blend_combos.json"))
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
    columns = subset_cols(column_blocks(full_dim), args.subset)
    x_train = x_train[:, columns]
    x_eval = x_eval[:, columns]
    print(
        f"  subset={args.subset}  feats={columns.size}  "
        f"train={len(y_train):,}  eval={len(y_eval):,}"
    )

    all_rows: list[dict] = []
    per_blend: dict[str, list[dict]] = {}

    for m in multipliers:
        cache_path = args.proba_dir / f"proba_pair_{args.subset}_m{m:g}.npz"
        p_xgb, p_cat = load_or_train_pair(
            cache_path=cache_path,
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            m=m,
            seed=int(args.seed),
            force=bool(args.force_recompute),
        )
        recipes = blend_recipes(p_xgb, p_cat)
        for blend_name, proba in recipes.items():
            for b in biases:
                pred = decode_probs(apply_class_log_bias(proba, GLOBAL_ORCHARDS, b))
                row = {
                    "blend": blend_name,
                    "subset": args.subset,
                    "m": float(m),
                    "b_orchard": float(b),
                    **compute_metrics(y_eval, pred),
                }
                all_rows.append(row)
                per_blend.setdefault(blend_name, []).append(row)

    # Summaries per blend
    lines = [
        f"XGB/CatBoost blend combination sweep ({args.subset})",
        f"m={multipliers}  biases={biases}",
        f"Production: acc>{BASELINE_ACCURACY}%  orchard>{BASELINE_ORCHARD_RECALL}%",
        f"Stretch: acc>={STRETCH_ACCURACY}%  orchard>={STRETCH_ORCHARD_RECALL}%",
        "",
        f"{'blend':14}  {'best_acc':>28}  {'closest_stretch':>28}  {'prod?':>5}  {'stretch?':>8}",
        "-" * 100,
    ]
    blend_summary = {}
    for blend_name, rows in per_blend.items():
        best_prod = pick_best(
            rows, acc_min=BASELINE_ACCURACY, orchard_min=BASELINE_ORCHARD_RECALL
        )
        best_stretch = pick_best(
            rows, acc_min=STRETCH_ACCURACY, orchard_min=STRETCH_ORCHARD_RECALL
        )
        best_acc = max(rows, key=lambda r: (r["accuracy"], r["orchard_recall"]))
        closest = min(rows, key=stretch_distance)
        blend_summary[blend_name] = {
            "best_production": best_prod,
            "best_stretch": best_stretch,
            "best_accuracy": best_acc,
            "closest_stretch": closest,
            "pareto": pareto_frontier(rows),
        }
        lines.append(
            f"{blend_name:14}  "
            f"acc={best_acc['accuracy']:5.2f}/R={best_acc['orchard_recall']:5.2f}"
            f"@m={best_acc['m']:g}/b={best_acc['b_orchard']:g}  "
            f"acc={closest['accuracy']:5.2f}/R={closest['orchard_recall']:5.2f}"
            f"@m={closest['m']:g}/b={closest['b_orchard']:g}  "
            f"{'Y' if best_prod else 'N':>5}  "
            f"{'Y' if best_stretch else 'N':>8}"
        )

    # Global best across all blends
    global_closest = min(all_rows, key=stretch_distance)
    global_best_acc = max(all_rows, key=lambda r: (r["accuracy"], r["orchard_recall"]))
    global_prod = pick_best(
        all_rows, acc_min=BASELINE_ACCURACY, orchard_min=BASELINE_ORCHARD_RECALL
    )
    global_stretch = pick_best(
        all_rows, acc_min=STRETCH_ACCURACY, orchard_min=STRETCH_ORCHARD_RECALL
    )
    global_pareto = pareto_frontier(all_rows)

    lines.extend(
        [
            "",
            "=== GLOBAL ===",
            f"Best production: {format_row(global_prod) if global_prod else 'none'}",
            f"Best stretch:    {format_row(global_stretch) if global_stretch else 'none'}",
            f"Closest stretch: {format_row(global_closest)}",
            f"Best accuracy:   {format_row(global_best_acc)}",
            f"Global Pareto ({len(global_pareto)} pts):",
        ]
    )
    for r in global_pareto[:20]:
        lines.append("  " + format_row(r))

    near80 = sorted(
        [r for r in all_rows if r["accuracy"] >= 80.0],
        key=lambda r: (-r["orchard_recall"], -r["accuracy"]),
    )[:15]
    lines.append("High-acc band (acc>=80%), by orchard then acc:")
    for r in near80:
        lines.append("  " + format_row(r))

    report = "\n".join(lines)
    print("\n" + report)
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(report + "\n", encoding="utf-8")
    payload = {
        "rows": all_rows,
        "per_blend": {
            k: {
                "best_production": v["best_production"],
                "best_stretch": v["best_stretch"],
                "best_accuracy": v["best_accuracy"],
                "closest_stretch": v["closest_stretch"],
                "pareto": v["pareto"],
            }
            for k, v in blend_summary.items()
        },
        "global_closest_stretch": global_closest,
        "global_best_accuracy": global_best_acc,
        "global_best_production": global_prod,
        "global_best_stretch": global_stretch,
        "global_pareto": global_pareto,
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output_log.resolve()}\nSaved: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()

"""Compact-layout meta frontier: XGB/Cat blends + orchard/meadow dual bias.

Recommended stretch path:
  features = transformer + msresnet + cereal + L3 + metadata (39-d)
  blends   = geo-mean | arithmetic(w_xgb=0.7)
  decode   = +b_orchard on orchards, -b_meadow on both meadow classes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from agricultureai.meta_features import (
    BASELINE_ACCURACY,
    BASELINE_ORCHARD_RECALL,
    COMPACT_FEATURE_DIM,
    STRETCH_ACCURACY,
    STRETCH_ORCHARD_RECALL,
    apply_orchard_meadow_log_bias,
    build_catboost,
    build_xgb,
    compute_metrics,
    decode_probs,
    fit_catboost,
    fit_xgb,
    geometric_mean_probs,
    load_feature_matrix,
    orchard_sample_weights,
    predict_proba_catboost,
    predict_proba_xgb,
    select_feature_layout,
    weighted_arithmetic_probs,
)
from agricultureai.proba_cache import load_npz, meta_proba_fingerprint, save_npz

DEFAULT_TRAIN = Path("artifacts/cached_logits_train.pt")
DEFAULT_EVAL = Path("artifacts/cached_logits_eval.pt")
DEFAULT_PROBA_DIR = Path("artifacts/pruned_proba_cache")


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-cache", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--layout", default="compact", choices=("compact", "minus_l2b_l2c", "full"))
    p.add_argument("--multipliers", default="1,2,3")
    p.add_argument("--orchard-biases", default="0,0.5,1,1.5,2,2.5,3")
    p.add_argument("--meadow-biases", default="0,0.25,0.5,0.75,1,1.5,2")
    p.add_argument(
        "--blends",
        default="geo_mean,arith_xgb0.7",
        help="Comma-separated: geo_mean,arith_mean,arith_xgb0.7,xgb_only,cat_only",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--proba-dir", type=Path, default=DEFAULT_PROBA_DIR)
    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached probabilities even if their fingerprint matches.",
    )
    p.add_argument("--output-log", type=Path, default=Path("artifacts/sweep_compact_dual_bias.log"))
    p.add_argument("--output-json", type=Path, default=Path("artifacts/sweep_compact_dual_bias.json"))
    return p.parse_args()


def train_pair(x_train, y_train, x_eval, *, m: float, seed: int):
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
    *, cache_path: Path, x_train, y_train, x_eval, m: float, seed: int, force: bool = False
):
    fingerprint = meta_proba_fingerprint(
        x_train, y_train, x_eval, seed=seed, kind="xgb_cat_pair", m=float(m)
    )
    cached = load_npz(cache_path, fingerprint, force=force)
    if cached is not None:
        print(f"Loaded {cache_path}", flush=True)
        return cached["p_xgb"].astype(np.float32), cached["p_cat"].astype(np.float32)
    print(f"Training XGB+CatBoost @ m={m:g} ...", flush=True)
    p_xgb, p_cat = train_pair(x_train, y_train, x_eval, m=m, seed=seed)
    save_npz(cache_path, fingerprint, p_xgb=p_xgb, p_cat=p_cat)
    print(f"Saved {cache_path.resolve()}", flush=True)
    return p_xgb, p_cat


def make_blend(name: str, p_xgb: np.ndarray, p_cat: np.ndarray) -> np.ndarray:
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


def pick_best(rows: List[dict], *, acc_min: float, orchard_min: float):
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


def pareto_frontier(rows: List[dict]) -> List[dict]:
    frontier, best_orchard = [], -1.0
    for r in sorted(rows, key=lambda x: (x["accuracy"], x["orchard_recall"]), reverse=True):
        if r["orchard_recall"] > best_orchard:
            frontier.append(r)
            best_orchard = r["orchard_recall"]
    return frontier


def format_row(r: dict) -> str:
    return (
        f"{r['blend']:12s}  m={r['m']:3g}  b_o={r['b_orchard']:4g}  b_m={r['b_meadow']:4g}  "
        f"acc={r['accuracy']:6.2f}%  orch_R={r['orchard_recall']:6.2f}%  "
        f"orch_P={r['orchard_precision']:6.2f}%"
    )


def main() -> None:
    args = parse_args()
    multipliers = parse_float_list(args.multipliers)
    orchard_biases = parse_float_list(args.orchard_biases)
    meadow_biases = parse_float_list(args.meadow_biases)
    blends = [b.strip() for b in args.blends.split(",") if b.strip()]
    np.random.seed(args.seed)

    print(f"Loading full features, layout={args.layout} ...")
    x_train_full, y_train = load_feature_matrix(args.train_cache, use_metadata=True)
    x_eval_full, y_eval = load_feature_matrix(args.eval_cache, use_metadata=True)
    x_train = select_feature_layout(x_train_full, args.layout)
    x_eval = select_feature_layout(x_eval_full, args.layout)
    print(
        f"  feats={x_train.shape[1]}  train={len(y_train):,}  eval={len(y_eval):,}"
        + (f"  (compact={COMPACT_FEATURE_DIM})" if args.layout == "compact" else "")
    )

    rows: List[dict] = []
    for m in multipliers:
        cache_path = args.proba_dir / f"proba_pair_{args.layout}_m{m:g}.npz"
        p_xgb, p_cat = load_or_train_pair(
            cache_path=cache_path,
            x_train=x_train,
            y_train=y_train,
            x_eval=x_eval,
            m=m,
            seed=int(args.seed),
            force=bool(args.force_recompute),
        )
        blend_probs = {name: make_blend(name, p_xgb, p_cat) for name in blends}
        for blend_name, proba in blend_probs.items():
            for b_o in orchard_biases:
                for b_m in meadow_biases:
                    pred = decode_probs(
                        apply_orchard_meadow_log_bias(
                            proba, b_orchard=b_o, b_meadow=b_m
                        )
                    )
                    rows.append(
                        {
                            "layout": args.layout,
                            "blend": blend_name,
                            "m": float(m),
                            "b_orchard": float(b_o),
                            "b_meadow": float(b_m),
                            **compute_metrics(y_eval, pred),
                        }
                    )

    best_prod = pick_best(
        rows, acc_min=BASELINE_ACCURACY, orchard_min=BASELINE_ORCHARD_RECALL
    )
    best_stretch = pick_best(
        rows, acc_min=STRETCH_ACCURACY, orchard_min=STRETCH_ORCHARD_RECALL
    )
    closest = min(rows, key=stretch_distance)
    best_acc = max(rows, key=lambda r: (r["accuracy"], r["orchard_recall"]))
    frontier = pareto_frontier(rows)
    near80 = sorted(
        [r for r in rows if r["accuracy"] >= 80.0],
        key=lambda r: (-r["orchard_recall"], -r["accuracy"]),
    )[:15]

    lines = [
        f"Compact dual-bias frontier ({args.layout})",
        f"blends={blends}  m={multipliers}",
        f"b_orchard={orchard_biases}  b_meadow={meadow_biases}",
        f"Production: acc>{BASELINE_ACCURACY}%  orchard>{BASELINE_ORCHARD_RECALL}%",
        f"Stretch: acc>={STRETCH_ACCURACY}%  orchard>={STRETCH_ORCHARD_RECALL}%",
        "",
        f"Best production: {format_row(best_prod) if best_prod else 'none'}",
        f"Best stretch:    {format_row(best_stretch) if best_stretch else 'none'}",
        f"Closest stretch: {format_row(closest)}",
        f"Best accuracy:   {format_row(best_acc)}",
        "",
        f"Pareto ({len(frontier)} pts):",
    ]
    for r in frontier[:20]:
        lines.append("  " + format_row(r))
    lines.append("High-acc band (acc>=80%), by orchard then acc:")
    for r in near80:
        lines.append("  " + format_row(r))

    report = "\n".join(lines)
    print("\n" + report)
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(report + "\n", encoding="utf-8")
    payload: Dict = {
        "rows": rows,
        "best_production": best_prod,
        "best_stretch": best_stretch,
        "closest_stretch": closest,
        "best_accuracy": best_acc,
        "pareto": frontier,
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output_log.resolve()}\nSaved: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()

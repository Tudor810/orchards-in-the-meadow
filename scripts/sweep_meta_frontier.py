"""2-D frontier sweep: orchard_multiplier m x orchard log-bias on blended probs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from agricultureai.meta_features import (
    BASELINE_ACCURACY,
    BASELINE_ORCHARD_RECALL,
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
DEFAULT_TRAIN = Path("artifacts/cached_logits_train.pt")
DEFAULT_EVAL = Path("artifacts/cached_logits_eval.pt")
DEFAULT_PROBA_CACHE = Path("artifacts/meta_proba_cache.npz")


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2-D sweep: orchard_multiplier x orchard log-bias on geo-mean XGB+CatBoost."
    )
    p.add_argument("--train-cache", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-cache", type=Path, default=DEFAULT_EVAL)
    p.add_argument(
        "--multipliers",
        default="15",
        help="Comma-separated orchard multipliers. Defaults to a single fixed m; "
        "the probability cache is built once and orchard log-bias does the tuning.",
    )
    p.add_argument("--orchard-biases", default="0,0.5,1,1.5,2,2.5,3,4,5,6,8")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--proba-cache", type=Path, default=DEFAULT_PROBA_CACHE)
    p.add_argument(
        "--force-recompute",
        "--refresh-proba-cache",
        dest="force_recompute",
        action="store_true",
        help="Ignore the probability cache even if its fingerprint matches.",
    )
    p.add_argument("--output-log", type=Path, default=Path("artifacts/sweep_meta_frontier.log"))
    p.add_argument("--output-json", type=Path, default=Path("artifacts/sweep_meta_frontier.json"))
    return p.parse_args()


def train_blended_probs(x_train, y_train, x_eval, *, m: float, seed: int) -> np.ndarray:
    weights = orchard_sample_weights(y_train, m)
    xgb = build_xgb(seed)
    fit_xgb(xgb, x_train, y_train, weights)
    cat = build_catboost(seed)
    fit_catboost(cat, x_train, y_train, weights)
    return geometric_mean_probs(predict_proba_xgb(xgb, x_eval), predict_proba_catboost(cat, x_eval))


def load_or_build_proba_cache(
    *, multipliers, x_train, y_train, x_eval, cache_path: Path, seed: int, force: bool
) -> Dict[float, np.ndarray]:
    fingerprint = meta_proba_fingerprint(
        x_train, y_train, x_eval,
        seed=seed, kind="geo_mean_xgb_cat", multipliers=sorted(multipliers),
    )
    cached = load_npz(cache_path, fingerprint, force=force)
    if cached is not None:
        out = {m: cached[f"m_{m:g}"].astype(np.float32) for m in multipliers}
        print(f"Loaded blended probs for {len(out)} m values from {cache_path}")
        return out
    out, save_dict = {}, {"multipliers": np.array(multipliers)}
    for m in multipliers:
        print(f"Training XGB+CatBoost @ m={m:g} ...", flush=True)
        out[m] = train_blended_probs(x_train, y_train, x_eval, m=m, seed=seed)
        save_dict[f"m_{m:g}"] = out[m]
    save_npz(cache_path, fingerprint, **save_dict)
    print(f"Saved probability cache: {cache_path.resolve()}")
    return out


def sweep_grid(y_eval, proba_by_m, multipliers, orchard_biases) -> List[Dict[str, float]]:
    rows = []
    for m in multipliers:
        for b in orchard_biases:
            pred = decode_probs(apply_class_log_bias(proba_by_m[m], GLOBAL_ORCHARDS, b))
            rows.append({"m": float(m), "b_orchard": float(b), **compute_metrics(y_eval, pred)})
    return rows


def pick_best(rows, *, acc_min, orchard_min, label) -> Tuple[Dict[str, float] | None, str]:
    feasible = [r for r in rows if r["accuracy"] >= acc_min and r["orchard_recall"] >= orchard_min]
    if not feasible:
        return None, f"No point met {label} (acc>={acc_min:.2f}%, orchard>={orchard_min:.2f}%)."
    return max(feasible, key=lambda r: (r["orchard_recall"], r["accuracy"], r["orchard_precision"])), ""


def pareto_frontier(rows):
    frontier, best_orchard = [], -1.0
    for r in sorted(rows, key=lambda x: (x["accuracy"], x["orchard_recall"]), reverse=True):
        if r["orchard_recall"] > best_orchard:
            frontier.append(r)
            best_orchard = r["orchard_recall"]
    return frontier


def format_row(r) -> str:
    return (
        f"m={r['m']:4g}  b={r['b_orchard']:4g}  acc={r['accuracy']:6.2f}%  "
        f"orch_R={r['orchard_recall']:6.2f}%  orch_P={r['orchard_precision']:6.2f}%  "
        f"wheat={r['wheat_recall']:6.2f}%"
    )


def main() -> None:
    args = parse_args()
    multipliers = parse_float_list(args.multipliers)
    orchard_biases = parse_float_list(args.orchard_biases)
    np.random.seed(args.seed)
    _, full_dim = feature_dims(use_metadata=True)
    print(f"Loading {full_dim}-d features ...")
    x_train, y_train = load_feature_matrix(args.train_cache, use_metadata=True)
    x_eval, y_eval = load_feature_matrix(args.eval_cache, use_metadata=True)
    print(f"  train={len(y_train):,}  eval={len(y_eval):,}")
    proba_by_m = load_or_build_proba_cache(
        multipliers=multipliers, x_train=x_train, y_train=y_train, x_eval=x_eval,
        cache_path=args.proba_cache, seed=args.seed, force=bool(args.force_recompute),
    )
    print(f"\nSweeping {len(multipliers)} x {len(orchard_biases)} grid (bias only) ...")
    rows = sweep_grid(y_eval, proba_by_m, multipliers, orchard_biases)
    best_prod, _ = pick_best(rows, acc_min=BASELINE_ACCURACY, orchard_min=BASELINE_ORCHARD_RECALL, label="production")
    best_stretch, stretch_msg = pick_best(rows, acc_min=STRETCH_ACCURACY, orchard_min=STRETCH_ORCHARD_RECALL, label="stretch")
    best_orchard = max(rows, key=lambda r: (r["orchard_recall"], r["accuracy"]))
    best_acc = max(rows, key=lambda r: (r["accuracy"], r["orchard_recall"]))
    frontier = pareto_frontier(rows)
    lines = [
        "Meta frontier sweep - geometric-mean(XGB, CatBoost) + orchard log-bias",
        f"Features: {full_dim}-d | Eval: {len(y_eval):,} samples",
        f"Production targets: acc>{BASELINE_ACCURACY}%, orchard>{BASELINE_ORCHARD_RECALL}%",
        f"Stretch targets: acc>={STRETCH_ACCURACY}%, orchard>={STRETCH_ORCHARD_RECALL}%",
        "",
        f"{'m':>6}  {'b':>5}  {'acc':>7}  {'orch_R':>8}  {'orch_P':>8}  {'wheat':>7}",
        "-" * 52,
    ]
    for r in sorted(rows, key=lambda x: (-x["orchard_recall"], -x["accuracy"])):
        lines.append(
            f"{r['m']:6g}  {r['b_orchard']:5g}  {r['accuracy']:6.2f}%  "
            f"{r['orchard_recall']:7.2f}%  {r['orchard_precision']:7.2f}%  {r['wheat_recall']:6.2f}%"
        )
    lines.extend(["", "=== SUMMARY ==="])
    lines.append("Best production dual-target: " + (format_row(best_prod) if best_prod else "none"))
    lines.append("Best accuracy on grid:       " + format_row(best_acc))
    lines.append("Best orchard recall:         " + format_row(best_orchard))
    lines.append("Best stretch dual-target:    " + (format_row(best_stretch) if best_stretch else stretch_msg))
    ref = best_prod or best_acc
    lines.extend([
        "", "=== GAP vs stretch (81%, 60%) ===", f"Reference: {format_row(ref)}",
        f"  acc gap:     {STRETCH_ACCURACY - ref['accuracy']:+.2f} pp",
        f"  orchard gap: {STRETCH_ORCHARD_RECALL - ref['orchard_recall']:+.2f} pp",
        f"  orchard precision @ ref: {ref['orchard_precision']:.2f}%",
        "", f"Pareto frontier ({len(frontier)} points):",
    ])
    for r in frontier[:12]:
        lines.append("  " + format_row(r))
    report = "\n".join(lines)
    print("\n" + report)
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(report + "\n", encoding="utf-8")
    args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output_log.resolve()}\nSaved: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
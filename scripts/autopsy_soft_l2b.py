"""Orchard FP autopsy + soft-L2b scale sweep (minus L2c, scaled L2b).

1. Autopsy predictions from cached ``minus_l2b_l2c`` probs at (m, b).
2. Train geo-mean(XGB, CatBoost) with L2c dropped and L2b columns scaled by
   ``alpha`` in {0, 0.1, 0.2, 0.3, 0.5, 1.0}; sweep orchard log-bias.
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
from agricultureai.pipeline import (
    GLOBAL_ORCHARDS,
    GLOBAL_PERMANENT_MEADOWS,
    GLOBAL_TEMPORARY_MEADOWS,
    NUM_GLOBAL_CLASSES,
)
from agricultureai.proba_cache import load_npz, meta_proba_fingerprint, save_npz
from sweep_pruned_frontier import build_subsets, load_or_train_proba

STRETCH_ACCURACY = 81.0
STRETCH_ORCHARD_RECALL = 60.0

CLASS_NAMES = (
    "barley",
    "wheat",
    "rapeseed",
    "corn",
    "sunflower",
    "orchards",
    "nuts",
    "perm_meadow",
    "temp_meadow",
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


def keep_columns_minus_l2c(blocks: dict[str, range]) -> np.ndarray:
    names = [n for n in blocks if n not in ("l2c_raw", "l2c_soft")]
    keep: list[int] = []
    for name in names:
        keep.extend(blocks[name])
    return np.array(sorted(keep), dtype=np.int64)


def scale_l2b(x: np.ndarray, blocks: dict[str, range], alpha: float) -> np.ndarray:
    out = x.copy()
    out[:, list(blocks["l2b"])] *= float(alpha)
    return out


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
    alpha = r.get("alpha", "")
    alpha_s = f"{alpha:g}" if isinstance(alpha, (int, float)) else str(alpha)
    return (
        f"alpha={alpha_s}  m={r['m']:4g}  b={r['b_orchard']:4g}  "
        f"acc={r['accuracy']:6.2f}%  orch_R={r['orchard_recall']:6.2f}%  "
        f"orch_P={r['orchard_precision']:6.2f}%  wheat={r['wheat_recall']:6.2f}%"
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


def stretch_distance(r: dict) -> tuple:
    acc_gap = max(0.0, STRETCH_ACCURACY - r["accuracy"])
    orch_gap = max(0.0, STRETCH_ORCHARD_RECALL - r["orchard_recall"])
    return (acc_gap + orch_gap, acc_gap, orch_gap, -r["accuracy"], -r["orchard_recall"])


def autopsy(y_true: np.ndarray, y_pred: np.ndarray, *, m: float, b: float) -> dict:
    metrics = compute_metrics(y_true, y_pred)
    orchard_pred = y_pred == GLOBAL_ORCHARDS
    orchard_true = y_true == GLOBAL_ORCHARDS
    fp_mask = orchard_pred & ~orchard_true
    fn_mask = orchard_true & (y_pred != GLOBAL_ORCHARDS)
    tp = int((orchard_pred & orchard_true).sum())
    fp = int(fp_mask.sum())
    fn = int(fn_mask.sum())

    fp_by_true = {
        CLASS_NAMES[c]: int((y_true[fp_mask] == c).sum())
        for c in range(NUM_GLOBAL_CLASSES)
        if c != GLOBAL_ORCHARDS
    }
    fn_by_pred = {
        CLASS_NAMES[c]: int((y_pred[fn_mask] == c).sum())
        for c in range(NUM_GLOBAL_CLASSES)
        if c != GLOBAL_ORCHARDS
    }

    meadow_ids = {GLOBAL_PERMANENT_MEADOWS, GLOBAL_TEMPORARY_MEADOWS}
    fp_meadow = int(sum(1 for t in y_true[fp_mask] if int(t) in meadow_ids))
    fn_to_meadow = int(sum(1 for p in y_pred[fn_mask] if int(p) in meadow_ids))

    return {
        "m": float(m),
        "b_orchard": float(b),
        **metrics,
        "orchard_tp": tp,
        "orchard_fp": fp,
        "orchard_fn": fn,
        "fp_by_true_class": fp_by_true,
        "fn_by_pred_class": fn_by_pred,
        "fp_meadow_share": float(fp_meadow / fp) if fp else 0.0,
        "fn_to_meadow_share": float(fn_to_meadow / fn) if fn else 0.0,
        "n_true_orchards": int(orchard_true.sum()),
        "n_pred_orchards": int(orchard_pred.sum()),
    }


def format_autopsy(a: dict) -> list[str]:
    lines = [
        f"Operating point m={a['m']:g} b={a['b_orchard']:g}",
        f"  acc={a['accuracy']:.2f}%  orch_R={a['orchard_recall']:.2f}%  "
        f"orch_P={a['orchard_precision']:.2f}%",
        f"  true_orchards={a['n_true_orchards']}  pred_orchards={a['n_pred_orchards']}  "
        f"TP={a['orchard_tp']}  FP={a['orchard_fp']}  FN={a['orchard_fn']}",
        f"  FP that are meadows: {a['fp_meadow_share']*100:.1f}%  "
        f"FN predicted as meadows: {a['fn_to_meadow_share']*100:.1f}%",
        "  False orchard positives by true class:",
    ]
    for name, count in sorted(a["fp_by_true_class"].items(), key=lambda x: -x[1]):
        if count:
            share = 100.0 * count / max(a["orchard_fp"], 1)
            lines.append(f"    {name:14s}  {count:6d}  ({share:5.1f}%)")
    lines.append("  Missed orchards predicted as:")
    for name, count in sorted(a["fn_by_pred_class"].items(), key=lambda x: -x[1]):
        if count:
            share = 100.0 * count / max(a["orchard_fn"], 1)
            lines.append(f"    {name:14s}  {count:6d}  ({share:5.1f}%)")
    return lines


def blend_two_models(
    proba_orchard: np.ndarray,
    proba_majority: np.ndarray,
    y_eval: np.ndarray,
    *,
    orchard_thresh: float,
    m: float,
    b: float,
) -> dict:
    """Use orchard specialist when its orchard prob exceeds thresh; else majority."""
    p_orch = apply_class_log_bias(proba_orchard, GLOBAL_ORCHARDS, b)
    p_maj = proba_majority
    use_orch = p_orch[:, GLOBAL_ORCHARDS] >= float(orchard_thresh)
    blended = p_maj.copy()
    blended[use_orch] = p_orch[use_orch]
    pred = decode_probs(blended)
    return {
        "alpha": "blend",
        "orchard_thresh": float(orchard_thresh),
        "m": float(m),
        "b_orchard": float(b),
        "n_orchard_specialist": int(use_orch.sum()),
        **compute_metrics(y_eval, pred),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-cache", type=Path, default=Path("artifacts/cached_logits_train.pt"))
    p.add_argument("--eval-cache", type=Path, default=Path("artifacts/cached_logits_eval.pt"))
    p.add_argument("--multiplier", type=float, default=1.0)
    p.add_argument("--autopsy-biases", default="0,0.5,1")
    p.add_argument("--orchard-biases", default="0,0.5,1,1.5,2,2.5,3")
    p.add_argument("--l2b-alphas", default="0,0.1,0.2,0.3,0.5,1.0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--proba-dir", type=Path, default=Path("artifacts/pruned_proba_cache"))
    p.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore cached probabilities even if their fingerprint matches.",
    )
    p.add_argument(
        "--blend-thresholds",
        default="0.05,0.1,0.15,0.2,0.3",
        help="Orchard-prob thresholds for two-model blend (specialist vs majority).",
    )
    p.add_argument("--output-log", type=Path, default=Path("artifacts/autopsy_soft_l2b.log"))
    p.add_argument("--output-json", type=Path, default=Path("artifacts/autopsy_soft_l2b.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    alphas = parse_float_list(args.l2b_alphas)
    biases = parse_float_list(args.orchard_biases)
    autopsy_biases = parse_float_list(args.autopsy_biases)
    blend_thresholds = parse_float_list(args.blend_thresholds)
    m = float(args.multiplier)
    np.random.seed(args.seed)

    _, full_dim = feature_dims(use_metadata=True)
    print(f"Loading {full_dim}-d features ...")
    x_train, y_train = load_feature_matrix(args.train_cache, use_metadata=True)
    x_eval, y_eval = load_feature_matrix(args.eval_cache, use_metadata=True)
    if x_train.shape[1] != full_dim:
        raise SystemExit(f"Expected {full_dim}-d; got {x_train.shape[1]}-d")
    print(f"  train={len(y_train):,}  eval={len(y_eval):,}")

    blocks = column_blocks(full_dim)
    keep = keep_columns_minus_l2c(blocks)
    # Remap L2b indices into the kept-column matrix.
    l2b_in_full = list(blocks["l2b"])
    keep_list = keep.tolist()
    l2b_in_kept = [keep_list.index(c) for c in l2b_in_full]

    lines: list[str] = [
        "Orchard FP autopsy + soft-L2b scale sweep",
        f"Base layout: minus_l2c ({keep.size}-d) with L2b scaled by alpha",
        f"m={m:g}  alphas={alphas}  biases={biases}",
        "",
        "=== AUTOPSY (minus_l2b_l2c cached probs) ===",
    ]

    autopsy_results = []
    # Same cache entry sweep_pruned_frontier.py writes; recomputed here if stale.
    proba_drop = load_or_train_proba(
        name="minus_l2b_l2c",
        columns=build_subsets(blocks)["minus_l2b_l2c"],
        x_train=x_train,
        y_train=y_train,
        x_eval=x_eval,
        m=m,
        seed=int(args.seed),
        proba_dir=args.proba_dir,
        force=bool(args.force_recompute),
    )
    for b in autopsy_biases:
        pred = decode_probs(apply_class_log_bias(proba_drop, GLOBAL_ORCHARDS, b))
        a = autopsy(y_eval, pred, m=m, b=b)
        autopsy_results.append(a)
        lines.extend(["", *format_autopsy(a)])

    # Soft-L2b sweep
    lines.extend(["", "=== SOFT L2b SCALE SWEEP ==="])
    all_rows: list[dict] = []
    x_train_kept = x_train[:, keep]
    x_eval_kept = x_eval[:, keep]

    proba_by_alpha: dict[float, np.ndarray] = {}
    for alpha in alphas:
        tag = f"soft_l2b_a{alpha:g}_minus_l2c"
        proba_path = args.proba_dir / f"proba_{tag}_m{m:g}.npz"
        xt = x_train_kept.copy()
        xe = x_eval_kept.copy()
        xt[:, l2b_in_kept] *= float(alpha)
        xe[:, l2b_in_kept] *= float(alpha)
        fingerprint = meta_proba_fingerprint(
            xt, y_train, xe, seed=int(args.seed), kind="geo_mean_xgb_cat", m=m
        )
        cached = load_npz(proba_path, fingerprint, force=bool(args.force_recompute))
        if cached is not None:
            print(f"Loaded {proba_path}", flush=True)
            proba = cached["proba"].astype(np.float32)
        else:
            print(f"Training alpha={alpha:g} @ m={m:g} ...", flush=True)
            proba = train_blended_probs(xt, y_train, xe, m=m, seed=int(args.seed))
            save_npz(proba_path, fingerprint, proba=proba)
            print(f"Saved {proba_path.resolve()}", flush=True)
        proba_by_alpha[float(alpha)] = proba

        for b in biases:
            pred = decode_probs(apply_class_log_bias(proba, GLOBAL_ORCHARDS, b))
            all_rows.append(
                {
                    "alpha": float(alpha),
                    "m": m,
                    "b_orchard": float(b),
                    **compute_metrics(y_eval, pred),
                }
            )

    best_prod = pick_best(
        all_rows, acc_min=BASELINE_ACCURACY, orchard_min=BASELINE_ORCHARD_RECALL
    )
    best_stretch = pick_best(
        all_rows, acc_min=STRETCH_ACCURACY, orchard_min=STRETCH_ORCHARD_RECALL
    )
    closest = min(all_rows, key=stretch_distance)
    best_acc = max(all_rows, key=lambda r: (r["accuracy"], r["orchard_recall"]))
    frontier = pareto_frontier(all_rows)

    lines.extend(
        [
            "",
            f"Best production: {format_row(best_prod) if best_prod else 'none'}",
            f"Best stretch:    {format_row(best_stretch) if best_stretch else 'none'}",
            f"Closest stretch: {format_row(closest)}",
            f"Best accuracy:   {format_row(best_acc)}",
            f"Pareto ({len(frontier)} pts):",
        ]
    )
    for r in frontier[:20]:
        lines.append("  " + format_row(r))

    near80 = sorted(
        [r for r in all_rows if r["accuracy"] >= 80.0],
        key=lambda r: (-r["orchard_recall"], -r["accuracy"]),
    )[:15]
    lines.append("High-acc band (acc>=80%), by orchard then acc:")
    for r in near80:
        lines.append("  " + format_row(r))

    # Two-model blend: orchard specialist = alpha=0 (drop L2b), majority = alpha=1
    lines.extend(["", "=== TWO-MODEL BLEND (alpha=0 specialist vs alpha=1 majority) ==="])
    blend_rows: list[dict] = []
    # alpha=0 coincides with minus_l2b_l2c only if L2b cols are zeroed but still present;
    # use the hard-drop cache as specialist (same info, fewer cols) for stability.
    p_spec = proba_drop
    if 1.0 not in proba_by_alpha:
        raise SystemExit("--l2b-alphas must include 1.0 (the majority model) for the blend.")
    p_maj = proba_by_alpha[1.0]

    for thresh in blend_thresholds:
        for b in biases:
            blend_rows.append(
                blend_two_models(
                    p_spec, p_maj, y_eval, orchard_thresh=thresh, m=m, b=b
                )
            )

    best_blend_prod = pick_best(
        blend_rows, acc_min=BASELINE_ACCURACY, orchard_min=BASELINE_ORCHARD_RECALL
    )
    best_blend_stretch = pick_best(
        blend_rows, acc_min=STRETCH_ACCURACY, orchard_min=STRETCH_ORCHARD_RECALL
    )
    closest_blend = min(blend_rows, key=stretch_distance)
    blend_frontier = pareto_frontier(blend_rows)
    lines.extend(
        [
            f"Best production: "
            + (
                f"thr={best_blend_prod['orchard_thresh']:g}  {format_row(best_blend_prod)}"
                if best_blend_prod
                else "none"
            ),
            f"Best stretch:    "
            + (
                f"thr={best_blend_stretch['orchard_thresh']:g}  {format_row(best_blend_stretch)}"
                if best_blend_stretch
                else "none"
            ),
            f"Closest stretch: thr={closest_blend['orchard_thresh']:g}  "
            + format_row(closest_blend),
            f"Pareto ({len(blend_frontier)} pts):",
        ]
    )
    for r in blend_frontier[:12]:
        lines.append(
            f"  thr={r['orchard_thresh']:g}  n_spec={r['n_orchard_specialist']}  "
            + format_row(r)
        )

    report = "\n".join(lines)
    print("\n" + report)
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(report + "\n", encoding="utf-8")
    payload = {
        "autopsy": autopsy_results,
        "soft_l2b_rows": all_rows,
        "soft_l2b_best_production": best_prod,
        "soft_l2b_best_stretch": best_stretch,
        "soft_l2b_closest_stretch": closest,
        "soft_l2b_pareto": frontier,
        "blend_rows": blend_rows,
        "blend_best_production": best_blend_prod,
        "blend_best_stretch": best_blend_stretch,
        "blend_closest_stretch": closest_blend,
        "blend_pareto": blend_frontier,
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.output_log.resolve()}\nSaved: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()

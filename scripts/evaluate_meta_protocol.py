"""Out-of-sample evaluation of the compact dual-bias meta-classifier on frh04.

``sweep_compact_dual_bias.py`` picks the decode configuration (blend, orchard
multiplier m, orchard / meadow log-bias) on frh04 and reports it on frh04. Here
the choice is cross-fitted over spatial blocks of frh04: every parcel is
predicted with the configuration a fixed rule selects on the *other* folds
(10 km blocks, 5 folds by default). The meta-classifier is still trained on
frh01-frh03 only; nothing is retrained, the sweep's probability caches are
re-decoded.

Selection rules (fixed in advance):
    accuracy_first   max accuracy with orchard recall >= 35%
    orchard_first    max orchard recall with accuracy >= 80.69%
    closest_stretch  min shortfall to (81% accuracy, 60% orchard recall)

Reports, per arm (cache variant) and seed: naive (selected on all of frh04) vs
cross-fitted metrics, spatial block-bootstrap 95% CIs, paired differences
between arms (paired by seed), per-class precision / recall /
F1, and orchard average precision.

Example::

    python scripts/evaluate_meta_protocol.py \\
        --arm aligned=artifacts/pruned_proba_cache_aligned \\
        --arm origcaches=artifacts/pruned_proba_cache_origcaches
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from agricultureai.evaluation.selection_protocol import (
    HEADLINE_KEYS,
    SELECTION_RULES,
    bootstrap_metric_samples,
    classification_metrics,
    confusion,
    cross_fit,
    per_class_from_confusion,
    percentile_ci,
    select_on,
    spatial_block_folds,
)
from agricultureai.meta_features import apply_orchard_meadow_log_bias, decode_probs
from agricultureai.pipeline import GLOBAL_ORCHARDS, NUM_GLOBAL_CLASSES

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from sweep_compact_dual_bias import make_blend, parse_float_list  # noqa: E402

CLASS_NAMES = (
    "barley", "wheat", "rapeseed", "corn", "sunflower",
    "orchards", "nuts", "perm_meadow", "temp_meadow",
)
METRIC_CRS = "EPSG:2154"  # Lambert-93, metres
DEFAULT_ARMS = ("msfix", "globalfix", "aligned", "l2aonly", "origcaches")
# NEW:OLD arm pairs whose paired differences are reported (one per fix, in order).
DEFAULT_PAIRS = ("msfix:aligned", "globalfix:msfix", "aligned:l2aonly", "aligned:origcaches")


# --------------------------------------------------------------------------- #
# Inputs                                                                       #
# --------------------------------------------------------------------------- #
def discover_arms(names) -> Dict[str, List[Path]]:
    arms = {}
    for name in names:
        base = Path(f"artifacts/pruned_proba_cache_{name}")
        dirs = ([base] if base.is_dir() else []) + sorted(
            Path("artifacts").glob(f"pruned_proba_cache_{name}_seed*")
        )
        if dirs:
            arms[name] = dirs
    return arms


def seed_label(proba_dir: Path) -> str:
    match = re.search(r"_seed(\d+)$", proba_dir.name)
    return f"seed{match.group(1)}" if match else "seed42"


def load_spatial_folds(
    path: Path,
    parcel_ids: np.ndarray,
    *,
    region: str,
    level: str,
    block_m: float,
    n_folds: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Fold and block ids per cache row; computed from parcel centroids once, then cached."""
    params = {"region": region, "block_m": block_m, "n_folds": n_folds, "seed": seed}
    if path.is_file():
        saved = np.load(path)
        same = json.loads(str(saved["params"])) == params
        if same and np.array_equal(saved["parcel_ids"], parcel_ids):
            return saved["folds"], saved["blocks"], params
        print(f"{path} does not match this cache / parameters; recomputing.")

    import breizhcrops as bzh

    print(f"Loading {region} parcel geometry for spatial folds ...")
    geo = bzh.BreizhCrops(region, level=level, load_timeseries=False).geodataframe()
    geo = geo.loc[~geo.geometry.isna()].set_index("id")
    geo = geo.loc[~geo.index.duplicated()].to_crs(METRIC_CRS)
    cent = geo.geometry.centroid
    lookup_x = dict(zip(geo.index.astype(np.int64), cent.x.to_numpy()))
    lookup_y = dict(zip(geo.index.astype(np.int64), cent.y.to_numpy()))
    x = np.array([lookup_x.get(int(i), np.nan) for i in parcel_ids])
    y = np.array([lookup_y.get(int(i), np.nan) for i in parcel_ids])
    folds, blocks = spatial_block_folds(x, y, block_size=block_m, n_folds=n_folds, seed=seed)
    np.savez(path, parcel_ids=parcel_ids, x=x, y=y, folds=folds, blocks=blocks, params=json.dumps(params))
    print(f"Saved spatial folds -> {path} ({int(np.isnan(x).sum())} parcels without geometry)")
    return folds, blocks, params


def config_grid(proba_dir: Path, args) -> Tuple[List[dict], np.ndarray, Dict[str, np.ndarray]]:
    """Every (m, blend, b_orchard, b_meadow) decode of one seed, in sweep order."""
    configs, preds, orchard_scores = [], [], {}
    for m in parse_float_list(args.multipliers):
        pair = np.load(proba_dir / f"proba_pair_{args.layout}_m{m:g}.npz")
        for blend in [b.strip() for b in args.blends.split(",") if b.strip()]:
            proba = make_blend(blend, pair["p_xgb"], pair["p_cat"])
            orchard_scores[f"{blend} m={m:g}"] = proba[:, GLOBAL_ORCHARDS]
            for b_o in parse_float_list(args.orchard_biases):
                for b_m in parse_float_list(args.meadow_biases):
                    biased = apply_orchard_meadow_log_bias(proba, b_orchard=b_o, b_meadow=b_m)
                    preds.append(decode_probs(biased).astype(np.int8))
                    configs.append({"blend": blend, "m": m, "b_orchard": b_o, "b_meadow": b_m})
    return configs, np.stack(preds), orchard_scores


def describe(cfg: dict) -> str:
    return f"{cfg['blend']} m={cfg['m']:g} b_o={cfg['b_orchard']:g} b_m={cfg['b_meadow']:g}"


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
class Report:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def __call__(self, line: str = "") -> None:
        print(line)
        self.lines.append(line)


def ci_text(samples: np.ndarray) -> str:
    lo, hi = percentile_ci(samples)
    return f"[{lo:5.2f},{hi:5.2f}]"


def main() -> None:
    args = parse_args()
    out = Report()
    payload = torch.load(args.eval_cache, map_location="cpu", weights_only=True)
    y = payload["labels"].numpy().astype(np.int64)
    parcel_ids = payload["parcel_ids"].numpy().astype(np.int64)
    meta = payload["meta"]
    region, level = meta["regions"][0], str(meta.get("level", "L2A"))

    arms = {}
    for spec in args.arm or []:
        name, dirs = spec.split("=", 1)
        arms[name] = [Path(d) for d in dirs.split(",")]
    if not arms:
        arms = discover_arms(DEFAULT_ARMS)
    if not arms:
        raise SystemExit("No proba caches found; run sweep_compact_dual_bias.py first.")

    folds, blocks, fold_params = load_spatial_folds(
        args.folds_file, parcel_ids, region=region, level=level,
        block_m=1000.0 * args.block_km, n_folds=args.n_folds, seed=args.fold_seed,
    )
    out(f"Out-of-sample decode selection on {region} ({len(y):,} parcels, {int((y == GLOBAL_ORCHARDS).sum())} orchards)")
    out(f"Spatial cross-fitting: {args.block_km:g} km blocks ({len(np.unique(blocks))} blocks), {args.n_folds} folds, seed {args.fold_seed}")
    for f in range(args.n_folds):
        sel = folds == f
        out(f"  fold {f}: {int(sel.sum()):6,} parcels, {int((y[sel] == GLOBAL_ORCHARDS).sum()):3d} orchards")
    out("Meta-classifier trained on frh01-frh03; decode config selected on the other folds only.")
    out(f"Rules: accuracy_first (max acc | orchard>=35), orchard_first (max orchard | acc>=80.69), closest_stretch")

    # ---- per arm / seed ------------------------------------------------------
    results: Dict[str, dict] = defaultdict(dict)
    cf_preds: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    ap_rows: Dict[str, Dict[str, float]] = {}
    for arm, dirs in arms.items():
        for proba_dir in dirs:
            key = f"{arm}/{seed_label(proba_dir)}"
            configs, preds, orchard_scores = config_grid(proba_dir, args)
            if preds.shape[1] != y.size:
                raise RuntimeError(f"{proba_dir}: {preds.shape[1]} rows, eval cache has {y.size}")
            ap_rows[key] = {
                name: 100.0 * average_precision_score(y == GLOBAL_ORCHARDS, s)
                for name, s in orchard_scores.items()
            }
            for rule_name, rule in SELECTION_RULES.items():
                k = select_on(preds, y, np.ones_like(y, dtype=bool), rule)
                cf, chosen = cross_fit(preds, y, folds, rule)
                cf_preds[rule_name][key] = cf
                results[rule_name][key] = {
                    "naive_config": configs[k],
                    "naive": classification_metrics(y, preds[k]),
                    "fold_configs": [configs[c] for c in chosen],
                    "cross_fitted": classification_metrics(y, cf),
                }

    # ---- bootstrap (shared resamples -> paired differences) ------------------
    boot = {
        rule_name: bootstrap_metric_samples(
            y, cf_preds[rule_name], n_boot=args.n_boot, seed=args.boot_seed, groups=blocks
        )
        for rule_name in SELECTION_RULES
    }

    # ---- report ---------------------------------------------------------------
    # The cache's msresnet_logits were extracted from zero-padded input and are
    # not a valid stand-alone baseline, so only the published references are quoted.
    out("\nReferences (L1C eval set, 122,614 parcels; not paired with these rows):")
    out("  MSResNet (stock transform, README):               acc=80.77 orchR=15.73 wheat=97.58")
    out("  README best (XGB m=25, 64-d, selected on frh04):  acc=80.76 orchR=35.80 wheat=97.57")

    for rule_name in SELECTION_RULES:
        out(f"\n=== Rule: {rule_name} " + "=" * 60)
        out(f"{'arm/seed':22s} | {'naive (selected on frh04)':27s} | cross-fitted, 95% block-bootstrap CI")
        out(f"{'':22s} | {'acc':>6s} {'orchR':>6s} {'orchP':>6s}  | "
            f"{'acc':>6s} {'CI':15s} {'orchR':>6s} {'CI':15s} {'orchP':>6s} {'orchF1':>6s} {'wheat':>6s} {'macroF1':>7s}")
        for key, r in results[rule_name].items():
            nv, cf, s = r["naive"], r["cross_fitted"], boot[rule_name][key]
            out(f"{key:22s} | {nv['accuracy']:6.2f} {nv['orchard_recall']:6.2f} {nv['orchard_precision']:6.2f}  | "
                f"{cf['accuracy']:6.2f} {ci_text(s['accuracy'])} {cf['orchard_recall']:6.2f} {ci_text(s['orchard_recall'])} "
                f"{cf['orchard_precision']:6.2f} {cf['orchard_f1']:6.2f} {cf['wheat_recall']:6.2f} {cf['macro_f1']:7.2f}")
        # seed spread per arm
        by_arm = defaultdict(list)
        for key, r in results[rule_name].items():
            by_arm[key.split("/")[0]].append(r["cross_fitted"])
        for arm, rows in by_arm.items():
            if len(rows) > 1:
                parts = [f"{k}={np.mean([r[k] for r in rows]):.2f}+/-{np.std([r[k] for r in rows], ddof=1):.2f}"
                         for k in ("accuracy", "orchard_recall", "orchard_precision", "macro_f1")]
                out(f"  {arm} over {len(rows)} seeds (mean+/-sd): " + "  ".join(parts))
        # chosen configs (stability)
        for key, r in results[rule_name].items():
            counts = defaultdict(int)
            for cfg in r["fold_configs"]:
                counts[describe(cfg)] += 1
            out(f"  {key}: naive pick {describe(r['naive_config'])}; fold picks "
                + "; ".join(f"{c} x{n}" for c, n in counts.items()))
        # paired differences
        out("  Paired differences (block bootstrap, same resamples):")
        keys = list(results[rule_name])
        pairs = [(a, b) for new, old in args.pairs for a in keys for b in keys
                 if a.split("/")[0] == new and b.split("/")[0] == old
                 and a.split("/")[1] == b.split("/")[1]]
        for a, b in pairs:
            sa, sb = boot[rule_name][a], boot[rule_name][b]
            parts = []
            for k in ("accuracy", "orchard_recall", "orchard_f1", "macro_f1"):
                d = sa[k] - sb[k]
                point = results[rule_name][a]["cross_fitted"][k] - results[rule_name][b]["cross_fitted"][k]
                lo, hi = percentile_ci(d)
                parts.append(f"{k} {point:+.2f} [{lo:+.2f},{hi:+.2f}] P(>0)={float((d > 0).mean()):.2f}")
            out(f"    {a} - {b}: " + " | ".join(parts))

    # per-class table for the headline arm
    head = next((k for k in results["accuracy_first"] if k.startswith("aligned/seed42")), None)
    head = head or next(iter(results["accuracy_first"]))
    for rule_name in ("accuracy_first", "orchard_first"):
        pc = per_class_from_confusion(confusion(y, cf_preds[rule_name][head]))
        out(f"\nPer-class (cross-fitted, {rule_name}, {head}):")
        out(f"  {'class':12s} {'support':>7s} {'prec':>6s} {'recall':>6s} {'F1':>6s}")
        for c in range(NUM_GLOBAL_CLASSES):
            out(f"  {CLASS_NAMES[c]:12s} {int(pc['support'][c]):7d} {100 * pc['precision'][c]:6.2f} "
                f"{100 * pc['recall'][c]:6.2f} {100 * pc['f1'][c]:6.2f}")

    out("\nOrchard average precision (threshold-free, from blended probabilities; no selection):")
    for key, rows in ap_rows.items():
        out(f"  {key:22s} " + "  ".join(f"{n}={v:.2f}" for n, v in rows.items()))

    report = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "eval_cache": str(args.eval_cache.resolve()),
        "folds": fold_params,
        "n_boot": args.n_boot,
        "arms": {a: [str(d) for d in ds] for a, ds in arms.items()},
        "orchard_ap": ap_rows,
        "rules": {
            rule: {
                key: {
                    **r,
                    "ci95": {k: percentile_ci(boot[rule][key][k]) for k in HEADLINE_KEYS},
                }
                for key, r in rows.items()
            }
            for rule, rows in results.items()
        },
    }
    args.output_json.write_text(json.dumps(report, indent=2))
    args.output_log.write_text("\n".join(out.lines) + "\n")
    print(f"\nSaved {args.output_json} and {args.output_log}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", help="NAME=PROBA_DIR[,PROBA_DIR...] (one dir per seed).")
    p.add_argument(
        "--pair",
        action="append",
        dest="pairs",
        help="NEW:OLD arms to compare per seed (repeatable; default: " + ", ".join(DEFAULT_PAIRS) + ").",
    )
    p.add_argument("--eval-cache", type=Path, default=Path("artifacts/cached_logits_eval.pt"))
    p.add_argument("--layout", default="compact")
    p.add_argument("--multipliers", default="1,2,3")
    p.add_argument("--blends", default="geo_mean,arith_xgb0.7")
    p.add_argument("--orchard-biases", default="0,0.5,1,1.5,2,2.5,3")
    p.add_argument("--meadow-biases", default="0,0.25,0.5,0.75,1,1.5,2")
    p.add_argument("--folds-file", type=Path, default=Path("artifacts/frh04_spatial_folds.npz"))
    p.add_argument("--block-km", type=float, default=10.0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--fold-seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--output-json", type=Path, default=Path("artifacts/meta_protocol_frh04.json"))
    p.add_argument("--output-log", type=Path, default=Path("artifacts/meta_protocol_frh04.log"))
    args = p.parse_args()
    raw_pairs = args.pairs or list(DEFAULT_PAIRS)
    if any(spec.count(":") != 1 for spec in raw_pairs):
        p.error("--pair takes NEW:OLD arm names.")
    args.pairs = [tuple(spec.split(":")) for spec in raw_pairs]
    return args


if __name__ == "__main__":
    main()

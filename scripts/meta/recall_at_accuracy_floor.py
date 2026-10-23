"""Orchard recall at an accuracy floor, cross-fitted, and orchard AP per meta-classifier arm.

For each arm and seed, the decode configuration (blend, m, orchard / meadow bias) that maximizes orchard
recall subject to accuracy >= ``--min-accuracy`` is chosen on four of the five spatial folds and applied
to the fifth (the same cross-fitting as ``evaluate_meta_protocol.py``). ``--pair NEW:OLD`` adds paired
block-bootstrap intervals of the per-seed differences. Orchard AP is that of the geometric-mean blend at
m = 1. This gives the "R at acc >= 81%" and "Orch. AP" columns of the paper's Table 2.

Example::

    A=artifacts/pruned_proba_cache_noexpens
    python scripts/meta/recall_at_accuracy_floor.py --arm final=$A,${A}_seed1,${A}_seed2 \\
        --arm none=... --pair final:none
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from agricultureai.evaluation.meta_protocol import config_grid, load_spatial_folds
from agricultureai.evaluation.selection_protocol import (
    bootstrap_metric_samples,
    classification_metrics,
    cross_fit,
    percentile_ci,
    select_orchard_first,
)
from agricultureai.pipeline import GLOBAL_ORCHARDS

KEYS = ("accuracy", "orchard_recall", "orchard_precision")


def main() -> None:
    args = parse_args()
    lines = []

    def out(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    payload = torch.load(args.eval_cache, map_location="cpu", weights_only=True)
    y = payload["labels"].numpy().astype(np.int64)
    meta = payload["meta"]
    folds, blocks, _ = load_spatial_folds(
        args.folds_file, payload["parcel_ids"].numpy().astype(np.int64), region=meta["regions"][0],
        level=str(meta.get("level", "L2A")), block_m=1000.0 * args.block_km, n_folds=args.n_folds, seed=args.fold_seed,
    )
    rule = lambda rows: select_orchard_first(rows, min_accuracy=args.min_accuracy)  # noqa: E731

    arms = {}
    for spec in args.arm:
        name, dirs = spec.split("=", 1)
        arms[name] = [Path(d) for d in dirs.split(",")]
    n_seeds = {len(d) for d in arms.values()}
    if len(n_seeds) != 1:
        raise SystemExit("Every arm needs the same number of seed directories.")

    preds, ap = {}, {}
    out(f"Orchard-first with accuracy >= {args.min_accuracy:g} (cross-fitted over {args.n_folds} spatial folds):")
    for name, dirs in arms.items():
        for s, proba_dir in enumerate(dirs):
            _, grid, orchard_scores = config_grid(proba_dir, args)
            cf, chosen = cross_fit(grid, y, folds, rule)
            preds[name, s] = cf
            ap[name, s] = 100.0 * average_precision_score(y == GLOBAL_ORCHARDS, orchard_scores["geo_mean m=1"])
            m = classification_metrics(y, cf)
            out(f"  {name:14s} {proba_dir.name:44s} acc={m['accuracy']:.2f} orchR={m['orchard_recall']:.2f} "
                f"orchP={m['orchard_precision']:.2f}  AP={ap[name, s]:.2f}  folds->{chosen}")
    out()
    for name, dirs in arms.items():
        rec = [classification_metrics(y, preds[name, s])["orchard_recall"] for s in range(len(dirs))]
        aps = [ap[name, s] for s in range(len(dirs))]
        out(f"  {name:14s} mean orchard recall {np.mean(rec):.2f} +/- {np.std(rec, ddof=1) if len(rec) > 1 else 0:.2f}"
            f"   mean orchard AP {np.mean(aps):.2f}")

    if args.pairs:
        out()
        out(f"Paired differences (NEW - OLD), 95% block-bootstrap intervals ({args.n_boot} resamples, seed {args.boot_seed}):")
    for spec in args.pairs:
        new, old = spec.split(":")
        for s, proba_dir in enumerate(arms[new]):
            boot = bootstrap_metric_samples(
                y, {"new": preds[new, s], "old": preds[old, s]}, n_boot=args.n_boot, seed=args.boot_seed,
                groups=blocks, keys=KEYS,
            )
            mn, mo = classification_metrics(y, preds[new, s]), classification_metrics(y, preds[old, s])
            parts = []
            for k in ("accuracy", "orchard_recall"):
                d = boot["new"][k] - boot["old"][k]
                lo, hi = percentile_ci(d)
                parts.append(f"{k} {mn[k] - mo[k]:+.2f} [{lo:+.2f},{hi:+.2f}] P(>0)={(d > 0).mean():.2f}")
            seed = re.search(r"_seed(\d+)$", proba_dir.name)
            out(f"  {new}-{old} seed {seed.group(1) if seed else 42}: " + " | ".join(parts))

    if args.output_log:
        args.output_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved {args.output_log}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True, help="NAME=PROBA_DIR[,PROBA_DIR...] (one dir per seed).")
    p.add_argument("--pair", dest="pairs", action="append", default=[], help="NEW:OLD arms to compare per seed.")
    p.add_argument("--min-accuracy", type=float, default=81.0)
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
    p.add_argument("--output-log", type=Path, default=None)
    args = p.parse_args()
    if any(spec.count(":") != 1 for spec in args.pairs):
        p.error("--pair takes NEW:OLD arm names.")
    return args


if __name__ == "__main__":
    main()

"""Multi-seed feature-layout ablation for the meta-classifier.

XGBoost only, fixed orchard multiplier, no decode bias, trained on the train
cache and scored on the eval cache, over several seeds, with caches and outputs
as arguments, and with leave-one-block-out from the production ``compact``
layout. Deltas are paired by seed (variant minus compact).

Example::

    python scripts/meta/ablate_meta_layouts.py --seeds 42,1,2 \\
        --output-log artifacts/ablate_meta_layouts.log --output-json artifacts/ablate_meta_layouts.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from agricultureai.evaluation.selection_protocol import classification_metrics
from agricultureai.meta_features import (
    COMPACT_BLOCK_NAMES,
    build_xgb,
    feature_column_indices,
    fit_booster,
    full_feature_blocks,
    load_feature_matrix,
    orchard_sample_weights,
    predict_xgb,
)

KEYS = ("accuracy", "orchard_recall", "orchard_precision", "orchard_f1", "macro_f1")
_ALL = list(full_feature_blocks())
LAYOUTS: Dict[str, List[str]] = {
    "full": _ALL,
    "minus_l2b_l2c": [n for n in _ALL if n not in ("l2b", "l2c_raw", "l2c_soft")],
    "compact": list(COMPACT_BLOCK_NAMES),
    "compact_no_meta": ["l3_raw", "cereal", "msresnet", "transformer"],
    "orchard_lean": ["l3_raw", "l3_soft", "cereal", "msresnet", "transformer"],
    "globals_meta": ["msresnet", "global_eng", "transformer", "metadata"],
    "drop_harm_and_noise": ["l3_raw", "cereal", "msresnet", "l3_soft", "transformer", "metadata"],
}
LAYOUTS.update(
    {f"compact_minus_{b}": [n for n in COMPACT_BLOCK_NAMES if n != b] for b in COMPACT_BLOCK_NAMES}
)


def fit_and_score(x_tr, y_tr, x_ev, y_ev, columns, *, m: float, seed: int) -> dict:
    clf = build_xgb(seed)
    fit_booster(clf, x_tr[:, columns], y_tr, orchard_sample_weights(y_tr, m))
    return classification_metrics(y_ev, predict_xgb(clf, x_ev[:, columns]).astype(np.int64))


def mean_sd(values: List[float]) -> str:
    sd = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return f"{np.mean(values):6.2f}+/-{sd:4.2f}"


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    x_tr, y_tr = load_feature_matrix(args.train_cache, use_metadata=True)
    x_ev, y_ev = load_feature_matrix(args.eval_cache, use_metadata=True)
    print(f"train={len(y_tr):,} eval={len(y_ev):,} feats={x_tr.shape[1]} m={args.orchard_multiplier:g} seeds={seeds}")

    rows: Dict[str, dict] = {}
    for name, blocks in LAYOUTS.items():
        columns = feature_column_indices(blocks)
        per_seed = []
        for seed in seeds:
            t0 = time.time()
            per_seed.append(fit_and_score(x_tr, y_tr, x_ev, y_ev, columns, m=args.orchard_multiplier, seed=seed))
            print(f"  {name:28s} seed={seed:<3d} acc={per_seed[-1]['accuracy']:.2f} "
                  f"orchR={per_seed[-1]['orchard_recall']:.2f} ({time.time() - t0:.0f}s)", flush=True)
        rows[name] = {"blocks": blocks, "n_features": int(columns.size), "per_seed": per_seed}

    base = rows["compact"]["per_seed"]
    lines = [
        "Meta-classifier feature-layout ablation (XGBoost only, no decode bias)",
        f"train={args.train_cache}  eval={args.eval_cache}",
        f"m={args.orchard_multiplier:g}  seeds={seeds}  (mean+/-sd over seeds; deltas paired vs compact)",
        "",
        f"{'layout':28s} {'feats':>5s}  {'accuracy':>12s} {'orchard R':>12s} {'orchard P':>12s} "
        f"{'macro F1':>12s}  {'d_acc':>12s} {'d_orchR':>12s}",
        "-" * 122,
    ]
    for name, r in rows.items():
        ps = r["per_seed"]
        d_acc = [a["accuracy"] - b["accuracy"] for a, b in zip(ps, base)]
        d_orch = [a["orchard_recall"] - b["orchard_recall"] for a, b in zip(ps, base)]
        r["mean"] = {k: float(np.mean([p[k] for p in ps])) for k in KEYS}
        r["delta_vs_compact"] = {"accuracy": d_acc, "orchard_recall": d_orch}
        if name == "compact_minus_l3_raw":
            lines.append("")
            lines.append("Leave one block out of compact:")
        lines.append(
            f"{name:28s} {r['n_features']:5d}  {mean_sd([p['accuracy'] for p in ps])} "
            f"{mean_sd([p['orchard_recall'] for p in ps])} {mean_sd([p['orchard_precision'] for p in ps])} "
            f"{mean_sd([p['macro_f1'] for p in ps])}  {mean_sd(d_acc)} {mean_sd(d_orch)}"
        )
    report = "\n".join(lines)
    print("\n" + report)
    args.output_log.write_text(report + "\n", encoding="utf-8")
    args.output_json.write_text(json.dumps({"seeds": seeds, "m": args.orchard_multiplier,
                                            "train_cache": str(args.train_cache),
                                            "eval_cache": str(args.eval_cache), "layouts": rows}, indent=2),
                                encoding="utf-8")
    print(f"\nSaved {args.output_log} and {args.output_json}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-cache", type=Path, default=Path("artifacts/cached_logits_train.pt"))
    p.add_argument("--eval-cache", type=Path, default=Path("artifacts/cached_logits_eval.pt"))
    p.add_argument("--seeds", default="42,1,2")
    p.add_argument("--orchard-multiplier", type=float, default=1.0)
    p.add_argument("--output-log", type=Path, default=Path("artifacts/ablate_meta_layouts.log"))
    p.add_argument("--output-json", type=Path, default=Path("artifacts/ablate_meta_layouts.json"))
    return p.parse_args()


if __name__ == "__main__":
    main()

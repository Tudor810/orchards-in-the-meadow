"""Meta-classifier training variants: stacking-leakage and extra-feature experiments.

Writes frh04 probabilities in the format of ``sweep_compact_dual_bias.py``
(``<proba-dir>/proba_pair_compact_m<m>.npz`` with ``p_xgb`` / ``p_cat``), so
``evaluate_meta_protocol.py`` scores every variant with the same out-of-sample
decode selection.

``--train-on cache`` (default) trains on the train cache (frh01-frh03), like the
production sweep. Its expert features are in-sample: the experts and the
pretrained MSResNet / Transformer were trained on those regions.

``--train-on frh04-oof`` trains on frh04 itself, cross-fitted over the spatial
folds of ``--folds-file``: each fold is predicted by models trained on the other
folds, so the meta-classifier learns from expert outputs that are genuinely out
of sample. (Caveat: this also trains inside the test region.) A class missing
from a training fold gets one copy of a random row relabelled to it with weight
1e-6, because XGBoost requires every class to be present.

``--extra NAME=TRAIN_NPZ,EVAL_NPZ`` appends a per-parcel feature block (e.g. from
``compute_phenology_features.py``); rows are checked against the caches' parcel ids.

Example::

    python scripts/train_meta_variants.py --train-on frh04-oof --seed 42 \\
        --proba-dir artifacts/pruned_proba_cache_oof
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from agricultureai.meta_features import (
    COMPACT_BLOCK_NAMES,
    build_catboost,
    build_xgb,
    fit_catboost,
    fit_xgb,
    full_feature_blocks,
    load_feature_matrix,
    orchard_sample_weights,
    predict_proba_catboost,
    predict_proba_xgb,
    select_feature_layout,
)
from agricultureai.pipeline import NUM_GLOBAL_CLASSES

MISSING_CLASS_WEIGHT = 1e-6


def compact_column_names() -> List[str]:
    blocks = full_feature_blocks()
    return [f"{b}_{i}" for b in COMPACT_BLOCK_NAMES for i in range(len(blocks[b]))]


def load_features(cache: Path, extras: List[Tuple[str, Path]], drop_blocks: Tuple[str, ...] = ()) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    x_full, y = load_feature_matrix(cache, use_metadata=True)
    x, names = select_feature_layout(x_full, "compact"), compact_column_names()
    if drop_blocks:  # ablation: remove whole compact blocks (e.g. the expert outputs)
        keep = [i for i, n in enumerate(names) if n.rsplit("_", 1)[0] not in drop_blocks]
        x, names = x[:, keep], [names[i] for i in keep]
    if extras:
        ids = torch.load(cache, map_location="cpu", weights_only=True)["parcel_ids"].numpy()
        for block, path in extras:
            data = np.load(path)
            if not np.array_equal(data["parcel_ids"], ids):
                raise RuntimeError(f"{path}: parcel ids do not match {cache}")
            x = np.hstack([x, data["features"].astype(np.float32)])
            names += [f"{block}:{c}" for c in data["columns"]]
    return x, y, names


def ensure_all_classes(x: np.ndarray, y: np.ndarray, w: np.ndarray, rng: np.random.Generator):
    missing = sorted(set(range(NUM_GLOBAL_CLASSES)) - set(np.unique(y).tolist()))
    if not missing:
        return x, y, w
    rows = rng.integers(0, len(y), size=len(missing))
    return (
        np.vstack([x, x[rows]]),
        np.concatenate([y, np.array(missing, dtype=y.dtype)]),
        np.concatenate([w, np.full(len(missing), MISSING_CLASS_WEIGHT, dtype=w.dtype)]),
    )


def fit_predict(x_tr, y_tr, x_ev, *, m: float, seed: int):
    w = orchard_sample_weights(y_tr, m)
    x_tr, y_tr, w = ensure_all_classes(x_tr, y_tr, w, np.random.default_rng(seed))
    xgb = build_xgb(seed)
    fit_xgb(xgb, x_tr, y_tr, w)
    cat = build_catboost(seed)
    fit_catboost(cat, x_tr, y_tr, w)
    return predict_proba_xgb(xgb, x_ev), predict_proba_catboost(cat, x_ev), xgb


def top_gain(model, names: List[str], k: int = 15) -> List[Tuple[str, float]]:
    gain = model.get_booster().get_score(importance_type="gain")
    scored = [(names[int(f[1:])], v) for f, v in gain.items()]
    total = sum(v for _, v in scored) or 1.0
    return [(n, v / total) for n, v in sorted(scored, key=lambda kv: -kv[1])[:k]]


def main() -> None:
    args = parse_args()
    extras = [(name, (Path(tr), Path(ev))) for name, tr, ev in args.extra]
    x_ev, y_ev, names = load_features(args.eval_cache, [(n, p[1]) for n, p in extras], args.drop_blocks)
    if args.train_on == "cache":
        x_tr, y_tr, _ = load_features(args.train_cache, [(n, p[0]) for n, p in extras], args.drop_blocks)
    else:
        folds_npz = np.load(args.folds_file)
        ids = torch.load(args.eval_cache, map_location="cpu", weights_only=True)["parcel_ids"].numpy()
        if not np.array_equal(folds_npz["parcel_ids"], ids):
            raise RuntimeError(f"{args.folds_file} does not match {args.eval_cache}")
        folds = folds_npz["folds"]
    print(f"train-on={args.train_on} seed={args.seed} feats={x_ev.shape[1]} eval={len(y_ev):,}", flush=True)

    args.proba_dir.mkdir(parents=True, exist_ok=True)
    report = {"train_on": args.train_on, "seed": args.seed, "extras": [e[0] for e in extras], "drop_blocks": list(args.drop_blocks),
              "n_features": int(x_ev.shape[1]), "multipliers": {}}
    summary = args.proba_dir / "variant.json"
    if summary.is_file():  # one multiplier per run: keep the others' entries
        report["multipliers"] = json.loads(summary.read_text()).get("multipliers", {})
    for m in args.multipliers:
        out = args.proba_dir / f"proba_pair_compact_m{m:g}.npz"
        if out.exists() and not args.force:
            raise SystemExit(f"{out} exists; pass --force to overwrite.")
        t0 = time.time()
        if args.train_on == "cache":
            p_xgb, p_cat, xgb = fit_predict(x_tr, y_tr, x_ev, m=m, seed=args.seed)
        else:
            p_xgb = np.zeros((len(y_ev), NUM_GLOBAL_CLASSES), np.float32)
            p_cat = np.zeros_like(p_xgb)
            for fold in np.unique(folds):
                held = folds == fold
                p_xgb[held], p_cat[held], xgb = fit_predict(
                    x_ev[~held], y_ev[~held], x_ev[held], m=m, seed=args.seed
                )
        np.savez(out, p_xgb=p_xgb.astype(np.float32), p_cat=p_cat.astype(np.float32),
                 variant=json.dumps({k: report[k] for k in ("train_on", "seed", "extras")} | {"m": m}))
        acc = 100.0 * float((p_xgb.argmax(1) == y_ev).mean())
        gains = top_gain(xgb, names)
        report["multipliers"][f"{m:g}"] = {"xgb_argmax_acc": acc, "top_gain": gains}
        print(f"  m={m:g}: XGB argmax acc={acc:.2f}% ({time.time() - t0:.0f}s) -> {out}", flush=True)
        print("    top XGB gain: " + ", ".join(f"{n} {v:.3f}" for n, v in gains[:10]), flush=True)
    summary.write_text(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-cache", type=Path, default=Path("artifacts/cached_logits_train.pt"))
    p.add_argument("--eval-cache", type=Path, default=Path("artifacts/cached_logits_eval.pt"))
    p.add_argument("--train-on", choices=("cache", "frh04-oof"), default="cache")
    p.add_argument("--folds-file", type=Path, default=Path("artifacts/frh04_spatial_folds.npz"))
    p.add_argument("--extra", action="append", default=[], help="NAME=TRAIN_NPZ,EVAL_NPZ (repeatable).")
    p.add_argument("--multipliers", default="1,2,3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--proba-dir", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--drop-blocks", default="", help="Comma-separated compact blocks to remove (l3_raw, cereal, msresnet, transformer, metadata).")
    args = p.parse_args()
    args.multipliers = [float(v) for v in args.multipliers.split(",") if v.strip()]
    args.drop_blocks = tuple(b.strip() for b in args.drop_blocks.split(",") if b.strip())
    parsed = []
    for spec in args.extra:
        name, _, paths = spec.partition("=")
        tr, _, ev = paths.partition(",")
        if not (name and tr and ev):
            p.error("--extra takes NAME=TRAIN_NPZ,EVAL_NPZ")
        parsed.append((name, tr, ev))
    args.extra = parsed
    return args


if __name__ == "__main__":
    main()

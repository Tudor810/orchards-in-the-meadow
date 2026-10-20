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

    python scripts/meta/train_meta_variants.py --train-on frh04-oof --seed 42 \\
        --proba-dir artifacts/pruned_proba_cache_oof
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from agricultureai.meta_training import fit_predict, load_features, top_gain
from agricultureai.pipeline import NUM_GLOBAL_CLASSES


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

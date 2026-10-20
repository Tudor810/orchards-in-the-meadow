"""Region rotation of the frozen design (rotation-safe variant, see FROZEN_DESIGN.md).

Each fold holds out one BreizhCrops region R and trains on the other three. The pretrained
BreizhCrops networks saw frh01-frh03, so they are replaced by our own L1C MSResNet trained per
fold; the L2A MSResNet is also trained per fold. Networks are trained and exported with
``train_series_classifier.py`` (``--export eval --eval-regions <Q> --export-tag <Q>`` for every
region Q) into ``<work>/nets/{l2a,l1c}_test<R>/logits_<Q>.npz``.

Stages:
    prep                     per-region id caches (eval-cache format), spatial folds, and the
                             frh04-fold network logits split per region from existing exports
    meta --test R --arm X    XGBoost + CatBoost on the training regions, probabilities on R
                             (arms: B = metadata + phenology + L1C net + L2A net,
                                    A = B without the L2A net, D = B without phenology)
    oof --test R             out-of-fold arm-B XGBoost scores on the training regions (verifier)
    verify --test R          orchard verifier on arm B (m=1 XGBoost), evaluated on R
Probabilities land in ``<work>/<R>/proba_<arm>[_seed<s>]/proba_pair_compact_m<m>.npz`` so
``scripts/meta/evaluate_meta_protocol.py --eval-cache <work>/ids_<R>.pt --folds-file <work>/folds_<R>.npz``
scores them with the usual out-of-sample protocol.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from agricultureai.evaluation.bootstrap import block_resample_indices
from agricultureai.evaluation.meta_protocol import load_spatial_folds
from agricultureai.evaluation.selection_protocol import (
    classification_metrics, cross_fit, percentile_ci, select_accuracy_first, select_orchard_first,
)
from agricultureai.meta_features import full_feature_blocks, load_feature_matrix
from agricultureai.meta_training import fit_predict, fit_xgb_scores
from agricultureai.verifier import (
    ORCHARD, ap, decode, geometry, monthly_means, neighbour_context, precision_at_recall, threshold_grid, verifier_scores,
)

REGIONS = ("frh01", "frh02", "frh03", "frh04")
ARMS = {"B": ("metadata", "pheno", "l1c", "l2a"), "A": ("metadata", "pheno", "l1c"), "D": ("metadata", "l1c", "l2a")}


def sfx(seed: int) -> str:
    return "" if seed == 42 else f"_seed{seed}"


def region_sizes():
    return {r: len(np.load(f"artifacts/series_store/{r}.npz")["parcel_ids"]) for r in REGIONS}


def split_rows(arr: np.ndarray, regions, sizes):
    out, start = {}, 0
    for r in regions:
        out[r] = arr[start:start + sizes[r]]
        start += sizes[r]
    assert start == len(arr), (start, len(arr))
    return out


# --------------------------------------------------------------------------- #
def stage_prep(args) -> None:
    w = args.work_dir
    sizes = region_sizes()
    for r in REGIONS:
        s = np.load(f"artifacts/series_store/{r}.npz")
        out = w / f"ids_{r}.pt"
        if not out.exists():
            torch.save({"parcel_ids": torch.from_numpy(s["parcel_ids"].astype(np.int64)),
                        "labels": torch.from_numpy(s["labels"].astype(np.int64)),
                        "meta": {"regions": [r], "level": "L2A"}}, out)
        load_spatial_folds(w / f"folds_{r}.npz", s["parcel_ids"].astype(np.int64), region=r, level="L2A",
                           block_m=10000.0, n_folds=5, seed=0)
    # frh04 fold: networks already trained on frh01-frh03 (seed 42), split their exports per region
    for kind, ck in (("l2a", "artifacts/checkpoints/msresnet_l2a_clear"), ("l1c", "artifacts/checkpoints/msresnet_l1c_all")):
        d = w / "nets" / f"{kind}_testfrh04"
        d.mkdir(parents=True, exist_ok=True)
        tr, ev = np.load(f"{ck}/logits_train.npz"), np.load(f"{ck}/logits_eval.npz")
        parts = split_rows(np.arange(len(tr["parcel_ids"])), REGIONS[:3], sizes) | {"frh04": None}
        for r, rows in parts.items():
            z = ev if rows is None else {k: tr[k][rows] for k in ("parcel_ids", "labels", "features")} | {"columns": tr["columns"]}
            np.savez(d / f"logits_{r}.npz", parcel_ids=z["parcel_ids"], labels=z["labels"],
                     features=z["features"], columns=z["columns"])
    print("prep done", flush=True)


# --------------------------------------------------------------------------- #
_CACHE = {}


def region_blocks(test: str):
    """Per region: labels and the feature blocks (metadata, pheno, l1c, l2a) for this fold."""
    if "base" not in _CACHE:
        sizes = region_sizes()
        meta_cols = list(full_feature_blocks()["metadata"])
        xtr, ytr = load_feature_matrix(Path("artifacts/cached_logits_train.pt"), use_metadata=True)
        xev, yev = load_feature_matrix(Path("artifacts/cached_logits_eval.pt"), use_metadata=True)
        ph_tr, ph_ev = np.load("artifacts/phenology_train.npz")["features"], np.load("artifacts/phenology_eval.npz")["features"]
        base = {}
        for r, rows in split_rows(np.arange(len(ytr)), REGIONS[:3], sizes).items():
            base[r] = {"y": ytr[rows], "metadata": xtr[rows][:, meta_cols], "pheno": ph_tr[rows]}
        base["frh04"] = {"y": yev, "metadata": xev[:, meta_cols], "pheno": ph_ev}
        for r in REGIONS:
            ids = np.load(f"artifacts/series_store/{r}.npz")["parcel_ids"]
            assert np.array_equal(torch.load(args_global.work_dir / f"ids_{r}.pt", weights_only=True)["parcel_ids"].numpy(), ids)
        _CACHE["base"] = base
    out = {}
    for r in REGIONS:
        blk = dict(_CACHE["base"][r])
        for kind in ("l1c", "l2a"):
            z = np.load(args_global.work_dir / "nets" / f"{kind}_test{test}" / f"logits_{r}.npz")
            assert np.array_equal(z["labels"], blk["y"])
            blk[kind] = z["features"].astype(np.float32)
        out[r] = blk
    return out


def stack(blocks, regions, groups):
    x = np.vstack([np.hstack([blocks[r][g] for g in groups]) for r in regions]).astype(np.float32)
    y = np.concatenate([blocks[r]["y"] for r in regions]).astype(np.int64)
    return x, y


def stage_meta(args) -> None:
    train = [r for r in REGIONS if r != args.test]
    blocks = region_blocks(args.test)
    x_tr, y_tr = stack(blocks, train, ARMS[args.arm])
    x_ev, y_ev = stack(blocks, [args.test], ARMS[args.arm])
    d = args.work_dir / args.test / f"proba_{args.arm}{sfx(args.seed)}"
    d.mkdir(parents=True, exist_ok=True)
    for m in args.multipliers:
        out = d / f"proba_pair_compact_m{m:g}.npz"
        if out.exists():
            continue
        t0 = time.time()
        p_xgb, p_cat, _ = fit_predict(x_tr, y_tr, x_ev, m=m, seed=args.seed)
        np.savez(out, p_xgb=p_xgb.astype(np.float32), p_cat=p_cat.astype(np.float32),
                 variant=json.dumps({"test": args.test, "arm": args.arm, "seed": args.seed, "m": m}))
        print(f"meta test={args.test} arm={args.arm} seed={args.seed} m={m:g}: feats {x_tr.shape[1]}, "
              f"XGB acc {100 * (p_xgb.argmax(1) == y_ev).mean():.2f}% ({time.time() - t0:.0f}s)", flush=True)


def stage_oof(args) -> None:
    out = args.work_dir / args.test / f"oof_B{sfx(args.seed)}.npz"
    if out.exists():
        return
    train = [r for r in REGIONS if r != args.test]
    blocks = region_blocks(args.test)
    probs = {}
    for q in train:
        rest = [r for r in train if r != q]
        x_tr, y_tr = stack(blocks, rest, ARMS["B"])
        x_q, _ = stack(blocks, [q], ARMS["B"])
        probs[q] = fit_xgb_scores(x_tr, y_tr, x_q, seed=args.seed)
        print(f"oof test={args.test} seed={args.seed} held={q}", flush=True)
    np.savez(out, **probs)


def stage_verify(args) -> None:
    t0 = time.time()
    test, train = args.test, [r for r in REGIONS if r != args.test]
    blocks = region_blocks(test)
    oof = np.load(args.work_dir / test / f"oof_B{sfx(args.seed)}.npz")
    p_ev = np.load(args.work_dir / test / f"proba_B{sfx(args.seed)}" / "proba_pair_compact_m1.npz")["p_xgb"]
    feats = {}
    for r in REGIONS:
        store = np.load(f"artifacts/series_store/{r}.npz")
        geo = geometry(r, store["parcel_ids"])
        proba = p_ev if r == test else oof[r]
        feats[r] = np.hstack([proba, blocks[r]["pheno"], monthly_means(store), geo[:, 2:].astype(np.float32),
                              neighbour_context(geo[:, :2], proba)])
    y_tr = np.concatenate([blocks[r]["y"] for r in train])
    p_tr = np.vstack([oof[r] for r in train])
    x_tr = np.vstack([feats[r] for r in train])
    y_ev = blocks[test]["y"]
    # keeps 90% of out-of-fold training orchards (verifier.candidate_threshold; literal 0.10 reproduces the logs)
    tau = float(np.quantile(p_tr[y_tr == ORCHARD, ORCHARD], 0.10))
    cand_tr, cand_ev = p_tr[:, ORCHARD] >= tau, p_ev[:, ORCHARD] >= tau
    score, _ = verifier_scores(x_tr, y_tr, cand_tr, feats[test], p_ev, cand_ev, args.seed)
    scores = {"meta": p_ev[:, ORCHARD].astype(np.float64), "verifier": score}

    folds = np.load(args.work_dir / f"folds_{test}.npz")
    y_bin = (y_ev == ORCHARD).astype(np.int8)
    boots = block_resample_indices(folds["blocks"], 1000)
    d = np.array([ap(y_bin[i], scores["verifier"][i]) - ap(y_bin[i], scores["meta"][i]) for i in boots])
    lo, hi = percentile_ci(d)
    rep = {"test": test, "seed": args.seed, "tau": tau, "candidates": int(cand_ev.sum()),
           "candidate_orchards": int(y_bin[cand_ev].sum()), "orchards": int(y_bin.sum()),
           "ap_diff_ci": [lo, hi]}
    rules = {"acc>=81": lambda r: select_orchard_first(r, min_accuracy=81.0), "accuracy-first": select_accuracy_first}
    for name, s in scores.items():
        row = {"ap": ap(y_bin, s), "p60": precision_at_recall(y_bin, s, 0.60), "p75": precision_at_recall(y_bin, s, 0.75)}
        preds = decode(p_ev, s, threshold_grid(s))
        for rname, rule in rules.items():
            cf, _ = cross_fit(preds, y_ev, folds["folds"], rule)
            m = classification_metrics(y_ev, cf)
            row[rname] = {k: m[k] for k in ("accuracy", "orchard_recall", "orchard_precision", "orchard_f1")}
        rep[name] = row
    (args.work_dir / test / f"verify{sfx(args.seed)}.json").write_text(json.dumps(rep, indent=2))
    print(f"verify test={test} seed={args.seed}: AP meta {rep['meta']['ap']:.2f} -> verifier {rep['verifier']['ap']:.2f} "
          f"[{lo:+.2f},{hi:+.2f}]; acc-first P {rep['meta']['accuracy-first']['orchard_precision']:.1f} -> "
          f"{rep['verifier']['accuracy-first']['orchard_precision']:.1f} ({time.time() - t0:.0f}s)", flush=True)


args_global = None


def main() -> None:
    global args_global
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("prep", "meta", "oof", "verify"))
    p.add_argument("--test", choices=REGIONS)
    p.add_argument("--arm", choices=tuple(ARMS), default="B")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--multipliers", default="1,2,3")
    p.add_argument("--work-dir", type=Path, default=Path("artifacts/rotation"))
    args = p.parse_args()
    args.multipliers = [float(v) for v in args.multipliers.split(",") if v.strip()]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args_global = args
    if args.stage != "prep" and not args.test:
        p.error("--test is required")
    {"prep": stage_prep, "meta": stage_meta, "oof": stage_oof, "verify": stage_verify}[args.stage](args)


if __name__ == "__main__":
    main()

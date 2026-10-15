"""Orchard verifier pilot: a second stage that re-checks the meta-classifier's orchard candidates.

Most false orchards are meadows. The verifier is a binary orchard-vs-rest model
trained only on the hard cases (parcels the meta-classifier scores as possible
orchards), with information the meta-classifier lacks: monthly clear-date band /
index means over the full series, parcel geometry, and what the neighbouring
parcels are predicted to be.

Stages (each fits in a short job; outputs cached under ``--work-dir``):

    oof --region R   meta-classifier XGBoost (m=1) trained on the other two
                     training regions, scoring region R (out-of-fold, so
                     training candidates look like test candidates)
    features         monthly means, geometry and neighbour context, train + frh04
    verify           candidate threshold, verifier per feature-group arm,
                     frh04 evaluation against thresholding the meta score alone

Test-side meta scores are the existing ``proba_pair_compact_m1.npz`` XGBoost
probabilities of the meta-classifier trained on all of frh01-frh03.

Example::

    python scripts/orchard_verifier.py oof --region frh01
    python scripts/orchard_verifier.py features
    python scripts/orchard_verifier.py verify
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_meta_variants import ensure_all_classes, load_features  # noqa: E402

from agricultureai.evaluation.selection_protocol import (  # noqa: E402
    classification_metrics,
    cross_fit,
    percentile_ci,
    select_accuracy_first,
    select_orchard_first,
)
from agricultureai.meta_features import build_xgb, fit_xgb, orchard_sample_weights, predict_proba_xgb  # noqa: E402

TRAIN_REGIONS = ("frh01", "frh02", "frh03")
ORCHARD, MEADOWS = 5, (7, 8)
BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
B4, B5, B8, B11 = 2, 3, 6, 8
METRIC_CRS = "EPSG:2154"
K_NEIGHBOURS = 10


def extras(args):
    ck = args.net_dir or f"artifacts/checkpoints/msresnet_l2a_clear{args.seed_suffix}"
    return [("pheno", (Path("artifacts/phenology_train.npz"), Path("artifacts/phenology_eval.npz"))),
            ("msclear", (Path(f"{ck}/logits_train.npz"), Path(f"{ck}/logits_eval.npz")))]


def region_rows(n_per_region):
    edges = np.concatenate([[0], np.cumsum(n_per_region)])
    return {r: np.arange(edges[i], edges[i + 1]) for i, r in enumerate(TRAIN_REGIONS)}


# --------------------------------------------------------------------------- #
# Stage 1: out-of-fold meta scores on the training regions                    #
# --------------------------------------------------------------------------- #
def stage_oof(args) -> None:
    out = args.work_dir / f"oof_{args.region}_seed{args.seed}.npz"
    if out.exists():
        raise SystemExit(f"{out} exists")
    x, y, _ = load_features(Path("artifacts/cached_logits_train.pt"), [(n, p[0]) for n, p in extras(args)], args.drop_blocks)
    sizes = [len(np.load(f"artifacts/series_store/{r}.npz")["parcel_ids"]) for r in TRAIN_REGIONS]
    rows = region_rows(sizes)
    held = rows[args.region]
    train = np.setdiff1d(np.arange(len(y)), held)
    t0 = time.time()
    w = orchard_sample_weights(y[train], 1.0)
    xt, yt, w = ensure_all_classes(x[train], y[train], w, np.random.default_rng(args.seed))
    xgb = build_xgb(args.seed)
    fit_xgb(xgb, xt, yt, w)
    p = predict_proba_xgb(xgb, x[held]).astype(np.float32)
    np.savez(out, rows=held, proba=p)
    rec = float((p[y[held] == ORCHARD].argmax(1) == ORCHARD).mean())
    print(f"oof {args.region}: {len(held):,} rows, argmax acc {100 * (p.argmax(1) == y[held]).mean():.2f}%, "
          f"orchard recall {100 * rec:.1f}% ({time.time() - t0:.0f}s) -> {out}", flush=True)


# --------------------------------------------------------------------------- #
# Stage 2: features the meta-classifier does not have                         #
# --------------------------------------------------------------------------- #
def monthly_means(store: np.lib.npyio.NpzFile) -> np.ndarray:
    """Per parcel x month (12): mean of the 10 bands + NDVI, NDRE, NDMI over clear dates (NaN if none)."""
    off, spec, doy, clear = store["offsets"], store["spectral"].astype(np.float64), store["doy"], store["clear"]
    n = len(off) - 1
    parcel = np.repeat(np.arange(n), np.diff(off))
    month = np.clip(((doy.astype(np.int64) - 1) * 12) // 366, 0, 11)
    eps = 1e-6
    ndvi = (spec[:, B8] - spec[:, B4]) / (spec[:, B8] + spec[:, B4] + eps)
    ndre = (spec[:, B8] - spec[:, B5]) / (spec[:, B8] + spec[:, B5] + eps)
    ndmi = (spec[:, B8] - spec[:, B11]) / (spec[:, B8] + spec[:, B11] + eps)
    vals = np.column_stack([spec, ndvi, ndre, ndmi])[clear]
    key = (parcel * 12 + month)[clear]
    sums = np.zeros((n * 12, vals.shape[1]))
    np.add.at(sums, key, vals)
    cnt = np.bincount(key, minlength=n * 12).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / cnt[:, None]
    mean[cnt == 0] = np.nan
    return mean.reshape(n, 12 * vals.shape[1]).astype(np.float32)


def monthly_names():
    chans = list(BANDS) + ["NDVI", "NDRE", "NDMI"]
    return [f"m{m + 1:02d}_{c}" for m in range(12) for c in chans]


def geometry(region: str, parcel_ids: np.ndarray) -> np.ndarray:
    """x, y (metres), area, perimeter, compactness, elongation per parcel id (NaN if no geometry)."""
    import breizhcrops as bzh

    geo = bzh.BreizhCrops(region, level="L2A", load_timeseries=False).geodataframe()
    geo = geo.loc[~geo.geometry.isna()].set_index("id")
    geo = geo.loc[~geo.index.duplicated()].to_crs(METRIC_CRS)
    g = geo.geometry
    cent = g.centroid
    rect = g.minimum_rotated_rectangle()
    coords = [np.asarray(r.exterior.coords) if r.geom_type == "Polygon" else None for r in rect]
    sides = np.array([[np.hypot(*(c[1] - c[0])), np.hypot(*(c[2] - c[1]))] if c is not None else [np.nan, np.nan]
                      for c in coords])
    with np.errstate(invalid="ignore", divide="ignore"):
        elong = sides.max(1) / sides.min(1)
    area, perim = g.area.to_numpy(), g.length.to_numpy()
    table = np.column_stack([cent.x, cent.y, area, perim, 4 * np.pi * area / perim ** 2, elong])
    lookup = dict(zip(geo.index.astype(np.int64), range(len(geo))))
    out = np.full((len(parcel_ids), table.shape[1]), np.nan, np.float64)
    hit = np.array([lookup.get(int(i), -1) for i in parcel_ids])
    out[hit >= 0] = table[hit[hit >= 0]]
    return out


def neighbour_context(xy: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """From the K nearest other parcels: mean / max orchard score, mean meadow score, mean distance."""
    from scipy.spatial import cKDTree

    out = np.full((len(xy), 4), np.nan, np.float32)
    ok = np.isfinite(xy).all(1)
    tree = cKDTree(xy[ok])
    dist, idx = tree.query(xy[ok], k=K_NEIGHBOURS + 1)
    dist, idx = dist[:, 1:], idx[:, 1:]  # drop self
    p_ok = proba[ok]
    orch = p_ok[idx, ORCHARD]
    mead = p_ok[idx][:, :, list(MEADOWS)].sum(-1)
    out[ok] = np.column_stack([orch.mean(1), orch.max(1), mead.mean(1), dist.mean(1)])
    return out


def stage_features(args) -> None:
    out = args.work_dir / f"features_seed{args.seed}.npz"
    if out.exists():
        raise SystemExit(f"{out} exists")
    t0 = time.time()
    tr_ids = torch.load("artifacts/cached_logits_train.pt", map_location="cpu", weights_only=True)["parcel_ids"].numpy()
    ev_ids = torch.load("artifacts/cached_logits_eval.pt", map_location="cpu", weights_only=True)["parcel_ids"].numpy()
    # meta scores: out-of-fold on train, the production model on frh04
    p_tr = np.zeros((len(tr_ids), 9), np.float32)
    for r in TRAIN_REGIONS:
        z = np.load(args.work_dir / f"oof_{r}_seed{args.seed}.npz")
        p_tr[z["rows"]] = z["proba"]
    p_ev = np.load(f"artifacts/pruned_proba_cache_{args.base_tag}{args.seed_suffix}/proba_pair_compact_m1.npz")["p_xgb"]
    blocks = {}
    for split, regions, ids, proba in (("train", TRAIN_REGIONS, tr_ids, p_tr), ("eval", ("frh04",), ev_ids, p_ev)):
        stores = [np.load(f"artifacts/series_store/{r}.npz") for r in regions]
        assert np.array_equal(np.concatenate([s["parcel_ids"] for s in stores]), ids)
        monthly = np.vstack([monthly_means(s) for s in stores])
        geo = np.vstack([geometry(r, s["parcel_ids"]) for r, s in zip(regions, stores)])
        neigh = np.vstack([neighbour_context(geo[rows, :2], proba[rows])
                           for rows in (region_rows([len(s["parcel_ids"]) for s in stores]).values()
                                        if split == "train" else [np.arange(len(ids))])])
        blocks[split] = dict(meta=proba, monthly=monthly, geometry=geo[:, 2:].astype(np.float32), neighbours=neigh)
        print(f"{split}: monthly {monthly.shape}, geometry missing {int(np.isnan(geo[:, 0]).sum())}, "
              f"({time.time() - t0:.0f}s)", flush=True)
    np.savez(out, **{f"{s}_{k}": v for s, d in blocks.items() for k, v in d.items()},
             monthly_names=np.array(monthly_names()))
    print(f"-> {out} ({time.time() - t0:.0f}s)", flush=True)


# --------------------------------------------------------------------------- #
# Stage 3: verifier + evaluation                                              #
# --------------------------------------------------------------------------- #
ARMS = {
    "meta": ("meta",),
    "+pheno": ("meta", "pheno"),
    "+monthly": ("meta", "pheno", "monthly"),
    "+geometry": ("meta", "pheno", "geometry"),
    "+neighbours": ("meta", "pheno", "neighbours"),
    "all": ("meta", "pheno", "monthly", "geometry", "neighbours"),
}


def build_verifier(seed: int):
    import xgboost as xgb

    return xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=2, tree_method="hist",
                             eval_metric="aucpr", random_state=seed, n_jobs=8)


def decode(p_meta: np.ndarray, score: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """One prediction row per threshold: orchard where score >= t, else the meta argmax without orchards."""
    rest = p_meta.copy()
    rest[:, ORCHARD] = -1
    base = rest.argmax(1).astype(np.int8)
    return np.stack([np.where(score >= t, ORCHARD, base).astype(np.int8) for t in thresholds])


def precision_at_recall(y_bin: np.ndarray, score: np.ndarray, recall: float) -> float:
    order = np.argsort(-score, kind="stable")
    tp = np.cumsum(y_bin[order])
    k = int(np.searchsorted(tp, np.ceil(recall * y_bin.sum())))
    return 100.0 * tp[k] / (k + 1)


def ap(y_bin: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return 100.0 * average_precision_score(y_bin, score)


def stage_verify(args) -> None:
    f = np.load(args.work_dir / f"features_seed{args.seed}.npz")
    tr = torch.load("artifacts/cached_logits_train.pt", map_location="cpu", weights_only=True)
    ev = torch.load("artifacts/cached_logits_eval.pt", map_location="cpu", weights_only=True)
    y_tr, y_ev = tr["labels"].numpy(), ev["labels"].numpy()
    ph_tr, ph_ev = np.load("artifacts/phenology_train.npz")["features"], np.load("artifacts/phenology_eval.npz")["features"]
    data = {s: {"meta": f[f"{s}_meta"], "pheno": ph, "monthly": f[f"{s}_monthly"],
                "geometry": f[f"{s}_geometry"], "neighbours": f[f"{s}_neighbours"]}
            for s, ph in (("train", ph_tr), ("eval", ph_ev))}
    p_tr, p_ev = data["train"]["meta"], data["eval"]["meta"]

    # candidates: the orchard-score threshold keeping 90% of out-of-fold training orchards
    tau = float(np.quantile(p_tr[y_tr == ORCHARD, ORCHARD], 1 - args.candidate_recall))
    cand_tr, cand_ev = p_tr[:, ORCHARD] >= tau, p_ev[:, ORCHARD] >= tau
    print(f"candidate threshold {tau:.4f}: train {cand_tr.sum():,} candidates "
          f"({(y_tr[cand_tr] == ORCHARD).sum()} orchards), frh04 {cand_ev.sum():,} "
          f"({(y_ev[cand_ev] == ORCHARD).sum()} of {(y_ev == ORCHARD).sum()} orchards, "
          f"{100 * (y_ev[cand_ev] == ORCHARD).mean():.1f}% precision; "
          f"meadows {np.isin(y_ev[cand_ev], MEADOWS).sum():,})", flush=True)

    folds = np.load("artifacts/frh04_spatial_folds.npz")
    assert np.array_equal(folds["parcel_ids"], ev["parcel_ids"].numpy())
    y_bin = (y_ev == ORCHARD).astype(np.int8)
    scores = {"meta score (no verifier)": p_ev[:, ORCHARD].astype(np.float64)}
    gains = {}
    for arm, groups in ARMS.items():
        t0 = time.time()
        x_tr = np.hstack([data["train"][g] for g in groups])[cand_tr]
        x_ev = np.hstack([data["eval"][g] for g in groups])[cand_ev]
        model = build_verifier(args.seed)
        model.fit(x_tr, (y_tr[cand_tr] == ORCHARD).astype(np.int8))
        s = p_ev[:, ORCHARD].astype(np.float64).copy()  # non-candidates keep the meta score (all below tau)
        s[cand_ev] = 1.0 + model.predict_proba(x_ev)[:, 1]
        scores[f"verifier: {arm}"] = s
        imp = model.get_booster().get_score(importance_type="gain")
        gains[arm] = sum(imp.values())
        print(f"  trained {arm:12s} on {x_tr.shape} ({time.time() - t0:.0f}s)", flush=True)

    # threshold-free: AP and precision at fixed recall, paired block bootstrap for AP
    _, gidx = np.unique(folds["blocks"], return_inverse=True)
    members = np.split(np.argsort(gidx, kind="stable"), np.cumsum(np.bincount(gidx))[:-1])
    rng = np.random.default_rng(0)
    boots = [np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
             for _ in range(args.n_boot)]
    base = scores["meta score (no verifier)"]
    report = {"tau": tau, "arms": {}}
    print(f"\nfrh04 orchard ranking (seed {args.seed}; AP diff vs meta score, {args.n_boot} block resamples):")
    print(f"{'arm':28s} {'AP':>6s} {'P@60%R':>7s} {'P@75%R':>7s}  AP diff [95% CI]")
    for name, s in scores.items():
        row = {"ap": ap(y_bin, s), "p60": precision_at_recall(y_bin, s, 0.60), "p75": precision_at_recall(y_bin, s, 0.75)}
        if s is not base:
            d = np.array([ap(y_bin[i], s[i]) - ap(y_bin[i], base[i]) for i in boots])
            lo, hi = percentile_ci(d)
            row |= {"ap_diff": row["ap"] - ap(y_bin, base), "ci": [lo, hi]}
            tail = f"{row['ap_diff']:+.2f} [{lo:+.2f},{hi:+.2f}]"
        else:
            tail = "-"
        report["arms"][name] = row
        print(f"{name:28s} {row['ap']:6.2f} {row['p60']:7.2f} {row['p75']:7.2f}  {tail}", flush=True)

    # an operating point, cross-fitted over the spatial folds, same decoding for every arm
    print("\nCross-fitted operating points (orchard where score >= t, else the meta argmax without orchards):")
    rules = {"acc>=81 max orchard": lambda r: select_orchard_first(r, min_accuracy=81.0),
             "accuracy-first": select_accuracy_first}
    for name, s in scores.items():
        qs = np.unique(np.quantile(s, np.linspace(0.90, 0.9995, 160)))
        preds = decode(p_ev, s, qs)
        for rname, rule in rules.items():
            cf, _ = cross_fit(preds, y_ev, folds["folds"], rule)
            m = classification_metrics(y_ev, cf)
            report["arms"][name][rname] = {k: m[k] for k in ("accuracy", "orchard_recall", "orchard_precision", "orchard_f1")}
            print(f"  {name:28s} {rname:20s} acc {m['accuracy']:.2f}  orchard R {m['orchard_recall']:.1f} "
                  f"P {m['orchard_precision']:.1f} F1 {m['orchard_f1']:.1f}", flush=True)
    (args.work_dir / f"verify_seed{args.seed}.json").write_text(json.dumps(report, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("oof", "features", "verify"))
    p.add_argument("--region", choices=TRAIN_REGIONS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--work-dir", type=Path, default=Path("artifacts/orchard_verifier"))
    p.add_argument("--candidate-recall", type=float, default=0.90)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--base-tag", default="phenomsclear", help="Meta-classifier arm whose m=1 XGBoost scores frh04.")
    p.add_argument("--drop-blocks", default="", help="Compact blocks the base arm drops (for the out-of-fold meta).")
    p.add_argument("--net-dir", default="", help="L2A network logits dir (default: msresnet_l2a_clear per seed).")
    args = p.parse_args()
    args.seed_suffix = "" if args.seed == 42 else f"_seed{args.seed}"
    args.drop_blocks = tuple(b for b in args.drop_blocks.split(",") if b)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "oof" and not args.region:
        p.error("oof needs --region")
    {"oof": stage_oof, "features": stage_features, "verify": stage_verify}[args.stage](args)


if __name__ == "__main__":
    main()

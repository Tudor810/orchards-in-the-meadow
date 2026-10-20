"""Second-stage orchard verifier: features, candidate selection, model, decoding.

Most false orchards are meadows. The verifier is a binary orchard-vs-rest XGBoost trained only on the
meta-classifier's orchard candidates (parcels whose meta orchard score passes a threshold that keeps 90% of
out-of-fold training orchards), with information the meta-classifier lacks: monthly clear-date band / index
means, parcel geometry and the neighbouring parcels' predictions. A candidate's score is 1 + the verifier's
probability, so candidates rank above every non-candidate, whose score stays the meta orchard probability.

Used by ``scripts/meta/orchard_verifier.py`` (frh04), ``scripts/rotation/`` (region rotation),
``scripts/meta/compute_model_metrics.py`` and ``paper/figure2.py``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from agricultureai.evaluation.selection_protocol import cross_fit

ORCHARD, MEADOWS = 5, (7, 8)
BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
B4, B5, B8, B11 = 2, 3, 6, 8
METRIC_CRS = "EPSG:2154"  # Lambert-93, metres
K_NEIGHBOURS = 10
CANDIDATE_RECALL = 0.90


# --------------------------------------------------------------------------- #
# Features the meta-classifier does not have                                  #
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


def monthly_names() -> list:
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


# --------------------------------------------------------------------------- #
# Candidates, model, scores                                                   #
# --------------------------------------------------------------------------- #
def candidate_threshold(p_meta_train: np.ndarray, y_train: np.ndarray, recall: float = CANDIDATE_RECALL) -> float:
    """Meta orchard-score threshold that keeps ``recall`` of the (out-of-fold) training orchards."""
    return float(np.quantile(p_meta_train[y_train == ORCHARD, ORCHARD], 1 - recall))


def build_verifier(seed: int):
    import xgboost as xgb

    return xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=2, tree_method="hist",
                             eval_metric="aucpr", random_state=seed, n_jobs=8)


def verifier_scores(x_train: np.ndarray, y_train: np.ndarray, cand_train: np.ndarray, x_eval: np.ndarray,
                    p_meta_eval: np.ndarray, cand_eval: np.ndarray, seed: int):
    """Fit on the training candidates; return (scores, model): 1 + verifier probability on evaluation candidates,
    the meta orchard probability elsewhere (all below the candidate threshold)."""
    model = build_verifier(seed)
    model.fit(x_train[cand_train], (y_train[cand_train] == ORCHARD).astype(np.int8))
    score = p_meta_eval[:, ORCHARD].astype(np.float64).copy()
    score[cand_eval] = 1.0 + model.predict_proba(x_eval[cand_eval])[:, 1]
    return score, model


# --------------------------------------------------------------------------- #
# Decoding and metrics                                                        #
# --------------------------------------------------------------------------- #
def decode(p_meta: np.ndarray, score: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """One prediction row per threshold: orchard where score >= t, else the meta argmax without orchards."""
    rest = p_meta.copy()
    rest[:, ORCHARD] = -1
    base = rest.argmax(1).astype(np.int8)
    return np.stack([np.where(score >= t, ORCHARD, base).astype(np.int8) for t in thresholds])


def threshold_grid(score: np.ndarray) -> np.ndarray:
    """Candidate decision thresholds: 160 quantiles of the score between its 90th and 99.95th percentile."""
    return np.unique(np.quantile(score, np.linspace(0.90, 0.9995, 160)))


def cross_fitted_predictions(p_meta: np.ndarray, score: np.ndarray, y: np.ndarray, folds: np.ndarray,
                             rule: Callable) -> np.ndarray:
    """Labels at the threshold ``rule`` selects on the other spatial folds (decode as in :func:`decode`)."""
    cf, _ = cross_fit(decode(p_meta, score, threshold_grid(score)), y, folds, rule)
    return cf


def precision_at_recall(y_bin: np.ndarray, score: np.ndarray, recall: float) -> float:
    order = np.argsort(-score, kind="stable")
    tp = np.cumsum(y_bin[order])
    k = int(np.searchsorted(tp, np.ceil(recall * y_bin.sum())))
    return 100.0 * tp[k] / (k + 1)


def ap(y_bin: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return 100.0 * average_precision_score(y_bin, score)

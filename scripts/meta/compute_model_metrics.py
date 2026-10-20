"""Confusion matrices and micro / macro precision, recall and F1 on frh04.

Systems (``--models``, default ``baseline,meta,final``):

    baseline  pretrained BreizhCrops MSResNet (L1C input; the eval cache's
              8-view TTA logits) -- the reference model of the paper
    meta      frozen meta-classifier (arm ``noexpens``: no expert blocks,
              phenology, 3-seed L2A network), accuracy-first decode
    final     meta + orchard verifier (FROZEN_DESIGN.md step 4), accuracy-first
              orchard threshold
    cascade   four-tier DMOE cascade decoded from the cached branch outputs, identity
              soft routing (intermediate option; the live evaluate_dmoe_pipeline.py run
              with 8-view inputs gives 79.65% / 21.74% orchard recall, 79.66% / 21.20% here)
    l2a_net   3-seed-averaged MSResNet trained on L2A, alone (intermediate option)

``meta`` and ``final`` are chosen out of sample as in ``scripts/meta/evaluate_meta_protocol.py``:
the accuracy-first rule (max accuracy with orchard recall >= 35%) picks the decode
settings / verifier threshold on four of five 10 km spatial folds and applies them
to the fifth. ``final`` retrains the ``all`` arm of ``orchard_verifier.py verify``
per seed (same features, hyper-parameters and seed) from its cached features.

All systems are scored on the same parcels: frh04 parcels without an L1C
counterpart (no baseline prediction) are left out of every system, so the
matrices are paired. For single-label classification micro precision = micro
recall = micro F1 = accuracy; macro averages are unweighted means over the nine
classes (macro F1 = mean of per-class F1), so the two tiny classes (sunflower 2,
nuts 11 parcels) weigh as much as corn.

Example::

    python scripts/meta/compute_model_metrics.py
    python scripts/meta/compute_model_metrics.py --models baseline,cascade,final --seeds 42
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from agricultureai.evaluation.meta_protocol import CLASS_NAMES, config_grid
from agricultureai.evaluation.selection_protocol import (
    confusion,
    cross_fit,
    per_class_from_confusion,
    select_accuracy_first,
)
from agricultureai.pipeline import GLOBAL_ORCHARDS, MEADOW_GLOBAL_IDS, NUM_GLOBAL_CLASSES, decode_predictions
from agricultureai.verifier import CANDIDATE_RECALL, candidate_threshold, cross_fitted_predictions, verifier_scores

DEFAULT_MODELS = "baseline,meta,final"
MODEL_LABELS = {
    "baseline": "Pretrained MSResNet (L1C, TTA-8)",
    "cascade": "Four-tier DMOE cascade",
    "l2a_net": "MSResNet trained on L2A (3-seed mean)",
    "meta": "Meta-classifier, accuracy-first",
    "final": "Meta + orchard verifier, accuracy-first",
}
SEEDED = {"meta", "final"}
# Decode grid of evaluate_meta_protocol.py (its defaults), compact layout.
META_GRID = Namespace(
    layout="compact",
    multipliers="1,2,3",
    blends="geo_mean,arith_xgb0.7",
    orchard_biases="0,0.5,1,1.5,2,2.5,3",
    meadow_biases="0,0.25,0.5,0.75,1,1.5,2",
)
VERIFIER_GROUPS = ("meta", "pheno", "monthly", "geometry", "neighbours")  # orchard_verifier.py arm "all"


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def summarize(cm: np.ndarray) -> Dict[str, float]:
    """Micro and macro precision / recall / F1 (percent) from a confusion matrix."""
    pc = per_class_from_confusion(cm)
    total, correct = float(cm.sum()), float(np.trace(cm))
    micro = 100.0 * correct / max(total, 1.0)
    return {
        "accuracy": micro,
        "micro_precision": micro,
        "micro_recall": micro,
        "micro_f1": micro,
        "macro_precision": 100.0 * float(pc["precision"].mean()),
        "macro_recall": 100.0 * float(pc["recall"].mean()),
        "macro_f1": 100.0 * float(pc["f1"].mean()),
        "orchard_precision": 100.0 * float(pc["precision"][GLOBAL_ORCHARDS]),
        "orchard_recall": 100.0 * float(pc["recall"][GLOBAL_ORCHARDS]),
        "orchard_f1": 100.0 * float(pc["f1"][GLOBAL_ORCHARDS]),
        **orchard_error_shares(cm),
    }


def orchard_error_shares(cm: np.ndarray) -> Dict[str, float]:
    """Share of false orchards that are meadows, and of missed orchards predicted as a meadow class."""
    meadows, o = list(MEADOW_GLOBAL_IDS), GLOBAL_ORCHARDS
    false_orch, missed = cm[:, o].sum() - cm[o, o], cm[o, :].sum() - cm[o, o]
    return {
        "false_orchards_meadow_pct": 100.0 * float(cm[meadows, o].sum()) / max(float(false_orch), 1.0),
        "missed_orchards_meadow_pct": 100.0 * float(cm[o, meadows].sum()) / max(float(missed), 1.0),
    }


# --------------------------------------------------------------------------- #
# Predictions (one label per frh04 cache row)                                  #
# --------------------------------------------------------------------------- #
class Inputs:
    """Lazily loaded frh04 inputs shared by the predictors."""

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.eval = torch.load(args.eval_cache, map_location="cpu", weights_only=True)
        self.y = self.eval["labels"].numpy().astype(np.int64)
        self.parcel_ids = self.eval["parcel_ids"].numpy().astype(np.int64)
        folds = np.load(args.folds_file)
        if not np.array_equal(folds["parcel_ids"], self.parcel_ids):
            raise RuntimeError(f"{args.folds_file} rows do not match {args.eval_cache}")
        self.folds = folds["folds"]
        self._y_train: Optional[np.ndarray] = None

    @property
    def y_train(self) -> np.ndarray:
        if self._y_train is None:
            train = torch.load(self.args.train_cache, map_location="cpu", weights_only=True)
            self._y_train = train["labels"].numpy().astype(np.int64)
        return self._y_train


def predict_baseline(inp: Inputs, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    logits = inp.eval["msresnet_logits"].numpy()
    valid = inp.eval["msresnet_l1c_matched"].numpy().astype(bool)
    return logits.argmax(1), valid


def predict_cascade(inp: Inputs, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    e = inp.eval
    pred = decode_predictions(
        e["l1_logits"], e["l2a_logits"], e["l2b_logits"], e["l2c_logits"], e["l3_distances"],
        t1=1.0, t2b=1.0, t2c=1.0, t3=1.0, tau_route=0.05, b_meadow=0.0, b_tree=0.0,
    ).numpy()
    return pred, np.ones_like(pred, dtype=bool)


def predict_l2a_net(inp: Inputs, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    net = np.load(inp.args.l2a_net_logits)
    if not np.array_equal(net["parcel_ids"], inp.parcel_ids):
        raise RuntimeError(f"{inp.args.l2a_net_logits} rows do not match the eval cache")
    pred = net["features"].argmax(1)
    return pred, np.ones_like(pred, dtype=bool)


def seed_dir(base: Path, seed: int) -> Path:
    return base if seed == 42 else base.with_name(f"{base.name}_seed{seed}")


def predict_meta(inp: Inputs, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    _, preds, _ = config_grid(seed_dir(inp.args.meta_proba_dir, seed), META_GRID)
    pred, _ = cross_fit(preds, inp.y, inp.folds, select_accuracy_first)
    return pred.astype(np.int64), np.ones_like(inp.y, dtype=bool)


def predict_final(inp: Inputs, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    feats = np.load(inp.args.verifier_dir / f"features_seed{seed}.npz")
    pheno = {s: np.load(p)["features"] for s, p in (("train", inp.args.pheno_train), ("eval", inp.args.pheno_eval))}
    data = {s: {"meta": feats[f"{s}_meta"], "pheno": pheno[s], "monthly": feats[f"{s}_monthly"],
                "geometry": feats[f"{s}_geometry"], "neighbours": feats[f"{s}_neighbours"]}
            for s in ("train", "eval")}
    y_tr, y_ev = inp.y_train, inp.y
    p_tr, p_ev = data["train"]["meta"], data["eval"]["meta"]
    tau = candidate_threshold(p_tr, y_tr, CANDIDATE_RECALL)
    cand_tr, cand_ev = p_tr[:, GLOBAL_ORCHARDS] >= tau, p_ev[:, GLOBAL_ORCHARDS] >= tau
    x_tr, x_ev = (np.hstack([data[s][g] for g in VERIFIER_GROUPS]) for s in ("train", "eval"))
    score, _ = verifier_scores(x_tr, y_tr, cand_tr, x_ev, p_ev, cand_ev, seed)
    pred = cross_fitted_predictions(p_ev, score, y_ev, inp.folds, select_accuracy_first)
    return pred.astype(np.int64), np.ones_like(y_ev, dtype=bool)


PREDICTORS: Dict[str, Callable[[Inputs, int], Tuple[np.ndarray, np.ndarray]]] = {
    "baseline": predict_baseline,
    "cascade": predict_cascade,
    "l2a_net": predict_l2a_net,
    "meta": predict_meta,
    "final": predict_final,
}


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def format_confusion(cm: np.ndarray) -> List[str]:
    short = [n[:6] for n in CLASS_NAMES]
    lines = ["  true \\ pred  " + "".join(f"{n:>8s}" for n in short)]
    for c in range(NUM_GLOBAL_CLASSES):
        lines.append(f"  {CLASS_NAMES[c]:12s}" + "".join(f"{int(v):8d}" for v in cm[c]))
    return lines


def format_per_class(cm: np.ndarray) -> List[str]:
    pc = per_class_from_confusion(cm)
    lines = [f"  {'class':12s} {'support':>8s} {'precision':>10s} {'recall':>8s} {'F1':>8s}"]
    for c in range(NUM_GLOBAL_CLASSES):
        lines.append(f"  {CLASS_NAMES[c]:12s} {int(pc['support'][c]):8d} {100 * pc['precision'][c]:10.2f} "
                     f"{100 * pc['recall'][c]:8.2f} {100 * pc['f1'][c]:8.2f}")
    return lines


def mean_sd(values: List[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:6.2f}"
    return f"{np.mean(values):6.2f}+/-{np.std(values, ddof=1):.2f}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=DEFAULT_MODELS, help=f"Comma-separated from {sorted(PREDICTORS)}.")
    p.add_argument("--seeds", default="42,1,2", help="Meta / verifier seeds; matrices are printed for the first.")
    p.add_argument("--eval-cache", type=Path, default=Path("artifacts/cached_logits_eval.pt"))
    p.add_argument("--train-cache", type=Path, default=Path("artifacts/cached_logits_train.pt"))
    p.add_argument("--folds-file", type=Path, default=Path("artifacts/frh04_spatial_folds.npz"))
    p.add_argument("--meta-proba-dir", type=Path, default=Path("artifacts/pruned_proba_cache_noexpens"))
    p.add_argument("--verifier-dir", type=Path, default=Path("artifacts/orchard_verifier_final"))
    p.add_argument("--pheno-train", type=Path, default=Path("artifacts/phenology_train.npz"))
    p.add_argument("--pheno-eval", type=Path, default=Path("artifacts/phenology_eval.npz"))
    p.add_argument("--l2a-net-logits", type=Path,
                   default=Path("artifacts/checkpoints/msresnet_l2a_clear_ens3/logits_eval.npz"))
    p.add_argument("--output-json", type=Path, default=Path("artifacts/model_metrics_frh04.json"))
    p.add_argument("--output-log", type=Path, default=Path("artifacts/model_metrics_frh04.log"))
    args = p.parse_args()
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in args.models if m not in PREDICTORS]
    if unknown:
        p.error(f"Unknown models {unknown}; choose from {sorted(PREDICTORS)}.")
    args.seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    return args


def main() -> None:
    args = parse_args()
    inp = Inputs(args)
    preds: Dict[str, Dict[int, np.ndarray]] = {}
    valid = np.ones_like(inp.y, dtype=bool)
    for name in args.models:
        preds[name] = {}
        for seed in (args.seeds if name in SEEDED else [args.seeds[0]]):
            print(f"Predicting {name} (seed {seed}) ...", flush=True)
            pred, ok = PREDICTORS[name](inp, seed)
            preds[name][seed] = pred
            valid &= ok

    y = inp.y[valid]
    lines: List[str] = [
        f"frh04 metrics ({datetime.now():%Y-%m-%d %H:%M}): {int(valid.sum()):,} of {inp.y.size:,} parcels "
        f"({int((y == GLOBAL_ORCHARDS).sum())} orchards); {int((~valid).sum())} without an L1C match "
        "left out of every system (paired).",
        "Micro P = R = F1 = accuracy (single-label); macro = unweighted mean over the 9 classes.",
        "meta / final: accuracy-first settings chosen on the other 4 of 5 spatial folds.",
    ]
    report: Dict[str, dict] = {"parcels": int(valid.sum()), "excluded": int((~valid).sum()),
                               "class_names": list(CLASS_NAMES), "models": {}}
    summary_rows = []
    for name in args.models:
        per_seed = {}
        for seed, pred in preds[name].items():
            cm = confusion(y, pred[valid])
            pc = per_class_from_confusion(cm)
            per_seed[seed] = {"confusion": cm.tolist(), "summary": summarize(cm),
                              "per_class": {k: v.tolist() for k, v in pc.items()}}
        first = per_seed[next(iter(per_seed))]
        seeds_txt = ", ".join(str(s) for s in per_seed)
        lines += ["", f"=== {MODEL_LABELS[name]} [{name}; seed{'s' if len(per_seed) > 1 else ''} {seeds_txt}] " + "=" * 20,
                  f"Confusion matrix (rows = true, cols = predicted; seed {next(iter(per_seed))}):"]
        lines += format_confusion(np.array(first["confusion"]))
        lines += ["Per-class:"] + format_per_class(np.array(first["confusion"]))
        s = first["summary"]
        lines.append(f"  micro P/R/F1 {s['micro_f1']:.2f}   macro P {s['macro_precision']:.2f}  "
                     f"R {s['macro_recall']:.2f}  F1 {s['macro_f1']:.2f}")
        report["models"][name] = {"label": MODEL_LABELS[name], "seeds": {str(k): v for k, v in per_seed.items()}}
        summary_rows.append((name, [r["summary"] for r in per_seed.values()]))

    keys = ("micro_f1", "macro_precision", "macro_recall", "macro_f1", "orchard_precision", "orchard_recall", "orchard_f1")
    heads = ("micro P=R=F1", "macro P", "macro R", "macro F1", "orch P", "orch R", "orch F1")
    lines += ["", "=== Summary (mean+/-sd over seeds where seeded) " + "=" * 20,
              f"  {'system':42s}" + "".join(f"{h:>14s}" for h in heads)]
    for name, rows in summary_rows:
        lines.append(f"  {MODEL_LABELS[name]:42s}" + "".join(f"{mean_sd([r[k] for r in rows]):>14s}" for k in keys))
    lines += ["", "=== Orchard errors (mean+/-sd over seeds) " + "=" * 20,
              f"  {'system':42s}{'false orchards that are meadows (%)':>38s}{'missed orchards called meadows (%)':>37s}"]
    for name, rows in summary_rows:
        lines.append(f"  {MODEL_LABELS[name]:42s}{mean_sd([r['false_orchards_meadow_pct'] for r in rows]):>38s}"
                     f"{mean_sd([r['missed_orchards_meadow_pct'] for r in rows]):>37s}")

    print("\n".join(lines))
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["created"] = datetime.now().isoformat(timespec="seconds")
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {args.output_log} and {args.output_json}")


if __name__ == "__main__":
    main()

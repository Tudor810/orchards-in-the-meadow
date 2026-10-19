"""Post-hoc diagnosis of the orchard verifier in the region rotation (not part of the frozen design).

Re-runs ``region_rotation.py verify`` in-process with variants, reusing one feature build per test region:
    base        identical to the frozen verifier (sanity check against verify[_seed<s>].json)
    share09     candidate threshold set so that 9% of parcels are candidates, in the training regions and
                in the test region separately (label-free; the frozen rule keeps 90% of training orchards)
    noneigh     frozen candidates, verifier without the neighbour-context features
    nomonthly   frozen candidates, verifier without the monthly clear-date means
Reports, per test region / seed / variant: candidate count and orchard coverage, orchard AP of the meta score
and of the verifier with the paired block-bootstrap interval of the difference (same resampling as
``stage_verify``), and accuracy-first precision / recall.

Usage: python scripts/diagnose_rotation_verifier.py --tests frh03 --variants base,share09,noneigh,nomonthly
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import region_rotation as rr  # noqa: E402
from orchard_verifier import ORCHARD, ap, build_verifier, decode, geometry, monthly_means, neighbour_context  # noqa: E402

from agricultureai.evaluation.selection_protocol import (  # noqa: E402
    classification_metrics, cross_fit, percentile_ci, select_accuracy_first,
)

GROUPS = ("meta", "pheno", "monthly", "geometry", "neigh")
DROP = {"base": (), "share09": (), "noneigh": ("neigh",), "nomonthly": ("monthly",)}
SHARE = 0.09


def run_region(test: str, seeds, variants, work: Path, out: list) -> None:
    t0 = time.time()
    rr.args_global = argparse.Namespace(work_dir=work)
    train = [r for r in rr.REGIONS if r != test]
    blocks = rr.region_blocks(test)
    stores = {r: np.load(f"artifacts/series_store/{r}.npz") for r in rr.REGIONS}
    geo = {r: geometry(r, stores[r]["parcel_ids"]) for r in rr.REGIONS}
    mon = {r: monthly_means(stores[r]) for r in rr.REGIONS}
    folds = np.load(work / f"folds_{test}.npz")
    y_ev = blocks[test]["y"]
    y_bin = (y_ev == ORCHARD).astype(np.int8)
    y_tr = np.concatenate([blocks[r]["y"] for r in train])
    _, gidx = np.unique(folds["blocks"], return_inverse=True)
    members = np.split(np.argsort(gidx, kind="stable"), np.cumsum(np.bincount(gidx))[:-1])
    rng = np.random.default_rng(0)
    boots = [np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))]) for _ in range(1000)]
    print(f"[{test}] features ready ({time.time() - t0:.0f}s)", flush=True)
    for seed in seeds:
        oof = np.load(work / test / f"oof_B{rr.sfx(seed)}.npz")
        p_ev = np.load(work / test / f"proba_B{rr.sfx(seed)}" / "proba_pair_compact_m1.npz")["p_xgb"]
        parts = {}
        for r in rr.REGIONS:
            proba = p_ev if r == test else oof[r]
            parts[r] = {"meta": proba, "pheno": blocks[r]["pheno"], "monthly": mon[r],
                        "geometry": geo[r][:, 2:].astype(np.float32), "neigh": neighbour_context(geo[r][:, :2], proba)}
        p_tr = np.vstack([oof[r] for r in train])
        meta_score = p_ev[:, ORCHARD].astype(np.float64)
        for variant in variants:
            t1 = time.time()
            groups = [g for g in GROUPS if g not in DROP[variant]]
            x_tr = np.vstack([np.hstack([parts[r][g] for g in groups]) for r in train])
            x_ev = np.hstack([parts[test][g] for g in groups])
            if variant == "share09":
                tau_tr = float(np.quantile(p_tr[:, ORCHARD], 1 - SHARE))
                tau_ev = float(np.quantile(p_ev[:, ORCHARD], 1 - SHARE))
            else:
                tau_tr = tau_ev = float(np.quantile(p_tr[y_tr == ORCHARD, ORCHARD], 0.10))
            cand_tr, cand_ev = p_tr[:, ORCHARD] >= tau_tr, p_ev[:, ORCHARD] >= tau_ev
            model = build_verifier(seed)
            model.fit(x_tr[cand_tr], (y_tr[cand_tr] == ORCHARD).astype(np.int8))
            score = meta_score.copy()
            score[cand_ev] = 1.0 + model.predict_proba(x_ev[cand_ev])[:, 1]
            d = np.array([ap(y_bin[i], score[i]) - ap(y_bin[i], meta_score[i]) for i in boots])
            lo, hi = percentile_ci(d)
            rows = {}
            for name, s in (("meta", meta_score), ("verifier", score)):
                preds = decode(p_ev, s, np.unique(np.quantile(s, np.linspace(0.90, 0.9995, 160))))
                cf, _ = cross_fit(preds, y_ev, folds["folds"], select_accuracy_first)
                m = classification_metrics(y_ev, cf)
                rows[name] = {"ap": ap(y_bin, s), "acc_first": {k: m[k] for k in ("accuracy", "orchard_precision", "orchard_recall")}}
            rec = {"test": test, "seed": seed, "variant": variant, "tau_train": tau_tr, "tau_test": tau_ev,
                   "candidates": int(cand_ev.sum()), "candidate_share": float(cand_ev.mean()),
                   "candidate_orchard_coverage": float(y_bin[cand_ev].sum() / y_bin.sum()),
                   "train_candidate_orchard_share": float((y_tr[cand_tr] == ORCHARD).mean()),
                   "ap_diff_ci": [lo, hi], **rows}
            out.append(rec)
            mv, vv = rows["meta"], rows["verifier"]
            print(f"{test} seed={seed:2d} {variant:9s} cand={rec['candidates']:6d} ({100 * rec['candidate_share']:4.1f}%, "
                  f"covers {100 * rec['candidate_orchard_coverage']:4.1f}% of orchards) | AP {mv['ap']:.2f} -> {vv['ap']:.2f} "
                  f"[{lo:+.2f},{hi:+.2f}] | acc-first P {mv['acc_first']['orchard_precision']:.1f} -> "
                  f"{vv['acc_first']['orchard_precision']:.1f}, R {mv['acc_first']['orchard_recall']:.1f} -> "
                  f"{vv['acc_first']['orchard_recall']:.1f} ({time.time() - t1:.0f}s)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tests", default="frh03")
    p.add_argument("--seeds", default="42,1,2")
    p.add_argument("--variants", default="base,share09,noneigh,nomonthly")
    p.add_argument("--work-dir", type=Path, default=Path("artifacts/rotation"))
    p.add_argument("--output", type=Path, default=Path("artifacts/rotation/verifier_diagnosis.json"))
    args = p.parse_args()
    variants = [v for v in args.variants.split(",") if v]
    unknown = set(variants) - set(DROP)
    if unknown:
        p.error(f"unknown variants: {sorted(unknown)}")
    out: list = json.loads(args.output.read_text()) if args.output.exists() else []
    for test in args.tests.split(","):
        run_region(test, [int(s) for s in args.seeds.split(",")], variants, args.work_dir, out)
        args.output.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

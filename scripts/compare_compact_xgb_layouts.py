"""Compare compact XGB feature layouts vs minus_l2b_l2c base."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agricultureai.meta_features import (
    METADATA_FEATURE_DIM,
    build_xgb,
    feature_dims,
    fit_xgb,
    load_feature_matrix,
    orchard_sample_weights,
    predict_xgb,
)
from agricultureai.pipeline import GLOBAL_ORCHARDS


def blocks(full_dim: int) -> dict[str, range]:
    meta_start = full_dim - METADATA_FEATURE_DIM
    return {
        "l1": range(0, 4),
        "l2a": range(4, 6),
        "l2b": range(6, 8),
        "l2c_raw": range(8, 10),
        "l3_raw": range(10, 15),
        "cereal": range(15, 22),
        "msresnet": range(22, 31),
        "global_eng": range(31, 41),
        "l2c_soft": range(41, 43),
        "l3_soft": range(43, 48),
        "transformer": range(48, 58),
        "metadata": range(meta_start, full_dim),
    }


def cols(b: dict[str, range], names: list[str]) -> np.ndarray:
    keep: list[int] = []
    for n in names:
        keep.extend(b[n])
    return np.array(sorted(keep), dtype=np.int64)


def eval_subset(xtr, ytr, xev, yev, c, *, m: float, seed: int) -> dict:
    clf = build_xgb(seed)
    fit_xgb(clf, xtr[:, c], ytr, orchard_sample_weights(ytr, m))
    pred = predict_xgb(clf, xev[:, c]).astype(np.int64)
    orch = yev == GLOBAL_ORCHARDS
    return {
        "n_features": int(c.size),
        "accuracy": float((pred == yev).mean() * 100.0),
        "orchard_recall": float((pred[orch] == GLOBAL_ORCHARDS).mean() * 100.0) if orch.any() else 0.0,
    }


def main() -> None:
    _, full_dim = feature_dims(use_metadata=True)
    xtr, ytr = load_feature_matrix(Path("artifacts/cached_logits_train.pt"), use_metadata=True)
    xev, yev = load_feature_matrix(Path("artifacts/cached_logits_eval.pt"), use_metadata=True)
    b = blocks(full_dim)
    m, seed = 1.0, 42

    layouts = {
        # Current best stretch-oriented base
        "minus_l2b_l2c": [
            n for n in b if n not in ("l2b", "l2c_raw", "l2c_soft")
        ],
        # Drop clearly-harmful + near-noise DMOE bits
        "compact_keep_meta": [
            "l3_raw", "cereal", "msresnet", "transformer", "metadata",
        ],
        # Acc-leaning compact (drop metadata which hurts orchard)
        "compact_no_meta": [
            "l3_raw", "cereal", "msresnet", "transformer",
        ],
        # Orchard-leaning: drop metadata, keep L3 soft too
        "orchard_lean": [
            "l3_raw", "l3_soft", "cereal", "msresnet", "transformer",
        ],
        # Globals-only control
        "globals_meta": ["msresnet", "global_eng", "transformer", "metadata"],
        # Drop only L2b+L2c+neutrals (keep L3 soft + cereal + ...)
        "drop_harm_and_noise": [
            "l3_raw", "cereal", "msresnet", "l3_soft", "transformer", "metadata",
        ],
    }

    rows = {}
    for name, names in layouts.items():
        print(f"Evaluating {name} ({len(names)} blocks) ...", flush=True)
        rows[name] = eval_subset(xtr, ytr, xev, yev, cols(b, names), m=m, seed=seed)
        rows[name]["blocks"] = names

    base = rows["minus_l2b_l2c"]
    lines = [
        "Compact XGB layout comparison (m=1, XGB only, no bias)",
        f"{'layout':22}  {'feats':>5}  {'acc':>7}  {'orch_R':>7}  {'d_acc':>7}  {'d_orch':>7}",
        "-" * 70,
    ]
    for name, r in rows.items():
        d_acc = r["accuracy"] - base["accuracy"]
        d_orch = r["orchard_recall"] - base["orchard_recall"]
        lines.append(
            f"{name:22}  {r['n_features']:5d}  {r['accuracy']:6.2f}%  "
            f"{r['orchard_recall']:6.2f}%  {d_acc:+6.2f}  {d_orch:+6.2f}"
        )
        lines.append(f"  blocks: {', '.join(r['blocks'])}")

    report = "\n".join(lines)
    print("\n" + report)
    Path("artifacts/compact_xgb_layouts.log").write_text(report + "\n", encoding="utf-8")
    Path("artifacts/compact_xgb_layouts.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

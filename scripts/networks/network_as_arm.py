"""Write a series network's frh04 logits as an ``evaluate_meta_protocol.py`` arm.

The network's softmax is stored as both ``p_xgb`` and ``p_cat`` (so every blend returns it unchanged) and
copied to ``proba_pair_compact_m{1,2,3}.npz`` (so the arm has the same decoding grid as the meta-classifier).
The network then gets the same orchard / meadow log-biases and the same out-of-sample selection rules: the
fair "end-to-end network + operating-point dial" baseline.

Example::

    python scripts/networks/network_as_arm.py artifacts/checkpoints/msresnet_l2a_clear/logits_eval.npz \\
        artifacts/pruned_proba_cache_l2anetdial
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def softmax(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float64) - z.max(1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logits", type=Path, help="logits_eval.npz from train_series_classifier.py --export eval")
    p.add_argument("proba_dir", type=Path)
    p.add_argument("--eval-cache", type=Path, default=Path("artifacts/cached_logits_eval.pt"))
    p.add_argument("--multipliers", default="1,2,3")
    p.add_argument("--logit-adjust", action="store_true",
                   help="Subtract log(class_weights) stored by a weighted training run (undoes the weighting's prior shift).")
    args = p.parse_args()

    net = np.load(args.logits)
    ids = torch.load(args.eval_cache, map_location="cpu", weights_only=True)["parcel_ids"].numpy()
    if not np.array_equal(net["parcel_ids"], ids):
        raise RuntimeError(f"{args.logits}: parcel ids do not match {args.eval_cache}")
    logits = net["features"].astype(np.float64)
    if args.logit_adjust:
        if "class_weights" not in net.files:
            raise RuntimeError(f"{args.logits} has no class_weights (not a weighted run)")
        logits = logits - np.log(net["class_weights"])[None, :]
    proba = softmax(logits)
    args.proba_dir.mkdir(parents=True, exist_ok=True)
    variant = json.dumps({"network": str(args.logits), "logit_adjust": args.logit_adjust, "note": "softmax as p_xgb and p_cat"})
    for m in (float(v) for v in args.multipliers.split(",")):
        np.savez(args.proba_dir / f"proba_pair_compact_m{m:g}.npz", p_xgb=proba, p_cat=proba, variant=variant)
    acc = 100.0 * float((proba.argmax(1) == net["labels"]).mean())
    print(f"{args.logits} -> {args.proba_dir} (argmax acc {acc:.2f}%)")


if __name__ == "__main__":
    main()

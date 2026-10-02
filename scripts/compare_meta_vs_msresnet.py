"""Side-by-side per-class metrics: DMOE meta-classifier vs pretrained MSResNet on frh04."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import breizhcrops as bzh
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import train_meta_classifier as tmc  # noqa: E402
from evaluate_pretrained_models import evaluate_metrics, pad_collate  # noqa: E402

CLASS_NAMES = tmc.GLOBAL_ID_TO_NAME


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 9) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    support = cm.sum(axis=1)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    rows = {}
    for c in range(num_classes):
        prec = 100.0 * _safe_div(int(tp[c]), int(tp[c] + fp[c]))
        rec = 100.0 * _safe_div(int(tp[c]), int(tp[c] + fn[c]))
        f1 = 100.0 * _safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0
        acc = 100.0 * _safe_div(int(tp[c]), int(support[c])) if support[c] else 0.0
        rows[c] = {
            "support": int(support[c]),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
    overall = accuracy_score(y_true, y_pred) * 100.0
    return {"overall_accuracy": overall, "classes": rows, "confusion": cm}


@torch.inference_mode()
def evaluate_msresnet(*, region: str, level: str, batch_size: int, device: torch.device) -> dict:
    print(f"Loading MSResNet on {region} level={level} ...")
    dataset = bzh.BreizhCrops(region, level=level)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: pad_collate(batch, map_to_super_class=False),
        pin_memory=torch.cuda.is_available(),
    )
    model = bzh.models.pretrained("MSResNet", device=device)
    metrics, per_class, _ = evaluate_metrics(model, loader, device=device, num_classes=9)
    rows = {}
    for c in range(9):
        rows[c] = {
            "support": per_class["support"][c],
            "accuracy": per_class["accuracy"][c],
            "precision": per_class["precision"][c],
            "recall": per_class["recall"][c],
            "f1": per_class["f1"][c],
        }
    return {
        "overall_accuracy": metrics["overall_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "classes": rows,
    }


def evaluate_meta(
    *,
    train_cache: Path,
    eval_cache: Path,
    orchard_multiplier: float,
    seed: int,
    use_metadata: bool,
) -> dict:
    print(f"Training meta-classifier (m={orchard_multiplier}) on {train_cache.name} ...")
    tmc.set_global_seed(seed)
    from agricultureai.meta_features import load_feature_matrix

    x_train, y_train = load_feature_matrix(train_cache, use_metadata=use_metadata)
    x_eval, y_eval = load_feature_matrix(eval_cache, use_metadata=use_metadata)
    clf = tmc.build_classifier("auto", seed=seed)
    tmc.fit_classifier(clf, x_train, y_train, orchard_multiplier=orchard_multiplier)
    y_pred = clf.predict(x_eval)
    return per_class_table(y_eval, y_pred)


def print_comparison(msresnet: dict, meta: dict, *, meta_label: str) -> None:
    print("\n" + "=" * 88)
    print(f"{'Class':<22} {'Support':>8}  {'MSResNet R':>10} {'Meta R':>10}  {'dRecall':>9}")
    print("-" * 88)
    for c in range(9):
        name = CLASS_NAMES[c][:20]
        sup = msresnet["classes"][c]["support"]
        ms_r = msresnet["classes"][c]["recall"]
        meta_r = meta["classes"][c]["recall"]
        delta = meta_r - ms_r
        sign = "+" if delta >= 0 else ""
        print(f"{name:<22} {sup:8d}  {ms_r:9.2f}% {meta_r:9.2f}%  {sign}{delta:8.2f}%")

    print("-" * 88)
    print(
        f"{'OVERALL':<22} {'':>8}  "
        f"{msresnet['overall_accuracy']:9.2f}% {meta['overall_accuracy']:9.2f}%  "
        f"{meta['overall_accuracy'] - msresnet['overall_accuracy']:+8.2f}%"
    )
    print("=" * 88)

    print(f"\nDetailed metrics (precision / recall / F1)")
    print(f"{'Class':<22} {'Model':<10} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print("-" * 60)
    for c in range(9):
        name = CLASS_NAMES[c][:20]
        for label, block in (("MSResNet", msresnet), (meta_label, meta)):
            row = block["classes"][c]
            print(
                f"{name if label == 'MSResNet' else '':<22} {label:<10} "
                f"{row['precision']:6.2f}% {row['recall']:6.2f}% {row['f1']:6.2f}%"
            )
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DMOE meta vs MSResNet per class on frh04.")
    parser.add_argument("--region", default="frh04")
    parser.add_argument("--level", default="L2A")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-cache", type=Path, default=tmc.DEFAULT_TRAIN_CACHE)
    parser.add_argument("--eval-cache", type=Path, default=tmc.DEFAULT_EVAL_CACHE)
    parser.add_argument(
        "--use-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--orchard-multiplier", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--meta-label",
        default="DMOE+Meta",
        help="Display name for the meta-classifier column.",
    )
    parser.add_argument(
        "--skip-msresnet",
        action="store_true",
        help="Skip MSResNet inference (reuse when MSResNet was already evaluated).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.skip_msresnet:
        print("Skipping MSResNet eval (--skip-msresnet).")
        msresnet = None
    else:
        msresnet = evaluate_msresnet(
            region=args.region,
            level=args.level,
            batch_size=args.batch_size,
            device=device,
        )
    meta = evaluate_meta(
        train_cache=args.train_cache.resolve(),
        eval_cache=args.eval_cache.resolve(),
        orchard_multiplier=float(args.orchard_multiplier),
        seed=int(args.seed),
        use_metadata=bool(args.use_metadata),
    )

    if msresnet is not None:
        print(
            f"\nMSResNet overall: {msresnet['overall_accuracy']:.2f}%  "
            f"(macro F1 {msresnet['macro_f1']:.2f}%)"
        )
    print(
        f"{args.meta_label} overall: {meta['overall_accuracy']:.2f}%  "
        f"(orchard recall {meta['classes'][5]['recall']:.2f}%)"
    )
    if msresnet is None:
        raise SystemExit(
            "Cannot print side-by-side comparison without MSResNet metrics. "
            "Re-run without --skip-msresnet."
        )
    print_comparison(msresnet, meta, meta_label=str(args.meta_label))


if __name__ == "__main__":
    main()

import argparse
import os

import torch
from tqdm import tqdm

from agricultureai.data.breizhcrops_cereal_expert import (
    CEREAL_EXPERT_ID_TO_NAME,
    CEREAL_EXPERT_RAW_IDS,
    CEREAL_EXPERT_TARGET_IDS,
    NUM_CEREAL_EXPERT_CLASSES,
    build_cereal_expert_loaders,
)
from agricultureai.evaluation.transformer import MAX_SEQ_LEN as DEFAULT_MAX_SEQ_LEN
from agricultureai.models.breizhcrops_ltae import BreizhCropsLTAEExpert
from agricultureai.training.checkpoints import load_checkpoint, model_state_dict_from_checkpoint
from agricultureai.training.trainer import ModelTrainer


def _safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def evaluate_cereal_expert(model, loader, device, id_to_class, num_classes=NUM_CEREAL_EXPERT_CLASSES):
    model.eval()
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.no_grad():
        for inputs, targets, _ in tqdm(loader, desc="Evaluating cereal expert", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device).long()
            preds = model(inputs).argmax(dim=1)

            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion[int(t), int(p)] += 1

    tp = confusion.diag()
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp
    support = confusion.sum(dim=1)

    precision_per_class = [
        100.0 * _safe_div(int(tp[c].item()), int((tp[c] + fp[c]).item())) for c in range(num_classes)
    ]
    recall_per_class = [
        100.0 * _safe_div(int(tp[c].item()), int((tp[c] + fn[c]).item())) for c in range(num_classes)
    ]
    macro_precision = sum(precision_per_class) / num_classes

    total = int(support.sum().item())
    correct = int(tp.sum().item())
    overall_acc = 100.0 * _safe_div(correct, total)

    print(f"\nOverall accuracy: {overall_acc:.2f}% ({correct}/{total})")
    print(f"Macro precision (Wheat vs Barley): {macro_precision:.2f}%\n")

    print("Per-class metrics:")
    for c in range(num_classes):
        name = id_to_class.get(c, str(c))
        acc = 100.0 * _safe_div(int(tp[c].item()), int(support[c].item()))
        print(
            f"  - class {c} ({name:<12}): "
            f"precision={precision_per_class[c]:6.2f}%  "
            f"recall={recall_per_class[c]:6.2f}%  "
            f"acc={acc:6.2f}%  "
            f"support={int(support[c].item())}"
        )

    print("\nConfusion matrix (rows=true, cols=pred):")
    print("              pred Barley  pred Wheat")
    for r in range(num_classes):
        row_name = id_to_class.get(r, str(r))
        print(f"  true {row_name:<8} {confusion[r].tolist()}")
    print(confusion.cpu().numpy())

    return macro_precision, confusion


def main():
    parser = argparse.ArgumentParser(
        description="Train Stage-2 Winter Cereal LTAE expert on raw ids 0 (barley) and 1 (wheat).",
    )
    parser.add_argument("--regions", nargs="+", default=["frh01", "frh02", "frh03"])
    parser.add_argument(
        "--val-regions",
        nargs="+",
        default=["frh04"],
        help="Held-out region(s) for epoch eval and best.pt (default: frh04).",
    )
    parser.add_argument(
        "--train-region-holdout",
        action="store_true",
        help="Legacy: 80/20 split per train region for val; do not use with --val-regions frh04.",
    )
    parser.add_argument("--level", default="L1C")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--target-ids",
        nargs="+",
        type=int,
        default=CEREAL_EXPERT_RAW_IDS,
        help=f"Raw BreizhCrops ids kept at load time (default: {CEREAL_EXPERT_RAW_IDS}).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="For quick runs: limit to first N samples per region (0 = full region).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="artifacts/checkpoints/expert_cereals",
    )
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument(
        "--no-cosine-lr",
        action="store_true",
        help="Disable cosine LR schedule (enabled by default).",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Random LTAE init instead of bzh.models.pretrained('ltae').",
    )
    parser.add_argument(
        "--eval-last-epoch",
        action="store_true",
        help="Final eval uses last epoch weights instead of best.pt.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log metrics and best checkpoint to Weights & Biases.")
    parser.add_argument("--wandb-project", default="dmoe-thesis")
    parser.add_argument("--wandb-run-name", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    id_to_class = dict(CEREAL_EXPERT_ID_TO_NAME)
    raw_target_ids = [int(x) for x in args.target_ids]

    val_regions = list(args.val_regions) if args.val_regions else []
    use_holdout = bool(args.train_region_holdout)
    if use_holdout and val_regions:
        raise SystemExit("Use either --val-regions (e.g. frh04) or --train-region-holdout, not both.")
    if not val_regions and not use_holdout:
        use_holdout = True

    train_loader, val_loader = build_cereal_expert_loaders(
        train_regions=args.regions,
        val_regions=val_regions,
        level=args.level,
        raw_target_ids=raw_target_ids,
        max_samples=int(args.max_samples),
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        use_train_region_holdout=use_holdout,
    )
    eval_set_name = "+".join(val_regions) if val_regions else "train_holdout"
    print(f"Train regions: {args.regions}  |  Val: {eval_set_name}")
    print(f"Raw filter ids: {raw_target_ids}  |  Expert classes: {CEREAL_EXPERT_TARGET_IDS}")

    model = BreizhCropsLTAEExpert(
        num_classes=NUM_CEREAL_EXPERT_CLASSES,
        sequencelength=args.max_seq_len,
        pretrained=not args.from_scratch,
        device=device,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Backbone arch: {model.arch}  |  pretrained init: {not args.from_scratch}  |  "
        f"trainable params: {trainable}/{total}"
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = None
    if not args.no_cosine_lr:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    architecture_config = {
        "model_type": "ltae",
        "expert": "cereals",
        "input_dim": 13,
        "channels": 13,
        "seq_len": int(args.max_seq_len),
        "num_classes": NUM_CEREAL_EXPERT_CLASSES,
        "pretrained": not args.from_scratch,
        "backbone_arch": model.arch,
        "level": args.level,
        "raw_target_ids": raw_target_ids,
        "target_ids": list(CEREAL_EXPERT_TARGET_IDS),
    }
    run_config = {
        "regions": list(args.regions),
        "val_regions": val_regions,
        "use_train_region_holdout": use_holdout,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "cosine_lr": not args.no_cosine_lr,
        "max_samples": int(args.max_samples),
    }
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        target_ids=CEREAL_EXPERT_TARGET_IDS,
        eval_set_name=eval_set_name,
        architecture_config=architecture_config,
        run_config=run_config,
        track_with_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=(args.wandb_run_name or None),
    )

    ckpt_dir = None if args.no_checkpoints else (args.checkpoint_dir or "").strip() or None
    metrics_csv = os.path.join(ckpt_dir, "train_metrics.csv") if ckpt_dir else None
    trainer.fit(
        epochs=args.epochs,
        checkpoint_dir=ckpt_dir,
        lr_scheduler=lr_scheduler,
        metrics_csv_path=metrics_csv,
    )

    if ckpt_dir and not args.eval_last_epoch:
        best_path = os.path.join(ckpt_dir, "best.pt")
        if os.path.isfile(best_path):
            checkpoint = load_checkpoint(best_path, map_location=device)
            model.load_state_dict(model_state_dict_from_checkpoint(checkpoint))
            print(f"Loaded {best_path} for final evaluation.")

    evaluate_cereal_expert(
        model,
        val_loader,
        device=device,
        id_to_class=id_to_class,
        num_classes=NUM_CEREAL_EXPERT_CLASSES,
    )


if __name__ == "__main__":
    main()

import argparse
import os

import torch

from agricultureai.data.breizhcrops_loaders import build_breizhcrops_loaders
from agricultureai.data.breizhcrops_super_classes import SUPER_CLASS_ID_TO_NAME
from agricultureai.evaluation.transformer import MAX_SEQ_LEN as DEFAULT_MAX_SEQ_LEN
from agricultureai.evaluation.transformer import evaluate_per_class
from agricultureai.models.breizhcrops_transformer import BreizhCropsTransformerClassifier
from agricultureai.training.checkpoints import load_checkpoint, model_state_dict_from_checkpoint
from agricultureai.training.trainer import ModelTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune bzh.models.pretrained('Transformer') as a Level-1 super-class router.",
    )
    parser.add_argument("--regions", nargs="+", default=["frh01", "frh02", "frh03"])
    parser.add_argument(
        "--val-regions",
        nargs="+",
        default=["frh04"],
        help="Held-out region(s) for epoch eval and best.pt (default: frh04). Empty list disables.",
    )
    parser.add_argument(
        "--train-region-holdout",
        action="store_true",
        help="Legacy: 80/20 split per train region for val; do not use with --val-regions frh04.",
    )
    parser.add_argument("--level", default="L2A", help="L2A (10 features, cloud-filtered) or L1C (13 features).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--lr", type=float, default=1.7e-4)
    parser.add_argument("--weight-decay", type=float, default=3.5e-6)
    parser.add_argument(
        "--no-freeze-backbone",
        action="store_true",
        help="Fine-tune the full Transformer (default: train only the new classification head).",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Random Transformer init instead of bzh.models.pretrained('Transformer').",
    )
    parser.add_argument(
        "--use-doy",
        action="store_true",
        help="Inject day-of-year embeddings and mask padded steps (native L2A path).",
    )
    parser.add_argument("--target-ids", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="For quick runs: limit to first N samples per region (0 = full region).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="artifacts/checkpoints/transformer",
        help="Directory for latest.pt, best.pt, train_metrics.csv.",
    )
    parser.add_argument("--no-checkpoints", action="store_true", help="Disable checkpoint writes.")
    parser.add_argument(
        "--final-save",
        default="",
        help="Optional path to save state_dict once after all epochs.",
    )
    parser.add_argument(
        "--no-cosine-lr",
        action="store_true",
        help="Disable cosine LR schedule (enabled by default).",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--eval-last-epoch",
        action="store_true",
        help="Final per-class eval uses last epoch weights instead of best.pt.",
    )
    parser.add_argument(
        "--init-weights",
        default="",
        help="Path to a .pt state_dict to load before training.",
    )
    parser.add_argument(
        "--resume-best",
        action="store_true",
        help="Load <checkpoint-dir>/best.pt before training. Ignored if --init-weights is set.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log metrics and best checkpoint to Weights & Biases.")
    parser.add_argument("--wandb-project", default="dmoe-thesis")
    parser.add_argument("--wandb-run-name", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    id_to_crop = dict(SUPER_CLASS_ID_TO_NAME)

    val_regions = list(args.val_regions) if args.val_regions else []
    use_holdout = bool(args.train_region_holdout)
    if use_holdout and val_regions:
        raise SystemExit("Use either --val-regions (e.g. frh04) or --train-region-holdout, not both.")
    if not val_regions and not use_holdout:
        use_holdout = True

    train_loader, val_loader = build_breizhcrops_loaders(
        train_regions=args.regions,
        val_regions=val_regions,
        level=args.level,
        target_ids=args.target_ids,
        max_samples=int(args.max_samples),
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        use_train_region_holdout=use_holdout,
        use_doy=args.use_doy,
    )
    eval_set_name = "+".join(val_regions) if val_regions else "train_holdout"
    print(f"Train regions: {args.regions}  |  Val: {eval_set_name}")

    num_classes = len(args.target_ids)
    channels = 13 if args.level == "L1C" else 10
    freeze_backbone = not args.no_freeze_backbone
    # The L1C pretrained backbone expects 13 channels; native L2A (10-ch) with
    # DOY embeddings must start from a random init.
    use_pretrained = not args.from_scratch and channels == 13
    if not args.from_scratch and not use_pretrained:
        print("L2A has 10 channels; disabling incompatible L1C pretrained weights.")
    model = BreizhCropsTransformerClassifier(
        num_classes=num_classes,
        input_dim=channels,
        freeze_backbone=freeze_backbone,
        pretrained=use_pretrained,
        use_doy=args.use_doy,
        device=device,
    )
    print(
        f"Backbone frozen: {freeze_backbone}  |  Pretrained init: {use_pretrained}  "
        f"|  DOY: {args.use_doy}"
    )

    init_path = (args.init_weights or "").strip()
    resume_checkpoint = None
    if not init_path and args.resume_best:
        init_path = os.path.join(os.path.abspath(args.checkpoint_dir), "best.pt")
    if init_path:
        if not os.path.isfile(init_path):
            raise SystemExit(f"Init weights not found: {init_path}")
        checkpoint = load_checkpoint(init_path, map_location=device)
        model.load_state_dict(model_state_dict_from_checkpoint(checkpoint), strict=True)
        if args.resume_best and not args.init_weights:
            resume_checkpoint = checkpoint
        print(f"Loaded initial weights from {init_path}")

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = None
    if not args.no_cosine_lr:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    architecture_config = {
        "model_type": "transformer",
        "expert": "level1_router",
        "input_dim": channels,
        "channels": channels,
        "seq_len": int(args.max_seq_len),
        "num_classes": num_classes,
        "freeze_backbone": freeze_backbone,
        "pretrained": use_pretrained,
        "use_doy": bool(args.use_doy),
        "level": args.level,
        "target_ids": list(args.target_ids),
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
        "label_smoothing": float(args.label_smoothing),
        "max_samples": int(args.max_samples),
    }
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        target_ids=args.target_ids,
        eval_set_name=eval_set_name,
        architecture_config=architecture_config,
        run_config=run_config,
        track_with_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=(args.wandb_run_name or None),
    )

    ckpt_dir = None if args.no_checkpoints else (args.checkpoint_dir or "").strip() or None
    final_save = (args.final_save or "").strip() or None
    metrics_csv = os.path.join(ckpt_dir, "train_metrics.csv") if ckpt_dir else None
    trainer.fit(
        epochs=args.epochs,
        save_path=final_save,
        checkpoint_dir=ckpt_dir,
        lr_scheduler=lr_scheduler,
        metrics_csv_path=metrics_csv,
        resume_checkpoint=resume_checkpoint,
    )

    if ckpt_dir and not args.eval_last_epoch:
        best_path = os.path.join(ckpt_dir, "best.pt")
        if os.path.isfile(best_path):
            checkpoint = load_checkpoint(best_path, map_location=device)
            model.load_state_dict(model_state_dict_from_checkpoint(checkpoint))
            print(f"Loaded {best_path} for final per-class evaluation.")

    evaluate_per_class(
        model,
        val_loader,
        device=device,
        id_to_crop=id_to_crop,
        num_classes=num_classes,
    )


if __name__ == "__main__":
    main()

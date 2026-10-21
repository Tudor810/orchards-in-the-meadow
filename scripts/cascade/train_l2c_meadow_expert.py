"""Train Level-2c meadow expert (Permanent vs Temporary) with MSResNet."""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from agricultureai.data.breizhcrops_l2c_meadow import (
    L2C_ID_TO_NAME,
    NUM_L2C_CLASSES,
    build_l2c_meadow_loaders,
)
from agricultureai.evaluation.transformer import MAX_SEQ_LEN as DEFAULT_MAX_SEQ_LEN
from agricultureai.models.breizhcrops_msresnet import BreizhCropsMSResNetExpert
from agricultureai.training.checkpoints import load_checkpoint, model_state_dict_from_checkpoint
from agricultureai.training.series_trainer import SeriesExpertTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train L2c meadow expert (PM vs TM) with MSResNet.")
    parser.add_argument("--regions", nargs="+", default=["frh01", "frh02", "frh03"])
    parser.add_argument("--val-regions", nargs="+", default=["frh04"])
    parser.add_argument("--train-region-holdout", action="store_true")
    parser.add_argument("--level", default="L2A")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dims", type=int, default=32)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--no-doy",
        action="store_true",
        help="Disable day-of-year embedding (legacy spectral-only MSResNet).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints/l2c_meadow_expert")
    parser.add_argument("--resume", default="")
    parser.add_argument("--no-cosine-lr", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    channels = 13 if args.level == "L1C" else 10

    train_loader, val_loader = build_l2c_meadow_loaders(
        train_regions=args.regions,
        val_regions=args.val_regions,
        level=args.level,
        max_samples=int(args.max_samples),
        batch_size=int(args.batch_size),
        max_seq_len=int(args.max_seq_len),
        use_train_region_holdout=bool(args.train_region_holdout),
        seed=int(args.seed),
    )

    use_doy = not bool(args.no_doy)
    print(f"L2c DOY embedding: {use_doy}")
    architecture_config = {
        "model_type": "msresnet",
        "role": "l2c_meadow",
        "num_classes": NUM_L2C_CLASSES,
        "input_dim": channels,
        "hidden_dims": int(args.hidden_dims),
        "pretrained": not bool(args.no_pretrained),
        "use_doy": use_doy,
        "doy_positional_encoding": use_doy,
        "balanced_sampling": True,
    }
    model = BreizhCropsMSResNetExpert(
        num_classes=NUM_L2C_CLASSES,
        input_dim=channels,
        hidden_dims=int(args.hidden_dims),
        pretrained=not bool(args.no_pretrained),
        use_doy=use_doy,
        device=device,
    )
    resume_best_val = -1.0
    resume_epoch_offset = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(model_state_dict_from_checkpoint(ckpt), strict=False)
        meta = ckpt.get("training_metadata") or {}
        resume_best_val = float(meta.get("val_accuracy", -1.0))
        resume_epoch_offset = int(meta.get("epoch", meta.get("best_epoch", 0)) or 0)
        print(
            f"Resumed from {args.resume} "
            f"(epoch={resume_epoch_offset}, val_acc={resume_best_val:.2f}%)"
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = None if args.no_cosine_lr else CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))

    run_config = {
        "regions": list(args.regions),
        "val_regions": list(args.val_regions),
        "level": args.level,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "class_names": L2C_ID_TO_NAME,
        "resume": args.resume or None,
        "resume_epoch_offset": resume_epoch_offset,
    }
    trainer = SeriesExpertTrainer(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        eval_set_name="+".join(args.val_regions),
        architecture_config=architecture_config,
        run_config=run_config,
    )
    metrics_csv = os.path.join(args.checkpoint_dir, "train_metrics.csv")
    trainer.fit(
        int(args.epochs),
        checkpoint_dir=args.checkpoint_dir,
        lr_scheduler=scheduler,
        metrics_csv_path=metrics_csv,
        initial_best_val=resume_best_val,
        epoch_offset=resume_epoch_offset,
    )


if __name__ == "__main__":
    main()

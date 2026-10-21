"""Train Level-2b anomaly gate (binary Grass vs Anomaly) with GL-TAE."""

from __future__ import annotations

import argparse
import os

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from agricultureai.data.breizhcrops_l2b_anomaly_gate import (
    L2B_ID_TO_NAME,
    NUM_L2B_CLASSES,
    build_l2b_anomaly_gate_loaders,
)
from agricultureai.evaluation.transformer import MAX_SEQ_LEN as DEFAULT_MAX_SEQ_LEN
from agricultureai.models.breizhcrops_gltae import BreizhCropsGLTAEExpert
from agricultureai.training.checkpoints import load_checkpoint, model_state_dict_from_checkpoint
from agricultureai.training.losses import FocalLoss
from agricultureai.training.series_trainer import SeriesExpertTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train L2b anomaly gate (Grass vs Anomaly).")
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
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--conv-kernel", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints/l2b_anomaly_gate")
    parser.add_argument("--resume", default="")
    parser.add_argument("--no-cosine-lr", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    channels = 13 if args.level == "L1C" else 10

    train_loader, val_loader = build_l2b_anomaly_gate_loaders(
        train_regions=args.regions,
        val_regions=args.val_regions,
        level=args.level,
        max_samples=int(args.max_samples),
        batch_size=int(args.batch_size),
        max_seq_len=int(args.max_seq_len),
        use_train_region_holdout=bool(args.train_region_holdout),
        seed=int(args.seed),
    )

    architecture_config = {
        "model_type": "gltae",
        "role": "l2b_anomaly_gate",
        "num_classes": NUM_L2B_CLASSES,
        "seq_len": int(args.max_seq_len),
        "input_dim": channels,
        "d_model": int(args.d_model),
        "n_heads": int(args.n_heads),
        "n_layers": int(args.n_layers),
        "dim_feedforward": int(args.dim_feedforward),
        "conv_kernel": int(args.conv_kernel),
        "dropout": float(args.dropout),
        "doy_positional_encoding": True,
        "masked_pooling": True,
        "weighted_anomaly_sampling": True,
    }
    model = BreizhCropsGLTAEExpert(
        num_classes=NUM_L2B_CLASSES,
        input_dim=channels,
        sequencelength=int(args.max_seq_len),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        dim_feedforward=int(args.dim_feedforward),
        conv_kernel_size=int(args.conv_kernel),
        dropout=float(args.dropout),
        device=device,
    )
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(model_state_dict_from_checkpoint(ckpt), strict=True)

    criterion = FocalLoss(gamma=float(args.focal_gamma))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = None if args.no_cosine_lr else CosineAnnealingLR(optimizer, T_max=max(1, int(args.epochs)))

    run_config = {
        "regions": list(args.regions),
        "val_regions": list(args.val_regions),
        "level": args.level,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "class_names": L2B_ID_TO_NAME,
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
    )


if __name__ == "__main__":
    main()

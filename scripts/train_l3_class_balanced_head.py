"""Train a frozen-backbone, class-balanced linear head for Level-3 routing.

Option #4 from the L3 architecture review: freeze the production ProtoNet
encoder and train only a 5-way linear classifier with inverse-frequency
class weights (plus optional orchard boost).
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from agricultureai.data.episodic_sampler import (
    LEVEL3_ID_TO_NAME,
    NUM_LEVEL3_CLASSES,
    Level3FewShotDataset,
)
from agricultureai.evaluation.transformer import MAX_SEQ_LEN
from agricultureai.models.proto_net import Level3LinearExpert, PrototypicalNetwork
from agricultureai.training.checkpoints import (
    architecture_config_from_checkpoint,
    build_checkpoint_payload,
    load_checkpoint,
    model_state_dict_from_checkpoint,
)


def _config_value(cfg: dict, key: str, default):
    return cfg.get(key, default) if isinstance(cfg, dict) else default


def load_frozen_protonet(
    path: Path, device: torch.device, *, channels: int
) -> PrototypicalNetwork:
    checkpoint = load_checkpoint(path, map_location=device)
    cfg = architecture_config_from_checkpoint(checkpoint)
    model = PrototypicalNetwork(
        in_channels=int(_config_value(cfg, "in_channels", channels)),
        hidden_dim=int(_config_value(cfg, "hidden_dim", 64)),
        encoder=str(_config_value(cfg, "encoder", "cnn")).lower(),
        device=device,
    )
    model.load_state_dict(model_state_dict_from_checkpoint(checkpoint), strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_level3_loaders(
    *,
    train_regions: list[str],
    val_regions: list[str],
    level: str,
    max_samples: int,
    max_seq_len: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, Level3FewShotDataset, Level3FewShotDataset]:
    train_ds = Level3FewShotDataset(
        regions=train_regions,
        level=level,
        max_samples=max_samples,
        max_seq_len=max_seq_len,
    )
    val_ds = Level3FewShotDataset(
        regions=val_regions,
        level=level,
        max_samples=max_samples,
        max_seq_len=max_seq_len,
    )

    def collate(batch):
        xs, ys = zip(*batch)
        return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_ds, val_ds


def compute_balanced_class_weights(
    labels: list[int],
    *,
    num_classes: int,
    orchard_class: int = 2,
    orchard_multiplier: float = 1.0,
) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / (float(num_classes) * counts)
    if orchard_multiplier != 1.0:
        weights[int(orchard_class)] *= float(orchard_multiplier)
    return weights


@torch.no_grad()
def evaluate(model: Level3LinearExpert, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        preds = model(x).argmax(dim=1)
        correct += int(preds.eq(y).sum().item())
        total += y.size(0)
    return 100.0 * correct / max(total, 1)


def train_epoch(
    model: Level3LinearExpert,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    model.encoder.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        correct += int(logits.argmax(dim=1).eq(y).sum().item())
        total += y.size(0)
    return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train frozen-backbone Level-3 linear head with class-balanced loss."
    )
    parser.add_argument("--regions", nargs="+", default=["frh01", "frh02", "frh03"])
    parser.add_argument("--val-regions", nargs="+", default=["frh04"])
    parser.add_argument("--level", default="L2A")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--orchard-multiplier",
        type=float,
        default=3.0,
        help="Extra weight on orchard class (local id 2) atop inverse-frequency.",
    )
    parser.add_argument(
        "--init-weights",
        type=Path,
        default=Path("artifacts/checkpoints/level3_protonet/best.pt"),
        help="Production ProtoNet encoder checkpoint (frozen).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts/checkpoints/level3_linear_head"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    channels = 13 if args.level.upper() == "L1C" else 10

    print(
        "Level-3 classes: "
        + ", ".join(f"{cid}={name}" for cid, name in LEVEL3_ID_TO_NAME.items())
    )

    train_loader, val_loader, train_ds, val_ds = build_level3_loaders(
        train_regions=list(args.regions),
        val_regions=list(args.val_regions),
        level=args.level,
        max_samples=int(args.max_samples),
        max_seq_len=int(args.max_seq_len),
        batch_size=int(args.batch_size),
    )
    print(f"Train samples: {len(train_ds):,}  |  Val samples: {len(val_ds):,}")

    encoder = load_frozen_protonet(args.init_weights.resolve(), device, channels=channels)
    model = Level3LinearExpert(encoder, num_classes=NUM_LEVEL3_CLASSES).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (linear head only): {trainable:,}")

    class_weights = compute_balanced_class_weights(
        train_ds.labels,
        num_classes=NUM_LEVEL3_CLASSES,
        orchard_multiplier=float(args.orchard_multiplier),
    ).to(device)
    print(
        "Class weights: "
        + ", ".join(
            f"{LEVEL3_ID_TO_NAME[i]}={class_weights[i].item():.3f}"
            for i in range(NUM_LEVEL3_CLASSES)
        )
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(int(args.epochs), 1))

    out_dir = args.checkpoint_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "train_metrics.csv"

    best_val = -1.0
    best_epoch = 0
    for epoch in range(int(args.epochs)):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] loss={train_loss:.4f} "
            f"train_acc={train_acc:.2f}% val_acc={val_acc:.2f}% lr={lr:.2e}"
        )

        write_header = not metrics_csv.is_file()
        with metrics_csv.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "lr"])
            writer.writerow([epoch + 1, train_loss, train_acc, val_acc, lr])

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch + 1
            payload = build_checkpoint_payload(
                model,
                architecture_config={
                    "model_type": "level3_linear_head",
                    "stage": "level3_frozen_backbone_balanced_head",
                    "encoder": model.encoder.encoder_name,
                    "in_channels": channels,
                    "hidden_dim": model.encoder.hidden_dim,
                    "num_classes": NUM_LEVEL3_CLASSES,
                    "init_encoder": str(args.init_weights.resolve()),
                    "orchard_multiplier": float(args.orchard_multiplier),
                    "level": args.level,
                },
                training_metadata={
                    "epochs": int(args.epochs),
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val,
                    "eval_set": "+".join(args.val_regions),
                    "train_regions": "+".join(args.regions),
                },
            )
            best_path = out_dir / "best.pt"
            torch.save(payload, best_path)
            prod_dir = Path("artifacts/checkpoints_frozen_backup/level3_linear_head_production")
            prod_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_path, prod_dir / f"best_epoch{best_epoch:02d}.pt")
            shutil.copy2(best_path, prod_dir / "best.pt")
            print(f"  -> Saved new best checkpoint (val_acc={val_acc:.2f}%)")

        scheduler.step()

    print(f"\nBest val accuracy: {best_val:.2f}% at epoch {best_epoch}")


if __name__ == "__main__":
    main()

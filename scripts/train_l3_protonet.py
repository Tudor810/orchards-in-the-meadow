"""Train the Level-3 ProtoNet with balanced, augmented episodes.

This is the rare-class rescuer for the DMOE pipeline. Episodic Prototypical
training with spectral/temporal augmentation is the regime that recovered
orchard recall on L1C; this script reproduces it for native 10-channel L2A.
Only ``best.pt`` and ``latest.pt`` are written (no per-epoch checkpoints).
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import torch

from agricultureai.data.episodic_sampler import (
    LEVEL3_ANOMALY_CLASSES,
    LEVEL3_RAW_IDS,
    NUM_LEVEL3_CLASSES,
    Level3FewShotDataset,
    build_episodic_dataloader,
)
from agricultureai.evaluation.transformer import MAX_SEQ_LEN
from agricultureai.models.proto_net import PrototypicalNetwork
from agricultureai.training.checkpoints import build_checkpoint_payload


def augment_episode(
    x: torch.Tensor,
    labels: torch.Tensor,
    *,
    augment_prob: float,
    anomaly_augment_prob: float,
    jitter_std: float,
    anomaly_jitter_std: float,
    gain_max_delta: float,
    time_mask_max_ratio: float,
    max_time_shift: int,
    channel_drop_prob: float,
) -> torch.Tensor:
    """Apply independent spectral/temporal perturbations to episode samples.

    Rare anomaly classes (orchards/nut trees/sunflowers) are augmented more
    aggressively so the encoder sees varied views of the few real examples.
    """
    out = x.clone()
    anomaly_classes = torch.tensor(
        LEVEL3_ANOMALY_CLASSES, device=labels.device, dtype=labels.dtype
    )
    is_anomaly = (labels[:, None] == anomaly_classes[None, :]).any(dim=1)
    probabilities = torch.where(
        is_anomaly,
        torch.full_like(labels, anomaly_augment_prob, dtype=torch.float32),
        torch.full_like(labels, augment_prob, dtype=torch.float32),
    )
    selected = torch.rand(labels.size(0), device=labels.device) < probabilities
    for row in selected.nonzero(as_tuple=False).flatten().tolist():
        std = anomaly_jitter_std if bool(is_anomaly[row]) else jitter_std
        out[row] += torch.randn_like(out[row]) * std
        gain = 1.0 + (2.0 * torch.rand(1, device=out.device) - 1.0) * gain_max_delta
        out[row] *= gain

        if max_time_shift > 0:
            shift = int(
                torch.randint(
                    -max_time_shift, max_time_shift + 1, (1,), device=out.device
                ).item()
            )
            out[row] = torch.roll(out[row], shifts=shift, dims=0)

        max_mask = int(round(out.size(1) * time_mask_max_ratio))
        if max_mask > 0:
            width = int(torch.randint(0, max_mask + 1, (1,), device=out.device).item())
            if width:
                start = int(
                    torch.randint(0, out.size(1) - width + 1, (1,), device=out.device).item()
                )
                out[row, start : start + width] = 0

        dropped = torch.rand(out.size(2), device=out.device) < channel_drop_prob
        if dropped.any():
            out[row, :, dropped] = 0
    return out


def run_episodes(
    model: PrototypicalNetwork,
    loader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    augmentation: dict | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_accuracy = 0.0
    for episode, (support_x, support_y, query_x, query_y) in enumerate(loader, 1):
        support_x = support_x.to(device, non_blocking=True)
        support_y = support_y.to(device, non_blocking=True)
        query_x = query_x.to(device, non_blocking=True)
        query_y = query_y.to(device, non_blocking=True)
        if training and augmentation is not None:
            support_x = augment_episode(support_x, support_y, **augmentation)
            query_x = augment_episode(query_x, query_y, **augmentation)

        # n_way is inferred per-episode so validation on frh04 (which lacks a
        # usable sunflower class) can run 4-way while training stays 5-way.
        n_way = int(support_y.max().item()) + 1
        if optimizer is not None:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            output = model.proto_episode(support_x, support_y, query_x, query_y, n_way)
            if optimizer is not None:
                output.loss.backward()
                optimizer.step()
        total_loss += float(output.loss.item())
        total_accuracy += float(output.accuracy.item() * 100.0)
        if training and episode % 50 == 0:
            print(
                f"  episode {episode}/{len(loader)} "
                f"loss={total_loss / episode:.4f} acc={total_accuracy / episode:.2f}%",
                flush=True,
            )
    count = max(len(loader), 1)
    return total_loss / count, total_accuracy / count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", nargs="+", default=["frh01", "frh02", "frh03"])
    parser.add_argument("--val-regions", nargs="+", default=["frh04"])
    parser.add_argument("--level", default="L2A")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--episodes-per-epoch", type=int, default=300)
    parser.add_argument("--val-episodes", type=int, default=100)
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--q-query", type=int, default=12)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--encoder", choices=("cnn", "gru"), default="cnn")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts/checkpoints/level3_protonet"),
    )
    parser.add_argument("--augment-prob", type=float, default=0.3)
    parser.add_argument("--anomaly-augment-prob", type=float, default=0.7)
    parser.add_argument("--jitter-std", type=float, default=0.01)
    parser.add_argument("--anomaly-jitter-std", type=float, default=0.02)
    parser.add_argument("--gain-max-delta", type=float, default=0.15)
    parser.add_argument("--time-mask-max-ratio", type=float, default=0.08)
    parser.add_argument("--max-time-shift", type=int, default=2)
    parser.add_argument("--channel-drop-prob", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_way != NUM_LEVEL3_CLASSES:
        raise SystemExit(
            f"Production Level-3 training requires n_way={NUM_LEVEL3_CLASSES}."
        )
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = 13 if args.level.upper() == "L1C" else 10
    print(f"Device: {device}; level={args.level}; channels={channels}")

    train_dataset = Level3FewShotDataset(
        regions=args.regions,
        level=args.level,
        max_samples=args.max_samples,
        max_seq_len=args.max_seq_len,
    )
    val_dataset = Level3FewShotDataset(
        regions=args.val_regions,
        level=args.level,
        max_samples=args.max_samples,
        max_seq_len=args.max_seq_len,
    )
    train_loader, train_sampler = build_episodic_dataloader(
        train_dataset,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        num_episodes=args.episodes_per_epoch,
        seed=args.seed,
    )

    # frh04 lacks a usable sunflower support pool, so validation runs with the
    # number of classes that actually qualify for k_shot+1 samples.
    val_counts = Counter(val_dataset.labels)
    val_n_way = min(
        args.n_way,
        sum(count >= args.k_shot + 1 for count in val_counts.values()),
    )
    if val_n_way < 2:
        raise RuntimeError(
            f"Validation data has only {val_n_way} episode-eligible classes: "
            f"{dict(sorted(val_counts.items()))}"
        )
    if val_n_way != args.n_way:
        print(
            f"Validation uses {val_n_way}-way episodes because "
            f"{'+'.join(args.val_regions)} has only {val_n_way} eligible classes."
        )
    print(
        f"Train samples={len(train_dataset):,}; val samples={len(val_dataset):,}; "
        f"episode={train_sampler.n_way}-way {train_sampler.k_shot}-shot "
        f"+ {train_sampler.q_query} query"
    )

    model = PrototypicalNetwork(
        in_channels=channels,
        hidden_dim=args.hidden_dim,
        encoder=args.encoder,
        dropout=args.dropout,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    augmentation = {
        "augment_prob": args.augment_prob,
        "anomaly_augment_prob": args.anomaly_augment_prob,
        "jitter_std": args.jitter_std,
        "anomaly_jitter_std": args.anomaly_jitter_std,
        "gain_max_delta": args.gain_max_delta,
        "time_mask_max_ratio": args.time_mask_max_ratio,
        "max_time_shift": args.max_time_shift,
        "channel_drop_prob": args.channel_drop_prob,
    }

    out_dir = args.checkpoint_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "train_metrics.csv"
    with metrics_path.open("w", newline="") as metrics_file:
        csv.writer(metrics_file).writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
        )

    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_episodes(
            model, train_loader, device, optimizer=optimizer, augmentation=augmentation
        )
        val_loader, _ = build_episodic_dataloader(
            val_dataset,
            n_way=val_n_way,
            k_shot=args.k_shot,
            q_query=args.q_query,
            num_episodes=args.val_episodes,
            seed=args.seed + 10_000,
        )
        val_loss, val_accuracy = run_episodes(
            model, val_loader, device, optimizer=None, augmentation=None
        )
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch}/{args.epochs}] train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.2f}% val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.2f}% lr={lr:.2e}",
            flush=True,
        )
        with metrics_path.open("a", newline="") as metrics_file:
            csv.writer(metrics_file).writerow(
                [epoch, train_loss, train_accuracy, val_loss, val_accuracy, lr]
            )

        payload = build_checkpoint_payload(
            model,
            architecture_config={
                "model_type": "protonet",
                "stage": "level3_few_shot_rescuer",
                "encoder": args.encoder,
                "in_channels": channels,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "seq_len": args.max_seq_len,
                "n_way": args.n_way,
                "k_shot": args.k_shot,
                "q_query": train_sampler.q_query,
                "level": args.level,
                "raw_target_ids": LEVEL3_RAW_IDS,
                "augmentation": {"enabled": True, **augmentation},
            },
            training_metadata={
                "epochs": args.epochs,
                "episodes_per_epoch": args.episodes_per_epoch,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "eval_set": "+".join(args.val_regions),
            },
        )
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["lr_scheduler_state_dict"] = scheduler.state_dict()
        torch.save(payload, out_dir / "latest.pt")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            payload["training_metadata"]["best_epoch"] = epoch
            payload["training_metadata"]["best_val_accuracy"] = best_accuracy
            torch.save(payload, out_dir / "best.pt")
            print(f"  -> New best episodic val accuracy: {best_accuracy:.2f}%")
        scheduler.step()

    print(f"Best episodic validation accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()

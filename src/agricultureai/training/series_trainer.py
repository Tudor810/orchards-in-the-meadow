"""Training loop for the day-of-year-aware cascade experts (L2b anomaly gate, L2c meadow expert).

Batches are ``(x, doy, padding_mask, target, _)`` as produced by ``pad_collate_breizhcrops_series``;
the model is called as ``model(x, doy=doy, padding_mask=padding_mask)``.
"""

from __future__ import annotations

import csv
import os

import torch

from agricultureai.training.checkpoints import build_checkpoint_payload


class SeriesExpertTrainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        *,
        eval_set_name: str,
        architecture_config: dict,
        run_config: dict,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.eval_set_name = eval_set_name
        self.architecture_config = architecture_config
        self.run_config = run_config

    def _payload(self, metadata: dict) -> dict:
        return build_checkpoint_payload(
            self.model,
            architecture_config=self.architecture_config,
            training_metadata={**self.run_config, **metadata, "eval_set": self.eval_set_name},
        )

    def _to_device(self, batch):
        x, doy, padding_mask, targets, _ = batch
        return (
            x.to(self.device, non_blocking=True),
            doy.to(self.device, non_blocking=True),
            padding_mask.to(self.device, non_blocking=True),
            targets.to(self.device, non_blocking=True).long(),
        )

    def train_epoch(self) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx % 5 == 0:
                print(f"  -> Processed batch {batch_idx}/{len(self.train_loader)}")
            x, doy, padding_mask, targets = self._to_device(batch)
            self.optimizer.zero_grad()
            logits = self.model(x, doy=doy, padding_mask=padding_mask)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())
            correct += int(logits.argmax(dim=1).eq(targets).sum().item())
            total += targets.size(0)
        return total_loss / max(1, len(self.train_loader)), 100.0 * correct / max(total, 1)

    def evaluate(self) -> float:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.test_loader:
                x, doy, padding_mask, targets = self._to_device(batch)
                logits = self.model(x, doy=doy, padding_mask=padding_mask)
                correct += int(logits.argmax(dim=1).eq(targets).sum().item())
                total += targets.size(0)
        return 100.0 * correct / max(total, 1)

    def fit(
        self,
        epochs: int,
        *,
        checkpoint_dir: str | None,
        lr_scheduler=None,
        metrics_csv_path: str | None = None,
        initial_best_val: float = -1.0,
        epoch_offset: int = 0,
    ) -> None:
        """Train for ``epochs``; ``initial_best_val`` and ``epoch_offset`` continue a resumed run."""
        best_val = float(initial_best_val)
        best_epoch = int(epoch_offset) if best_val >= 0 else 0
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        for epoch in range(epochs):
            global_epoch = int(epoch_offset) + epoch + 1
            train_loss, train_acc = self.train_epoch()
            val_acc = self.evaluate()
            lr = self.optimizer.param_groups[0]["lr"]
            resumed = f" (global {global_epoch})" if epoch_offset else ""
            print(
                f"Epoch [{epoch + 1}/{epochs}]{resumed} loss={train_loss:.4f} "
                f"train_acc={train_acc:.2f}% val_acc={val_acc:.2f}% lr={lr:.2e}"
            )
            if metrics_csv_path:
                write_header = not os.path.isfile(metrics_csv_path)
                with open(metrics_csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(["epoch", "train_loss", "train_acc", "val_acc", "eval_set", "lr"])
                    w.writerow([global_epoch, train_loss, train_acc, val_acc, self.eval_set_name, lr])
            if checkpoint_dir:
                meta = {
                    "epoch": global_epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "learning_rate": lr,
                }
                torch.save(self._payload(meta), os.path.join(checkpoint_dir, "latest.pt"))
                if val_acc > best_val:
                    best_val = val_acc
                    best_epoch = global_epoch
                    torch.save(
                        self._payload({**meta, "best_epoch": best_epoch}),
                        os.path.join(checkpoint_dir, "best.pt"),
                    )
                    print(f"  -> New best val acc {best_val:.2f}% (saved best.pt)")
            if lr_scheduler is not None:
                lr_scheduler.step()

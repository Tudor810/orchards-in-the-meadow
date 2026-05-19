import csv
import os
import time

import torch


class ModelTrainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        target_ids,
        *,
        eval_set_name: str = "validation",
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.target_ids = target_ids
        self.eval_set_name = eval_set_name

    def _filter_and_remap(self, inputs, targets):
        """Isolates target crops and remaps their IDs to 0, 1, 2..."""
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for target_id in self.target_ids:
            mask |= targets == target_id

        filtered_inputs = inputs[mask]
        filtered_targets = targets[mask]

        remapped_targets = torch.zeros_like(filtered_targets)
        for new_id, original_id in enumerate(self.target_ids):
            remapped_targets[filtered_targets == original_id] = new_id

        return filtered_inputs, remapped_targets

    @staticmethod
    def _unpack_batch(batch):
        inputs, targets, _ = batch[:3]
        return inputs, targets

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx % 5 == 0:
                print(f"  -> Processed batch {batch_idx}/{len(self.train_loader)}")

            inputs, targets = self._unpack_batch(batch)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs, targets = self._filter_and_remap(inputs, targets)

            if targets.size(0) == 0:
                continue

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        acc = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / max(1, len(self.train_loader))
        return avg_loss, acc

    def evaluate(self, per_class: bool = False):
        self.model.eval()
        correct = 0
        total = 0

        num_classes = len(self.target_ids)
        correct_per_class = torch.zeros(num_classes, dtype=torch.long)
        total_per_class = torch.zeros(num_classes, dtype=torch.long)
        confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

        n_test = len(self.test_loader)
        print(
            f"  -> Evaluating on {self.eval_set_name} set ({n_test} batches, "
            f"no output until next line if this is slow)..."
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                if batch_idx % 2000 == 0:
                    print(f"  -> Test batch {batch_idx}/{n_test}")

                inputs, targets = self._unpack_batch(batch)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs, targets = self._filter_and_remap(inputs, targets)

                if targets.size(0) == 0:
                    continue

                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if per_class:
                    for c in range(num_classes):
                        mask = targets == c
                        if mask.any():
                            total_per_class[c] += mask.sum().item()
                            correct_per_class[c] += (predicted[mask] == c).sum().item()

                    for t, p in zip(targets.view(-1), predicted.view(-1)):
                        confusion[int(t), int(p)] += 1

        overall = 100.0 * correct / total if total > 0 else 0.0
        if not per_class:
            return overall

        per_class_acc = [
            (100.0 * int(correct_per_class[c].item()) / int(total_per_class[c].item()))
            if int(total_per_class[c].item()) > 0
            else 0.0
            for c in range(num_classes)
        ]
        return overall, per_class_acc, confusion.cpu()

    def fit(self, epochs, save_path=None, checkpoint_dir=None, lr_scheduler=None, metrics_csv_path=None):
        print(f"Starting training for {epochs} epochs on {self.device}...")
        print(f"Targeting specific crop IDs: {self.target_ids}")

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Per-epoch checkpoints enabled: {checkpoint_dir}")

        best_test_acc = -1.0
        best_epoch = 0

        for epoch in range(epochs):
            start_time = time.time()
            train_loss, train_acc = self.train_epoch()
            test_acc = self.evaluate()
            epoch_time = time.time() - start_time
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Acc ({self.eval_set_name}): {test_acc:.2f}% | "
                f"LR: {lr:.2e}"
            )

            if metrics_csv_path:
                parent = os.path.dirname(os.path.abspath(metrics_csv_path))
                if parent:
                    os.makedirs(parent, exist_ok=True)
                write_header = not os.path.isfile(metrics_csv_path)
                with open(metrics_csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(["epoch", "train_loss", "train_acc", "val_acc", "eval_set", "lr"])
                    w.writerow([
                        epoch + 1,
                        f"{train_loss:.6f}",
                        f"{train_acc:.6f}",
                        f"{test_acc:.6f}",
                        self.eval_set_name,
                        f"{lr:.8e}",
                    ])

            if checkpoint_dir:
                epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1:03d}.pt")
                latest_path = os.path.join(checkpoint_dir, "latest.pt")
                torch.save(self.model.state_dict(), epoch_path)
                torch.save(self.model.state_dict(), latest_path)
                print(f"  -> Saved checkpoint: {epoch_path}")

                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_epoch = epoch + 1
                    best_path = os.path.join(checkpoint_dir, "best.pt")
                    torch.save(self.model.state_dict(), best_path)
                    print(f"  -> New best val acc on {self.eval_set_name} {best_test_acc:.4f}% (saved {best_path})")

            if lr_scheduler is not None:
                lr_scheduler.step()

        if checkpoint_dir and best_epoch:
            print(f"Best run: epoch {best_epoch}  val_acc ({self.eval_set_name})={best_test_acc:.4f}%")

        if save_path:
            parent = os.path.dirname(os.path.abspath(save_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            print(f"Model successfully saved to {save_path}")

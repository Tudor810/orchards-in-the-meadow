import torch
import time
import os


class ModelTrainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device, target_ids):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.target_ids = target_ids

    def _filter_and_remap(self, inputs, targets):
        """Isolates target crops and remaps their IDs to 0, 1, 2..."""
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for target_id in self.target_ids:
            mask |= (targets == target_id)

        filtered_inputs = inputs[mask]
        filtered_targets = targets[mask]

        remapped_targets = torch.zeros_like(filtered_targets)
        for new_id, original_id in enumerate(self.target_ids):
            remapped_targets[filtered_targets == original_id] = new_id

        return filtered_inputs, remapped_targets

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):

            if batch_idx % 5 == 0:
                print(f"  -> Processed batch {batch_idx}/{len(self.train_loader)}")

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

        acc = 100. * correct / total if total > 0 else 0.0
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

        with torch.no_grad():
            for inputs, targets, _ in self.test_loader:
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

        overall = 100. * correct / total if total > 0 else 0.0
        if not per_class:
            return overall

        per_class_acc = [
            (100.0 * int(correct_per_class[c].item()) / int(total_per_class[c].item()))
            if int(total_per_class[c].item()) > 0
            else 0.0
            for c in range(num_classes)
        ]
        return overall, per_class_acc, confusion.cpu()

    def fit(self, epochs, save_path=None):
        print(f"Starting training for {epochs} epochs on {self.device}...")
        print(f"Targeting specific crop IDs: {self.target_ids}")

        for epoch in range(epochs):
            start_time = time.time()
            train_loss, train_acc = self.train_epoch()
            test_acc = self.evaluate()
            epoch_time = time.time() - start_time

            print(f"Epoch [{epoch + 1}/{epochs}] "
                  f"Time: {epoch_time:.2f}s | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}%")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            print(f"Model successfully saved to {save_path}")
"""Tests for the shared L2b/L2c training loop on a tiny synthetic problem."""

from __future__ import annotations

import csv

import torch
from torch import nn

from agricultureai.training.series_trainer import SeriesExpertTrainer


class _MeanPool(nn.Module):
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x, doy=None, padding_mask=None):
        keep = (~padding_mask).unsqueeze(-1).float()
        return self.head((x * keep).sum(1) / keep.sum(1).clamp(min=1.0))


def _batches(n_batches: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_batches):
        y = torch.randint(0, 2, (16,), generator=g)
        x = torch.randn(16, 10, 4, generator=g) + 2.0 * y.view(-1, 1, 1).float()
        doy = torch.arange(10).repeat(16, 1)
        mask = torch.zeros(16, 10, dtype=torch.bool)
        mask[:, 8:] = True
        out.append((x, doy, mask, y, None))
    return out


def _trainer(model):
    return SeriesExpertTrainer(
        model, _batches(4, 0), _batches(2, 1), nn.CrossEntropyLoss(),
        torch.optim.Adam(model.parameters(), lr=0.05), torch.device("cpu"),
        eval_set_name="synthetic", architecture_config={"model_type": "test"}, run_config={"lr": 0.05},
    )


def test_fit_learns_and_writes_checkpoints_and_log(tmp_path):
    torch.manual_seed(0)
    trainer = _trainer(_MeanPool(4, 2))
    log = tmp_path / "metrics.csv"
    trainer.fit(3, checkpoint_dir=str(tmp_path / "ckpt"), metrics_csv_path=str(log))
    assert trainer.evaluate() > 90.0
    rows = list(csv.reader(log.open()))
    assert rows[0] == ["epoch", "train_loss", "train_acc", "val_acc", "eval_set", "lr"]
    assert [r[0] for r in rows[1:]] == ["1", "2", "3"]
    best = torch.load(tmp_path / "ckpt" / "best.pt", weights_only=False)
    latest = torch.load(tmp_path / "ckpt" / "latest.pt", weights_only=False)
    assert latest["training_metadata"]["epoch"] == 3
    assert best["training_metadata"]["eval_set"] == "synthetic"
    assert best["training_metadata"]["val_accuracy"] == max(float(r[3]) for r in rows[1:])


def test_resume_continues_epoch_numbers_and_keeps_better_best(tmp_path):
    torch.manual_seed(0)
    trainer = _trainer(_MeanPool(4, 2))
    log = tmp_path / "metrics.csv"
    trainer.fit(2, checkpoint_dir=str(tmp_path), metrics_csv_path=str(log), initial_best_val=100.1, epoch_offset=5)
    assert [r[0] for r in list(csv.reader(log.open()))[1:]] == ["6", "7"]
    assert not (tmp_path / "best.pt").exists()  # nothing beat the resumed best
    assert torch.load(tmp_path / "latest.pt", weights_only=False)["training_metadata"]["epoch"] == 7

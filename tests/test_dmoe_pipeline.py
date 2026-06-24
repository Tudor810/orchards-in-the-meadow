import torch
import torch.nn as nn

from agricultureai.models.proto_net import PrototypicalNetwork
from agricultureai.pipeline import (
    GLOBAL_CORN,
    GLOBAL_RAPESEED,
    SUPER_CEREAL_GROUP,
    SUPER_CORN,
    SUPER_GRASSLAND_GROUP,
    SUPER_RAPESEED,
    DMOEPipeline,
)

SEQ_LEN, CHANNELS = 12, 13


class _ConstLogits(nn.Module):
    """Stub model returning a fixed logit row broadcast over the batch."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("row", torch.tensor(logits, dtype=torch.float32))

    def forward(self, x, *_, **__):
        return self.row.unsqueeze(0).expand(x.size(0), -1).clone()


class _RowRouter(nn.Module):
    """Stub L1 that maps a sentinel value in x[:, 0, 0] to a chosen super-class."""

    def forward(self, x):
        super_ids = x[:, 0, 0].long().clamp(0, 3)
        logits = torch.full((x.size(0), 4), -10.0)
        logits[torch.arange(x.size(0)), super_ids] = 10.0
        return logits


def _default_l2c():
    # Permanent meadow local 0 -> global 7.
    return _ConstLogits([10.0, -10.0])


def _default_l2b():
    # Grass branch dominates; anomaly suppressed.
    return _ConstLogits([5.0, -5.0])


def _make_pipeline(l1, l2a, l2b, l2c=None, *, hidden_dim=8, threshold=0.20):
    l3 = PrototypicalNetwork(in_channels=CHANNELS, hidden_dim=hidden_dim, encoder="cnn")
    protos = torch.randn(5, hidden_dim)
    return DMOEPipeline(
        l1,
        l2a,
        l2b,
        l2c if l2c is not None else _default_l2c(),
        l3,
        support_prototypes=protos,
        anomaly_threshold=threshold,
    )


def test_rapeseed_and_corn_translate_to_global_raw_ids():
    l1 = _RowRouter()
    l2a = _ConstLogits([10.0, -10.0])
    pipe = _make_pipeline(l1, l2a, _default_l2b())

    x = torch.zeros(2, SEQ_LEN, CHANNELS)
    x[0, 0, 0] = SUPER_RAPESEED
    x[1, 0, 0] = SUPER_CORN

    preds = pipe(x)
    assert preds.tolist() == [GLOBAL_RAPESEED, GLOBAL_CORN]


def test_cereal_branch_maps_local_to_global():
    l1 = _RowRouter()
    l2a = _ConstLogits([-10.0, 10.0])
    pipe = _make_pipeline(l1, l2a, _default_l2b())

    x = torch.zeros(3, SEQ_LEN, CHANNELS)
    x[:, 0, 0] = SUPER_CEREAL_GROUP
    preds = pipe(x)
    assert preds.tolist() == [1, 1, 1]


def test_grassland_below_threshold_resolves_to_meadow():
    l1 = _RowRouter()
    l2a = _ConstLogits([10.0, -10.0])
    l2b = _ConstLogits([5.0, -5.0])          # grass branch
    l2c = _ConstLogits([-5.0, 5.0])          # temporary meadow -> global 8
    pipe = _make_pipeline(l1, l2a, l2b, l2c, threshold=0.20)

    x = torch.zeros(4, SEQ_LEN, CHANNELS)
    x[:, 0, 0] = SUPER_GRASSLAND_GROUP
    preds = pipe(x)
    assert preds.tolist() == [8, 8, 8, 8]
    assert pipe.last_l3_count == 0


def test_grassland_above_threshold_routes_to_l3():
    l1 = _RowRouter()
    l2a = _ConstLogits([10.0, -10.0])
    l2b = _ConstLogits([-5.0, 10.0])         # anomaly branch
    pipe = _make_pipeline(l1, l2a, l2b, threshold=0.20)

    x = torch.zeros(6, SEQ_LEN, CHANNELS)
    x[:, 0, 0] = SUPER_GRASSLAND_GROUP
    pipe(x)

    assert pipe.last_l3_count == 6


def test_mixed_batch_preserves_original_positions():
    l1 = _RowRouter()
    l2a = _ConstLogits([-10.0, 10.0])
    l2b = _ConstLogits([5.0, -5.0])
    l2c = _ConstLogits([-5.0, 5.0])
    pipe = _make_pipeline(l1, l2a, l2b, l2c)

    x = torch.zeros(4, SEQ_LEN, CHANNELS)
    x[0, 0, 0] = SUPER_CORN
    x[1, 0, 0] = SUPER_CEREAL_GROUP
    x[2, 0, 0] = SUPER_RAPESEED
    x[3, 0, 0] = SUPER_GRASSLAND_GROUP

    preds = pipe(x)
    assert preds.tolist() == [3, 1, 2, 8]
    assert preds.shape == (4,)


def test_pipeline_runs_under_no_grad_and_returns_long():
    l1 = _RowRouter()
    l2a = _ConstLogits([10.0, -10.0])
    pipe = _make_pipeline(l1, l2a, _default_l2b())

    x = torch.randn(5, SEQ_LEN, CHANNELS)
    x[:, 0, 0] = SUPER_GRASSLAND_GROUP
    preds = pipe(x)
    assert preds.dtype == torch.long
    assert not preds.requires_grad

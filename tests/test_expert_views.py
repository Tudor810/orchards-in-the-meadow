"""Tests for resampled expert views (L2a / L3 fed like their stock-transform training)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from agricultureai.data.breizhcrops_series import (
    BreizhCropsParcelDataset,
    BreizhCropsParcelViewsDataset,
    pad_collate_with_views,
)
from agricultureai.pipeline import (
    GLOBAL_BARLEY,
    GLOBAL_WHEAT,
    SUPER_CEREAL_GROUP,
    SUPER_GRASSLAND_GROUP,
    DMOEPipeline,
)

T, C = 12, 10


class _Router(nn.Module):
    """L1 stub: super-class from the sentinel in ``x[:, 0, 0]``."""

    def forward(self, x):
        logits = torch.full((x.size(0), 4), -10.0)
        logits[torch.arange(x.size(0)), x[:, 0, 0].long().clamp(0, 3)] = 10.0
        return logits


class _Const(nn.Module):
    def __init__(self, row):
        super().__init__()
        self.register_buffer("row", torch.tensor(row, dtype=torch.float32))

    def forward(self, x, *_, **__):
        return self.row.expand(x.size(0), -1).clone()


class _MeanChannelCereal(nn.Module):
    """L2a stub: wheat logit = mean of channel 1, barley = its negative."""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.clone())
        s = x[:, :, 1].mean(dim=1)
        return torch.stack([-s, s], dim=1)


class _RecordingL3(nn.Module):
    """L3 stub exposing ``pseudo_distances`` (like the linear head)."""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def pseudo_distances(self, x):
        self.inputs.append(x.clone())
        return x[:, :, 2].mean(dim=1, keepdim=True) * torch.arange(1.0, 6.0)


def _pipeline(l2a=None, l2b=None, l3=None):
    return DMOEPipeline(
        _Router(),
        l2a or _MeanChannelCereal(),
        l2b or _Const([-5.0, 10.0]),  # anomaly branch -> L3 runs
        _Const([10.0, -10.0]),
        l3 or _RecordingL3(),
        support_prototypes=torch.zeros(5, 4),
    )


def test_single_identical_view_reproduces_legacy_probabilities():
    torch.manual_seed(0)
    x = torch.randn(6, T, C)
    x[:3, 0, 0] = SUPER_CEREAL_GROUP
    x[3:, 0, 0] = SUPER_GRASSLAND_GROUP
    pipe = _pipeline()
    legacy = pipe.forward_probs(x)
    with_views = pipe.forward_probs(x, expert_views=x.unsqueeze(1))
    assert torch.allclose(legacy, with_views)


def test_l2a_averages_logits_over_views_instead_of_padded_input():
    x = torch.zeros(2, T, C)  # padded input says nothing: barley/wheat tie
    x[:, 0, 0] = SUPER_CEREAL_GROUP
    views = torch.zeros(2, 3, 45, C)
    views[:, 0, :, 1] = 3.0
    views[:, 1:, :, 1] = -0.5  # mean logit over views = (3 - 0.5 - 0.5) / 3 > 0
    l2a = _MeanChannelCereal()
    pipe = _pipeline(l2a=l2a)
    probs = pipe.forward_probs(x, expert_views=views)
    assert l2a.inputs[-1].shape == (6, 45, C)  # B * V views, not x
    expected = torch.softmax(torch.tensor([-2.0 / 3, 2.0 / 3]), dim=0)
    assert torch.allclose(probs[:, [GLOBAL_BARLEY, GLOBAL_WHEAT]], expected.expand(2, -1), atol=1e-4)
    assert pipe(x, expert_views=views).tolist() == [GLOBAL_WHEAT, GLOBAL_WHEAT]
    assert pipe(x).tolist() == [GLOBAL_BARLEY, GLOBAL_BARLEY]  # legacy tie -> barley


def test_l3_averages_distances_over_views():
    x = torch.zeros(2, T, C)
    x[:, 0, 0] = SUPER_GRASSLAND_GROUP
    views = torch.zeros(2, 4, 45, C)
    views[0, :, :, 2] = torch.tensor([1.0, 2.0, 3.0, 6.0]).view(4, 1)  # mean 3
    l3 = _RecordingL3()
    pipe = _pipeline(l3=l3)
    pipe.forward_probs(x, expert_views=views)
    assert l3.inputs[-1].shape == (8, 45, C)
    assert pipe.last_l3_count == 2
    # Recompute what the pipeline should have used for parcel 0.
    expected = l3.pseudo_distances(views[0]).mean(dim=0)
    assert torch.allclose(expected, 3.0 * torch.arange(1.0, 6.0))


class _DoyRouter(_Router):
    """L1 stub flagged as DOY-aware; records what it was called with."""

    use_doy = True

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, doy=None, padding_mask=None):
        self.calls.append((doy, padding_mask))
        return super().forward(x)


def test_doy_aware_l1_receives_doy_and_mask():
    l1 = _DoyRouter()
    pipe = DMOEPipeline(
        l1, _MeanChannelCereal(), _Const([5.0, -5.0]), _Const([10.0, -10.0]), _RecordingL3(),
        support_prototypes=torch.zeros(5, 4),
    )
    x = torch.zeros(2, T, C)
    doy = torch.arange(1, T + 1).expand(2, -1)
    mask = torch.ones(2, T, dtype=torch.bool)
    pipe.forward_probs(x, doy=doy, padding_mask=mask)
    seen_doy, seen_mask = l1.calls[-1]
    assert torch.equal(seen_doy, doy) and torch.equal(seen_mask, mask)
    with pytest.raises(ValueError):
        pipe.forward_probs(x)  # all-zero DOY fallback would misroute cereals


def test_expert_views_shape_is_validated():
    pipe = _pipeline()
    x = torch.zeros(2, T, C)
    with pytest.raises(ValueError):
        pipe.forward_probs(x, expert_views=torch.zeros(2, 45, C))
    with pytest.raises(ValueError):
        pipe.forward_probs(x, expert_views=torch.zeros(3, 2, 45, C))
    with pytest.raises(ValueError):
        pipe.forward_probs(x, expert_views=torch.zeros(2, 2, 45, C + 1))


# --------------------------------------------------------------------------- #
# Views dataset                                                                #
# --------------------------------------------------------------------------- #
_L2A_COLUMNS = 14  # doa, 10 spectral bands, CLD, EDG, SAT


class _FakeBreizhCrops:
    """Minimal stand-in exposing what ``load_raw_series_from_h5`` needs."""

    level = "L2A"
    h5path = None

    def __init__(self, lengths):
        rng = np.random.default_rng(0)
        self.X_list = []
        for t in lengths:
            raw = rng.uniform(100, 5000, size=(t, _L2A_COLUMNS))
            raw[:, 0] = pd.date_range("2017-01-01", periods=t, freq="5D").asi8
            self.X_list.append(raw)
        n = len(lengths)
        self.index = pd.DataFrame({"path": [f"p{i}" for i in range(n)], "CODE_CULTU": ["BTH"] * n, "id": np.arange(100, 100 + n)})
        self.mapping = pd.DataFrame({"id": [1]}, index=["BTH"])


def test_views_dataset_keeps_padded_item_and_adds_seeded_views():
    base = _FakeBreizhCrops([20, 60, 45])
    plain = BreizhCropsParcelDataset(base, [0, 1, 2], max_seq_len=45)
    with_views = BreizhCropsParcelViewsDataset(base, [0, 1, 2], draws=4, sampling_seed=7, max_seq_len=45)
    for i in range(3):
        item = with_views[i]
        for a, b in zip(plain[i][:2], item[:2]):
            assert torch.equal(a, b)
        assert item[2:4] == plain[i][2:4]
        views = item[4]
        assert views.shape == (4, 45, C)
        series = torch.from_numpy(base.X_list[i][:, 1:11].astype(np.float32) * 1e-4)
        # every view step is a real observation of the parcel (no zero padding)
        assert all(any(torch.equal(row, s) for s in series) for row in views[0])

    # seeded per parcel: independent of access order
    reversed_views = [with_views[i][4] for i in (2, 1, 0)][::-1]
    assert all(torch.equal(with_views[i][4], reversed_views[i]) for i in range(3))
    other_seed = BreizhCropsParcelViewsDataset(base, [0], draws=4, sampling_seed=8, max_seq_len=45)
    assert not torch.equal(other_seed[0][4], with_views[0][4])


def test_pad_collate_with_views_stacks_views():
    base = _FakeBreizhCrops([20, 60])
    ds = BreizhCropsParcelViewsDataset(base, [0, 1], draws=3, max_seq_len=45)
    x, doy, mask, y, ids, views = pad_collate_with_views([ds[0], ds[1]], max_seq_len=45)
    assert x.shape == (2, 45, C) and views.shape == (2, 3, 45, C)
    assert mask[0].sum() == 20 and mask[1].sum() == 45
    assert y.tolist() == [1, 1] and ids == [100, 101]

"""Tests for re-extracting pretrained global-model logits (``scripts/patch_global_logits.py``)."""

from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from agricultureai.data.breizhcrops_series import resample_series_views, split_raw_series

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import patch_global_logits as pgl  # noqa: E402

L1C_COLUMNS = 17


def _first_channel_model(x):
    """Stub model: every 'logit' is the mean of the first spectral channel."""
    return x[:, :, 0].mean(dim=1, keepdim=True).repeat(1, 9)


def test_region_logits_average_model_output_over_seeded_views(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    raws = [rng.uniform(100, 5000, size=(t, L1C_COLUMNS)) for t in (10, 50, 0)]
    h5path = tmp_path / "region.h5"
    with h5py.File(h5path, "w") as h5:
        for i, raw in enumerate(raws):
            h5.create_dataset(f"p{i}", data=raw)

    class FakeBreizhCrops:
        def __init__(self, region, level):
            self.index = pd.DataFrame({"path": [f"p{i}" for i in range(len(raws))]})
            self.h5path = str(h5path)

    monkeypatch.setattr(pgl.bzh, "BreizhCrops", FakeBreizhCrops)
    out = pgl.region_logits(
        {"msresnet": _first_channel_model}, "frh04", level="L1C", method="tta", draws=4,
        seed=0, seq_len=45, batch_size=2, device=torch.device("cpu"),
    )["msresnet"]
    assert out.shape == (3, 9)

    ref_rng = np.random.default_rng([0, zlib.crc32(b"frh04")])
    for i, raw in enumerate(raws):
        views = resample_series_views(split_raw_series(raw, "L1C")[0], 45, method="tta", draws=4, rng=ref_rng)
        assert out[i, 0] == pytest.approx(views[:, :, 0].mean(), rel=1e-5)
    assert np.all(out[2] == 0.0)  # empty parcel -> all-zero views


def _cache(path: Path, ids, labels, old_matched):
    torch.save(
        {
            "labels": torch.tensor(labels),
            "parcel_ids": torch.tensor(ids),
            "msresnet_logits": torch.zeros(len(ids), 9),
            "msresnet_l1c_matched": torch.tensor(old_matched),
            "transformer_logits": torch.full((len(ids), 9), 7.0),
            "meta": {"regions": ["frh04"], "level": "L2A"},
        },
        path,
    )


def _args(**overrides):
    args = argparse.Namespace(
        models=["msresnet"], method="tta", draws=8, seed=0, max_samples=0,
        output_suffix="_msfix", compare_only=False, source_level="L1C",
    )
    vars(args).update(overrides)
    return args


def _patch_sources(monkeypatch, target_ids, target_labels):
    # Source (L1C) has parcels 11 and 12 plus an extra one; target parcel 10 has no L1C twin.
    source = {
        "ids": np.array([12, 11, 99]),
        "labels": np.array([5, 1, 3]),
        "msresnet": np.array([np.full(9, 2.0), np.full(9, 1.0), np.full(9, 9.0)], dtype=np.float32),
        "transformer": np.zeros((3, 9), dtype=np.float32),
    }
    monkeypatch.setattr(
        pgl, "parcel_ids_and_labels",
        lambda regions, level, max_samples=0: (torch.tensor(target_ids), torch.tensor(target_labels)),
    )
    monkeypatch.setattr(pgl, "load_or_compute_region", lambda region, args, loader: source)


def test_patch_cache_aligns_by_parcel_id_and_writes_a_copy(tmp_path, monkeypatch):
    ids, labels = [10, 11, 12], [7, 1, 5]
    src = tmp_path / "cached_logits_eval.pt"
    _cache(src, ids, labels, [False, True, True])
    before = src.read_bytes()
    _patch_sources(monkeypatch, ids, labels)

    pgl.patch_cache(src, _args(), lambda: {})
    assert src.read_bytes() == before
    out = torch.load(tmp_path / "cached_logits_eval_msfix.pt", weights_only=True)
    assert torch.equal(out["msresnet_logits"][:, 0], torch.tensor([0.0, 1.0, 2.0]))  # 10 unmatched -> zeros
    assert out["msresnet_l1c_matched"].tolist() == [False, True, True]
    assert torch.equal(out["transformer_logits"], torch.full((3, 9), 7.0))  # not in --models: untouched
    assert out["meta"]["msresnet"] == "pretrained_L1C_tta8_aligned"
    assert out["meta"]["global_logits_patch_source"] == str(src)


def test_patch_cache_compare_only_writes_nothing(tmp_path, monkeypatch):
    ids, labels = [10, 11, 12], [7, 1, 5]
    src = tmp_path / "cached_logits_eval.pt"
    _cache(src, ids, labels, [False, True, True])
    _patch_sources(monkeypatch, ids, labels)
    pgl.patch_cache(src, _args(compare_only=True, output_suffix=""), lambda: {})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cached_logits_eval.pt"]


def test_patch_cache_refuses_a_different_match_mask(tmp_path, monkeypatch):
    ids, labels = [10, 11, 12], [7, 1, 5]
    src = tmp_path / "cached_logits_eval.pt"
    _cache(src, ids, labels, [True, True, True])  # cached mask claims parcel 10 matched
    _patch_sources(monkeypatch, ids, labels)
    with pytest.raises(RuntimeError, match="match mask"):
        pgl.patch_cache(src, _args(), lambda: {})


def test_in_place_patch_keeps_verified_backup(tmp_path, monkeypatch):
    ids, labels = [10, 11, 12], [7, 1, 5]
    src = tmp_path / "cached_logits_eval.pt"
    _cache(src, ids, labels, [False, True, True])
    before = src.read_bytes()
    _patch_sources(monkeypatch, ids, labels)
    pgl.patch_cache(src, _args(output_suffix=""), lambda: {})
    backup = tmp_path / "cached_logits_eval.pre_global_patch.pt"
    assert backup.read_bytes() == before
    assert torch.load(src, weights_only=True)["meta"]["global_logits_backup"] == str(backup)

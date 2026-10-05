"""Tests for cloud screening and the cloud-aware series store used by the pilot."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from agricultureai.data.cloud_mask import L2A_B2, L2A_B4, L2A_B8, L2A_CLD, L2A_DOA, clear_observation_mask

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_series_classifier as tsc  # noqa: E402


def _raw(doys, ndvi, cld, blue=300.0):
    t = len(doys)
    raw = np.full((t, 14), 1000.0)
    raw[:, L2A_DOA] = (np.datetime64("2017-01-01", "ns") + (np.array(doys) - 1).astype("timedelta64[D]")).astype(np.int64)
    raw[:, L2A_B8] = 1000.0 * (1 + np.array(ndvi))
    raw[:, L2A_B4] = 1000.0 * (1 - np.array(ndvi))
    raw[:, L2A_B2] = blue
    raw[:, L2A_CLD] = cld
    raw[:, 12] = 0.0  # EDG
    return raw


def test_mask_drops_cloudy_bright_and_spiky_dates():
    doys = [100, 110, 120, 130, 140, 150]
    ndvi = [0.8, 0.8, 0.3, 0.8, 0.8, 0.8]  # day 120 dips and recovers at once
    raw = _raw(doys, ndvi, cld=[0, 0, 0, 0, 0.5, 0])
    raw[5, L2A_B2] = 3000.0  # bright: cloud missed by the flag
    keep = clear_observation_mask(raw, max_cloud=0.1)
    assert keep.tolist() == [True, True, False, True, False, False]
    # a dip whose next kept date is beyond the gap window is left alone
    raw = _raw([100, 110, 120, 150], [0.8, 0.8, 0.3, 0.8], cld=[0, 0, 0, 0])
    assert clear_observation_mask(raw).all()


def test_mask_keeps_a_cut_that_does_not_recover_at_once():
    raw = _raw([100, 110, 120, 130], [0.8, 0.8, 0.35, 0.45], cld=[0, 0, 0, 0])
    assert clear_observation_mask(raw).all()


def _store(tmp_path, clear_flags):
    lengths = [len(c) for c in clear_flags]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    spectral = np.arange(offsets[-1] * 10, dtype=np.float32).reshape(-1, 10)
    np.savez(tmp_path / "r.npz", offsets=offsets, spectral=spectral, doy=np.zeros(offsets[-1], np.int16),
             clear=np.concatenate(clear_flags), labels=np.arange(len(lengths)), parcel_ids=np.arange(len(lengths)) + 100)
    return tmp_path / "r.npz", spectral


def test_store_uses_clear_dates_and_falls_back_to_all(tmp_path):
    path, spectral = _store(tmp_path, [np.array([1, 0, 1, 0], bool), np.array([0, 0, 0], bool), np.zeros(0, bool)])
    clear = tsc.SeriesStore([path], "clear")
    assert clear.n_dates.tolist() == [2, 3, 0]  # parcel 1 has no clear date -> all its dates
    every = tsc.SeriesStore([path], "all")
    assert every.n_dates.tolist() == [4, 3, 0]
    x = clear.batch(np.array([0, 1, 2]), np.random.default_rng(0)).numpy()
    assert x.shape == (3, tsc.SEQ_LEN, tsc.CHANNELS)
    assert set(x[0, :, 0].tolist()) <= {spectral[0, 0], spectral[2, 0]}  # only clear rows 0 and 2
    assert not x[2].any()  # no dates at all -> zeros

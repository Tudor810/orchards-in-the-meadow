"""Tests for cloud screening and the cloud-aware series store used by the pilot."""

from __future__ import annotations


import numpy as np
import torch

from agricultureai.data.cloud_mask import L2A_B2, L2A_B4, L2A_B8, L2A_CLD, L2A_DOA, clear_observation_mask

import build_series_store as bss
import train_series_classifier as tsc


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


def test_l1c_rows_keep_the_l2a_bands():
    raw = np.tile(np.arange(17, dtype=float) * 1000.0, (3, 1))  # value = 1000 x raw column
    raw[:, 16] = np.datetime64("2017-03-01", "ns").astype(np.int64)  # doa
    raw[1, 1] = np.nan  # B10 is not kept but still drops the date, as in BreizhCrops
    spectral, doa, clear = bss.parcel_rows(raw, "L1C")
    assert spectral.shape == (2, 10) and len(doa) == 2 and not clear.any()
    l1c_columns = [4, 5, 6, 7, 8, 9, 10, 11, 2, 3]  # B2-B8A, B11, B12 in the raw L1C layout
    np.testing.assert_allclose(spectral[0], 0.1 * np.array(l1c_columns), rtol=1e-6)


def test_vegetation_indices_values_and_padding():
    x = np.zeros((1, 2, 10), np.float32)  # second date stays padded (all zeros)
    x[0, 0] = [0.05, 0.08, 0.04, 0.10, 0.25, 0.30, 0.40, 0.42, 0.20, 0.10]  # B2..B8A, B11, B12
    v = tsc.vegetation_indices(x)
    assert v.shape == (1, 2, len(tsc.INDEX_NAMES))
    ndvi, ndmi, msi = (v[0, 0, tsc.INDEX_NAMES.index(n)] for n in ("NDVI", "NDMI", "MSI"))
    np.testing.assert_allclose([ndvi, ndmi, msi], [(0.40 - 0.04) / 0.44, (0.40 - 0.20) / 0.60, 0.20 / 0.40], rtol=1e-5)
    assert not v[0, 1].any()  # padded date -> zeros, no NaN or inf
    assert np.isfinite(v).all()


def test_store_batch_appends_indices(tmp_path):
    path, _ = _store(tmp_path, [np.array([1, 1, 1], bool), np.array([1, 0], bool)])
    x = tsc.SeriesStore([path], "all", indices=True).batch(np.array([0, 1]), np.random.default_rng(0)).numpy()
    assert x.shape == (2, tsc.SEQ_LEN, tsc.CHANNELS + len(tsc.INDEX_NAMES))
    model = tsc.build_model(0, indices=True).eval()
    assert model(torch.from_numpy(x)).shape == (2, tsc.NUM_CLASSES)


def test_class_weights_modes():
    labels = np.repeat(np.arange(tsc.NUM_CLASSES), [400, 100, 100, 400, 1, 25, 2, 400, 400])
    assert tsc.class_weights("none", labels) is None
    w = tsc.class_weights("orchard", labels, orchard_weight=10.0)
    assert w[tsc.ORCHARD] == 10.0 and np.allclose(np.delete(w, tsc.ORCHARD), 1.0)
    w = tsc.class_weights("sqrt_inv", labels, cap=10.0)
    assert w.max() <= 10.0 and w[0] < w[1] < w[tsc.ORCHARD]  # rarer class, larger weight
    assert w[4] == 10.0  # one-parcel class hits the cap

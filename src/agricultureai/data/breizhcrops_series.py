"""BreizhCrops parcel time-series helpers: spectral bands + Day-of-Year (DOY).

The stock ``BreizhCrops`` default transform drops the ``doa`` column and applies
random subsampling. Grassland / GL-TAE training needs calendar-aware positional
features and explicit padding masks, so this module loads the raw HDF5 rows,
splits reflectance from acquisition timestamps, and pads batches with aligned
``doy`` and ``padding_mask`` tensors.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import breizhcrops as bzh

from agricultureai.evaluation.transformer import MAX_SEQ_LEN as DEFAULT_MAX_SEQ_LEN

# Mirrors breizhcrops.datasets.breizhcrops.SELECTED_BANDS / get_default_transform.
_L1C_ALL_BANDS: List[str] = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9",
    "B10", "B11", "B12", "QA10", "QA20", "QA60", "doa",
]
_L2A_ALL_BANDS: List[str] = [
    "doa", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
    "CLD", "EDG", "SAT",
]
_L1C_SPECTRAL_BANDS: List[str] = [
    "B1", "B10", "B11", "B12", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9",
]
_L2A_SPECTRAL_BANDS: List[str] = [
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
]

_REFLECTANCE_SCALE = 1e-4


def spectral_band_names(level: str) -> List[str]:
    level = level.upper()
    if level == "L1C":
        return list(_L1C_SPECTRAL_BANDS)
    if level == "L2A":
        return list(_L2A_SPECTRAL_BANDS)
    raise ValueError(f"Unsupported BreizhCrops level {level!r}; use L1C or L2A.")


def doa_values_to_doy(doa: np.ndarray) -> np.ndarray:
    """Convert BreizhCrops ``doa`` nanosecond timestamps to day-of-year [1, 366]."""
    if doa.size == 0:
        return np.zeros(0, dtype=np.int64)
    dt = pd.to_datetime(doa)
    doy = dt.dayofyear.to_numpy(dtype=np.int64)
    return np.clip(doy, 1, 366)


def load_raw_series_from_h5(
    base_ds: bzh.BreizhCrops,
    base_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one parcel's reflectance ``(T, C)`` and DOY ``(T,)`` from the HDF5 store."""
    row = base_ds.index.iloc[int(base_index)]
    if base_ds.X_list is not None:
        raw = np.asarray(base_ds.X_list[int(base_index)])
    else:
        with h5py.File(base_ds.h5path, "r") as dataset:
            raw = np.array(dataset[row.path])

    spectral, doa = split_raw_series(raw, str(base_ds.level))
    return spectral, doa_values_to_doy(doa)


def split_raw_series(raw: np.ndarray, level: str) -> Tuple[np.ndarray, np.ndarray]:
    """Split one raw HDF5 parcel row into reflectance ``(T, C)`` and raw ``doa`` ``(T,)``."""
    if level.upper() == "L1C":
        all_bands, spectral_bands = _L1C_ALL_BANDS, _L1C_SPECTRAL_BANDS
    else:
        all_bands, spectral_bands = _L2A_ALL_BANDS, _L2A_SPECTRAL_BANDS

    spectral_idxs = np.array([all_bands.index(b) for b in spectral_bands], dtype=np.int64)
    doa_idx = all_bands.index("doa")

    # Drop timesteps with NaNs in spectral bands (matches BreizhCrops.load).
    spectral = raw[:, spectral_idxs]
    doa = raw[:, doa_idx]
    valid = ~np.isnan(spectral).any(axis=1)
    spectral = spectral[valid]
    doa = doa[valid]

    return spectral.astype(np.float32) * _REFLECTANCE_SCALE, doa


def subsample_series(
    x: np.ndarray,
    doy: np.ndarray,
    max_seq_len: int,
    *,
    train: bool,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample or truncate to ``max_seq_len`` timesteps (sorted by time)."""
    t = x.shape[0]
    if t == 0:
        c = x.shape[1] if x.ndim == 2 else 0
        return np.zeros((0, c), dtype=np.float32), np.zeros(0, dtype=np.int64)

    if t <= max_seq_len:
        return x, doy

    if train:
        gen = rng if rng is not None else np.random.default_rng()
        replace = t < max_seq_len
        idxs = gen.choice(t, size=max_seq_len, replace=replace)
        idxs.sort()
    else:
        idxs = np.arange(max_seq_len)

    return x[idxs], doy[idxs]


def resample_indices(
    n_steps: int,
    length: int,
    *,
    method: str = "random",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Time indices that resample an ``n_steps`` series to exactly ``length`` steps.

    Models trained through the stock BreizhCrops transform (``Subset`` of
    ``bzh.BreizhCrops``, e.g. the L2a cereal LTAE) never saw zero padding:
    ``"random"`` reproduces ``breizhcrops.datasets.breizhcrops.get_default_transform``
    (sorted draw, with replacement when ``n_steps < length``); ``"linspace"`` is a
    deterministic counterpart that spreads repeats of short series evenly.
    """
    if n_steps <= 0:
        raise ValueError("Cannot resample an empty series.")
    if method == "random":
        gen = rng if rng is not None else np.random.default_rng()
        return np.sort(gen.choice(n_steps, size=length, replace=n_steps < length))
    if method == "linspace":
        return np.linspace(0, n_steps - 1, length).round().astype(np.int64)
    raise ValueError(f"Unknown resample method {method!r}; use 'random' or 'linspace'.")


SAMPLING_METHODS: Tuple[str, ...] = ("zero_pad", "linspace", "random", "tta")


def resample_series_views(
    x: np.ndarray,
    length: int,
    *,
    method: str,
    draws: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Fixed-length views ``(V, length, C)`` of one ``(T, C)`` series.

    ``zero_pad`` reproduces cache extraction (first ``length`` steps, zero-padded,
    as ``BreizhCropsParcelDataset(train=False)`` + ``pad_collate_breizhcrops_series``);
    ``linspace`` and ``random`` give one :func:`resample_indices` view; ``tta``
    gives ``draws`` random views for the caller to average. An empty series
    yields all-zero views and consumes no randomness.
    """
    if method not in SAMPLING_METHODS:
        raise ValueError(f"Unknown sampling method {method!r}; use one of {SAMPLING_METHODS}.")
    n_views = int(draws) if method == "tta" else 1
    if n_views < 1:
        raise ValueError(f"draws must be >= 1; got {draws}")
    t, c = x.shape
    out = np.zeros((n_views, length, c), dtype=np.float32)
    if t == 0:
        return out
    if method == "zero_pad":
        out[0, : min(t, length)] = x[:length]
    elif method == "linspace":
        out[0] = x[resample_indices(t, length, method="linspace")]
    else:
        for view in range(n_views):
            out[view] = x[resample_indices(t, length, method="random", rng=rng)]
    return out


def pad_series_with_doy(
    x: torch.Tensor,
    doy: torch.Tensor,
    *,
    max_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad/truncate one parcel to ``(max_seq_len, C)``, ``(max_seq_len,)``, mask.

    Returns:
        padded_x: ``(max_seq_len, C)`` float.
        padded_doy: ``(max_seq_len,)`` long, ``0`` on padded positions.
        padding_mask: ``(max_seq_len,)`` bool, ``True`` = valid observation.
    """
    if x.dim() != 2:
        raise ValueError(f"Expected x shape (T, C); got {tuple(x.shape)}")
    if doy.dim() != 1 or doy.size(0) != x.size(0):
        raise ValueError(f"doy length {doy.size(0)} != x length {x.size(0)}")

    t, c = x.shape
    if t >= max_seq_len:
        valid_len = max_seq_len
        padded_x = x[:max_seq_len, :]
        padded_doy = doy[:max_seq_len]
    else:
        valid_len = t
        pad_rows = max_seq_len - t
        padded_x = torch.cat([x, torch.zeros(pad_rows, c, dtype=x.dtype)], dim=0)
        padded_doy = torch.cat(
            [doy, torch.zeros(pad_rows, dtype=torch.long)],
            dim=0,
        )

    padding_mask = torch.zeros(max_seq_len, dtype=torch.bool)
    padding_mask[:valid_len] = True
    return padded_x, padded_doy, padding_mask


class BreizhCropsParcelDataset(Dataset):
    """``BreizhCrops`` parcels with aligned spectral series and DOY vectors.

    Each item is ``(x, doy, y, parcel_id)`` where ``x`` is ``(T, C)`` float,
    ``doy`` is ``(T,)`` long in ``[1, 366]``, ``y`` is the raw class id, and
  ``parcel_id`` is the field identifier from the index.
    """

    def __init__(
        self,
        base_ds: bzh.BreizhCrops,
        indices: Sequence[int],
        *,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        train: bool = False,
        seed: int = 0,
    ) -> None:
        self.base_ds = base_ds
        self.indices = [int(i) for i in indices]
        self.max_seq_len = int(max_seq_len)
        self.train = bool(train)
        self._rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        base_idx = self.indices[int(idx)]
        spectral, doy_np = load_raw_series_from_h5(self.base_ds, base_idx)
        return self._item(base_idx, spectral, doy_np)

    def _item(
        self, base_idx: int, spectral: np.ndarray, doy_np: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        spectral, doy_np = subsample_series(
            spectral,
            doy_np,
            self.max_seq_len,
            train=self.train,
            rng=self._rng if self.train else None,
        )
        x = torch.from_numpy(spectral).float()
        doy = torch.from_numpy(doy_np).long()

        row = self.base_ds.index.iloc[base_idx]
        y = int(self.base_ds.mapping.loc[row["CODE_CULTU"]].id)
        parcel_id = int(row.id)
        return x, doy, y, parcel_id


class BreizhCropsParcelViewsDataset(BreizhCropsParcelDataset):
    """:class:`BreizhCropsParcelDataset` items plus resampled views of the series.

    Items are ``(x, doy, y, parcel_id, views)`` with ``views`` ``(V, max_seq_len, C)``
    from :func:`resample_series_views`, for experts trained on the stock
    BreizhCrops transform. Views are seeded per parcel from
    ``(sampling_seed, base index)``, so they do not depend on loading order.
    """

    def __init__(
        self,
        base_ds: bzh.BreizhCrops,
        indices: Sequence[int],
        *,
        method: str = "tta",
        draws: int = 8,
        sampling_seed: int = 0,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    ) -> None:
        super().__init__(base_ds, indices, max_seq_len=max_seq_len, train=False)
        self.method = method
        self.draws = int(draws)
        self.sampling_seed = int(sampling_seed)

    def __getitem__(self, idx: int):
        base_idx = self.indices[int(idx)]
        spectral, doy_np = load_raw_series_from_h5(self.base_ds, base_idx)
        rng = np.random.default_rng([self.sampling_seed, base_idx])
        views = resample_series_views(
            spectral, self.max_seq_len, method=self.method, draws=self.draws, rng=rng
        )
        return (*self._item(base_idx, spectral, doy_np), torch.from_numpy(views))


def pad_collate_with_views(
    batch: Sequence[tuple],
    *,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> tuple:
    """:func:`pad_collate_breizhcrops_series` plus stacked ``(B, V, T, C)`` views."""
    padded = pad_collate_breizhcrops_series([item[:4] for item in batch], max_seq_len=max_seq_len)
    return (*padded, torch.stack([item[4] for item in batch], dim=0))


def infer_padding_mask_from_features(
    x: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Heuristic mask: padded BreizhCrops rows are exact spectral zeros."""
    return x.abs().sum(dim=-1) > eps


def pad_collate_breizhcrops_series(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor, int, int]],
    *,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    map_label: Optional[Callable[[int], int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """Collate ``(x, doy, y, id)`` items into padded batch tensors.

    Returns:
        x: ``(B, T, C)``
        doy: ``(B, T)`` long, ``0`` on padded steps.
        padding_mask: ``(B, T)`` bool, ``True`` = valid (PyTorch attention: invert for
            ``src_key_padding_mask``).
        labels: ``(B,)`` long.
        ids: list of parcel ids.
    """
    xs, doys, ys, ids = zip(*batch)
    padded_xs: List[torch.Tensor] = []
    padded_doys: List[torch.Tensor] = []
    padding_masks: List[torch.Tensor] = []

    for x, doy in zip(xs, doys):
        px, pd, pm = pad_series_with_doy(x, doy, max_seq_len=max_seq_len)
        padded_xs.append(px)
        padded_doys.append(pd)
        padding_masks.append(pm)

    labels = [int(map_label(y) if map_label is not None else y) for y in ys]
    return (
        torch.stack(padded_xs, dim=0),
        torch.stack(padded_doys, dim=0),
        torch.stack(padding_masks, dim=0),
        torch.tensor(labels, dtype=torch.long),
        list(ids),
    )

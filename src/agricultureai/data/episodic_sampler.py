"""Level-3 Few-Shot Rescuer: data plumbing for episodic Prototypical training.

Stage 3 of the DMOE pipeline receives a *noisy bucket* forwarded by the Level-2
Grassland Router (high-recall threshold 0.20). That bucket mixes two true
anomaly families with false-positive meadow leakage. Level 3 treats the problem
as a 5-way few-shot metric-learning task over the raw BreizhCrops grassland
family (raw ids 4-8).

This module provides:

* ``RAW_TO_LEVEL3`` / ``LEVEL3_ID_TO_NAME`` -- the 5-way label space.
* ``Level3FewShotDataset`` -- a thin ``Dataset`` that filters BreizhCrops to the
  5 Level-3 classes, pads every parcel series to a fixed ``(T, C)`` shape, and
  exposes a parallel ``labels`` list so episode sampling never has to touch the
  time-series tensors.
* ``EpisodicBatchSampler`` -- a ``torch.utils.data.Sampler`` that yields one
  flat index list per episode (a support block followed by a query block).
* ``episodic_collate_fn`` / ``build_episodic_dataloader`` -- glue that reshapes a
  fetched episode into ``support_x`` / ``query_x`` tensors with relative
  (0..n_way-1) label indices.
"""

from __future__ import annotations

import random
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

import breizhcrops as bzh

from agricultureai.data.breizhcrops_loaders import filtered_indices
from agricultureai.data.breizhcrops_series import load_raw_series_from_h5, resample_series_views
from agricultureai.evaluation.transformer import MAX_SEQ_LEN as DEFAULT_MAX_SEQ_LEN

# Raw BreizhCrops ids that make up the Level-2 grassland family. These are the
# only ids that can ever reach the Level-3 rescuer.
LEVEL3_RAW_IDS: List[int] = [7, 8, 5, 6, 4]

# Raw BreizhCrops id -> Level-3 relative class id. The first two are the
# "rejection anchors" (false-positive meadow leakage); the last three are the
# true anomalies the rescuer must precisely separate.
RAW_TO_LEVEL3: Dict[int, int] = {
    7: 0,  # permanent meadows  -> Rejection Anchor
    8: 1,  # temporary meadows  -> Rejection Anchor
    5: 2,  # orchards           -> True Anomaly
    6: 3,  # nut trees          -> True Anomaly
    4: 4,  # sunflowers         -> True Anomaly
}

LEVEL3_ID_TO_NAME: Dict[int, str] = {
    0: "Permanent Meadows",
    1: "Temporary Meadows",
    2: "Orchards",
    3: "Nut Trees",
    4: "Sunflowers",
}

NUM_LEVEL3_CLASSES = len(LEVEL3_ID_TO_NAME)

# Classes that represent false-positive grassland leakage the rescuer should
# push back (i.e. "reject") rather than promote to a true anomaly sub-class.
LEVEL3_REJECTION_CLASSES: Tuple[int, int] = (0, 1)
LEVEL3_ANOMALY_CLASSES: Tuple[int, int, int] = (2, 3, 4)


def map_raw_to_level3(label: int) -> int:
    """Map a raw BreizhCrops id (4-8) to its Level-3 relative class id (0-4)."""
    try:
        return RAW_TO_LEVEL3[int(label)]
    except KeyError as exc:
        raise ValueError(
            f"Raw label {label} is outside Level-3 grassland ids {LEVEL3_RAW_IDS}"
        ) from exc


def _pad_or_truncate(x: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    """Force a parcel series to a fixed ``(max_seq_len, C)`` shape.

    ``x`` arrives as ``(T, C)``. Sequences longer than ``max_seq_len`` are
    clipped; shorter ones are zero-padded along the time axis. The channel
    dimension ``C`` is preserved untouched.
    """
    t, c = x.shape  # (T, C)
    if t >= max_seq_len:
        return x[:max_seq_len, :]  # (max_seq_len, C)
    padding = torch.zeros(max_seq_len - t, c, dtype=x.dtype)  # (pad, C)
    return torch.cat([x, padding], dim=0)  # (max_seq_len, C)


class Level3FewShotDataset(Dataset):
    """BreizhCrops parcels filtered to the 5 Level-3 classes.

    Each item is a ``(x, y)`` tuple where ``x`` is a padded ``(T, C)`` float
    tensor and ``y`` is the Level-3 relative class id (0-4). The parallel
    ``labels`` list mirrors ``y`` for every index so episode samplers can build
    class -> index maps without ever materialising the time-series tensors.
    """

    def __init__(
        self,
        *,
        regions: Sequence[str],
        level: str = "L1C",
        max_samples: int = 0,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        raw_ids: Sequence[int] = LEVEL3_RAW_IDS,
    ) -> None:
        self.max_seq_len = int(max_seq_len)
        self.raw_ids = [int(x) for x in raw_ids]

        # We keep (base_dataset, base_index) references rather than copying
        # tensors; BreizhCrops lazily decodes parcels in __getitem__.
        self._samples: List[Tuple[object, int]] = []
        self.labels: List[int] = []

        max_n = int(max_samples)
        for region in regions:
            print(f"Loading region={region} level={level} (Level-3 few-shot filter) ...")
            ds = bzh.BreizhCrops(region, level=level)
            kept = filtered_indices(ds, self.raw_ids, max_n)
            if not kept:
                raise RuntimeError(f"No Level-3 samples after filtering for region={region}.")
            for base_idx in kept:
                raw_label = int(ds.index.iloc[int(base_idx)]["classid"])
                self._samples.append((ds, int(base_idx)))
                self.labels.append(map_raw_to_level3(raw_label))

        if not self._samples:
            raise RuntimeError("Level3FewShotDataset is empty after filtering.")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        base_ds, base_idx = self._samples[int(idx)]
        x, _raw_y, _parcel_id = base_ds[base_idx]  # x: (T, C)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        x = _pad_or_truncate(x.float(), self.max_seq_len)  # (max_seq_len, C)
        return x, int(self.labels[int(idx)])

    def raw_series(self, idx: int) -> np.ndarray:
        """Full ``(T, C)`` reflectance series of item ``idx``.

        ``__getitem__`` goes through the stock BreizhCrops transform (an unseeded
        random 45-step draw); this returns the series before any resampling so
        callers can resample it reproducibly.
        """
        base_ds, base_idx = self._samples[int(idx)]
        spectral, _doy = load_raw_series_from_h5(base_ds, base_idx)
        return spectral

    def class_to_indices(self) -> Dict[int, List[int]]:
        """Return ``{level3_class_id: [dataset indices]}`` for sampling."""
        mapping: Dict[int, List[int]] = defaultdict(list)
        for dataset_idx, label in enumerate(self.labels):
            mapping[int(label)].append(dataset_idx)
        return dict(mapping)


class EpisodicBatchSampler(Sampler[List[int]]):
    """Yield one flat index list per Prototypical episode.

    Layout of every yielded list (length ``n_way * (k_shot + q_eff)``)::

        [ class0_support x k_shot, class1_support x k_shot, ... ,   <- support block
          class0_query  x q_eff,  class1_query  x q_eff,  ... ]    <- query block

    Keeping the support block strictly before the query block, and grouping by
    class within each block, lets ``episodic_collate_fn`` recover relative
    labels with a single ``repeat_interleave`` and no per-item bookkeeping.

    Graceful scale-down
    -------------------
    Prototypical networks need ``k_shot + q_query`` samples per class, but the
    rarest true-anomaly classes (orchards / nut trees) can be tiny. At init we
    therefore clamp the effective query size to::

        q_eff = min(q_query, min_eligible_class_count - k_shot)

    so every episode stays rectangular (and thus trivially collatable) while
    never requesting more queries than the rarest eligible class can supply.
    Classes that cannot even provide ``k_shot + 1`` samples are dropped from the
    eligible pool.
    """

    def __init__(
        self,
        labels: Sequence[int],
        *,
        n_way: int = 5,
        k_shot: int = 5,
        q_query: int = 15,
        num_episodes: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if n_way < 2:
            raise ValueError(f"n_way must be >= 2; got {n_way}")
        if k_shot < 1:
            raise ValueError(f"k_shot must be >= 1; got {k_shot}")
        if q_query < 1:
            raise ValueError(f"q_query must be >= 1; got {q_query}")

        self.n_way = int(n_way)
        self.k_shot = int(k_shot)
        self.requested_q_query = int(q_query)
        self.num_episodes = int(num_episodes)
        self._rng = random.Random(seed)

        # class -> dataset indices, restricted to classes large enough to give
        # at least one support and one query sample.
        per_class: Dict[int, List[int]] = defaultdict(list)
        for dataset_idx, label in enumerate(labels):
            per_class[int(label)].append(int(dataset_idx))

        min_needed = self.k_shot + 1
        self.class_to_indices: Dict[int, List[int]] = {
            cls: idxs for cls, idxs in per_class.items() if len(idxs) >= min_needed
        }
        self.eligible_classes: List[int] = sorted(self.class_to_indices.keys())

        if len(self.eligible_classes) < self.n_way:
            raise ValueError(
                f"Need >= {self.n_way} classes with >= {min_needed} samples each "
                f"for n_way={self.n_way} episodes; only {len(self.eligible_classes)} "
                f"qualify ({self.eligible_classes})."
            )

        smallest_eligible = min(len(idxs) for idxs in self.class_to_indices.values())
        # q_eff is shared across classes within an episode so the batch stays
        # rectangular; it can only shrink relative to the requested value.
        self.q_query = max(1, min(self.requested_q_query, smallest_eligible - self.k_shot))
        if self.q_query < self.requested_q_query:
            print(
                f"[EpisodicBatchSampler] scaling q_query {self.requested_q_query} -> "
                f"{self.q_query}; rarest eligible class only has {smallest_eligible} "
                f"samples (k_shot={self.k_shot})."
            )

    def __len__(self) -> int:
        return self.num_episodes

    def __iter__(self):
        for _ in range(self.num_episodes):
            chosen = self._rng.sample(self.eligible_classes, self.n_way)

            support_block: List[int] = []
            query_block: List[int] = []
            for cls in chosen:
                pool = self.class_to_indices[cls]
                # Sample support + query disjointly within the class.
                picked = self._rng.sample(pool, self.k_shot + self.q_query)
                support_block.extend(picked[: self.k_shot])
                query_block.extend(picked[self.k_shot :])

            # Support block first (n_way*k_shot), then query block
            # (n_way*q_query). Order is what episodic_collate_fn relies on.
            yield support_block + query_block


def episodic_collate_fn(
    batch: Sequence[Tuple[torch.Tensor, int]],
    *,
    n_way: int,
    k_shot: int,
    q_query: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reshape one fetched episode into support/query tensors.

    ``batch`` is the ordered list produced by ``EpisodicBatchSampler`` after
    ``Dataset.__getitem__`` resolution: ``n_way*k_shot`` support items followed
    by ``n_way*q_query`` query items, grouped by class within each block.

    Returns:
        support_x: ``(n_way * k_shot, seq_len, channels)``
        support_y: ``(n_way * k_shot,)`` relative class ids in ``[0, n_way)``
        query_x:   ``(n_way * q_query, seq_len, channels)``
        query_y:   ``(n_way * q_query,)`` relative class ids in ``[0, n_way)``
    """
    n_support = n_way * k_shot
    expected = n_support + n_way * q_query
    if len(batch) != expected:
        raise ValueError(
            f"episodic_collate_fn received {len(batch)} items but expected "
            f"{expected} (n_way={n_way}, k_shot={k_shot}, q_query={q_query})."
        )

    # Stack the raw per-item series. Each x is (seq_len, channels).
    xs = torch.stack([item[0] for item in batch], dim=0)  # (episode, seq_len, channels)

    support_x = xs[:n_support]  # (n_way*k_shot, seq_len, channels)
    query_x = xs[n_support:]    # (n_way*q_query, seq_len, channels)

    # Relative labels follow directly from the grouped layout: the support block
    # is [0]*k, [1]*k, ... and the query block is [0]*q, [1]*q, ....
    class_ids = torch.arange(n_way, dtype=torch.long)  # (n_way,)
    support_y = class_ids.repeat_interleave(k_shot)  # (n_way*k_shot,)
    query_y = class_ids.repeat_interleave(q_query)   # (n_way*q_query,)

    return support_x, support_y, query_x, query_y


def build_episodic_dataloader(
    dataset: Level3FewShotDataset,
    *,
    n_way: int = 5,
    k_shot: int = 5,
    q_query: int = 15,
    num_episodes: int = 100,
    seed: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, EpisodicBatchSampler]:
    """Wrap a ``Level3FewShotDataset`` into an episodic ``DataLoader``.

    The returned loader yields ``(support_x, support_y, query_x, query_y)``
    tuples (see ``episodic_collate_fn``). The sampler is returned alongside so
    callers can read the *effective* ``q_query`` after graceful scale-down.
    """
    sampler = EpisodicBatchSampler(
        dataset.labels,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        num_episodes=num_episodes,
        seed=seed,
    )
    collate = partial(
        episodic_collate_fn,
        n_way=sampler.n_way,
        k_shot=sampler.k_shot,
        q_query=sampler.q_query,  # effective (possibly scaled-down) value
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, sampler


def reference_support_indices(
    labels: Sequence[int],
    *,
    k_shot: int,
    n_way: int = NUM_LEVEL3_CLASSES,
    seed: int = 0,
) -> List[Tuple[int, int]]:
    """Deterministic ``(dataset_index, class_id)`` picks for the reference pool.

    Up to ``k_shot`` items per class, grouped by class in ``0..n_way-1`` order.
    """
    rng = torch.Generator().manual_seed(int(seed))

    class_to_indices: Dict[int, List[int]] = defaultdict(list)
    for dataset_idx, label in enumerate(labels):
        class_to_indices[int(label)].append(dataset_idx)

    picks: List[Tuple[int, int]] = []
    for class_id in range(n_way):
        pool = class_to_indices.get(class_id, [])

        actual_k_shot = min(k_shot, len(pool))
        if actual_k_shot < k_shot:
            name = LEVEL3_ID_TO_NAME.get(class_id, str(class_id))
            print(
                f"[Warning] Reference pool requested {k_shot} samples for class {class_id} "
                f"({name}), but only {len(pool)} are available. Capping at {actual_k_shot}."
            )
        # Deterministic shuffle, then take the first k_shot indices.
        perm = torch.randperm(len(pool), generator=rng).tolist()
        picks.extend((pool[i], class_id) for i in perm[:actual_k_shot])
    return picks


def build_reference_support_pool(
    dataset: Level3FewShotDataset,
    *,
    k_shot: int,
    n_way: int = NUM_LEVEL3_CLASSES,
    seed: int = 0,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Pick a fixed, clean ``k_shot``-per-class reference pool.

    These are the "confirmed ground-truth examples" used to lock the L3
    prototype coordinates at inference time. Selection is deterministic (fixed
    seed), but each item goes through ``dataset.__getitem__`` (the stock
    BreizhCrops transform, an unseeded random 45-step draw), so the series fed
    to the encoder differ between runs.

    Returns:
        support_x: ``(n_way * k_shot, seq_len, channels)``
        support_y: ``(n_way * k_shot,)`` relative class ids in ``[0, n_way)``
    """
    picks = reference_support_indices(dataset.labels, k_shot=k_shot, n_way=n_way, seed=seed)
    support_x = torch.stack([dataset[idx][0] for idx, _ in picks], dim=0)       # (n_way*k_shot, T, C)
    support_y = torch.tensor([class_id for _, class_id in picks], dtype=torch.long)  # (n_way*k_shot,)
    return support_x, support_y


def build_resampled_support_pool(
    dataset: Level3FewShotDataset,
    *,
    k_shot: int,
    method: str,
    draws: int = 8,
    sampling_seed: int = 0,
    seq_len: int = DEFAULT_MAX_SEQ_LEN,
    n_way: int = NUM_LEVEL3_CLASSES,
    seed: int = 0,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Reproducible counterpart of :func:`build_reference_support_pool`.

    Same parcels, but each series is resampled with
    :func:`~agricultureai.data.breizhcrops_series.resample_series_views`
    (seeded); with ``method="tta"`` every one of the ``draws`` views is its own
    support sample.

    Returns:
        support_x: ``(n_support_series, seq_len, channels)``
        support_y: ``(n_support_series,)`` relative class ids in ``[0, n_way)``
    """
    rng = np.random.default_rng(sampling_seed)
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for idx, class_id in reference_support_indices(dataset.labels, k_shot=k_shot, n_way=n_way, seed=seed):
        views = resample_series_views(dataset.raw_series(idx), seq_len, method=method, draws=draws, rng=rng)
        xs.append(views)
        ys.extend([class_id] * views.shape[0])
    return torch.from_numpy(np.concatenate(xs)), torch.tensor(ys, dtype=torch.long)

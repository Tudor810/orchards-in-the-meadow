"""Per-observation cloud screening for BreizhCrops L2A parcel series.

The L2A HDF5 rows carry a per-date cloud fraction ``CLD`` in [0, 1] (share of the
parcel flagged cloudy) and an edge / no-data fraction ``EDG``. About 60% of
dates have ``CLD > 0``. :func:`clear_observation_mask` combines three tests:

1. cloud fraction  ``CLD <= max_cloud``
2. brightness      blue reflectance ``B2 < max_blue`` (thick cloud and haze the
                   flag misses are bright in the blue band)
3. NDVI spike      a date whose NDVI drops more than ``spike`` below *both*
                   neighbouring kept dates, each within ``spike_gap_days``, is a
                   short dip that recovers at once: a shadow or thin cloud, not a
                   cut (regrowth after mowing takes weeks)

Rows with NaN spectral values are never kept.
"""

from __future__ import annotations

import numpy as np

from agricultureai.data.breizhcrops_series import doa_values_to_doy

# L2A HDF5 column order (breizhcrops SELECTED_BANDS["L2A"]).
L2A_DOA, L2A_B2, L2A_B4, L2A_B8, L2A_CLD, L2A_EDG = 0, 1, 3, 7, 11, 12
L2A_SPECTRAL = slice(1, 11)


def clear_observation_mask(
    raw: np.ndarray,
    *,
    max_cloud: float = 0.0,
    max_blue: float = 1200.0,
    max_edge: float = 0.5,
    spike: float = 0.15,
    spike_gap_days: int = 20,
) -> np.ndarray:
    """Boolean ``(T,)`` mask of usable dates for one raw L2A row block ``(T, 14)``.

    ``max_blue`` is in raw reflectance units (x 1e-4); ``spike <= 0`` disables the
    NDVI spike test.
    """
    keep = ~np.isnan(raw[:, L2A_SPECTRAL]).any(axis=1)
    keep &= raw[:, L2A_CLD] <= max_cloud
    keep &= raw[:, L2A_EDG] <= max_edge
    keep &= raw[:, L2A_B2] < max_blue
    if spike <= 0 or keep.sum() < 3:
        return keep

    idx = np.flatnonzero(keep)
    doy = doa_values_to_doy(raw[idx, L2A_DOA])
    order = np.argsort(doy, kind="stable")
    idx, doy = idx[order], doy[order]
    b4, b8 = raw[idx, L2A_B4], raw[idx, L2A_B8]
    ndvi = (b8 - b4) / np.maximum(b8 + b4, 1e-6)
    prev_drop = ndvi[1:-1] < ndvi[:-2] - spike
    next_drop = ndvi[1:-1] < ndvi[2:] - spike
    close = (np.diff(doy)[:-1] <= spike_gap_days) & (np.diff(doy)[1:] <= spike_gap_days)
    keep[idx[1:-1][prev_drop & next_drop & close]] = False
    return keep

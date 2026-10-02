"""Fingerprinted ``.npz`` caches for meta-classifier eval probabilities.

The sweep scripts train XGBoost + CatBoost on cached branch features and store
the eval-set probabilities under ``artifacts/`` keyed only by subset name and
orchard multiplier. A stored file is reused only when its fingerprint matches
the exact feature matrices, labels, seed, multiplier, model hyperparameters and
library versions of the current run; anything else is a cache miss. Feature-code
changes and re-patched logit caches therefore invalidate stale probabilities
automatically. ``force=True`` (``--force-recompute``) always recomputes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

FINGERPRINT_KEY = "fingerprint"


def array_digest(*arrays: np.ndarray) -> str:
    """SHA-256 over dtype, shape and bytes; NaNs are canonicalized first."""
    h = hashlib.sha256()
    for arr in arrays:
        arr = np.ascontiguousarray(arr)
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.where(np.isnan(arr), np.nan, arr).astype(arr.dtype, copy=False)
        h.update(f"{arr.dtype.str}{arr.shape}".encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def meta_proba_fingerprint(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    seed: int,
    **spec: Any,
) -> str:
    """Fingerprint everything that determines cached XGB/CatBoost eval probabilities.

    ``spec`` carries the caller's cache identity (e.g. ``kind``, ``m``); the
    model hyperparameters come from ``build_xgb`` / ``build_catboost`` so edits
    there also invalidate old files.
    """
    import catboost
    import xgboost

    from agricultureai.meta_features import build_catboost, build_xgb

    payload = {
        "data": array_digest(x_train, y_train, x_eval),
        "seed": int(seed),
        "xgb_params": build_xgb(seed).get_params(),
        "cat_params": build_catboost(seed).get_params(),
        "xgboost": xgboost.__version__,
        "catboost": catboost.__version__,
        **spec,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def load_npz(
    path: Path,
    fingerprint: str,
    *,
    force: bool = False,
) -> Optional[Dict[str, np.ndarray]]:
    """Return the cached arrays if ``path`` holds a matching fingerprint, else ``None``."""
    if not path.is_file():
        return None
    if force:
        print(f"--force-recompute: ignoring {path}", flush=True)
        return None
    with np.load(path) as data:
        stored = str(data[FINGERPRINT_KEY]) if FINGERPRINT_KEY in data.files else None
        if stored != fingerprint:
            reason = "no fingerprint" if stored is None else "fingerprint mismatch"
            print(f"Stale probability cache ({reason}): {path} -> recomputing", flush=True)
            return None
        return {key: data[key] for key in data.files if key != FINGERPRINT_KEY}


def save_npz(path: Path, fingerprint: str, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, **{FINGERPRINT_KEY: np.asarray(fingerprint)})

"""Safe in-place patching of the cached_logits_*.pt files: verified backups and atomic saves."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def backup_once(cache_path: Path, suffix: str) -> Path:
    """Copy ``cache_path`` to ``<stem><suffix>`` unless that backup already exists; never overwrite it."""
    backup = cache_path.with_name(cache_path.stem + suffix)
    if backup.exists():
        print(f"Backup already exists, keeping it: {backup}")
        return backup
    shutil.copy2(cache_path, backup)
    if sha256_file(backup) != sha256_file(cache_path):
        raise RuntimeError(f"Backup verification failed for {backup}")
    print(f"Backup (sha256 verified): {backup}")
    return backup


def output_path(cache_path: Path, suffix: str) -> Path:
    return cache_path.with_name(f"{cache_path.stem}{suffix}{cache_path.suffix}") if suffix else cache_path


def save_atomic(payload, target: Path) -> None:
    """torch.save to a temporary file, then rename, so an interrupted save never truncates ``target``."""
    tmp = target.with_name(target.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)

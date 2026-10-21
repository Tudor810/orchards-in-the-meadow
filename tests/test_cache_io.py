"""Tests for the cache backup and save helpers used by the patch_* scripts."""

from __future__ import annotations

from pathlib import Path

import torch

from agricultureai.cache_io import backup_once, output_path, save_atomic


def test_backup_once_verifies_and_never_overwrites(tmp_path):
    cache = tmp_path / "cached_logits_eval.pt"
    cache.write_bytes(b"original")
    backup = backup_once(cache, ".pre_l3_patch.pt")
    assert backup.name == "cached_logits_eval.pre_l3_patch.pt"
    assert backup.read_bytes() == b"original"
    cache.write_bytes(b"patched")
    assert backup_once(cache, ".pre_l3_patch.pt") == backup
    assert backup.read_bytes() == b"original"


def test_output_path_suffix():
    cache = Path("artifacts/cached_logits_train.pt")
    assert output_path(cache, "_l3fix") == Path("artifacts/cached_logits_train_l3fix.pt")
    assert output_path(cache, "") == cache


def test_save_atomic_replaces_target_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "cached_logits_eval.pt"
    target.write_bytes(b"old")
    save_atomic({"labels": torch.arange(3)}, target)
    assert torch.equal(torch.load(target, weights_only=True)["labels"], torch.arange(3))
    assert [f.name for f in tmp_path.iterdir()] == ["cached_logits_eval.pt"]

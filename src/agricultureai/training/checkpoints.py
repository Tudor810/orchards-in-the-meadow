"""Checkpoint helpers for self-describing AgricultureAi model files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

CHECKPOINT_FORMAT_VERSION = 1
STATE_DICT_KEY = "model_state_dict"
ARCHITECTURE_CONFIG_KEY = "architecture_config"
TRAINING_METADATA_KEY = "training_metadata"


def build_checkpoint_payload(
    model: torch.nn.Module,
    *,
    architecture_config: Mapping[str, Any] | None = None,
    training_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pack model weights together with the config needed to rebuild the model."""
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        STATE_DICT_KEY: model.state_dict(),
        ARCHITECTURE_CONFIG_KEY: dict(architecture_config or {}),
        TRAINING_METADATA_KEY: dict(training_metadata or {}),
    }


def is_self_describing_checkpoint(obj: Any) -> bool:
    return isinstance(obj, dict) and STATE_DICT_KEY in obj


def load_checkpoint(path: str | Path, *, map_location: Any = None) -> dict[str, Any]:
    """Load a checkpoint and normalize legacy raw state_dict files."""
    loaded = torch.load(str(path), map_location=map_location)
    if is_self_describing_checkpoint(loaded):
        return loaded
    return {
        "checkpoint_format_version": 0,
        STATE_DICT_KEY: loaded,
        ARCHITECTURE_CONFIG_KEY: {},
        TRAINING_METADATA_KEY: {},
    }


def model_state_dict_from_checkpoint(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    return checkpoint[STATE_DICT_KEY]


def architecture_config_from_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    return dict(checkpoint.get(ARCHITECTURE_CONFIG_KEY) or {})


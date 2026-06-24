"""LTAE expert for Stage-2 DMOE specialists (barley vs wheat)."""

from __future__ import annotations

import importlib
from typing import Callable, Optional, Tuple, Type

import breizhcrops as bzh
import torch
import torch.nn as nn

from agricultureai.models.ltae import ParcelLTAEClassifier


def _try_pretrained_ltae(device: torch.device) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
    for name in ("ltae", "LTAE"):
        try:
            return bzh.models.pretrained(name, device=device), "ltae"
        except ValueError:
            continue
    return None, None


def _import_breizhcrops_ltae_class() -> Optional[Type[torch.nn.Module]]:
    for module_path, class_name in (
        ("breizhcrops.models.LTAE", "LTAE"),
        ("breizhcrops.models.ltae", "LTAE"),
    ):
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
    return None


def _build_breizhcrops_ltae_classifier(
    ltae_cls: Type[torch.nn.Module],
    *,
    num_classes: int,
    input_dim: int,
    sequencelength: int,
) -> torch.nn.Module:
    """Instantiate package LTAE; supports common constructor signatures."""
    attempts: Tuple[Callable[..., torch.nn.Module], ...] = (
        lambda: ltae_cls(in_channels=input_dim, num_classes=num_classes, len_max_seq=sequencelength),
        lambda: ltae_cls(input_dim=input_dim, num_classes=num_classes, sequencelength=sequencelength),
        lambda: ltae_cls(in_channels=input_dim, num_classes=num_classes),
        lambda: ltae_cls(input_dim=input_dim, num_classes=num_classes),
    )
    last_error: Optional[Exception] = None
    for factory in attempts:
        try:
            return factory()
        except TypeError as exc:
            last_error = exc
            continue
    raise TypeError(
        f"Could not construct {ltae_cls.__name__} with known signatures"
    ) from last_error


def _build_fresh_ltae(
    *,
    num_classes: int,
    sequencelength: int,
    input_dim: int = 13,
) -> Tuple[torch.nn.Module, str]:
    ltae_cls = _import_breizhcrops_ltae_class()
    if ltae_cls is not None:
        model = _build_breizhcrops_ltae_classifier(
            ltae_cls,
            num_classes=num_classes,
            input_dim=input_dim,
            sequencelength=sequencelength,
        )
        if not hasattr(model, "head") and not _has_logits_output(model):
            _replace_classification_head(model, num_classes, "ltae")
        return model, "ltae"

    return (
        ParcelLTAEClassifier(
            num_classes,
            input_dim=input_dim,
            len_max_seq=sequencelength,
        ),
        "ltae",
    )


def _has_logits_output(model: torch.nn.Module) -> bool:
    return hasattr(model, "head") or hasattr(model, "outlinear") or hasattr(model, "decoder")


def _load_ltae_backbone(
    device: torch.device,
    *,
    pretrained: bool,
    num_classes: int,
    sequencelength: int,
) -> Tuple[torch.nn.Module, str]:
    if pretrained:
        model, arch = _try_pretrained_ltae(device)
        if model is not None:
            return model, arch
        print("LTAE pretrained weights unavailable; initializing LTAE from scratch.")
    else:
        print("Initializing LTAE from scratch (--from-scratch).")

    return _build_fresh_ltae(num_classes=num_classes, sequencelength=sequencelength)


def _replace_classification_head(model: torch.nn.Module, num_classes: int, arch: str) -> None:
    if isinstance(model, ParcelLTAEClassifier):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        return

    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        return

    if hasattr(model, "outlinear"):
        in_features = model.outlinear.in_features
        model.outlinear = nn.Linear(in_features, num_classes)
        return

    if hasattr(model, "decoder") and isinstance(model.decoder, nn.Sequential):
        last = model.decoder[-1]
        if isinstance(last, nn.Linear):
            in_features = last.in_features
            layers = list(model.decoder.children())
            layers[-1] = nn.Linear(in_features, num_classes)
            model.decoder = nn.Sequential(*layers)
            return

    if hasattr(model, "logsoftmax"):
        if isinstance(model.logsoftmax, nn.Sequential):
            last = model.logsoftmax[-1]
            if isinstance(last, nn.Linear):
                in_features = last.in_features
                model.logsoftmax = nn.Linear(in_features, num_classes)
                return
        if isinstance(model.logsoftmax, nn.Linear):
            in_features = model.logsoftmax.in_features
            model.logsoftmax = nn.Linear(in_features, num_classes)
            return

    raise TypeError(f"Unsupported LTAE backbone type for head replacement: {type(model)}")


class BreizhCropsLTAEExpert(nn.Module):
    """
    Winter-cereal expert using the L-TAE temporal attention encoder.

    Tries `bzh.models.pretrained("ltae")` first; otherwise builds a fresh LTAE
    (from `breizhcrops.models.LTAE` when available, else the local implementation).
  """

    def __init__(
        self,
        num_classes: int,
        *,
        sequencelength: int = 45,
        pretrained: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        dev = device or torch.device("cpu")

        self.model, self.arch = _load_ltae_backbone(
            dev,
            pretrained=pretrained,
            num_classes=self.num_classes,
            sequencelength=sequencelength,
        )
        if not isinstance(self.model, ParcelLTAEClassifier):
            _replace_classification_head(self.model, self.num_classes, self.arch)

        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        if device is not None:
            self.to(device)

    def forward(self, inputs: torch.Tensor, **_) -> torch.Tensor:
        if isinstance(self.model, ParcelLTAEClassifier):
            return self.model(inputs)

        if hasattr(self.model, "head"):
            return self.model.head(self.model.encoder(inputs)) if hasattr(self.model, "encoder") else self.model(inputs)

        return self.model(inputs)

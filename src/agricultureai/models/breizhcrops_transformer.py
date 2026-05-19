"""Fine-tune the BreizhCrops pretrained Transformer for super-class routing."""

from __future__ import annotations

from typing import Optional

import breizhcrops as bzh
import torch
import torch.nn as nn
from breizhcrops.models.TransformerModel import TransformerModel


class BreizhCropsTransformerClassifier(nn.Module):
    """
    Wrapper around `bzh.models.pretrained("Transformer")` with a replaceable
    classification head. Returns logits (not log-probabilities) for CrossEntropyLoss.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.freeze_backbone = bool(freeze_backbone)

        if pretrained:
            self.model = bzh.models.pretrained("Transformer", device=device or torch.device("cpu"))
        else:
            self.model = TransformerModel(input_dim=13, num_classes=9)
            if device is not None:
                self.model.to(device)

        in_features = self.model.outlinear.in_features
        self.model.outlinear = nn.Linear(in_features, self.num_classes)
        self._set_backbone_frozen(self.freeze_backbone)

        if device is not None:
            self.to(device)

    def _set_backbone_frozen(self, freeze: bool) -> None:
        for name, param in self.model.named_parameters():
            param.requires_grad = (not freeze) or name.startswith("outlinear.")

    def forward(self, inputs: torch.Tensor, **_) -> torch.Tensor:
        x = self.model.inlinear(inputs)
        x = self.model.relu(x)
        x = x.transpose(0, 1)
        x = self.model.transformerencoder(x)
        x = x.transpose(0, 1)
        x = x.max(dim=1)[0]
        x = self.model.relu(x)
        return self.model.outlinear(x)

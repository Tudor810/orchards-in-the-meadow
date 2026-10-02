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
        input_dim: int = 10,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        use_doy: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_dim = int(input_dim)
        self.freeze_backbone = bool(freeze_backbone)
        self.use_doy = bool(use_doy)

        if self.use_doy:
            self.doy_embed = nn.Embedding(367, self.input_dim)
            nn.init.normal_(self.doy_embed.weight, mean=0.0, std=0.02)

        if pretrained:
            self.model = bzh.models.pretrained("Transformer", device=device or torch.device("cpu"))
        else:
            self.model = TransformerModel(input_dim=input_dim, num_classes=9)
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

    def forward(
        self,
        inputs: torch.Tensor,
        doy: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        x = inputs
        if self.use_doy:
            if doy is None:
                doy = torch.zeros(
                    x.size(0), x.size(1), dtype=torch.long, device=x.device
                )
            if doy.shape != (x.size(0), x.size(1)):
                raise ValueError(
                    f"doy shape {tuple(doy.shape)} != ({x.size(0)}, {x.size(1)})"
                )
            x = x + self.doy_embed(doy.clamp(min=0, max=366))

        x = self.model.inlinear(x)
        x = self.model.relu(x)
        x = x.transpose(0, 1)
        # PyTorch attention masks padded steps where the mask is True, so invert
        # the valid-observation mask (True = valid) coming from the loaders.
        src_key_padding_mask = None if padding_mask is None else ~padding_mask
        x = self.model.transformerencoder(x, src_key_padding_mask=src_key_padding_mask)
        x = x.transpose(0, 1)
        if padding_mask is None:
            pooled = x.max(dim=1)[0]
        else:
            masked = x.masked_fill(~padding_mask.unsqueeze(-1), float("-inf"))
            pooled = masked.max(dim=1).values
            pooled = torch.where(pooled.isinf(), torch.zeros_like(pooled), pooled)
        x = self.model.relu(pooled)
        return self.model.outlinear(x)

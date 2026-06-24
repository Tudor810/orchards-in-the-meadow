"""BreizhCrops MSResNet expert wrapper for the L2c meadow binary head."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

import breizhcrops as bzh


class BreizhCropsMSResNetExpert(nn.Module):
    """Multi-scale ResNet time-series classifier (BreizhCrops baseline architecture).

    Expects input ``(B, T, C)`` (same layout as other AgricultureAi experts).
    When ``use_doy=True``, day-of-year indices ``(B, T)`` in ``[1, 366]`` (``0`` on
    pads) are embedded and added to the spectral channels before the MSResNet backbone,
    matching the calendar-aware path used by GL-TAE.

    Returns raw logits ``(B, num_classes)`` for pipeline softmax composition.
    """

    def __init__(
        self,
        *,
        num_classes: int = 2,
        input_dim: int = 13,
        hidden_dims: int = 32,
        pretrained: bool = True,
        use_doy: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.input_dim = int(input_dim)
        self.hidden_dims = int(hidden_dims)
        self.use_doy = bool(use_doy)

        if self.use_doy:
            self.doy_embed = nn.Embedding(367, self.input_dim)
            nn.init.normal_(self.doy_embed.weight, mean=0.0, std=0.02)

        if pretrained:
            base = bzh.models.pretrained("MSResNet", device="cpu")
            self.backbone = base
            if self.backbone.fc.out_features != self.num_classes:
                in_features = int(self.backbone.fc.in_features)
                self.backbone.fc = nn.Linear(in_features, self.num_classes)
        else:
            from breizhcrops.models.MSResNet import MSResNet

            self.backbone = MSResNet(
                input_dim=self.input_dim,
                num_classes=self.num_classes,
                hidden_dims=self.hidden_dims,
            )

        if device is not None:
            self.to(device)

    def forward(
        self,
        x: torch.Tensor,
        doy: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        del padding_mask  # MSResNet pools over interpolated length; pads are zero spectral.
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
        logits, _ = self.backbone._logits(x)
        return logits

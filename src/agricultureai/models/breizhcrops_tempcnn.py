"""TempCNN expert models for Stage-2 DMOE specialists."""

from __future__ import annotations

from typing import Optional

import breizhcrops as bzh
import torch
import torch.nn as nn
from breizhcrops.models.TempCNN import TempCNN


class BreizhCropsTempCNNExpert(nn.Module):
    """
    TempCNN with a replaceable linear head returning logits for CrossEntropyLoss.

    By default loads `bzh.models.pretrained("TempCNN")` as the backbone init,
    replaces the 9-class LogSoftmax head with `num_classes` logits, and keeps
    every layer trainable.
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

        if pretrained:
            self.model = bzh.models.pretrained("TempCNN", device=device or torch.device("cpu"))
        else:
            self.model = TempCNN(input_dim=13, num_classes=9, sequencelength=sequencelength)
            if device is not None:
                self.model.to(device)

        hidden = self.model.hidden_dims
        self.model.logsoftmax = nn.Linear(4 * hidden, self.num_classes)
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        if device is not None:
            self.to(device)

    def forward(self, inputs: torch.Tensor, **_) -> torch.Tensor:
        x = inputs.transpose(1, 2)
        x = self.model.conv_bn_relu1(x)
        x = self.model.conv_bn_relu2(x)
        x = self.model.conv_bn_relu3(x)
        x = self.model.flatten(x)
        x = self.model.dense(x)
        return self.model.logsoftmax(x)

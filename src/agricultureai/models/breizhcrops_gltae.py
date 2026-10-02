"""
Global-Local Temporal Attention Encoder (GL-TAE) expert for Stage-2 grassland
classification on BreizhCrops Sentinel-2 time-series.

GL-TAE fuses both views:

    Input (B, T, C_in=13) + DOY (B, T) + padding_mask (B, T)
      -> Local 1-D Convolutional block      (sharp local temporal features)
      -> Day-of-year embedding            (calendar phenology)
      -> Multi-head Self-Attention encoder  (global context; ignores padding)
      -> Masked mean + max pooling over time
      -> Linear classifier                  (raw logits, FocalLoss-compatible)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _LocalConvBlock(nn.Module):
    """Two-stage 1-D conv block extracting local temporal features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd to preserve sequence length; got {kernel_size}"
            )
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.act(self.bn2(self.conv2(x)))
        return x.transpose(1, 2)


def _masked_mean_pool(
    x: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked mean over time: ``padding_mask`` True = valid timestep."""
    mask_float = padding_mask.float().unsqueeze(-1)  # (B, T, 1)
    return (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-9)


def _masked_max_pool(
    x: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked max over time; padded positions are set to ``-inf`` before max."""
    if not bool(padding_mask.any()):
        return x.max(dim=1).values
    masked = x.masked_fill(~padding_mask.unsqueeze(-1), float("-inf"))
    max_pool = masked.max(dim=1).values
    return torch.where(max_pool.isinf(), torch.zeros_like(max_pool), max_pool)


class BreizhCropsGLTAEExpert(nn.Module):
    """Stage-2 grassland expert: Global-Local Temporal Attention Encoder."""

    def __init__(
        self,
        num_classes: int,
        *,
        input_dim: int = 10,
        sequencelength: int = 45,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        dim_feedforward: int = 256,
        conv_kernel_size: int = 5,
        dropout: float = 0.2,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive; got {num_classes}")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )

        self.num_classes = int(num_classes)
        self.input_dim = int(input_dim)
        self.sequencelength = int(sequencelength)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.conv_kernel_size = int(conv_kernel_size)
        self.dropout_p = float(dropout)

        self.local_conv = _LocalConvBlock(
            in_channels=self.input_dim,
            out_channels=self.d_model,
            kernel_size=self.conv_kernel_size,
            dropout=self.dropout_p,
        )

        # DOY in [0, 366]: index 0 reserved for padded positions (masked out).
        self.doy_embed = nn.Embedding(367, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=int(dim_feedforward),
            dropout=self.dropout_p,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
        )
        self.encoder_norm = nn.LayerNorm(self.d_model)

        self.classifier = nn.Sequential(
            nn.Linear(2 * self.d_model, self.d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.d_model, self.num_classes),
        )

        self._init_weights()

        if device is not None:
            self.to(device)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.doy_embed.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        doy: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Spectral tensor ``(B, T, C_in)``.
            doy: Day-of-year integers ``(B, T)`` in ``[1, 366]``; ``0`` on pads.
            padding_mask: ``(B, T)`` bool, ``True`` = valid observation.

        Returns:
            Raw class logits ``(B, num_classes)``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"GL-TAE expects (B, T, C) input; got shape {tuple(x.shape)}"
            )
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"GL-TAE input channel mismatch: expected C_in={self.input_dim}, "
                f"got {x.size(-1)}"
            )

        batch_size, seq_len, _ = x.shape

        if padding_mask is None:
            padding_mask = torch.ones(
                batch_size, seq_len, dtype=torch.bool, device=x.device
            )
        if doy is None:
            doy = torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.device)

        if padding_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"padding_mask shape {tuple(padding_mask.shape)} != "
                f"({batch_size}, {seq_len})"
            )
        if doy.shape != (batch_size, seq_len):
            raise ValueError(
                f"doy shape {tuple(doy.shape)} != ({batch_size}, {seq_len})"
            )

        pooled = self.encode_pooled(x, doy=doy, padding_mask=padding_mask)
        return self.classifier(pooled)

    def encode_pooled(
        self,
        x: torch.Tensor,
        doy: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return penultimate pooled embedding ``(B, 2 * d_model)`` before the classifier."""
        if x.dim() != 3:
            raise ValueError(
                f"GL-TAE expects (B, T, C) input; got shape {tuple(x.shape)}"
            )
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"GL-TAE input channel mismatch: expected C_in={self.input_dim}, "
                f"got {x.size(-1)}"
            )

        batch_size, seq_len, _ = x.shape

        if padding_mask is None:
            padding_mask = torch.ones(
                batch_size, seq_len, dtype=torch.bool, device=x.device
            )
        if doy is None:
            doy = torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.device)

        if padding_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"padding_mask shape {tuple(padding_mask.shape)} != "
                f"({batch_size}, {seq_len})"
            )
        if doy.shape != (batch_size, seq_len):
            raise ValueError(
                f"doy shape {tuple(doy.shape)} != ({batch_size}, {seq_len})"
            )

        local_feats = self.local_conv(x)
        doy_idx = doy.clamp(min=0, max=366)
        feats = local_feats + self.doy_embed(doy_idx)

        key_padding_mask = ~padding_mask
        global_feats = self.global_encoder(
            feats,
            src_key_padding_mask=key_padding_mask,
        )
        global_feats = self.encoder_norm(global_feats)

        mean_pool = _masked_mean_pool(global_feats, padding_mask)
        max_pool = _masked_max_pool(global_feats, padding_mask)
        return torch.cat([mean_pool, max_pool], dim=1)

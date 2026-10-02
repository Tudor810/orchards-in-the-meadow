"""
Lightweight Temporal Attention Encoder (L-TAE) for parcel time series.

Adapted from VSainteuf/lightweight-temporal-attention-pytorch (MIT).
Used when `breizhcrops` does not ship `models.LTAE` or pretrained LTAE weights.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


class LTAE(nn.Module):
  """Sequence-to-embedding encoder (multi-head temporal self-attention)."""

  def __init__(
      self,
      in_channels: int = 128,
      n_head: int = 16,
      d_k: int = 8,
      n_neurons: Optional[Sequence[int]] = None,
      dropout: float = 0.2,
      d_model: Optional[int] = 256,
      T: int = 1000,
      len_max_seq: int = 24,
      positions=None,
      return_att: bool = False,
  ):
    super().__init__()
    self.in_channels = in_channels
    self.positions = positions
    self.n_neurons = copy.deepcopy(list(n_neurons or [256, 128]))
    self.return_att = return_att

    if positions is None:
      positions = len_max_seq + 1

    if d_model is not None:
      self.d_model = d_model
      self.inconv = nn.Sequential(
          nn.Conv1d(in_channels, d_model, 1),
          nn.LayerNorm((d_model, len_max_seq)),
      )
    else:
      self.d_model = in_channels
      self.inconv = None

    sin_tab = get_sinusoid_encoding_table(positions, self.d_model // n_head, T=T)
    self.register_buffer(
        "position_enc_table",
        torch.cat([sin_tab for _ in range(n_head)], dim=1),
        persistent=False,
    )
    self.position_enc = nn.Embedding.from_pretrained(self.position_enc_table, freeze=True)

    self.inlayernorm = nn.LayerNorm(self.in_channels)
    self.outlayernorm = nn.LayerNorm(self.n_neurons[-1])
    self.attention_heads = MultiHeadAttention(n_head=n_head, d_k=d_k, d_in=self.d_model)
    assert self.n_neurons[0] == self.d_model

    layers: List[nn.Module] = []
    for i in range(len(self.n_neurons) - 1):
      layers.extend([
          nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]),
          nn.LayerNorm(self.n_neurons[i + 1]),
          nn.ReLU(),
      ])
    self.mlp = nn.Sequential(*layers)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor):
    sz_b, seq_len, _ = x.shape
    x = self.inlayernorm(x)

    if self.inconv is not None:
      x = self.inconv(x.permute(0, 2, 1)).permute(0, 2, 1)

    if self.positions is None:
      src_pos = torch.arange(1, seq_len + 1, dtype=torch.long, device=x.device).expand(sz_b, seq_len)
    else:
      src_pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).expand(sz_b, seq_len)

    enc_output = x + self.position_enc(src_pos)
    enc_output, attn = self.attention_heads(enc_output, enc_output, enc_output)
    enc_output = enc_output.permute(1, 0, 2).contiguous().view(sz_b, -1)
    enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))

    if self.return_att:
      return enc_output, attn
    return enc_output


class ParcelLTAEClassifier(nn.Module):
  """BreizhCrops parcel series (N, T, C) -> class logits via L-TAE."""

  def __init__(
      self,
      num_classes: int,
      *,
      input_dim: int = 10,
      len_max_seq: int = 45,
      n_head: int = 16,
      d_k: int = 8,
      d_model: int = 256,
      n_neurons: Optional[Sequence[int]] = None,
      dropout: float = 0.2,
  ):
    super().__init__()
    neurons = list(n_neurons or [256, 128])
    self.encoder = LTAE(
        in_channels=input_dim,
        n_head=n_head,
        d_k=d_k,
        n_neurons=neurons,
        dropout=dropout,
        d_model=d_model,
        len_max_seq=len_max_seq,
    )
    self.head = nn.Linear(neurons[-1], int(num_classes))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.head(self.encoder(x))


class MultiHeadAttention(nn.Module):
  def __init__(self, n_head: int, d_k: int, d_in: int):
    super().__init__()
    self.n_head = n_head
    self.d_k = d_k
    self.d_in = d_in
    self.Q = nn.Parameter(torch.zeros((n_head, d_k)))
    nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / d_k))
    self.fc1_k = nn.Linear(d_in, n_head * d_k)
    nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / d_k))
    self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

  def forward(self, q, k, v):
    d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
    sz_b, seq_len, _ = q.size()

    q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(-1, d_k)
    k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
    k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)
    v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(n_head * sz_b, seq_len, -1)
    output, attn = self.attention(q, k, v)
    attn = attn.view(n_head, sz_b, 1, seq_len).squeeze(dim=2)
    output = output.view(n_head, sz_b, 1, d_in // n_head).squeeze(dim=2)
    return output, attn


class ScaledDotProductAttention(nn.Module):
  def __init__(self, temperature: float, attn_dropout: float = 0.1):
    super().__init__()
    self.temperature = temperature
    self.dropout = nn.Dropout(attn_dropout)
    self.softmax = nn.Softmax(dim=2)

  def forward(self, q, k, v):
    attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / self.temperature
    attn = self.softmax(attn)
    attn = self.dropout(attn)
    output = torch.matmul(attn, v)
    return output, attn


def get_sinusoid_encoding_table(positions, d_hid: int, T: int = 1000) -> torch.Tensor:
  if isinstance(positions, int):
    positions = list(range(positions))

  def cal_angle(position, hid_idx):
    return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

  def get_posi_angle_vec(position):
    return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

  sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions], dtype=np.float32)
  sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
  sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
  return torch.from_numpy(sinusoid_table)

"""Prototypical Network for the Level-3 Few-Shot Rescuer.

This is the metric-learning core of Stage 3 of the DMOE pipeline. It maps
BreizhCrops parcel time-series ``(B, T, C)`` into a low-dimensional embedding
space, builds one prototype (class centroid) per class from a small labelled
support set, and classifies queries by *negative squared Euclidean distance* to
those prototypes (Snell et al., 2017, "Prototypical Networks for Few-shot
Learning").

Two interchangeable embedding backbones are provided:

* ``TemporalCNNEncoder`` -- a lightweight 3-layer 1-D temporal CNN (default).
* ``GRUEncoder`` -- a small bidirectional GRU, useful when phenological ordering
  matters more than sharp local events.

``PrototypicalNetwork`` wraps a chosen backbone and exposes ``forward`` /
``proto_episode`` which return the log-softmax distribution over the negative
distances together with the cross-entropy loss and batch accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EpisodeOutput:
    """Container for a single episode / inference forward pass.

    Attributes:
        log_p_y: ``(num_query, n_way)`` log-softmax over negative distances.
        loss: scalar cross-entropy (NLL of ``log_p_y`` vs. query labels).
        accuracy: scalar query accuracy in ``[0, 1]``.
        distances: ``(num_query, n_way)`` squared Euclidean distances.
        prototypes: ``(n_way, embedding_dim)`` class centroids.
    """

    log_p_y: torch.Tensor
    loss: torch.Tensor
    accuracy: torch.Tensor
    distances: torch.Tensor
    prototypes: torch.Tensor


class TemporalCNNEncoder(nn.Module):
    """3-layer 1-D temporal CNN: ``(B, T, C) -> (B, hidden_dim)``.

    Each block is Conv1d -> BatchNorm -> ReLU with odd-kernel "same" padding so
    the temporal length is preserved through the stack. A final adaptive
    average pool collapses the time axis into a fixed-width embedding,
    independent of the input sequence length.
    """

    def __init__(
        self,
        *,
        in_channels: int = 10,
        hidden_dim: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive; got {hidden_dim}")
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd; got {kernel_size}")
        padding = kernel_size // 2

        # Channel widths grow then settle at hidden_dim: C -> 32 -> 64 -> hidden.
        c1, c2 = 32, 64
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size, padding=padding),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size, padding=padding),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(c2, hidden_dim, kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Collapse the (variable) time axis to length 1.
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.hidden_dim = int(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, C, T): Conv1d convolves over the last (time) axis.
        x = x.transpose(1, 2)            # (B, C, T)
        x = self.block1(x)               # (B, 32, T)
        x = self.block2(x)               # (B, 64, T)
        x = self.block3(x)               # (B, hidden_dim, T)
        x = self.temporal_pool(x)        # (B, hidden_dim, 1)
        # Drop the singleton time axis -> flat embedding.
        return x.squeeze(-1)             # (B, hidden_dim)


class GRUEncoder(nn.Module):
    """Bidirectional GRU backbone: ``(B, T, C) -> (B, hidden_dim)``.

    The final forward and backward hidden states are concatenated and linearly
    projected to ``hidden_dim`` so it is a drop-in alternative to the CNN.
    """

    def __init__(
        self,
        *,
        in_channels: int = 10,
        hidden_dim: int = 64,
        rnn_hidden: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 2 * rnn_hidden because the GRU is bidirectional.
        self.project = nn.Linear(2 * rnn_hidden, hidden_dim)
        self.hidden_dim = int(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h_n: (num_layers * 2, B, rnn_hidden).
        _, h_n = self.gru(x)
        # Take the last layer's forward (h_n[-2]) and backward (h_n[-1]) states.
        last_fwd = h_n[-2]               # (B, rnn_hidden)
        last_bwd = h_n[-1]               # (B, rnn_hidden)
        pooled = torch.cat([last_fwd, last_bwd], dim=1)  # (B, 2*rnn_hidden)
        return self.project(pooled)      # (B, hidden_dim)


def squared_euclidean_distance(
    queries: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    """Pairwise squared Euclidean distance between queries and prototypes.

    Args:
        queries: ``(num_query, embedding_dim)``.
        prototypes: ``(n_way, embedding_dim)``.

    Returns:
        ``(num_query, n_way)`` where entry ``[i, k] = ||q_i - c_k||^2``.
    """
    # Broadcast to (num_query, n_way, embedding_dim) then reduce the feature
    # axis: unsqueeze(1) adds the prototype axis, unsqueeze(0) adds the query
    # axis.
    diff = queries.unsqueeze(1) - prototypes.unsqueeze(0)  # (num_query, n_way, D)
    return diff.pow(2).sum(dim=2)                          # (num_query, n_way)


def squared_mahalanobis_distance(
    queries: torch.Tensor,
    prototypes: torch.Tensor,
    inv_vars: torch.Tensor,
) -> torch.Tensor:
    """Pairwise squared *diagonal* Mahalanobis distance.

    Each class gets its own diagonal covariance, so the unit ball around a
    prototype becomes an axis-aligned ellipsoid: wide, noisy classes (meadows)
    stretch along their high-variance dimensions instead of bleeding across a
    rigid spherical boundary into compact classes (orchards/nuts).

    With ``inv_vars`` filled with ones this reduces exactly to
    :func:`squared_euclidean_distance`.

    Args:
        queries: ``(num_query, embedding_dim)``.
        prototypes: ``(n_way, embedding_dim)``.
        inv_vars: ``(n_way, embedding_dim)`` per-class inverse diagonal
            variances (``1 / sigma^2``), strictly positive.

    Returns:
        ``(num_query, n_way)`` where entry
        ``[i, k] = sum_d (q_id - c_kd)^2 / sigma^2_kd``.
    """
    # (Nq, 1, D) - (1, K, D) -> (Nq, K, D); inv_vars broadcasts over queries.
    diff = queries.unsqueeze(1) - prototypes.unsqueeze(0)   # (num_query, n_way, D)
    return (diff.pow(2) * inv_vars.unsqueeze(0)).sum(dim=2)  # (num_query, n_way)


def compute_prototypes(
    support_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    n_way: int,
) -> torch.Tensor:
    """Average support embeddings per class into ``n_way`` centroids.

    Args:
        support_embeddings: ``(n_way * k_shot, embedding_dim)``.
        support_labels: ``(n_way * k_shot,)`` relative ids in ``[0, n_way)``.
        n_way: number of classes (prototypes) to produce.

    Returns:
        ``(n_way, embedding_dim)`` prototype matrix, row ``k`` = centroid of
        class ``k``.
    """
    embedding_dim = support_embeddings.size(1)
    prototypes = support_embeddings.new_zeros((n_way, embedding_dim))  # (n_way, D)
    for class_id in range(n_way):
        mask = support_labels == class_id           # (n_way*k_shot,)
        if not bool(mask.any()):
            raise ValueError(
                f"No support samples for class {class_id}; cannot form a prototype."
            )
        prototypes[class_id] = support_embeddings[mask].mean(dim=0)
    return prototypes


def compute_prototypes_and_variances(
    support_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    n_way: int,
    *,
    shrinkage: float = 0.5,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-class centroids plus shrinkage-regularized diagonal variances.

    Few-shot supports give unstable raw variance estimates (a 1-shot class has
    variance exactly zero), so every class variance is shrunk toward the pooled
    within-class variance over the whole support set (Simple-CNAPS style)::

        sigma^2_k = shrinkage * sigma^2_k_hat
                    + (1 - shrinkage) * sigma^2_pooled
                    + eps

    ``shrinkage=1.0`` keeps the raw per-class estimates; ``shrinkage=0.0``
    gives every class the same pooled diagonal covariance.

    Args:
        support_embeddings: ``(n_way * k_shot, embedding_dim)``.
        support_labels: ``(n_way * k_shot,)`` relative ids in ``[0, n_way)``.
        n_way: number of classes.
        shrinkage: blend factor ``alpha`` in ``[0, 1]`` (default 0.5).
        eps: safety floor added to every variance (default 1e-4).

    Returns:
        prototypes: ``(n_way, embedding_dim)`` class centroids.
        variances: ``(n_way, embedding_dim)`` shrunk diagonal variances,
        strictly positive (invert for :func:`squared_mahalanobis_distance`).
    """
    if not 0.0 <= float(shrinkage) <= 1.0:
        raise ValueError(f"shrinkage must be in [0, 1]; got {shrinkage}")
    if float(eps) <= 0.0:
        raise ValueError(f"eps must be > 0; got {eps}")

    embedding_dim = support_embeddings.size(1)
    prototypes = support_embeddings.new_zeros((n_way, embedding_dim))   # (K, D)
    raw_variances = support_embeddings.new_zeros((n_way, embedding_dim))  # (K, D)
    counts = support_embeddings.new_zeros(n_way)                        # (K,)

    for class_id in range(n_way):
        mask = support_labels == class_id            # (n_way*k_shot,)
        if not bool(mask.any()):
            raise ValueError(
                f"No support samples for class {class_id}; cannot form a prototype."
            )
        class_emb = support_embeddings[mask]         # (k, D)
        prototypes[class_id] = class_emb.mean(dim=0)
        # Population (biased) variance: well-defined for k == 1 (zero), where
        # the shrinkage term below supplies all of the spread.
        raw_variances[class_id] = class_emb.var(dim=0, unbiased=False)
        counts[class_id] = class_emb.size(0)

    # Pooled within-class variance: count-weighted average of the per-class
    # deviations, i.e. the mean squared distance of every support item to its
    # own class centroid, computed per embedding dimension.
    pooled = (raw_variances * counts.unsqueeze(1)).sum(dim=0) / counts.sum()  # (D,)

    variances = (
        float(shrinkage) * raw_variances
        + (1.0 - float(shrinkage)) * pooled.unsqueeze(0)
        + float(eps)
    )                                                # (K, D)
    return prototypes, variances


def prototypical_loss(
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    prototypes: torch.Tensor,
) -> EpisodeOutput:
    """Negative-distance log-softmax + cross-entropy + accuracy.

    Args:
        query_embeddings: ``(num_query, embedding_dim)``.
        query_labels: ``(num_query,)`` relative ids in ``[0, n_way)``.
        prototypes: ``(n_way, embedding_dim)``.

    Returns:
        ``EpisodeOutput`` with log-probabilities, scalar loss, and accuracy.
    """
    distances = squared_euclidean_distance(query_embeddings, prototypes)  # (Nq, n_way)
    # Closer prototype -> smaller distance -> larger logit, hence the negation.
    log_p_y = F.log_softmax(-distances, dim=1)                            # (Nq, n_way)
    loss = F.nll_loss(log_p_y, query_labels)                              # scalar
    preds = log_p_y.argmax(dim=1)                                         # (Nq,)
    accuracy = (preds == query_labels).float().mean()                     # scalar
    return EpisodeOutput(
        log_p_y=log_p_y,
        loss=loss,
        accuracy=accuracy,
        distances=distances,
        prototypes=prototypes,
    )


class PrototypicalNetwork(nn.Module):
    """Embedding backbone + prototypical classification head.

    The module is parameter-light on purpose: all class-specific behaviour comes
    from the support set at inference time, so the same trained encoder can be
    re-pointed at any 5 reference prototypes inside the live DMOE pipeline.
    """

    def __init__(
        self,
        *,
        in_channels: int = 10,
        hidden_dim: int = 64,
        encoder: str = "cnn",
        dropout: float = 0.2,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        encoder = encoder.lower()
        if encoder == "cnn":
            self.encoder: nn.Module = TemporalCNNEncoder(
                in_channels=in_channels, hidden_dim=hidden_dim, dropout=dropout
            )
        elif encoder == "gru":
            self.encoder = GRUEncoder(
                in_channels=in_channels, hidden_dim=hidden_dim, dropout=dropout
            )
        else:
            raise ValueError(f"Unknown encoder '{encoder}'; use 'cnn' or 'gru'.")

        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.encoder_name = encoder

        if device is not None:
            self.to(device)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Project a batch ``(B, T, C)`` into the embedding space ``(B, D)``."""
        if x.dim() != 3:
            raise ValueError(f"Expected (B, T, C) input; got shape {tuple(x.shape)}")
        if x.size(-1) != self.in_channels:
            raise ValueError(
                f"Channel mismatch: encoder expects C={self.in_channels}, "
                f"got {x.size(-1)}"
            )
        return self.encoder(x)

    def proto_episode(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        n_way: int,
    ) -> EpisodeOutput:
        """Full prototypical pass over one episode.

        Shapes:
            support_x: ``(n_way * k_shot, seq_len, channels)``
            support_y: ``(n_way * k_shot,)``
            query_x:   ``(num_query, seq_len, channels)``
            query_y:   ``(num_query,)``
        """
        # 1. Embed support and query items in the shared metric space.
        support_emb = self.embed(support_x)  # (n_way*k_shot, D)
        query_emb = self.embed(query_x)      # (num_query, D)

        # 2. Collapse k_shot support embeddings per class into one centroid.
        prototypes = compute_prototypes(support_emb, support_y, n_way)  # (n_way, D)

        # 3. Distances -> log-softmax -> CE loss + accuracy.
        return prototypical_loss(query_emb, query_y, prototypes)

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        n_way: int,
    ) -> EpisodeOutput:
        return self.proto_episode(support_x, support_y, query_x, query_y, n_way)

    @torch.no_grad()
    def prototypes_from_support(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        """Lock down prototype coordinates from a fixed reference support pool.

        Used by the production inference path: the 5 prototypes are computed
        once from clean ground-truth examples and then reused for the entire
        incoming query stream.
        """
        self.eval()
        support_emb = self.embed(support_x)  # (n_way*k_shot, D)
        return compute_prototypes(support_emb, support_y, n_way)  # (n_way, D)

    @torch.no_grad()
    def prototypes_and_inv_variances_from_support(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        n_way: int,
        *,
        shrinkage: float = 0.5,
        eps: float = 1e-4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lock prototypes *and* inverse diagonal variances from a support pool.

        The Mahalanobis counterpart of :meth:`prototypes_from_support`: returns
        everything the live DMOE pipeline needs to register as buffers for the
        elliptical (diagonal-covariance) metric.

        Args:
            support_x: ``(n_way * k_shot, seq_len, channels)``.
            support_y: ``(n_way * k_shot,)`` relative ids in ``[0, n_way)``.
            n_way: number of classes.
            shrinkage: per-class vs. pooled variance blend (default 0.5).
            eps: variance floor (default 1e-4).

        Returns:
            prototypes: ``(n_way, embedding_dim)``.
            inv_variances: ``(n_way, embedding_dim)`` strictly positive
            ``1 / sigma^2`` for :func:`squared_mahalanobis_distance`.
        """
        self.eval()
        support_emb = self.embed(support_x)  # (n_way*k_shot, D)
        prototypes, variances = compute_prototypes_and_variances(
            support_emb, support_y, n_way, shrinkage=shrinkage, eps=eps
        )
        return prototypes, 1.0 / variances

    @torch.no_grad()
    def classify_against_prototypes(
        self,
        query_x: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Predict relative class ids for a query batch given fixed prototypes.

        Args:
            query_x: ``(B, seq_len, channels)``.
            prototypes: ``(n_way, embedding_dim)``.

        Returns:
            ``(B,)`` predicted relative class ids (nearest-prototype).
        """
        self.eval()
        query_emb = self.embed(query_x)                              # (B, D)
        distances = squared_euclidean_distance(query_emb, prototypes)  # (B, n_way)
        # Nearest prototype == smallest distance == largest negative distance.
        return distances.argmin(dim=1)                               # (B,)


class Level3LinearExpert(nn.Module):
    """Frozen ProtoNet encoder + trainable linear head (Option #4).

    Stores ``-logits`` as pseudo-distances so existing DMOE decode
    ``softmax(-d / t3)`` and meta features ``-l3_distances`` stay compatible.
    """

    def __init__(
        self,
        encoder: PrototypicalNetwork,
        *,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.head = nn.Linear(int(encoder.hidden_dim), int(num_classes))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.embed(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder.embed(x))

    @torch.no_grad()
    def pseudo_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Negative logits shaped like L3 prototype distances (lower = closer)."""
        self.eval()
        return -self.forward(x)

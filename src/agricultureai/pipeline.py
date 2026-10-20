"""Stage-4 System Integration: the unified 4-Tier DMOE inference pipeline.

``DMOEPipeline`` chains five standalone models via soft probabilistic routing:

    x (B, T, C)
      |
      v
    [L1 Generalist] -> P_L1 over 4 super-classes
      |-- P(Rapeseed) / P(Corn) ---------------------------> global raw {2, 3}
      |-- P(Cereal) > tau -> [L2a] -> barley / wheat       -> global {0, 1}
      `-- P(Grass) > tau -> [L2b Anomaly Gate]
              |-- P(Grass) branch > tau -> [L2c Meadow]     -> global {7, 8}
              `-- P(Anomaly) branch > tau -> [L3 ProtoNet]  -> global {4, 5, 6}
                  (L3 meadow locals {0,1} accumulate onto {7,8} as rejection mass)

Joint composition (per sample)::

    P(PM) = P_L1(grass) * P_L2b(grass) * P_L2c(PM)
    P(TM) = P_L1(grass) * P_L2b(grass) * P_L2c(TM)
    P(tree_k) = P_L1(grass) * P_L2b(anomaly) * P_L3(tree_k)
    P(meadow_k) += P_L1(grass) * P_L2b(anomaly) * P_L3(meadow_k)   # L3 rejection

Temperatures ``t1``, ``t2b``, ``t2c``, ``t3`` scale each stage before the product.
Cost-sensitive decode adds ``b_meadow`` / ``b_tree`` in log-space before argmax.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from agricultureai.data.breizhcrops_l2b_anomaly_gate import L2B_ANOMALY_LOCAL, L2B_GRASS_LOCAL
from agricultureai.data.breizhcrops_l2c_meadow import MEADOW_LOCAL_TO_GLOBAL
from agricultureai.data.breizhcrops_series import infer_padding_mask_from_features
from agricultureai.models.breizhcrops_gltae import BreizhCropsGLTAEExpert
from agricultureai.models.proto_net import (
    squared_euclidean_distance,
    squared_mahalanobis_distance,
)

# --- Super-class space (L1 Generalist output) ---------------------------------
SUPER_CEREAL_GROUP = 0
SUPER_RAPESEED = 1
SUPER_CORN = 2
SUPER_GRASSLAND_GROUP = 3

# --- Global BreizhCrops raw ids (final prediction space) ----------------------
GLOBAL_BARLEY = 0
GLOBAL_WHEAT = 1
GLOBAL_RAPESEED = 2
GLOBAL_CORN = 3
GLOBAL_SUNFLOWER = 4
GLOBAL_ORCHARDS = 5
GLOBAL_NUTS = 6
GLOBAL_PERMANENT_MEADOWS = 7
GLOBAL_TEMPORARY_MEADOWS = 8

NUM_GLOBAL_CLASSES = 9

MEADOW_GLOBAL_IDS: Tuple[int, int] = (GLOBAL_PERMANENT_MEADOWS, GLOBAL_TEMPORARY_MEADOWS)
TREE_GLOBAL_IDS: Tuple[int, int, int] = (GLOBAL_SUNFLOWER, GLOBAL_ORCHARDS, GLOBAL_NUTS)

# --- Local-expert -> global raw id mappings -----------------------------------
CEREAL_LOCAL_TO_GLOBAL: Dict[int, int] = {
    0: GLOBAL_BARLEY,
    1: GLOBAL_WHEAT,
}

L3_LOCAL_TO_GLOBAL: Dict[int, int] = {
    0: GLOBAL_PERMANENT_MEADOWS,
    1: GLOBAL_TEMPORARY_MEADOWS,
    2: GLOBAL_ORCHARDS,
    3: GLOBAL_NUTS,
    4: GLOBAL_SUNFLOWER,
}

# Legacy aliases (3-tier grassland expert); kept for older scripts/tests.
GRASSLAND_LOCAL_TO_GLOBAL = MEADOW_LOCAL_TO_GLOBAL
GRASSLAND_ANOMALY_LOCAL_INDEX = L2B_ANOMALY_LOCAL


def _lookup_table(mapping: Dict[int, int], size: int) -> torch.Tensor:
    lut = torch.zeros(size, dtype=torch.long)
    for local_id, global_id in mapping.items():
        lut[int(local_id)] = int(global_id)
    return lut


class DMOEPipeline(nn.Module):
    """Master orchestrator chaining L1 -> {L2a, L2b -> L2c / L3} via soft routing."""

    def __init__(
        self,
        l1_model: nn.Module,
        l2a_cereal: nn.Module,
        l2b_anomaly_gate: nn.Module,
        l2c_meadow: nn.Module,
        l3_protonet: nn.Module,
        support_prototypes: torch.Tensor,
        *,
        support_inv_variances: Optional[torch.Tensor] = None,
        anomaly_threshold: float = 0.20,
        t1: float = 1.0,
        t2b: float = 1.0,
        t2c: float = 1.0,
        t3: float = 1.0,
        b_meadow: float = 0.0,
        b_tree: float = 0.0,
        tau_route: float = 0.05,
    ) -> None:
        super().__init__()
        if support_prototypes.dim() != 2:
            raise ValueError(
                f"support_prototypes must be (n_way, hidden_dim); got "
                f"{tuple(support_prototypes.shape)}"
            )
        if support_prototypes.size(0) != len(L3_LOCAL_TO_GLOBAL):
            raise ValueError(
                f"support_prototypes must have {len(L3_LOCAL_TO_GLOBAL)} rows; got "
                f"{support_prototypes.size(0)}"
            )
        for name, value in (("t1", t1), ("t2b", t2b), ("t2c", t2c), ("t3", t3)):
            if float(value) <= 0.0:
                raise ValueError(f"{name} must be > 0; got {value}")
        if not 0.0 <= float(tau_route) < 1.0:
            raise ValueError(f"tau_route must be in [0, 1); got {tau_route}")

        self.l1_model = l1_model
        self.l2a_cereal = l2a_cereal
        self.l2b_anomaly_gate = l2b_anomaly_gate
        self.l2c_meadow = l2c_meadow
        self.l3_protonet = l3_protonet

        self.anomaly_threshold = float(anomaly_threshold)
        self.t1 = float(t1)
        self.t2b = float(t2b)
        self.t2c = float(t2c)
        self.t3 = float(t3)
        self.b_meadow = float(b_meadow)
        self.b_tree = float(b_tree)
        self.tau_route = float(tau_route)

        self.register_buffer("support_prototypes", support_prototypes.detach().clone())

        if support_inv_variances is not None:
            if support_inv_variances.shape != support_prototypes.shape:
                raise ValueError(
                    f"support_inv_variances shape "
                    f"{tuple(support_inv_variances.shape)} must match "
                    f"support_prototypes shape {tuple(support_prototypes.shape)}"
                )
            if not bool((support_inv_variances > 0).all()):
                raise ValueError("support_inv_variances must be strictly positive")
            self.register_buffer(
                "support_inv_variances", support_inv_variances.detach().clone()
            )
        else:
            self.register_buffer("support_inv_variances", None)

        self.register_buffer(
            "cereal_local_to_global_lut",
            _lookup_table(CEREAL_LOCAL_TO_GLOBAL, size=2),
            persistent=False,
        )
        self.register_buffer(
            "meadow_local_to_global_lut",
            _lookup_table(MEADOW_LOCAL_TO_GLOBAL, size=2),
            persistent=False,
        )
        self.register_buffer(
            "l3_local_to_global_lut",
            _lookup_table(L3_LOCAL_TO_GLOBAL, size=len(L3_LOCAL_TO_GLOBAL)),
            persistent=False,
        )

        self.last_l3_count: int = 0

    def _l2b_forward(
        self,
        grass_x: torch.Tensor,
        doy: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if padding_mask is None:
            padding_mask = infer_padding_mask_from_features(grass_x)
        if doy is None:
            doy = torch.zeros(
                grass_x.size(0),
                grass_x.size(1),
                dtype=torch.long,
                device=grass_x.device,
            )
        if isinstance(self.l2b_anomaly_gate, BreizhCropsGLTAEExpert):
            return self.l2b_anomaly_gate(grass_x, doy, padding_mask)
        return self.l2b_anomaly_gate(grass_x)

    def _l1_forward(
        self,
        x: torch.Tensor,
        doy: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # A DOY-aware L1 silently falls back to all-zero DOY when none is given,
        # which misroutes cereals; require it, as the cache extraction passes it.
        if getattr(self.l1_model, "use_doy", False):
            if doy is None:
                raise ValueError("L1 model uses day-of-year; pass doy (and padding_mask).")
            return self.l1_model(x, doy=doy, padding_mask=padding_mask)
        return self.l1_model(x)

    def _l3_distances(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.l3_protonet, "pseudo_distances"):
            return self.l3_protonet.pseudo_distances(x)
        embeddings = self.l3_protonet.embed(x)
        if self.support_inv_variances is not None:
            return squared_mahalanobis_distance(
                embeddings, self.support_prototypes, self.support_inv_variances
            )
        return squared_euclidean_distance(embeddings, self.support_prototypes)

    @staticmethod
    def _mean_over_views(fn, x: torch.Tensor, views: Optional[torch.Tensor]) -> torch.Tensor:
        """``fn(x)``, or the mean of ``fn`` over resampled ``views`` ``(B, V, T, C)``."""
        if views is None:
            return fn(x)
        b, v = views.shape[:2]
        return fn(views.reshape(b * v, *views.shape[2:])).view(b, v, -1).mean(dim=1)

    @torch.inference_mode()
    def forward_probs(
        self,
        x: torch.Tensor,
        doy: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        expert_views: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Joint 9-class probabilities.

        ``x`` / ``doy`` / ``padding_mask`` are the zero-padded series L1, L2b and
        L2c were trained on. L2a and L3 were trained through the stock
        BreizhCrops transform (fixed-length resampling, no padding); pass
        ``expert_views`` ``(B, V, T, C)`` with resampled views of each parcel and
        their logits / distances are averaged over the V views. Without it they
        see ``x`` (legacy behaviour).
        """
        if x.dim() != 3:
            raise ValueError(f"Expected (B, T, C) input; got shape {tuple(x.shape)}")
        if expert_views is not None and (
            expert_views.dim() != 4
            or expert_views.size(0) != x.size(0)
            or expert_views.size(-1) != x.size(-1)
        ):
            raise ValueError(
                f"expert_views must be (B, V, T, C) matching x {tuple(x.shape)}; "
                f"got {tuple(expert_views.shape)}"
            )

        batch_size = x.size(0)
        self.last_l3_count = 0
        probs = x.new_zeros((batch_size, NUM_GLOBAL_CLASSES))

        l1_logits = self._l1_forward(x, doy, padding_mask)
        l1_probs = torch.softmax(l1_logits / self.t1, dim=1)
        p_cereal = l1_probs[:, SUPER_CEREAL_GROUP]
        p_grass = l1_probs[:, SUPER_GRASSLAND_GROUP]

        probs[:, GLOBAL_RAPESEED] = l1_probs[:, SUPER_RAPESEED]
        probs[:, GLOBAL_CORN] = l1_probs[:, SUPER_CORN]

        # --- L2a cereal branch ------------------------------------------------
        run_l2a = p_cereal > self.tau_route
        if bool(run_l2a.any()):
            cereal_rows = run_l2a.nonzero(as_tuple=True)[0]
            cereal_logits = self._mean_over_views(
                self.l2a_cereal,
                x[run_l2a],
                expert_views[run_l2a] if expert_views is not None else None,
            )
            cereal_probs = torch.softmax(cereal_logits, dim=1)
            cereal_joint = cereal_probs * p_cereal[run_l2a].unsqueeze(1)
            probs[
                cereal_rows.unsqueeze(1),
                self.cereal_local_to_global_lut.unsqueeze(0),
            ] = cereal_joint

        # --- L2b anomaly gate + L2c meadow + L3 rescue ------------------------
        run_l2b = p_grass > self.tau_route
        if bool(run_l2b.any()):
            grass_rows = run_l2b.nonzero(as_tuple=True)[0]
            grass_x = x[run_l2b]
            grass_doy = doy[run_l2b] if doy is not None else None
            grass_mask = padding_mask[run_l2b] if padding_mask is not None else None
            grass_views = expert_views[run_l2b] if expert_views is not None else None

            gate_logits = self._l2b_forward(grass_x, grass_doy, grass_mask)
            gate_probs = torch.softmax(gate_logits / self.t2b, dim=1)
            p_grass_sub = p_grass[run_l2b]
            p_l2b_grass = gate_probs[:, L2B_GRASS_LOCAL]
            p_l2b_anomaly = gate_probs[:, L2B_ANOMALY_LOCAL]

            grass_path_mass = p_grass_sub * p_l2b_grass
            anomaly_mass = p_grass_sub * p_l2b_anomaly

            # L2c meadow expert on the grass branch mass.
            run_l2c = grass_path_mass > self.tau_route
            if bool(run_l2c.any()):
                l2c_rows = grass_rows[run_l2c]
                meadow_logits = self.l2c_meadow(
                    grass_x[run_l2c],
                    doy=grass_doy[run_l2c] if grass_doy is not None else None,
                    padding_mask=grass_mask[run_l2c] if grass_mask is not None else None,
                )
                meadow_probs = torch.softmax(meadow_logits / self.t2c, dim=1)
                meadow_joint = meadow_probs * grass_path_mass[run_l2c].unsqueeze(1)
                probs[
                    l2c_rows.unsqueeze(1),
                    self.meadow_local_to_global_lut.unsqueeze(0),
                ] = meadow_joint

            # L3 ProtoNet on the anomaly branch mass (meadow locals accumulate).
            run_l3 = anomaly_mass > self.tau_route
            if bool(run_l3.any()):
                l3_rows = grass_rows[run_l3]
                distances = self._mean_over_views(
                    self._l3_distances,
                    grass_x[run_l3],
                    grass_views[run_l3] if grass_views is not None else None,
                )
                l3_probs = torch.softmax(-distances / self.t3, dim=1)
                l3_joint = l3_probs * anomaly_mass[run_l3].unsqueeze(1)
                probs[
                    l3_rows.unsqueeze(1),
                    self.l3_local_to_global_lut.unsqueeze(0),
                ] += l3_joint
                self.last_l3_count = int(run_l3.sum().item())

        return probs

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        doy: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        expert_views: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        probs = self.forward_probs(
            x, doy=doy, padding_mask=padding_mask, expert_views=expert_views
        )
        log_p = torch.log(probs + 1e-9)
        log_p[:, list(MEADOW_GLOBAL_IDS)] += self.b_meadow
        log_p[:, list(TREE_GLOBAL_IDS)] += self.b_tree
        return log_p.argmax(dim=-1)

# --------------------------------------------------------------------------- #
# Decoding cached branch outputs (no model forward pass)                      #
# --------------------------------------------------------------------------- #
CEREAL_COLS = [CEREAL_LOCAL_TO_GLOBAL[i] for i in range(len(CEREAL_LOCAL_TO_GLOBAL))]
MEADOW_COLS = list(MEADOW_GLOBAL_IDS)
L3_COLS = [L3_LOCAL_TO_GLOBAL[i] for i in range(len(L3_LOCAL_TO_GLOBAL))]


def decode_predictions(
    l1_logits: torch.Tensor,
    l2a_logits: torch.Tensor,
    l2b_logits: torch.Tensor,
    l2c_logits: torch.Tensor,
    l3_distances: torch.Tensor,
    *,
    t1: float,
    t2b: float,
    t2c: float,
    t3: float,
    tau_route: float,
    b_meadow: float,
    b_tree: float,
) -> torch.Tensor:
    """Vectorized 4-tier DMOEPipeline forward_probs + cost-sensitive decode."""
    n = l1_logits.size(0)
    probs = l1_logits.new_zeros((n, NUM_GLOBAL_CLASSES))

    l1_probs = torch.softmax(l1_logits / t1, dim=1)
    p_cereal = l1_probs[:, SUPER_CEREAL_GROUP]
    p_grass = l1_probs[:, SUPER_GRASSLAND_GROUP]

    probs[:, GLOBAL_RAPESEED] = l1_probs[:, SUPER_RAPESEED]
    probs[:, GLOBAL_CORN] = l1_probs[:, SUPER_CORN]

    gate_l2a = (p_cereal > tau_route).to(probs.dtype)
    l2a_probs = torch.softmax(l2a_logits, dim=1)
    cereal_joint = l2a_probs * (p_cereal * gate_l2a).unsqueeze(1)
    probs[:, CEREAL_COLS] = cereal_joint

    gate_l2b = (p_grass > tau_route).to(probs.dtype)
    l2b_probs = torch.softmax(l2b_logits / t2b, dim=1)
    grass_mass = p_grass * gate_l2b
    p_l2b_grass = l2b_probs[:, L2B_GRASS_LOCAL]
    p_l2b_anomaly = l2b_probs[:, L2B_ANOMALY_LOCAL]

    grass_path_mass = grass_mass * p_l2b_grass
    anomaly_mass = grass_mass * p_l2b_anomaly

    gate_l2c = (grass_path_mass > tau_route).to(probs.dtype)
    l2c_probs = torch.softmax(l2c_logits / t2c, dim=1)
    meadow_joint = l2c_probs * (grass_path_mass * gate_l2c).unsqueeze(1)
    probs[:, MEADOW_COLS] = meadow_joint

    gate_l3 = (anomaly_mass > tau_route).to(probs.dtype)
    l3_probs = torch.softmax(-l3_distances / t3, dim=1)
    l3_joint = l3_probs * (anomaly_mass * gate_l3).unsqueeze(1)
    probs[:, L3_COLS] += l3_joint

    log_p = torch.log(probs + 1e-9)
    log_p[:, list(MEADOW_GLOBAL_IDS)] += b_meadow
    log_p[:, list(TREE_GLOBAL_IDS)] += b_tree
    return log_p.argmax(dim=-1)

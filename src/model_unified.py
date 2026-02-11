"""
Engine C — Unified 1D-2D Coupled HeteroGNN-GRU  (Tier 2 / Fallback).

Architecture
------------
A **Heterogeneous Graph Neural Network** with per-node-type GRU cells,
explicitly modelling the physical coupling between the underground pipe
network (1D) and the surface terrain (2D).

The graph consumed by this model is produced by
``graph_builder_unified.build_unified_graph()`` and contains **four
directed edge types**:

    1. ``(node_1d, pipe_to, node_1d)``     — bidirectional pipe flow
    2. ``(node_2d, surface_to, node_2d)``  — surface mesh adjacency
    3. ``(node_1d, surcharges_to, node_2d)``— surcharge (pipe → street)
    4. ``(node_2d, drains_to, node_1d)``   — drainage  (street → pipe)

Separating surcharge from drainage lets the model learn distinct
message-passing functions for pressure-driven overflow vs gravity-driven
inlet flow — they are fundamentally different physics.

Target Space (v4 — Depth-Based)
-------------------------------
The model predicts **depth** (water height above a physical reference):
  - 1D: ``depth = WSE − invert_elevation``  (height above pipe invert)
  - 2D: ``depth = WSE − min_elevation``     (water depth above ground)

This eliminates the train-inference mismatch that existed in the old
anomaly-based approach (WSE − WSE(t=0)), where dynamic features were
computed differently during teacher forcing vs student forcing.  Now:
  - 1D dynamic features: ``relative_depth = depth`` (identical to target),
    ``fill_ratio = depth / capacity``
  - 2D dynamic features: ``rainfall, water_depth = depth, water_volume``

Recovery at inference: ``WSE = predicted_depth + elevation_reference``

Data Flow (single timestep)
---------------------------
1. **Feature Assembly** — concatenate static node features (from
   ``data[nt].x``) with the current timestep's dynamic features
   (from ``data[nt].dynamic[t]``) to form the per-node input.
2. **Node Encoders** — separate ``Linear`` projections map the
   heterogeneous feature spaces (1D: 5+3=8, 2D: 9+3=12) into a
   shared hidden dimension ``H``.
3. **HeteroConv Stack** — ``L`` rounds of heterogeneous message
   passing with *edge-conditioned* convolutions.  Each layer uses
   ``GCNConv`` for pipes, ``SAGEConv`` for surfaces, and ``SAGEConv``
   for the two coupling directions.  Residual connections + LayerNorm
   stabilise deep stacks.
4. **Temporal GRU** — per-node-type GRU cells maintain temporal memory
   across autoregressive steps.  Hidden state ``h`` carries the
   "volume of water" context from previous timesteps.
5. **Prediction Heads** — separate 2-layer MLPs decode the GRU output
   into scalar water-level predictions for 1D and 2D nodes.

Autoregressive Inference
------------------------
During the *spin-up* phase (t=1…10 in the test set), ground-truth
dynamic features are fed to build up the GRU hidden states.
During the *prediction* phase (t=11…end), the model's own predictions
are fed back as the dynamic input for the next step.

The ``rollout()`` method implements both phases and supports
**Scheduled Sampling** (curriculum learning) for training: a
``teacher_forcing_ratio`` controls the probability of using
ground-truth vs model predictions at each step.

Owner : Member C (Lead Architect)
See   : IMPLEMENTATION_PLAN.md → Task 2.4, PROJECT_BIBLE.md §6 Tier 2
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv


# =====================================================================
#  Constants
# =====================================================================

# Edge type tuples (must match graph_builder_unified exactly)
_PIPE_ET: Tuple[str, str, str] = ("node_1d", "pipe_to", "node_1d")
_SURFACE_ET: Tuple[str, str, str] = ("node_2d", "surface_to", "node_2d")
_SURCHARGE_ET: Tuple[str, str, str] = ("node_1d", "surcharges_to", "node_2d")
_DRAINAGE_ET: Tuple[str, str, str] = ("node_2d", "drains_to", "node_1d")


# =====================================================================
#  Building Blocks
# =====================================================================

class _NodeEncoder(nn.Module):
    """Projects heterogeneous node features into a shared hidden space.

    Supports fusing static features (``x``, fixed across time) with
    per-timestep dynamic features, then projecting to ``hidden_channels``.

    Parameters
    ----------
    static_dim : int
        Dimensionality of the static feature vector (e.g. 5 for 1D).
    dynamic_dim : int
        Dimensionality of the per-timestep dynamic feature vector
        (e.g. 3 for 1D: relative_depth, fill_ratio, inlet_flow).
    hidden_channels : int
        Output dimensionality.
    """

    def __init__(
        self,
        static_dim: int,
        dynamic_dim: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        in_dim = static_dim + dynamic_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(negative_slope=0.01),
        )

    def forward(self, x_static: Tensor, x_dynamic: Tensor) -> Tensor:
        """Concatenate static + dynamic and project.

        Parameters
        ----------
        x_static  : Tensor [N, F_static]
        x_dynamic : Tensor [N, F_dynamic]  (for a single timestep)

        Returns
        -------
        Tensor [N, H]
        """
        return self.proj(torch.cat([x_static, x_dynamic], dim=-1))


class _HeteroGNNLayer(nn.Module):
    """One layer of heterogeneous message passing over all 4 edge types.

    Uses ``GCNConv`` for intra-domain edges (pipes, surface) where the
    graph is typically regular, and ``SAGEConv`` for inter-domain coupling
    edges where the degree distribution is highly irregular (a few pipes
    connect to many surface cells).

    Includes LayerNorm + residual connection for training stability
    in deep stacks.

    Parameters
    ----------
    hidden_channels : int
        Input and output channel count (residual requires same dim).
    dropout : float
        Dropout probability applied after activation.
    """

    def __init__(self, hidden_channels: int, dropout: float = 0.1) -> None:
        super().__init__()

        # Build per-edge-type convolutions
        self.conv = HeteroConv(
            {
                # Intra-1D: GCN captures neighbourhood aggregation
                # over regular pipe topology
                _PIPE_ET: GCNConv(
                    hidden_channels, hidden_channels, add_self_loops=False
                ),
                # Intra-2D: SAGEConv handles large irregular mesh
                _SURFACE_ET: SAGEConv(
                    hidden_channels, hidden_channels
                ),
                # Inter-domain: surcharge (1D → 2D)
                _SURCHARGE_ET: SAGEConv(
                    (hidden_channels, hidden_channels), hidden_channels
                ),
                # Inter-domain: drainage (2D → 1D)
                _DRAINAGE_ET: SAGEConv(
                    (hidden_channels, hidden_channels), hidden_channels
                ),
            },
            aggr="sum",
        )

        # Per-node-type normalisation (applied after aggregation)
        self.norm_1d = nn.LayerNorm(hidden_channels)
        self.norm_2d = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[str, Tensor]:
        """Forward pass: conv → residual + norm → LeakyReLU → dropout.

        Parameters
        ----------
        x_dict : dict
            ``{"node_1d": [N_1d, H], "node_2d": [N_2d, H]}``
        edge_index_dict : dict
            Edge indices for all 4 edge types.

        Returns
        -------
        dict  ``{"node_1d": [N_1d, H], "node_2d": [N_2d, H]}``
        """
        # Message passing (all edge types simultaneously)
        out_dict = self.conv(x_dict, edge_index_dict)

        # Residual + LayerNorm + activation  (per node type)
        result: Dict[str, Tensor] = {}
        for ntype, norm in [("node_1d", self.norm_1d), ("node_2d", self.norm_2d)]:
            if ntype in out_dict:
                h = norm(out_dict[ntype] + x_dict[ntype])  # residual
                h = F.leaky_relu(h, negative_slope=0.01)
                h = self.dropout(h)
                result[ntype] = h
            else:
                # Fallback: if no edges touch this node type (degenerate graph)
                result[ntype] = x_dict[ntype]

        return result


class _PredictionHead(nn.Module):
    """2-layer MLP that decodes GRU output into scalar water level.

    Architecture: Linear → LeakyReLU → Linear → output.
    LeakyReLU is critical for 2D nodes that are mostly dry.

    Parameters
    ----------
    hidden_channels : int
    """

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, h: Tensor) -> Tensor:
        """Decode hidden state to scalar prediction.

        Parameters
        ----------
        h : Tensor [N, H]

        Returns
        -------
        Tensor [N]
        """
        return self.mlp(h).squeeze(-1)


# =====================================================================
#  Main Model
# =====================================================================

class UnifiedFloodModel(nn.Module):
    """Heterogeneous GNN-GRU for coupled 1D-2D flood prediction.

    This model explicitly models the physical interaction between the
    underground pipe network and the surface terrain via heterogeneous
    message-passing edges.  It is the "Heavy Weapon" deployed when
    the decoupled Twin Engines fail on surcharge-dominated events.

    Parameters
    ----------
    in_channels_1d_static : int
        Number of static features per 1D node (default 5).
    in_channels_1d_dynamic : int
        Number of dynamic features per 1D node per timestep (default 3).
    in_channels_2d_static : int
        Number of static features per 2D node (default 9).
    in_channels_2d_dynamic : int
        Number of dynamic features per 2D node per timestep (default 3).
    hidden_channels : int
        Dimensionality of GRU hidden state and GNN embeddings (default 64).
    num_gnn_layers : int
        Number of heterogeneous message-passing layers (default 3).
    num_gru_layers : int
        Number of stacked GRU layers (default 1).
    dropout : float
        Dropout rate between GNN layers and in GRU (default 0.1).

    Attributes
    ----------
    encoder_1d, encoder_2d : _NodeEncoder
        Project heterogeneous features into shared hidden space.
    gnn_layers : nn.ModuleList[_HeteroGNNLayer]
        Stack of heterogeneous message-passing layers.
    gru_1d, gru_2d : nn.GRUCell (or stacked)
        Per-node-type temporal memory.
    head_1d, head_2d : _PredictionHead
        Decode GRU output → scalar water level.

    Example
    -------
    >>> from src.graph_builder_unified import build_unified_graph, get_feature_dims
    >>> hetero = build_unified_graph(sample)
    >>> dims = get_feature_dims(hetero)
    >>> model = UnifiedFloodModel(**dims)
    >>> preds_1d, preds_2d, h_1d, h_2d = model.step(hetero, t=0)
    """

    def __init__(
        self,
        in_channels_1d_static: int = 5,
        in_channels_1d_dynamic: int = 3,
        in_channels_2d_static: int = 9,
        in_channels_2d_dynamic: int = 3,
        hidden_channels: int = 64,
        num_gnn_layers: int = 3,
        num_gru_layers: int = 1,
        dropout: float = 0.1,
        **kwargs: Any,  # absorb extra keys from get_feature_dims()
    ) -> None:
        super().__init__()

        # Store hyperparameters for serialisation / logging
        self.hidden_channels = hidden_channels
        self.num_gnn_layers = num_gnn_layers
        self.num_gru_layers = num_gru_layers
        self.in_channels_1d_static = in_channels_1d_static
        self.in_channels_1d_dynamic = in_channels_1d_dynamic
        self.in_channels_2d_static = in_channels_2d_static
        self.in_channels_2d_dynamic = in_channels_2d_dynamic

        # ── Node Encoders ──────────────────────────────────────────
        self.encoder_1d = _NodeEncoder(
            in_channels_1d_static, in_channels_1d_dynamic, hidden_channels
        )
        self.encoder_2d = _NodeEncoder(
            in_channels_2d_static, in_channels_2d_dynamic, hidden_channels
        )

        # ── Heterogeneous GNN Stack ────────────────────────────────
        self.gnn_layers = nn.ModuleList([
            _HeteroGNNLayer(hidden_channels, dropout=dropout)
            for _ in range(num_gnn_layers)
        ])

        # ── Temporal GRU Cells (per node type) ─────────────────────
        # We use GRUCell (not GRU module) for step-by-step control
        # during autoregressive rollout.  For multi-layer GRU we
        # stack cells manually.
        self.gru_1d = nn.ModuleList([
            nn.GRUCell(hidden_channels, hidden_channels)
            for _ in range(num_gru_layers)
        ])
        self.gru_2d = nn.ModuleList([
            nn.GRUCell(hidden_channels, hidden_channels)
            for _ in range(num_gru_layers)
        ])

        # Layer norms after GRU for stable hidden-state evolution
        self.gru_norm_1d = nn.LayerNorm(hidden_channels)
        self.gru_norm_2d = nn.LayerNorm(hidden_channels)

        # ── Prediction Heads ───────────────────────────────────────
        self.head_1d = _PredictionHead(hidden_channels)
        self.head_2d = _PredictionHead(hidden_channels)

        # ── Dropout for GRU inter-layer ────────────────────────────
        self.gru_dropout = nn.Dropout(dropout) if num_gru_layers > 1 else nn.Identity()

        # ── Initialisation ─────────────────────────────────────────
        self._init_weights()

    # ------------------------------------------------------------------ #
    #  Weight Initialisation
    # ------------------------------------------------------------------ #

    def _init_weights(self) -> None:
        """Xavier uniform for linear layers; orthogonal for GRU.

        Orthogonal init for recurrent weights is a well-known trick to
        improve gradient flow in long autoregressive rollouts (prevents
        vanishing/exploding gradients better than default).
        """
        for name, param in self.named_parameters():
            if "weight" in name:
                if "gru" in name:
                    # Orthogonal init for GRU recurrent weights
                    if param.dim() >= 2:
                        nn.init.orthogonal_(param)
                elif param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    # ------------------------------------------------------------------ #
    #  Hidden State Management
    # ------------------------------------------------------------------ #

    def init_hidden(
        self,
        n_1d: int,
        n_2d: int,
        device: torch.device | None = None,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Create zero-initialised GRU hidden states.

        Returns
        -------
        (h_1d_layers, h_2d_layers)
            Each is a list of length ``num_gru_layers`` containing
            tensors of shape ``[N, H]``.
        """
        if device is None:
            device = next(self.parameters()).device

        h_1d = [
            torch.zeros(n_1d, self.hidden_channels, device=device)
            for _ in range(self.num_gru_layers)
        ]
        h_2d = [
            torch.zeros(n_2d, self.hidden_channels, device=device)
            for _ in range(self.num_gru_layers)
        ]
        return h_1d, h_2d

    # ------------------------------------------------------------------ #
    #  Single-Step Forward
    # ------------------------------------------------------------------ #

    def step(
        self,
        data: HeteroData,
        t: int,
        h_1d: Optional[List[Tensor]] = None,
        h_2d: Optional[List[Tensor]] = None,
        *,
        override_dynamic_1d: Optional[Tensor] = None,
        override_dynamic_2d: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor]]:
        """Run a single autoregressive step at timestep ``t``.

        This is the core computational unit.  The ``rollout()`` method
        calls this in a loop.

        Parameters
        ----------
        data : HeteroData
            The heterogeneous graph from ``build_unified_graph()``.
        t : int
            Current timestep index into ``data[nt].dynamic``.
        h_1d, h_2d : list[Tensor] or None
            GRU hidden states from the previous step.  If ``None``,
            initialised to zeros.
        override_dynamic_1d : Tensor or None
            If provided, used instead of ``data['node_1d'].dynamic[t]``.
            This enables feeding model predictions back during
            autoregressive inference.
        override_dynamic_2d : Tensor or None
            Same, for 2D nodes.

        Returns
        -------
        (pred_1d, pred_2d, h_1d_new, h_2d_new)
            pred_1d : Tensor [N_1d]  — predicted water levels
            pred_2d : Tensor [N_2d]  — predicted water levels
            h_1d_new, h_2d_new : updated GRU hidden states
        """
        n_1d = data["node_1d"].num_nodes
        n_2d = data["node_2d"].num_nodes
        device = data["node_1d"].x.device

        # ── 0. Initialise hidden states if needed ─────────────────
        if h_1d is None or h_2d is None:
            h_1d, h_2d = self.init_hidden(n_1d, n_2d, device)

        # ── 1. Assemble per-timestep input features ───────────────
        x_static_1d = data["node_1d"].x                    # [N_1d, Fs]
        x_static_2d = data["node_2d"].x                    # [N_2d, Fs]

        if override_dynamic_1d is not None:
            x_dyn_1d = override_dynamic_1d                  # [N_1d, Fd]
        else:
            x_dyn_1d = data["node_1d"].dynamic[t]           # [N_1d, Fd]

        if override_dynamic_2d is not None:
            x_dyn_2d = override_dynamic_2d                  # [N_2d, Fd]
        else:
            x_dyn_2d = data["node_2d"].dynamic[t]           # [N_2d, Fd]

        # ── 2. Node Encoding ──────────────────────────────────────
        z_1d = self.encoder_1d(x_static_1d, x_dyn_1d)      # [N_1d, H]
        z_2d = self.encoder_2d(x_static_2d, x_dyn_2d)      # [N_2d, H]

        x_dict: Dict[str, Tensor] = {
            "node_1d": z_1d,
            "node_2d": z_2d,
        }

        # ── 3. Build edge_index_dict ──────────────────────────────
        edge_index_dict: Dict[Tuple[str, str, str], Tensor] = {}
        for et in [_PIPE_ET, _SURFACE_ET, _SURCHARGE_ET, _DRAINAGE_ET]:
            if et in data.edge_types:
                edge_index_dict[et] = data[et].edge_index

        # ── 4. Heterogeneous GNN Stack ────────────────────────────
        for gnn_layer in self.gnn_layers:
            x_dict = gnn_layer(x_dict, edge_index_dict)

        # ── 5. Temporal GRU (forced fp32) ──────────────────────────
        # GRU hidden states persist across the full rollout.  Under
        # fp16 autocast they can drift beyond 65504 and overflow to
        # NaN.  Force fp32 for GRU + prediction heads.  The GNN
        # layers above still benefit from fp16.
        h_1d_new: List[Tensor] = []
        h_2d_new: List[Tensor] = []

        gru_input_1d = x_dict["node_1d"].float()
        gru_input_2d = x_dict["node_2d"].float()

        with torch.amp.autocast(device.type, enabled=False):
            for layer_idx in range(self.num_gru_layers):
                h_1d_layer = self.gru_1d[layer_idx](
                    gru_input_1d, h_1d[layer_idx].float()
                )
                h_2d_layer = self.gru_2d[layer_idx](
                    gru_input_2d, h_2d[layer_idx].float()
                )

                h_1d_new.append(h_1d_layer)
                h_2d_new.append(h_2d_layer)

                # Inter-layer dropout (only between stacked layers)
                if layer_idx < self.num_gru_layers - 1:
                    gru_input_1d = self.gru_dropout(h_1d_layer)
                    gru_input_2d = self.gru_dropout(h_2d_layer)

            # Normalise final hidden state for stable autoregressive rollout
            h_final_1d = self.gru_norm_1d(h_1d_new[-1])
            h_final_2d = self.gru_norm_2d(h_2d_new[-1])

            # ── 6. Prediction Heads ───────────────────────────────
            pred_1d = self.head_1d(h_final_1d)              # [N_1d]
            pred_2d = self.head_2d(h_final_2d)              # [N_2d]

        return pred_1d, pred_2d, h_1d_new, h_2d_new

    # ------------------------------------------------------------------ #
    #  Autoregressive Rollout
    # ------------------------------------------------------------------ #

    def rollout(
        self,
        data: HeteroData,
        *,
        spinup_steps: int = 10,
        teacher_forcing_ratio: float = 0.0,
        prediction_steps: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Full autoregressive rollout over an event.

        This implements the competition inference protocol:
        1. **Spin-up** (t=0…spinup_steps-1): Feed ground-truth dynamic
           features to build up GRU hidden states.  Predictions are
           generated but *not* scored.
        2. **Prediction** (t=spinup_steps…end): Feed model's own
           predictions back as dynamic input.  These are the scored
           predictions.

        During training, ``teacher_forcing_ratio`` controls curriculum
        learning (Scheduled Sampling):
        - 1.0 = always use ground truth (teacher forcing)
        - 0.0 = always use model predictions (student forcing)
        - 0.5 = 50% chance of each (stochastic mix)

        Parameters
        ----------
        data : HeteroData
            Full event graph from ``build_unified_graph()``.
        spinup_steps : int
            Number of warm-up steps using ground truth (default 10).
        teacher_forcing_ratio : float
            Probability of using ground truth during prediction phase
            (default 0.0 = full autoregressive).
        prediction_steps : int or None
            If given, only predict this many steps after spinup.
            None = predict until end of event.

        Returns
        -------
        (preds_1d, preds_2d)
            preds_1d : Tensor [T_total, N_1d]
            preds_2d : Tensor [T_total, N_2d]
            Predictions for *all* timesteps (spinup + prediction).
        """
        T = data.num_timesteps
        n_1d = data["node_1d"].num_nodes
        n_2d = data["node_2d"].num_nodes
        device = data["node_1d"].x.device

        if prediction_steps is not None:
            T = min(T, spinup_steps + prediction_steps)

        # Pre-allocate output tensors
        all_preds_1d = torch.zeros(T, n_1d, device=device)
        all_preds_2d = torch.zeros(T, n_2d, device=device)

        # Initialise hidden states
        h_1d, h_2d = self.init_hidden(n_1d, n_2d, device)

        for t in range(T):
            if t < spinup_steps:
                # ── Spin-up: always use ground truth ──────────────
                pred_1d, pred_2d, h_1d, h_2d = self.step(
                    data, t, h_1d, h_2d
                )
            else:
                # ── Prediction phase: scheduled sampling ──────────
                use_teacher = (
                    self.training
                    and teacher_forcing_ratio > 0.0
                    and torch.rand(1).item() < teacher_forcing_ratio
                )

                if use_teacher:
                    # Teacher forcing: use ground truth
                    pred_1d, pred_2d, h_1d, h_2d = self.step(
                        data, t, h_1d, h_2d
                    )
                else:
                    # Student forcing: construct dynamic input from
                    # previous predictions
                    dyn_1d, dyn_2d = self._build_feedback_dynamic(
                        data, t, all_preds_1d[t - 1], all_preds_2d[t - 1]
                    )
                    pred_1d, pred_2d, h_1d, h_2d = self.step(
                        data, t, h_1d, h_2d,
                        override_dynamic_1d=dyn_1d,
                        override_dynamic_2d=dyn_2d,
                    )

            # Clamp predictions to physically reasonable depth ranges.
            # 1D depth: can be slightly negative (pipe below invert) but
            # bounded by max pipe capacity (~25m) + surcharge margin.
            # 2D depth: physically >= 0 (water on surface), upper bounded.
            pred_1d = pred_1d.clamp(-2.0, 30.0)
            pred_2d = pred_2d.clamp(-0.5, 15.0)

            all_preds_1d[t] = pred_1d
            all_preds_2d[t] = pred_2d

            # Early NaN detection: break immediately to avoid wasting
            # compute on a doomed rollout.  The trainer's NaN guard
            # will skip this event.
            if torch.isnan(pred_1d).any() or torch.isnan(pred_2d).any():
                break

        return all_preds_1d, all_preds_2d

    # ------------------------------------------------------------------ #
    #  Push-Forward Training
    # ------------------------------------------------------------------ #

    def pushforward_rollout(
        self,
        data: HeteroData,
        start_t: int,
        K: int,
        teacher_forcing_ratio: float = 0.0,
        h_1d: Optional[List[Tensor]] = None,
        h_2d: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Roll out K steps from a starting point for push-forward loss.

        Push-forward training computes loss on an entire *trajectory*
        rather than just t+1.  This penalises the model for drifting
        away from the true trajectory over time, which directly combats
        autoregressive instability.

        .. math::

            Loss = \\sum_{k=0}^{K-1}
                \\text{SRMSE}(\\hat{y}_{t+k}, y_{t+k})

        Parameters
        ----------
        data : HeteroData
        start_t : int
            First timestep of the rollout window.
        K : int
            Number of steps to roll out.
        teacher_forcing_ratio : float
            Scheduled sampling ratio (0.0 = fully autoregressive).
        h_1d, h_2d : list[Tensor] or None
            Initial hidden states.  Pass pre-warmed states from spinup.

        Returns
        -------
        (preds_1d, preds_2d, targets_1d, targets_2d)
            preds_*   : Tensor [K, N_*]  — model predictions
            targets_* : Tensor [K, N_*]  — ground truth (for loss)
        """
        T = data.num_timesteps
        n_1d = data["node_1d"].num_nodes
        n_2d = data["node_2d"].num_nodes
        device = data["node_1d"].x.device

        # Clamp K to available timesteps
        K = min(K, T - start_t)
        if K <= 0:
            raise ValueError(
                f"Cannot rollout {K} steps from t={start_t} "
                f"(event has {T} timesteps)."
            )

        if h_1d is None or h_2d is None:
            h_1d, h_2d = self.init_hidden(n_1d, n_2d, device)

        preds_1d = torch.zeros(K, n_1d, device=device)
        preds_2d = torch.zeros(K, n_2d, device=device)

        prev_pred_1d: Optional[Tensor] = None
        prev_pred_2d: Optional[Tensor] = None

        for k in range(K):
            t = start_t + k

            use_teacher = (
                k == 0  # first step always uses ground truth
                or (
                    self.training
                    and teacher_forcing_ratio > 0.0
                    and torch.rand(1).item() < teacher_forcing_ratio
                )
            )

            if use_teacher or prev_pred_1d is None:
                pred_1d, pred_2d, h_1d, h_2d = self.step(
                    data, t, h_1d, h_2d
                )
            else:
                dyn_1d, dyn_2d = self._build_feedback_dynamic(
                    data, t, prev_pred_1d, prev_pred_2d
                )
                pred_1d, pred_2d, h_1d, h_2d = self.step(
                    data, t, h_1d, h_2d,
                    override_dynamic_1d=dyn_1d,
                    override_dynamic_2d=dyn_2d,
                )

            # Clamp to physically reasonable depth ranges.
            pred_1d = pred_1d.clamp(-2.0, 30.0)
            pred_2d = pred_2d.clamp(-0.5, 15.0)

            preds_1d[k] = pred_1d
            preds_2d[k] = pred_2d
            # Allow gradient flow through AR loop (BPTT) so the model
            # can learn error-correcting dynamics.  Combined with
            # gradient clipping (norm=1.0), this is stable.
            prev_pred_1d = pred_1d
            prev_pred_2d = pred_2d

            # Early NaN detection
            if torch.isnan(pred_1d).any() or torch.isnan(pred_2d).any():
                break

        # Extract targets
        end_t = start_t + K
        targets_1d = data["node_1d"].y[start_t:end_t]  # [K, N_1d]
        targets_2d = data["node_2d"].y[start_t:end_t]  # [K, N_2d]

        return preds_1d, preds_2d, targets_1d, targets_2d

    # ------------------------------------------------------------------ #
    #  Feedback Construction
    # ------------------------------------------------------------------ #

    def _build_feedback_dynamic(
        self,
        data: HeteroData,
        t: int,
        prev_pred_1d: Tensor,
        prev_pred_2d: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Build dynamic input features from model predictions.

        **Depth-based (v4):** The model predicts *depth* directly:
          - 1D: depth = WSE − invert_elevation
          - 2D: depth = WSE − min_elevation

        This is a major simplification over the old anomaly-based
        approach: the predicted depth IS the dynamic feature.  No
        anomaly→absolute→relative conversion chain needed, which
        eliminates the train-inference mismatch bug.

        1D Dynamic features: [relative_depth, fill_ratio, inlet_flow]
          - ``relative_depth = predicted_depth``  (identical to target)
          - ``fill_ratio = clamp(depth / capacity, 0, 5)``
          - ``inlet_flow``: ground truth if available, else zero

        2D Dynamic features: [rainfall, water_depth, water_volume]
          - ``rainfall``: always available from data (known forcing)
          - ``water_depth = predicted_depth``  (identical to target)
          - ``water_volume = area × relu(depth)``

        Parameters
        ----------
        data : HeteroData
        t : int
            Current timestep.
        prev_pred_1d : Tensor [N_1d]
            Model's *depth* prediction from the previous step.
        prev_pred_2d : Tensor [N_2d]
            Model's *depth* prediction from the previous step.

        Returns
        -------
        (dyn_1d, dyn_2d)
            dyn_1d : Tensor [N_1d, F_1d_dyn]
            dyn_2d : Tensor [N_2d, F_2d_dyn]
        """
        device = prev_pred_1d.device

        # ── Force fp32 for physics computations ────────────────
        prev_pred_1d = prev_pred_1d.float()
        prev_pred_2d = prev_pred_2d.float()

        # ── 1D feedback ───────────────────────────────────────────
        # Depth IS relative_depth — no conversion needed.
        capacity = data["node_1d"].capacity.to(device).float()  # [N_1d]

        relative_depth = prev_pred_1d  # depth = WSE - invert_elev
        fill_ratio = torch.clamp(
            relative_depth / capacity.clamp(min=1e-3),
            min=0.0, max=5.0,
        )

        # inlet_flow: use ground truth if available (driven by
        # rainfall which is known), otherwise zero
        T_avail = data["node_1d"].dynamic.size(0)
        if t < T_avail:
            # inlet_flow is the 3rd dynamic feature (index 2)
            inlet_flow = data["node_1d"].dynamic[t, :, 2].to(device)
        else:
            inlet_flow = torch.zeros_like(prev_pred_1d)

        dyn_1d = torch.stack([relative_depth, fill_ratio, inlet_flow], dim=-1)

        # ── 2D feedback ───────────────────────────────────────────
        # Rainfall is always available (known forcing)
        if t < data["node_2d"].dynamic.size(0):
            rainfall = data["node_2d"].dynamic[t, :, 0].to(device)
        else:
            rainfall = torch.zeros(data["node_2d"].num_nodes, device=device)

        # water_depth = predicted depth (identical to target space)
        water_depth = prev_pred_2d

        # water_volume = area × depth (depth must be non-negative)
        area_2d = data["node_2d"].x[:, 0].to(device).float()
        water_volume_approx = area_2d * F.relu(prev_pred_2d)

        dyn_2d = torch.stack(
            [rainfall, water_depth, water_volume_approx], dim=-1
        )

        return dyn_1d, dyn_2d

    # ------------------------------------------------------------------ #
    #  Convenience: construct from graph dims
    # ------------------------------------------------------------------ #

    @classmethod
    def from_graph(
        cls,
        data: HeteroData,
        hidden_channels: int = 64,
        num_gnn_layers: int = 3,
        num_gru_layers: int = 1,
        dropout: float = 0.1,
    ) -> "UnifiedFloodModel":
        """Construct a model from a sample HeteroData object.

        Automatically reads feature dimensions from the graph.

        Parameters
        ----------
        data : HeteroData
            A sample graph from ``build_unified_graph()``.
        hidden_channels, num_gnn_layers, num_gru_layers, dropout
            Architecture hyperparameters.

        Returns
        -------
        UnifiedFloodModel
            Initialised model with correct input dimensions.

        Example
        -------
        >>> hetero = build_unified_graph(sample)
        >>> model = UnifiedFloodModel.from_graph(hetero, hidden_channels=128)
        """
        from src.graph_builder_unified import get_feature_dims

        dims = get_feature_dims(data)
        return cls(
            in_channels_1d_static=dims["in_channels_1d_static"],
            in_channels_1d_dynamic=dims["in_channels_1d_dynamic"],
            in_channels_2d_static=dims["in_channels_2d_static"],
            in_channels_2d_dynamic=dims["in_channels_2d_dynamic"],
            hidden_channels=hidden_channels,
            num_gnn_layers=num_gnn_layers,
            num_gru_layers=num_gru_layers,
            dropout=dropout,
        )

    # ------------------------------------------------------------------ #
    #  Model Summary
    # ------------------------------------------------------------------ #

    def summarise(self) -> str:
        """Return a human-readable model summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (
            f"UnifiedFloodModel\n"
            f"  Hidden channels : {self.hidden_channels}\n"
            f"  GNN layers      : {self.num_gnn_layers}\n"
            f"  GRU layers      : {self.num_gru_layers}\n"
            f"  Input dims      : 1D={self.in_channels_1d_static}+{self.in_channels_1d_dynamic}  "
            f"2D={self.in_channels_2d_static}+{self.in_channels_2d_dynamic}\n"
            f"  Total params    : {total_params:,}\n"
            f"  Trainable       : {trainable:,}\n"
        )

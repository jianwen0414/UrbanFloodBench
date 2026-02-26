"""
model_unified — UnifiedHeteroModel: HeteroGNN + HeteroGRU (v5).

Predicts **Delta Logits** per node per timestep.
Outputs are passed through ``tanh`` in the training loop to
produce smoothly bounded physical deltas (no gradient death).

Recovery in the training loop:
    raw_delta     = tanh(logit) × max_delta
    pred_depth_t  = prev_depth_{t-1} + raw_delta
    pred_wse_t    = pred_depth_t + node_elevation

Owner : Member C (Lead Architect)
See   : IMPLEMENTATION_PLAN.md → Task 2.4, PROJECT_BIBLE.md §6

Architecture
------------
1.  **Encoders** — Separate ``Linear`` projections for 1D and 2D
    node features → shared hidden dimension ``H``.
2.  **Processor** — ``L`` layers of ``HeteroConv`` using ``SAGEConv``
    for all four edge types (pipe, spread, link_1→2, link_2→1).
    Residual connections + LayerNorm for deep-stack stability.
3.  **Recurrent** — Per-node-type ``GRUCell`` maintains temporal
    memory across autoregressive steps.
4.  **Decoders** — Separate 3-layer MLP heads with **input skip
    connection**: decoder receives ``[GRU_output; raw_features]``
    so it can directly condition on current depth, flow, rainfall.

Edge Types (must match graph_builder_unified)
---------------------------------------------
    ('1d', 'pipe',   '1d')  — bidirectional pipe flow
    ('2d', 'spread', '2d')  — surface mesh adjacency
    ('1d', 'link',   '2d')  — surcharge (1D → 2D)
    ('2d', 'link',   '1d')  — drainage  (2D → 1D)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HeteroConv, SAGEConv


# =====================================================================
#  Constants — edge type tuples (match graph_builder_unified)
# =====================================================================
_PIPE_ET: Tuple[str, str, str] = ("1d", "pipe", "1d")
_SPREAD_ET: Tuple[str, str, str] = ("2d", "spread", "2d")
_LINK_12_ET: Tuple[str, str, str] = ("1d", "link", "2d")
_LINK_21_ET: Tuple[str, str, str] = ("2d", "link", "1d")


# =====================================================================
#  Main Model
# =====================================================================

class UnifiedHeteroModel(nn.Module):
    """Heterogeneous GNN-GRU predicting delta-depth (Δd_t).

    The model outputs the *change* in depth at each timestep, not
    the absolute water surface elevation.  This keeps outputs in a
    small, well-conditioned range and eliminates the need for the
    model to learn absolute elevation baselines.

    Parameters
    ----------
    in_channels_1d : int
        Feature dimension per 1D node per timestep (default 3).
    in_channels_2d : int
        Feature dimension per 2D node per timestep (default 3).
    hidden_channels : int
        Hidden dimension for GNN and GRU (default 192).
    num_gnn_layers : int
        Number of HeteroConv message-passing layers (default 3).
    dropout : float
        Dropout probability (default 0.05).
    """

    def __init__(
        self,
        in_channels_1d: int = 3,
        in_channels_2d: int = 3,
        hidden_channels: int = 256,  # EDA: lag-1 autocorr 0.9996 → larger GRU
        num_gnn_layers: int = 3,
        dropout: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels_1d = in_channels_1d
        self.in_channels_2d = in_channels_2d
        self.hidden_channels = hidden_channels
        self.num_gnn_layers = num_gnn_layers

        # ── Node Encoders ─────────────────────────────────────────
        self.encoder_1d = nn.Sequential(
            nn.Linear(in_channels_1d, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.encoder_2d = nn.Sequential(
            nn.Linear(in_channels_2d, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.LeakyReLU(negative_slope=0.01),
        )

        # ── HeteroConv GNN Stack ──────────────────────────────────
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms_1d = nn.ModuleList()
        self.gnn_norms_2d = nn.ModuleList()

        for _ in range(num_gnn_layers):
            conv = HeteroConv(
                {
                    _PIPE_ET: SAGEConv(hidden_channels, hidden_channels),
                    _SPREAD_ET: SAGEConv(hidden_channels, hidden_channels),
                    _LINK_12_ET: SAGEConv(
                        (hidden_channels, hidden_channels), hidden_channels
                    ),
                    _LINK_21_ET: SAGEConv(
                        (hidden_channels, hidden_channels), hidden_channels
                    ),
                },
                aggr="sum",
            )
            self.gnn_layers.append(conv)
            self.gnn_norms_1d.append(nn.LayerNorm(hidden_channels))
            self.gnn_norms_2d.append(nn.LayerNorm(hidden_channels))

        self.dropout = nn.Dropout(dropout)

        # ── GRU Cells (one per node type) ─────────────────────────
        self.gru_1d = nn.GRUCell(hidden_channels, hidden_channels)
        self.gru_2d = nn.GRUCell(hidden_channels, hidden_channels)

        # LayerNorm after GRU for stable hidden-state evolution
        self.gru_norm_1d = nn.LayerNorm(hidden_channels)
        self.gru_norm_2d = nn.LayerNorm(hidden_channels)

        # ── Decoder Heads → scalar Δd (delta depth) ──────────────
        # Input skip connection: decoder sees [GRU_output; raw_features]
        # so it can directly condition on current depth/flow/rainfall.
        dec_in_1d = hidden_channels + in_channels_1d
        dec_in_2d = hidden_channels + in_channels_2d

        self.decoder_1d = nn.Sequential(
            nn.Linear(dec_in_1d, hidden_channels // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_channels // 4, 1),
        )
        self.decoder_2d = nn.Sequential(
            nn.Linear(dec_in_2d, hidden_channels // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_channels // 4, 1),
        )

        # ── Weight Initialisation ─────────────────────────────────
        self._init_weights()

    # ------------------------------------------------------------------ #
    #  Weight Initialisation
    # ------------------------------------------------------------------ #

    def _init_weights(self) -> None:
        """Xavier uniform for linear layers; orthogonal for GRU.

        Orthogonal init for recurrent weights prevents vanishing /
        exploding gradients during long autoregressive rollouts.
        """
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                if "gru" in name:
                    nn.init.orthogonal_(param)
                else:
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
        device: torch.device,
    ) -> Dict[str, Tensor]:
        """Create zero-initialised GRU hidden states.

        Returns
        -------
        dict
            ``{'1d': [N_1d, H], '2d': [N_2d, H]}``
        """
        return {
            "1d": torch.zeros(n_1d, self.hidden_channels, device=device),
            "2d": torch.zeros(n_2d, self.hidden_channels, device=device),
        }

    # ------------------------------------------------------------------ #
    #  Forward Pass (single timestep)
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        hidden_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Single-timestep forward pass.

        Parameters
        ----------
        x_dict : dict
            ``{'1d': [N_1d, F_1d], '2d': [N_2d, F_2d]}``
            Per-node normalised features for the current timestep.
        edge_index_dict : dict
            Edge indices for all edge types, keyed by 3-tuples.
        hidden_dict : dict or None
            ``{'1d': [N_1d, H], '2d': [N_2d, H]}``
            GRU hidden states from the previous timestep.
            If ``None``, initialised to zeros.

        Returns
        -------
        delta_dict : dict
            ``{'1d': [N_1d], '2d': [N_2d]}``
            Predicted delta-depth for each node.
        next_hidden_dict : dict
            ``{'1d': [N_1d, H], '2d': [N_2d, H]}``
            Updated GRU hidden states.
        """
        n_1d = x_dict["1d"].size(0)
        n_2d = x_dict["2d"].size(0)
        device = x_dict["1d"].device

        # ── 0. Initialise hidden if needed ────────────────────────
        if hidden_dict is None:
            hidden_dict = self.init_hidden(n_1d, n_2d, device)

        # ── 1. Encode ─────────────────────────────────────────────
        z: Dict[str, Tensor] = {
            "1d": self.encoder_1d(x_dict["1d"]),   # [N_1d, H]
            "2d": self.encoder_2d(x_dict["2d"]),   # [N_2d, H]
        }

        # ── 2. GNN Message Passing (L layers) ─────────────────────
        for i, conv in enumerate(self.gnn_layers):
            out = conv(z, edge_index_dict)

            # Residual + LayerNorm + activation + dropout
            z_new: Dict[str, Tensor] = {}
            for ntype, norm_layer in [
                ("1d", self.gnn_norms_1d[i]),
                ("2d", self.gnn_norms_2d[i]),
            ]:
                if ntype in out:
                    h = norm_layer(out[ntype] + z[ntype])      # residual
                    h = F.leaky_relu(h, negative_slope=0.01)
                    h = self.dropout(h)
                    z_new[ntype] = h
                else:
                    z_new[ntype] = z[ntype]  # fallback if no edges
            z = z_new

        # ── 3. GRU (forced fp32 for numerical safety) ─────────────
        #    Under AMP, hidden states can drift beyond fp16 max
        #    (65504) and overflow to NaN during long rollouts.
        h_1d = self.gru_1d(
            z["1d"].float(), hidden_dict["1d"].float()
        )
        h_2d = self.gru_2d(
            z["2d"].float(), hidden_dict["2d"].float()
        )

        h_1d_norm = self.gru_norm_1d(h_1d)
        h_2d_norm = self.gru_norm_2d(h_2d)

        # ── 4. Decode → delta depth (with input skip) ─────────────
        #    Concatenate raw input features so the decoder can
        #    directly see current depth, flow, rainfall etc.
        dec_in_1d = torch.cat([h_1d_norm, x_dict["1d"]], dim=-1)
        dec_in_2d = torch.cat([h_2d_norm, x_dict["2d"]], dim=-1)

        delta_1d = self.decoder_1d(dec_in_1d).squeeze(-1)  # [N_1d]
        delta_2d = self.decoder_2d(dec_in_2d).squeeze(-1)  # [N_2d]

        delta_dict = {"1d": delta_1d, "2d": delta_2d}
        next_hidden_dict = {"1d": h_1d, "2d": h_2d}

        return delta_dict, next_hidden_dict

    # ------------------------------------------------------------------ #
    #  Convenience
    # ------------------------------------------------------------------ #

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summarise(self) -> str:
        """Human-readable model summary."""
        total = sum(p.numel() for p in self.parameters())
        trainable = self.count_parameters()
        return (
            f"UnifiedHeteroModel (Delta-Depth)\n"
            f"  Hidden channels : {self.hidden_channels}\n"
            f"  GNN layers      : {self.num_gnn_layers}\n"
            f"  Input dims      : 1D={self.in_channels_1d}  "
            f"2D={self.in_channels_2d}\n"
            f"  Total params    : {total:,}\n"
            f"  Trainable       : {trainable:,}\n"
        )

    def __repr__(self) -> str:
        return (
            f"UnifiedHeteroModel(in_1d={self.in_channels_1d}, "
            f"in_2d={self.in_channels_2d}, "
            f"hidden={self.hidden_channels}, "
            f"gnn_layers={self.num_gnn_layers})"
        )

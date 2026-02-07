"""
Engine C — Unified 1D-2D Coupled Model (Fallback / Tier 2).

Architecture
------------
HeteroGNN-GRU operating on a heterogeneous graph with three edge types:
    1. pipe  → pipe   (1D internal flow)
    2. cell  → cell   (2D surface spread)
    3. node_1d ↔ node_2d  (1D-2D coupling / surcharge exchange)

The ``HeteroData`` input is constructed by
``graph_builder_unified.build_unified_graph()``.

When to deploy
--------------
This model is the "Heavy Weapon." Use it only if the decoupled
Twin Engines (Engine A + Engine B) fail to capture complex flooding
events — specifically **surcharge**, where pressurized pipes force
water onto the surface through manholes.

Owner: Member C (Lead Architect)
See: IMPLEMENTATION_PLAN.md → Task 2.4
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# ── Imports (uncomment when implementing) ────────────────────────────
# import torch
# import torch.nn as nn
# from torch_geometric.data import HeteroData
# from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv


# =====================================================================
#  Model
# =====================================================================

class UnifiedFloodModel:
    """Heterogeneous GNN-GRU for coupled 1D-2D flood prediction.

    This model explicitly models the physical interaction between the
    underground pipe network and the surface terrain via heterogeneous
    message-passing edges.

    Parameters
    ----------
    in_channels_1d : int
        Number of input features per 1D node
        (e.g. relative_depth, capacity, inlet_flow, rain).
    in_channels_2d : int
        Number of input features per 2D node
        (e.g. water_level, rainfall, water_volume).
    hidden_channels : int
        Dimensionality of the GRU hidden state and GNN embeddings.
    num_gnn_layers : int
        Number of heterogeneous message-passing layers.
    dropout : float
        Dropout rate applied between GNN layers.

    Architecture Overview
    ---------------------
    1. **Node Encoders** — separate linear projections for 1D and 2D
       node features into a shared hidden dimension.
    2. **HeteroConv Layers** — ``num_gnn_layers`` rounds of
       heterogeneous message passing across all three edge types.
    3. **Temporal GRU** — per-node-type GRU cells that maintain
       temporal memory across autoregressive steps.
    4. **Prediction Heads** — separate linear decoders for 1D and 2D
       water level predictions.

    Forward Signature (planned)
    ---------------------------
    ``forward(hetero_data, h_1d, h_2d) -> (pred_1d, pred_2d, h_1d, h_2d)``

    Where:
        - ``hetero_data`` : HeteroData from graph_builder_unified
        - ``h_1d``, ``h_2d`` : GRU hidden states (None on first call)
        - Returns predictions + updated hidden states for the next
          autoregressive step.
    """

    # TODO: Implement in Phase 2 (Task 2.4) — only if Twin Engines
    #       fail on surcharge events.
    #
    # Implementation checklist:
    #   [ ] Node encoders (Linear) for 1D and 2D feature spaces
    #   [ ] HeteroConv with GCNConv (pipe→pipe), SAGEConv (cell→cell),
    #       and a coupling conv (1d↔2d)
    #   [ ] Separate GRU cells for 1D and 2D hidden states
    #   [ ] Prediction heads (Linear → water_level)
    #   [ ] LeakyReLU activations (critical for dry 2D nodes)
    #   [ ] Scheduled Sampling support (teacher_forcing_ratio arg)
    #   [ ] Push-forward loss over K-step trajectories

    pass

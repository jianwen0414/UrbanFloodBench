"""
graph_builder_unified — Construct a HeteroData object for the coupled
1D-2D Unified Engine.

Consumes the sample dict from ``FloodDataset.__getitem__()`` and
produces a ``torch_geometric.data.HeteroData`` graph with **three
explicit edge types** for heterogeneous message passing.

Owner: Member C (Lead Architect)
See: IMPLEMENTATION_PLAN.md → Task 1.4

Edge Types
----------
1. (``node_1d``, ``pipe_flow``,  ``node_1d``) — bidirectional pipe
   connectivity from ``1d_edge_index.csv``.
2. (``node_2d``, ``surface_flow``, ``node_2d``) — surface mesh
   adjacency from ``2d_edge_index.csv``.
3. (``node_1d``, ``couples``, ``node_2d``) — physical coupling edges
   from ``1d2d_connections.csv``  (also reversed: ``node_2d`` →
   ``node_1d`` for bidirectional exchange).

When to Use
-----------
This graph is consumed by ``UnifiedFloodModel`` (model_unified.py)
and is deployed only if the decoupled Twin Engines fail to capture
surcharge events.

Input DataFrames (from FloodDataset sample)
-------------------------------------------
All static + dynamic DataFrames from the sample dict, plus
``1d2d_conn`` for the coupling edges.

Output (HeteroData)
-------------------
- ``data['node_1d'].x``     : 1D static features  [N_1d, F]
- ``data['node_2d'].x``     : 2D static features  [N_2d, F]
- ``data['node_1d'].y``     : 1D water level targets [T, N_1d]
- ``data['node_2d'].y``     : 2D water level targets [T, N_2d]
- ``data[edge_type].edge_index`` : per edge type [2, E_type]
"""

from __future__ import annotations

from typing import Any, Dict

# ── Imports (uncomment when implementing) ────────────────────────────
# import numpy as np
# import pandas as pd
# import torch
# from torch_geometric.data import HeteroData


def build_unified_graph(sample: Dict[str, Any]) -> Any:
    """Convert a FloodDataset sample into a heterogeneous 1D-2D graph.

    Parameters
    ----------
    sample : dict
        A single sample from ``FloodDataset.__getitem__()``.
        Uses all static/dynamic keys plus ``1d2d_conn``.

    Returns
    -------
    torch_geometric.data.HeteroData
        Heterogeneous graph with 1D nodes, 2D nodes, and three edge
        types (pipe, surface, coupling).

    Implementation Steps
    --------------------
    1. Build 1D node features (same as graph_builder_1d: capacity,
       relative depth, etc.).
    2. Build 2D node features (same as graph_builder_2d: elevation,
       roughness, etc.; but NO dist_to_drain — coupling is explicit).
    3. Create pipe edges (bidirectional) from ``edge_index_1d``.
    4. Create surface edges from ``edge_index_2d``.
    5. Create coupling edges from ``1d2d_conn``:
       - ``node_1d`` → ``node_2d``  (surcharge: water exits pipe)
       - ``node_2d`` → ``node_1d``  (drainage: water enters pipe)
    6. Assemble ``HeteroData`` with node/edge stores.
    """
    # TODO: Member C — implement for Task 1.4 (only if Tier 1 fails)
    raise NotImplementedError("build_unified_graph not yet implemented")

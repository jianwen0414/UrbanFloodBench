"""
graph_builder_1d — Construct a PyG Data object for the 1D pipe network.

Consumes the sample dict from ``FloodDataset.__getitem__()`` and
produces a homogeneous ``torch_geometric.data.Data`` graph with
**bidirectional** edges (critical for backwater-flow simulation).

Owner: Member A
See: IMPLEMENTATION_PLAN.md → Task 1.2

Input DataFrames (from FloodDataset sample)
-------------------------------------------
- ``static_1d_nodes``  : node_idx, depth, invert_elevation,
                         surface_elevation, base_area
- ``dynamic_1d_nodes`` : timestep, node_idx, water_level, inlet_flow
                         (long format)
- ``edge_index_1d``    : edge_idx, from_node, to_node
- ``static_1d_edges``  : edge_idx, length, diameter, shape, roughness,
                         slope

Output (PyG Data)
-----------------
- ``data.x``          : Static node features  [N, F_static]
- ``data.edge_index`` : Bidirectional edges    [2, 2*E]
- ``data.edge_attr``  : Edge features          [2*E, F_edge]
- ``data.y``          : Water level targets    [T, N]
- ``data.dynamic``    : Dynamic input features [T, N, F_dyn]

Required Physics Features (from PROJECT_BIBLE.md §6)
-----------------------------------------------------
- Capacity       = surface_elevation − invert_elevation
- Relative Depth = water_level − invert_elevation
- Edges MUST be bidirectional (u→v AND v→u)
"""

from __future__ import annotations

from typing import Any, Dict

# ── Imports (uncomment when implementing) ────────────────────────────
# import numpy as np
# import pandas as pd
# import torch
# from torch_geometric.data import Data


def build_1d_graph(sample: Dict[str, Any]) -> Any:
    """Convert a FloodDataset sample into a 1D pipe graph.

    Parameters
    ----------
    sample : dict
        A single sample from ``FloodDataset.__getitem__()``.
        Expected keys: ``static_1d_nodes``, ``dynamic_1d_nodes``,
        ``edge_index_1d``, ``static_1d_edges``.

    Returns
    -------
    torch_geometric.data.Data
        Homogeneous graph for the 1D pipe network.

    Implementation Steps
    --------------------
    1. Extract static node features; compute ``capacity``.
    2. Pivot dynamic DataFrame from long → wide:
       ``(T, N)`` matrix of water levels.
    3. Compute ``relative_depth = water_level − invert_elevation``.
    4. Build bidirectional ``edge_index`` from ``from_node``/``to_node``:
       stack [from→to] and [to→from].
    5. Attach edge attributes (diameter, roughness, slope) — duplicated
       for both directions.
    6. Assemble ``Data(x=..., edge_index=..., y=..., ...)``.
    """
    # TODO: Member A — implement for Task 1.2
    raise NotImplementedError("build_1d_graph not yet implemented")

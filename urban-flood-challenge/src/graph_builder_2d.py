"""
graph_builder_2d — Construct a PyG Data object for the 2D surface mesh.

Consumes the sample dict from ``FloodDataset.__getitem__()`` and
produces a homogeneous ``torch_geometric.data.Data`` graph with the
"Soft Coupling" feature: **distance to nearest 1D drain node**.

Owner: Member B
See: IMPLEMENTATION_PLAN.md → Task 1.3

Input DataFrames (from FloodDataset sample)
-------------------------------------------
- ``static_2d_nodes``  : node_idx, position_x, position_y, area,
                         roughness, min_elevation, elevation, aspect,
                         curvature, flow_accumulation
- ``dynamic_2d_nodes`` : timestep, node_idx, rainfall, water_level,
                         water_volume  (long format)
- ``edge_index_2d``    : edge_idx, from_node, to_node
- ``static_2d_edges``  : edge_idx, face_length, length, slope
- ``static_1d_nodes``  : node_idx, position_x, position_y, ...
                         (needed for dist_to_drain computation)
- ``1d2d_conn``        : connection_idx, node_1d, node_2d

Output (PyG Data)
-----------------
- ``data.x``          : Static node features  [N, F_static]
                        (includes dist_to_drain)
- ``data.edge_index`` : Mesh adjacency        [2, E]
- ``data.edge_attr``  : Edge features          [E, F_edge]
- ``data.y``          : Water level targets    [T, N]
- ``data.dynamic``    : Dynamic input features [T, N, F_dyn]

Required Physics Features (from PROJECT_BIBLE.md §6)
-----------------------------------------------------
- dist_to_drain : Euclidean distance from each 2D node to its nearest
                  1D node (using 1d2d_connections.csv + node positions)
- Z-scored elevation : elevation relative to immediate neighbours
                       (highlights depressions / sinks)
- Activation: LeakyReLU (most 2D nodes are dry → ReLU kills gradients)
"""

from __future__ import annotations

from typing import Any, Dict

# ── Imports (uncomment when implementing) ────────────────────────────
# import numpy as np
# import pandas as pd
# import torch
# from torch_geometric.data import Data
# from scipy.spatial import cKDTree  # for efficient nearest-neighbour


def build_2d_graph(sample: Dict[str, Any]) -> Any:
    """Convert a FloodDataset sample into a 2D surface mesh graph.

    Parameters
    ----------
    sample : dict
        A single sample from ``FloodDataset.__getitem__()``.
        Expected keys: ``static_2d_nodes``, ``dynamic_2d_nodes``,
        ``edge_index_2d``, ``static_2d_edges``, ``static_1d_nodes``,
        ``1d2d_conn``.

    Returns
    -------
    torch_geometric.data.Data
        Homogeneous graph for the 2D surface mesh.

    Implementation Steps
    --------------------
    1. Extract static node features (area, roughness, elevation, ...).
    2. Compute ``dist_to_drain``:
       a. Get 1D node positions from ``static_1d_nodes``.
       b. Get 2D node positions from ``static_2d_nodes``.
       c. For each 2D node, find the nearest 1D node (cKDTree or brute
          force) and compute Euclidean distance.
    3. Optionally compute Z-scored elevation relative to graph
       neighbours.
    4. Pivot dynamic DataFrame from long → wide:
       ``(T, N)`` matrices for water_level, rainfall, etc.
    5. Build ``edge_index`` from ``from_node``/``to_node``.
    6. Attach edge attributes (face_length, length, slope).
    7. Assemble ``Data(x=..., edge_index=..., y=..., ...)``.
    """
    # TODO: Member B — implement for Task 1.3
    raise NotImplementedError("build_2d_graph not yet implemented")

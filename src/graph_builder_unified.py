"""
graph_builder_unified — Production-grade Heterogeneous Graph Builder
for the Coupled 1D-2D Unified Flood Model (Tier 2).

Constructs a ``torch_geometric.data.HeteroData`` object that the
``UnifiedFloodModel`` (model_unified.py) consumes via heterogeneous
message passing (``HeteroConv``).

Owner : Member C (Lead Architect)
See   : IMPLEMENTATION_PLAN.md → Task 1.4, PROJECT_BIBLE.md §3-4, §6

Node Types
----------
- ``node_1d`` — manholes / junctions in the pipe network.
- ``node_2d`` — mesh cells on the surface terrain.

Edge Types (4 directed types, physically motivated)
----------------------------------------------------
1. ``(node_1d, pipe_to, node_1d)`` — **bidirectional** pipe flow.
   Derived from ``1d_edge_index.csv``.  Both directions are included
   to model backwater (reverse) flow.
2. ``(node_2d, surface_to, node_2d)`` — surface mesh adjacency from
   ``2d_edge_index.csv``.  Made bidirectional (deduplicated).
3. ``(node_1d, surcharges_to, node_2d)`` — physical 1D→2D coupling.
   Models water erupting from a pipe onto the surface when the
   hydraulic head exceeds the surface elevation.
4. ``(node_2d, drains_to, node_1d)`` — physical 2D→1D coupling.
   Models surface water draining into the pipe network via inlets.

Separating the coupling into two directed edge types lets the model
learn *distinct* message-passing functions for surcharge (pressure-
driven, fast) versus drainage (gravity-driven, slower).

Physics-Informed Feature Engineering
-------------------------------------
1D Static Features (per node):
  - ``capacity``  = surface_elevation − invert_elevation
  - ``depth``, ``base_area``, ``invert_elevation``, ``surface_elevation``

1D Dynamic Features (T, N_1d, F):
  - ``relative_depth`` = water_level − invert_elevation
  - ``fill_ratio``     = clamp(relative_depth / capacity)
  - ``inlet_flow``     (auxiliary — zero-filled after spin-up)

2D Static Features (per node):
  - ``area``, ``roughness``, ``min_elevation``, ``elevation``
  - ``relative_elevation`` = elevation − min_elevation
  - ``aspect_sin``, ``aspect_cos`` (periodic encoding of aspect angle)
  - ``curvature``, ``flow_accumulation``
  (No ``dist_to_drain`` — coupling is *explicit* via graph edges.)

2D Dynamic Features (T, N_2d, F):
  - ``rainfall`` (always available — the only known forcing at test time)
  - ``water_level``
  - ``water_volume``

Coupling Edge Features:
  - ``elevation_diff`` = surface_elev_1d − centroid_elev_2d
  - ``euclidean_dist`` = ‖pos_1d − pos_2d‖₂
  - ``capacity_1d``    (surcharge threshold of the connected 1D node)
  - ``area_2d``        (absorption capacity of the connected 2D cell)
  - ``dx``, ``dy``     (directional offset)

Output Layout (HeteroData)
--------------------------
- ``data['node_1d'].x``          : [N_1d, F_1d_static]
- ``data['node_1d'].y``          : [T, N_1d]  — water level targets
- ``data['node_1d'].dynamic``    : [T, N_1d, F_1d_dyn]
- ``data['node_1d'].invert_elev``: [N_1d]     — for back-conversion
- ``data['node_1d'].capacity``   : [N_1d]     — for fill_ratio at inference
- ``data['node_2d'].x``          : [N_2d, F_2d_static]
- ``data['node_2d'].y``          : [T, N_2d]
- ``data['node_2d'].dynamic``    : [T, N_2d, F_2d_dyn]
- ``data[(src, rel, dst)].edge_index`` : [2, E] per edge type
- ``data[(src, rel, dst)].edge_attr``  : [E, F_edge] per edge type
- ``data.model_id``              : str
- ``data.event_id``              : str
- ``data.num_timesteps``         : int
- ``data.feature_names_*``       : list[str]  — for introspection
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


# =====================================================================
#  Constants
# =====================================================================

_EPS: float = 1e-8

# Feature name registries (single source of truth for column order)
_1D_STATIC_FEATURE_NAMES: List[str] = [
    "capacity",
    "depth",
    "base_area",
    "invert_elevation",
    "surface_elevation",
]

_1D_DYNAMIC_FEATURE_NAMES: List[str] = [
    "relative_depth",
    "fill_ratio",
    "inlet_flow",
]

_2D_STATIC_FEATURE_NAMES: List[str] = [
    "area",
    "roughness",
    "min_elevation",
    "elevation",
    "relative_elevation",
    "aspect_sin",
    "aspect_cos",
    "curvature",
    "flow_accumulation",
]

_2D_DYNAMIC_FEATURE_NAMES: List[str] = [
    "rainfall",
    "water_level",
    "water_volume",
]

_PIPE_EDGE_FEATURE_NAMES: List[str] = [
    "length",
    "diameter",
    "roughness",
    "slope",
]

_SURFACE_EDGE_FEATURE_NAMES: List[str] = [
    "face_length",
    "length",
    "slope",
]

_COUPLING_EDGE_FEATURE_NAMES: List[str] = [
    "elevation_diff",
    "euclidean_dist",
    "capacity_1d",
    "area_2d",
    "dx",
    "dy",
]


# =====================================================================
#  Internal helpers
# =====================================================================

def _validate_dataframe(
    df: pd.DataFrame,
    required_cols: List[str],
    name: str,
) -> None:
    """Raise ``ValueError`` if *df* is missing required columns."""
    if df.empty:
        raise ValueError(f"DataFrame '{name}' is empty.")
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame '{name}' is missing columns: {sorted(missing)}. "
            f"Available: {sorted(df.columns)}"
        )


def _make_bidirectional(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    *,
    deduplicate: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Convert directed edges to bidirectional, optionally deduplicating.

    Parameters
    ----------
    edge_index : Tensor [2, E]
    edge_attr  : Tensor [E, F] or None
    deduplicate : bool
        If ``True``, remove duplicate (u, v) pairs that arise when
        the source data already contains both directions.

    Returns
    -------
    (edge_index, edge_attr)  with up to 2*E edges.
    """
    # Stack forward + reverse
    rev = edge_index.flip(0)
    bi_index = torch.cat([edge_index, rev], dim=1)  # [2, 2E]

    bi_attr: Optional[torch.Tensor] = None
    if edge_attr is not None:
        bi_attr = torch.cat([edge_attr, edge_attr], dim=0)  # [2E, F]

    if deduplicate:
        # Canonical encoding: pack (u, v) into a single int64 key
        max_id = int(bi_index.max().item()) + 1
        packed = bi_index[0].long() * max_id + bi_index[1].long()

        # Keep the first occurrence of each unique edge
        _, first_indices = torch.unique(packed, return_inverse=True)
        # first_indices maps each element to its unique-group index.
        # We want the first position for each group.
        keep = torch.zeros(bi_index.size(1), dtype=torch.bool)
        seen_groups: set[int] = set()
        for i in range(bi_index.size(1)):
            g = first_indices[i].item()
            if g not in seen_groups:
                seen_groups.add(g)
                keep[i] = True

        bi_index = bi_index[:, keep]
        if bi_attr is not None:
            bi_attr = bi_attr[keep]

    return bi_index, bi_attr


def _pivot_dynamic_to_tensor(
    df: pd.DataFrame,
    n_nodes: int,
    timestep_col: str,
    node_col: str,
    value_cols: List[str],
) -> torch.Tensor:
    """Pivot a long-format dynamic DataFrame into a dense (T, N, F) tensor.

    Missing entries are filled with ``0.0`` (physically: no flow / dry).

    Parameters
    ----------
    df : DataFrame
        Long-format with columns [timestep_col, node_col, *value_cols].
    n_nodes : int
        Total number of nodes (used to allocate the dense tensor).
    timestep_col, node_col : str
        Column names for timestep index and node index.
    value_cols : list[str]
        Ordered list of feature columns to extract.

    Returns
    -------
    Tensor of shape (T, N, len(value_cols)).
    """
    if df.empty:
        # Return a single-timestep zero tensor as fallback
        return torch.zeros(1, n_nodes, len(value_cols), dtype=torch.float32)

    timesteps = sorted(df[timestep_col].unique())
    n_timesteps = len(timesteps)
    t_map = {t: i for i, t in enumerate(timesteps)}

    tensor = torch.zeros(n_timesteps, n_nodes, len(value_cols), dtype=torch.float32)

    # Vectorised fill: map timesteps and node indices to dense indices
    t_indices = df[timestep_col].map(t_map).values.astype(np.intp)
    n_indices = df[node_col].values.astype(np.intp)

    for f_idx, col in enumerate(value_cols):
        if col not in df.columns:
            warnings.warn(
                f"Dynamic column '{col}' not found — filled with zeros."
            )
            continue
        vals = df[col].values.astype(np.float32)
        # Handle NaN → 0
        nan_mask = np.isnan(vals)
        if nan_mask.any():
            warnings.warn(
                f"NaN detected in dynamic column '{col}' "
                f"({nan_mask.sum()} values) — replaced with 0."
            )
            vals[nan_mask] = 0.0
        tensor[t_indices, n_indices, f_idx] = torch.from_numpy(vals)

    return tensor


def _extract_targets(
    df: pd.DataFrame,
    n_nodes: int,
    timestep_col: str = "timestep",
    node_col: str = "node_idx",
    target_col: str = "water_level",
) -> torch.Tensor:
    """Extract target water levels as a dense (T, N) tensor."""
    if df.empty:
        return torch.zeros(1, n_nodes, dtype=torch.float32)

    timesteps = sorted(df[timestep_col].unique())
    n_timesteps = len(timesteps)
    t_map = {t: i for i, t in enumerate(timesteps)}

    tensor = torch.zeros(n_timesteps, n_nodes, dtype=torch.float32)
    t_indices = df[timestep_col].map(t_map).values.astype(np.intp)
    n_indices = df[node_col].values.astype(np.intp)
    vals = df[target_col].values.astype(np.float32)

    nan_mask = np.isnan(vals)
    if nan_mask.any():
        warnings.warn(
            f"NaN in target '{target_col}' ({nan_mask.sum()} values) "
            "— forward-filled then zero-filled."
        )
        vals = pd.Series(vals).ffill().fillna(0.0).values.astype(np.float32)

    tensor[t_indices, n_indices] = torch.from_numpy(vals)
    return tensor


# =====================================================================
#  1D Feature Engineering
# =====================================================================

def _build_1d_static_features(
    static_nodes: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute 1D node static features from ``1d_nodes_static.csv``.

    Returns
    -------
    x : Tensor [N_1d, 5]
        Static features: capacity, depth, base_area, invert_elev, surface_elev.
    invert_elev : Tensor [N_1d]
        Invert elevation (needed for relative_depth at inference).
    capacity : Tensor [N_1d]
        Capacity (needed for fill_ratio at inference).
    """
    _validate_dataframe(
        static_nodes,
        ["node_idx", "depth", "invert_elevation", "surface_elevation", "base_area"],
        "static_1d_nodes",
    )

    # Sort by node index for deterministic ordering
    df = static_nodes.sort_values("node_idx").reset_index(drop=True)

    invert_elev = df["invert_elevation"].values.astype(np.float32)
    surface_elev = df["surface_elevation"].values.astype(np.float32)

    # Physics: Capacity = Surface Elevation − Invert Elevation
    # This is the maximum water depth before the manhole floods.
    capacity = surface_elev - invert_elev
    # Clamp capacity ≥ ε to avoid division-by-zero in fill_ratio
    capacity = np.maximum(capacity, _EPS)

    depth = df["depth"].values.astype(np.float32)
    base_area = df["base_area"].values.astype(np.float32)

    # Stack into feature matrix: [N, 5]
    x = np.column_stack([capacity, depth, base_area, invert_elev, surface_elev])

    # Replace any residual NaN with 0
    nan_count = np.isnan(x).sum()
    if nan_count > 0:
        warnings.warn(f"1D static features: {nan_count} NaN values → 0.")
        np.nan_to_num(x, copy=False)

    return (
        torch.from_numpy(x).float(),
        torch.from_numpy(invert_elev),
        torch.from_numpy(capacity),
    )


def _build_1d_dynamic_features(
    dynamic_nodes: pd.DataFrame,
    n_nodes: int,
    invert_elev: torch.Tensor,
    capacity: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 1D dynamic features and targets.

    Dynamic features per timestep:
      - relative_depth = water_level − invert_elevation
      - fill_ratio     = clamp(relative_depth / capacity, 0, 5)
      - inlet_flow

    Parameters
    ----------
    dynamic_nodes : DataFrame
        Long-format: timestep, node_idx, water_level, inlet_flow.
    n_nodes : int
    invert_elev : Tensor [N_1d]
    capacity : Tensor [N_1d]

    Returns
    -------
    dynamic : Tensor [T, N_1d, 3]
    targets : Tensor [T, N_1d]
    """
    # Extract raw water_level and inlet_flow
    targets = _extract_targets(
        dynamic_nodes, n_nodes,
        timestep_col="timestep", node_col="node_idx",
        target_col="water_level",
    )  # [T, N]

    # Build raw dynamic columns (inlet_flow)
    raw_cols = ["inlet_flow"]
    raw_dyn = _pivot_dynamic_to_tensor(
        dynamic_nodes, n_nodes,
        timestep_col="timestep", node_col="node_idx",
        value_cols=raw_cols,
    )  # [T, N, 1]

    # Compute derived physics features
    # relative_depth: how high the water is above the pipe invert [T, N]
    relative_depth = targets - invert_elev.unsqueeze(0)  # broadcast [1, N]
    # fill_ratio: fraction of capacity used; >1 indicates surcharge [T, N]
    fill_ratio = torch.clamp(relative_depth / capacity.unsqueeze(0), min=0.0, max=5.0)

    # Stack: [T, N, 3] = [relative_depth, fill_ratio, inlet_flow]
    dynamic = torch.stack(
        [relative_depth, fill_ratio, raw_dyn[..., 0]],
        dim=-1,
    )

    return dynamic, targets


# =====================================================================
#  2D Feature Engineering
# =====================================================================

def _build_2d_static_features(
    static_nodes: pd.DataFrame,
) -> torch.Tensor:
    """Compute 2D node static features from ``2d_nodes_static.csv``.

    Returns
    -------
    x : Tensor [N_2d, 9]
        Features: area, roughness, min_elevation, elevation,
        relative_elevation, aspect_sin, aspect_cos, curvature,
        flow_accumulation.
    """
    _validate_dataframe(
        static_nodes,
        [
            "node_idx", "area", "roughness", "min_elevation",
            "elevation", "aspect", "curvature", "flow_accumulation",
        ],
        "static_2d_nodes",
    )

    df = static_nodes.sort_values("node_idx").reset_index(drop=True)

    area = df["area"].values.astype(np.float32)
    roughness = df["roughness"].values.astype(np.float32)
    min_elev = df["min_elevation"].values.astype(np.float32)
    elev = df["elevation"].values.astype(np.float32)

    # Physics: relative elevation highlights depressions (potential sinks)
    relative_elev = elev - min_elev

    # Periodic encoding of aspect angle (degrees → sin/cos)
    aspect_rad = np.deg2rad(df["aspect"].values.astype(np.float32))
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)

    curvature = df["curvature"].values.astype(np.float32)
    flow_acc = df["flow_accumulation"].values.astype(np.float32)

    x = np.column_stack([
        area, roughness, min_elev, elev, relative_elev,
        aspect_sin, aspect_cos, curvature, flow_acc,
    ])

    nan_count = np.isnan(x).sum()
    if nan_count > 0:
        warnings.warn(f"2D static features: {nan_count} NaN values → 0.")
        np.nan_to_num(x, copy=False)

    return torch.from_numpy(x).float()


def _build_2d_dynamic_features(
    dynamic_nodes: pd.DataFrame,
    n_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 2D dynamic features and targets.

    Dynamic features per timestep: rainfall, water_level, water_volume.

    Returns
    -------
    dynamic : Tensor [T, N_2d, 3]
    targets : Tensor [T, N_2d]
    """
    targets = _extract_targets(
        dynamic_nodes, n_nodes,
        timestep_col="timestep", node_col="node_idx",
        target_col="water_level",
    )

    dynamic = _pivot_dynamic_to_tensor(
        dynamic_nodes, n_nodes,
        timestep_col="timestep", node_col="node_idx",
        value_cols=["rainfall", "water_level", "water_volume"],
    )

    return dynamic, targets


# =====================================================================
#  Edge Construction
# =====================================================================

def _build_pipe_edges(
    edge_index_df: pd.DataFrame,
    edge_static_df: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build bidirectional pipe edges with physical attributes.

    Returns
    -------
    edge_index : Tensor [2, 2*E]  (bidirectional, deduplicated)
    edge_attr  : Tensor [2*E, 4]  (length, diameter, roughness, slope)
    """
    _validate_dataframe(
        edge_index_df, ["edge_idx", "from_node", "to_node"], "edge_index_1d"
    )

    # Sort by edge_idx for alignment with edge_static
    ei_df = edge_index_df.sort_values("edge_idx").reset_index(drop=True)

    src = torch.tensor(ei_df["from_node"].values, dtype=torch.long)
    dst = torch.tensor(ei_df["to_node"].values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)  # [2, E]

    # Edge attributes
    edge_attr: Optional[torch.Tensor] = None
    if not edge_static_df.empty:
        es_df = edge_static_df.sort_values("edge_idx").reset_index(drop=True)
        attr_cols = ["length", "diameter", "roughness", "slope"]
        available_cols = [c for c in attr_cols if c in es_df.columns]
        if available_cols:
            vals = es_df[available_cols].values.astype(np.float32)
            np.nan_to_num(vals, copy=False)
            edge_attr = torch.from_numpy(vals)

    # Make bidirectional (deduplicate in case of existing reverse edges)
    edge_index, edge_attr = _make_bidirectional(
        edge_index, edge_attr, deduplicate=True
    )

    return edge_index, edge_attr if edge_attr is not None else torch.empty(edge_index.size(1), 0)


def _build_surface_edges(
    edge_index_df: pd.DataFrame,
    edge_static_df: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build bidirectional surface mesh edges with attributes.

    Returns
    -------
    edge_index : Tensor [2, E_bi]
    edge_attr  : Tensor [E_bi, 3]  (face_length, length, slope)
    """
    _validate_dataframe(
        edge_index_df, ["edge_idx", "from_node", "to_node"], "edge_index_2d"
    )

    ei_df = edge_index_df.sort_values("edge_idx").reset_index(drop=True)

    src = torch.tensor(ei_df["from_node"].values, dtype=torch.long)
    dst = torch.tensor(ei_df["to_node"].values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    edge_attr: Optional[torch.Tensor] = None
    if not edge_static_df.empty:
        es_df = edge_static_df.sort_values("edge_idx").reset_index(drop=True)
        attr_cols = ["face_length", "length", "slope"]
        available_cols = [c for c in attr_cols if c in es_df.columns]
        if available_cols:
            vals = es_df[available_cols].values.astype(np.float32)
            np.nan_to_num(vals, copy=False)
            edge_attr = torch.from_numpy(vals)

    edge_index, edge_attr = _make_bidirectional(
        edge_index, edge_attr, deduplicate=True
    )

    return edge_index, edge_attr if edge_attr is not None else torch.empty(edge_index.size(1), 0)


def _build_coupling_edges(
    conn_df: pd.DataFrame,
    static_1d: pd.DataFrame,
    static_2d: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build physics-informed coupling edges between 1D and 2D nodes.

    Creates two directed edge types:
      - surcharge  (1D → 2D): water exits pipes onto the surface
      - drainage   (2D → 1D): surface water enters the pipe network

    Both directions carry the same physical edge features but the model
    learns *distinct* convolution weights for each, reflecting the
    different physics of surcharge (pressure-driven) vs drainage
    (gravity-driven).

    Edge Features (per connection):
      - elevation_diff  = surface_elev_1d − centroid_elev_2d
      - euclidean_dist  = ‖pos_1d − pos_2d‖₂
      - capacity_1d     = capacity of the connected 1D node
      - area_2d         = area of the connected 2D cell
      - dx, dy          = directional offset  (pos_1d − pos_2d)

    Returns
    -------
    edge_index_1d_to_2d : Tensor [2, C]
    edge_index_2d_to_1d : Tensor [2, C]
    edge_attr_1d_to_2d  : Tensor [C, 6]
    edge_attr_2d_to_1d  : Tensor [C, 6]
    """
    _validate_dataframe(
        conn_df, ["node_1d", "node_2d"], "1d2d_conn"
    )

    if conn_df.empty:
        empty_ei = torch.empty(2, 0, dtype=torch.long)
        empty_ea = torch.empty(0, len(_COUPLING_EDGE_FEATURE_NAMES), dtype=torch.float32)
        return empty_ei, empty_ei.clone(), empty_ea, empty_ea.clone()

    # Sort static DataFrames by node_idx for O(1) lookup
    s1d = static_1d.sort_values("node_idx").reset_index(drop=True)
    s2d = static_2d.sort_values("node_idx").reset_index(drop=True)

    # Build lookup dictionaries for positions and attributes
    # 1D node attributes indexed by node_idx
    pos_x_1d = s1d.set_index("node_idx")["position_x"].to_dict()
    pos_y_1d = s1d.set_index("node_idx")["position_y"].to_dict()
    surf_elev_1d = s1d.set_index("node_idx")["surface_elevation"].to_dict()
    invert_elev_1d = s1d.set_index("node_idx")["invert_elevation"].to_dict()

    # 2D node attributes indexed by node_idx
    pos_x_2d = s2d.set_index("node_idx")["position_x"].to_dict()
    pos_y_2d = s2d.set_index("node_idx")["position_y"].to_dict()
    elev_2d = s2d.set_index("node_idx")["elevation"].to_dict()
    area_2d_dict = s2d.set_index("node_idx")["area"].to_dict()

    # Build edge index and features
    src_1d: List[int] = []
    dst_2d: List[int] = []
    features: List[List[float]] = []

    for _, row in conn_df.iterrows():
        n1d = int(row["node_1d"])
        n2d = int(row["node_2d"])

        # Skip connections to nodes not in our static data
        if n1d not in pos_x_1d or n2d not in pos_x_2d:
            warnings.warn(
                f"Coupling edge ({n1d} → {n2d}) references missing node — skipped."
            )
            continue

        x1, y1 = pos_x_1d[n1d], pos_y_1d[n1d]
        x2, y2 = pos_x_2d[n2d], pos_y_2d[n2d]

        dx = x1 - x2
        dy = y1 - y2
        dist = np.sqrt(dx ** 2 + dy ** 2 + _EPS)

        # Elevation difference: positive → 1D node is higher than 2D
        elev_diff = surf_elev_1d[n1d] - elev_2d[n2d]

        # Capacity of the 1D node (surcharge threshold)
        cap_1d = surf_elev_1d[n1d] - invert_elev_1d[n1d]

        # Area of the 2D cell (absorption capacity)
        a_2d = area_2d_dict[n2d]

        src_1d.append(n1d)
        dst_2d.append(n2d)
        features.append([elev_diff, dist, cap_1d, a_2d, dx, dy])

    if not src_1d:
        empty_ei = torch.empty(2, 0, dtype=torch.long)
        empty_ea = torch.empty(0, len(_COUPLING_EDGE_FEATURE_NAMES), dtype=torch.float32)
        return empty_ei, empty_ei.clone(), empty_ea, empty_ea.clone()

    # 1D → 2D (surcharge direction)
    src_t = torch.tensor(src_1d, dtype=torch.long)
    dst_t = torch.tensor(dst_2d, dtype=torch.long)
    ei_1d_to_2d = torch.stack([src_t, dst_t], dim=0)  # [2, C]

    # 2D → 1D (drainage direction) — same connections, reversed
    ei_2d_to_1d = torch.stack([dst_t, src_t], dim=0)  # [2, C]

    # Edge features (same physical properties for both directions;
    # the model learns direction-specific weights via separate conv layers)
    feat_arr = np.array(features, dtype=np.float32)
    edge_attr = torch.from_numpy(feat_arr)

    return ei_1d_to_2d, ei_2d_to_1d, edge_attr, edge_attr.clone()


# =====================================================================
#  Public API
# =====================================================================

def build_unified_graph(
    sample: Dict[str, Any],
    *,
    include_edge_dynamic: bool = False,
) -> HeteroData:
    """Convert a ``FloodDataset`` sample into a heterogeneous 1D-2D graph.

    This is the main entry point for the Unified Engine's data pipeline.
    It consumes all static and dynamic DataFrames from the sample dict
    produced by ``FloodDataset.__getitem__()`` and returns a fully
    assembled ``HeteroData`` object ready for ``UnifiedFloodModel``.

    Parameters
    ----------
    sample : dict
        A single sample from ``FloodDataset.__getitem__()``.
        Required keys (from ``FloodDataset._STATIC_FILES`` and
        ``_DYNAMIC_FILES``):

        Static : ``static_1d_nodes``, ``static_2d_nodes``,
                 ``static_1d_edges``, ``static_2d_edges``,
                 ``edge_index_1d``, ``edge_index_2d``, ``1d2d_conn``
        Dynamic: ``dynamic_1d_nodes``, ``dynamic_2d_nodes``
        Meta   : ``model_id``, ``event_id``

    include_edge_dynamic : bool, optional
        If ``True``, also attach dynamic edge features (flow, velocity)
        from ``dynamic_1d_edges`` to the pipe edge store.  These are
        auxiliary targets — useful for multi-task learning but not
        required for standard water-level prediction.  Default ``False``.

    Returns
    -------
    torch_geometric.data.HeteroData
        Fully populated heterogeneous graph.  See module docstring for
        the complete attribute layout.

    Raises
    ------
    ValueError
        If required DataFrames or columns are missing.

    Examples
    --------
    >>> from src.dataset import FloodDataset
    >>> from src.graph_builder_unified import build_unified_graph
    >>> ds = FloodDataset("data", mode="train")
    >>> sample = ds[0]
    >>> hetero = build_unified_graph(sample)
    >>> hetero.node_types    # ['node_1d', 'node_2d']
    >>> hetero.edge_types    # 4 directed edge types
    >>> hetero['node_1d'].x.shape   # [N_1d, 5]
    >>> hetero['node_1d'].y.shape   # [T, N_1d]
    """

    # ------------------------------------------------------------------
    # 0.  Extract DataFrames from sample
    # ------------------------------------------------------------------
    static_1d_nodes: pd.DataFrame = sample["static_1d_nodes"]
    static_2d_nodes: pd.DataFrame = sample["static_2d_nodes"]
    static_1d_edges: pd.DataFrame = sample["static_1d_edges"]
    static_2d_edges: pd.DataFrame = sample["static_2d_edges"]
    edge_index_1d: pd.DataFrame   = sample["edge_index_1d"]
    edge_index_2d: pd.DataFrame   = sample["edge_index_2d"]
    conn_1d2d: pd.DataFrame       = sample["1d2d_conn"]
    dynamic_1d: pd.DataFrame      = sample["dynamic_1d_nodes"]
    dynamic_2d: pd.DataFrame      = sample["dynamic_2d_nodes"]

    model_id: str = sample.get("model_id", "unknown")
    event_id: str = sample.get("event_id", "unknown")

    n_1d = len(static_1d_nodes)
    n_2d = len(static_2d_nodes)

    # ------------------------------------------------------------------
    # 1.  1D node features
    # ------------------------------------------------------------------
    x_1d, invert_elev, capacity = _build_1d_static_features(static_1d_nodes)
    dyn_1d, y_1d = _build_1d_dynamic_features(dynamic_1d, n_1d, invert_elev, capacity)

    # ------------------------------------------------------------------
    # 2.  2D node features
    # ------------------------------------------------------------------
    x_2d = _build_2d_static_features(static_2d_nodes)
    dyn_2d, y_2d = _build_2d_dynamic_features(dynamic_2d, n_2d)

    # ------------------------------------------------------------------
    # 3.  Synchronise timestep counts
    # ------------------------------------------------------------------
    # 1D and 2D dynamic data may have slightly different timestep
    # counts in edge cases.  Truncate to the shorter sequence so that
    # targets and dynamics are always aligned.
    t_1d, t_2d = y_1d.size(0), y_2d.size(0)
    if t_1d != t_2d:
        warnings.warn(
            f"Timestep mismatch: 1D has {t_1d}, 2D has {t_2d}. "
            f"Truncating to min={min(t_1d, t_2d)}."
        )
        t_min = min(t_1d, t_2d)
        y_1d = y_1d[:t_min]
        dyn_1d = dyn_1d[:t_min]
        y_2d = y_2d[:t_min]
        dyn_2d = dyn_2d[:t_min]

    num_timesteps = y_1d.size(0)

    # ------------------------------------------------------------------
    # 4.  Pipe edges (bidirectional)
    # ------------------------------------------------------------------
    pipe_ei, pipe_ea = _build_pipe_edges(edge_index_1d, static_1d_edges)

    # ------------------------------------------------------------------
    # 5.  Surface edges (bidirectional)
    # ------------------------------------------------------------------
    surf_ei, surf_ea = _build_surface_edges(edge_index_2d, static_2d_edges)

    # ------------------------------------------------------------------
    # 6.  Coupling edges (1D↔2D, two directed types)
    # ------------------------------------------------------------------
    coup_ei_12, coup_ei_21, coup_ea_12, coup_ea_21 = _build_coupling_edges(
        conn_1d2d, static_1d_nodes, static_2d_nodes
    )

    # ------------------------------------------------------------------
    # 7.  Assemble HeteroData
    # ------------------------------------------------------------------
    data = HeteroData()

    # ---- 1D nodes ----------------------------------------------------
    data["node_1d"].x = x_1d                       # [N_1d, 5]
    data["node_1d"].y = y_1d                       # [T, N_1d]
    data["node_1d"].dynamic = dyn_1d               # [T, N_1d, 3]
    data["node_1d"].invert_elev = invert_elev      # [N_1d]
    data["node_1d"].capacity = capacity             # [N_1d]
    data["node_1d"].num_nodes = n_1d

    # ---- 2D nodes ----------------------------------------------------
    data["node_2d"].x = x_2d                       # [N_2d, 9]
    data["node_2d"].y = y_2d                       # [T, N_2d]
    data["node_2d"].dynamic = dyn_2d               # [T, N_2d, 3]
    data["node_2d"].num_nodes = n_2d

    # ---- Pipe edges (1D internal) ------------------------------------
    data["node_1d", "pipe_to", "node_1d"].edge_index = pipe_ei
    data["node_1d", "pipe_to", "node_1d"].edge_attr = pipe_ea

    # ---- Surface edges (2D internal) ---------------------------------
    data["node_2d", "surface_to", "node_2d"].edge_index = surf_ei
    data["node_2d", "surface_to", "node_2d"].edge_attr = surf_ea

    # ---- Coupling: surcharge path (1D → 2D) -------------------------
    data["node_1d", "surcharges_to", "node_2d"].edge_index = coup_ei_12
    data["node_1d", "surcharges_to", "node_2d"].edge_attr = coup_ea_12

    # ---- Coupling: drainage path (2D → 1D) --------------------------
    data["node_2d", "drains_to", "node_1d"].edge_index = coup_ei_21
    data["node_2d", "drains_to", "node_1d"].edge_attr = coup_ea_21

    # ------------------------------------------------------------------
    # 8.  Optional: dynamic edge features (flow, velocity) for pipes
    # ------------------------------------------------------------------
    if include_edge_dynamic:
        dyn_edges_df: pd.DataFrame = sample.get("dynamic_1d_edges", pd.DataFrame())
        if not dyn_edges_df.empty and {"timestep", "edge_idx", "flow", "velocity"}.issubset(dyn_edges_df.columns):
            n_edges_raw = len(static_1d_edges)
            pipe_edge_dyn = _pivot_dynamic_to_tensor(
                dyn_edges_df, n_edges_raw,
                timestep_col="timestep", node_col="edge_idx",
                value_cols=["flow", "velocity"],
            )  # [T, E_raw, 2]
            # Duplicate for bidirectional: reverse edges get same features
            # (flow sign should ideally be flipped for reverse, but we
            #  leave this to the model to learn direction-awareness)
            n_pipe = pipe_ei.size(1)
            if pipe_edge_dyn.size(1) * 2 <= n_pipe:
                # Bidirectional duplication
                pipe_edge_dyn = torch.cat(
                    [pipe_edge_dyn, pipe_edge_dyn], dim=1
                )
            data["node_1d", "pipe_to", "node_1d"].edge_dynamic = pipe_edge_dyn[:num_timesteps]

    # ------------------------------------------------------------------
    # 9.  Metadata (non-tensor attributes for bookkeeping)
    # ------------------------------------------------------------------
    data.model_id = model_id
    data.event_id = event_id
    data.num_timesteps = num_timesteps

    # Feature name registries for downstream introspection
    data.feature_names_1d_static = _1D_STATIC_FEATURE_NAMES
    data.feature_names_1d_dynamic = _1D_DYNAMIC_FEATURE_NAMES
    data.feature_names_2d_static = _2D_STATIC_FEATURE_NAMES
    data.feature_names_2d_dynamic = _2D_DYNAMIC_FEATURE_NAMES
    data.feature_names_pipe_edge = _PIPE_EDGE_FEATURE_NAMES
    data.feature_names_surface_edge = _SURFACE_EDGE_FEATURE_NAMES
    data.feature_names_coupling_edge = _COUPLING_EDGE_FEATURE_NAMES

    return data


# =====================================================================
#  Utilities (for downstream consumers)
# =====================================================================

def get_feature_dims(data: HeteroData) -> Dict[str, int]:
    """Return a summary of feature dimensions for model initialisation.

    Useful for constructing ``UnifiedFloodModel(in_channels_1d=...,
    in_channels_2d=...)``.

    Returns
    -------
    dict with keys:
      - ``in_channels_1d_static``
      - ``in_channels_1d_dynamic``
      - ``in_channels_2d_static``
      - ``in_channels_2d_dynamic``
      - ``pipe_edge_dim``
      - ``surface_edge_dim``
      - ``coupling_edge_dim``
      - ``n_1d``, ``n_2d``, ``num_timesteps``
    """
    return {
        "in_channels_1d_static":  data["node_1d"].x.size(-1),
        "in_channels_1d_dynamic": data["node_1d"].dynamic.size(-1),
        "in_channels_2d_static":  data["node_2d"].x.size(-1),
        "in_channels_2d_dynamic": data["node_2d"].dynamic.size(-1),
        "pipe_edge_dim":     data["node_1d", "pipe_to", "node_1d"].edge_attr.size(-1),
        "surface_edge_dim":  data["node_2d", "surface_to", "node_2d"].edge_attr.size(-1),
        "coupling_edge_dim": data["node_1d", "surcharges_to", "node_2d"].edge_attr.size(-1),
        "n_1d":              data["node_1d"].num_nodes,
        "n_2d":              data["node_2d"].num_nodes,
        "num_timesteps":     data.num_timesteps,
    }


def summarise_graph(data: HeteroData) -> str:
    """Return a human-readable summary of the HeteroData object.

    Useful for logging and sanity-checking after graph construction.
    """
    lines = [
        f"╔══ Unified Graph  (Model {data.model_id}, Event {data.event_id}) ══",
        f"║  Timesteps : {data.num_timesteps}",
        f"║  1D Nodes  : {data['node_1d'].num_nodes:>6d}   "
        f"static={list(data['node_1d'].x.shape)}  "
        f"dynamic={list(data['node_1d'].dynamic.shape)}",
        f"║  2D Nodes  : {data['node_2d'].num_nodes:>6d}   "
        f"static={list(data['node_2d'].x.shape)}  "
        f"dynamic={list(data['node_2d'].dynamic.shape)}",
    ]

    for et in data.edge_types:
        store = data[et]
        ei_shape = list(store.edge_index.shape)
        ea_shape = list(store.edge_attr.shape) if hasattr(store, "edge_attr") else "—"
        label = f"({et[0]}, {et[1]}, {et[2]})"
        lines.append(f"║  Edge {label:<45s}  index={ei_shape}  attr={ea_shape}")

    lines.append("╚" + "═" * 70)
    return "\n".join(lines)

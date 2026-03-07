"""
graph_builder_unified — Depth-Based, Normalised HeteroGraph Builder (v6).

Constructs a ``torch_geometric.data.HeteroData`` for the Unified
1D-2D coupled flood model with **depth-based features**, **rich
static features**, **depth lags**, and **per-model z-score
normalisation**.

Owner : Member C (Lead Architect)
See   : IMPLEMENTATION_PLAN.md, PROJECT_BIBLE.md §4-6

Key Design Decisions (v6 — from v5)
------------------------------------
1.  **min_elevation** for 2D WSE→Depth conversion (not centroid).
    Only 0.3% depths negative (vs 93.6% with centroid_elevation).
2.  **Rich 2D static features**: area, roughness, aspect, curvature,
    flow_accumulation, elev_rel_neighbors, dist_to_drain, is_connected.
3.  **Depth lags** (t-2, t-3, t-4) for explicit temporal context.
4.  **1D enrichment**: base_area added.
5.  Per-model z-score normalisation (unchanged from v5).
6.  Raw **min_elevation** stored in ``.elev`` for physics-compliant
    loss recovery: ``pred_wse = pred_depth + min_elevation``.

Node & Feature Layout
---------------------
``data['1d'].x``  : [T, N_1d, 7]
    [depth, inlet_flow, lag1, lag2, lag3, capacity, base_area]
``data['1d'].y``  : [T, N_1d]   — target WSE (absolute)
``data['1d'].depth``: [T, N_1d] — raw depth (for delta targets)
``data['1d'].elev``: [N_1d]     — invert_elevation (raw)

``data['2d'].x``  : [T, N_2d, 19]
    [depth, rainfall, lag1, lag2, lag3, rain_rolling_mean, rain_delta,
     rain_lag2, elevation, min_elevation, slope, area, roughness,
     aspect, curvature, flow_acc, elev_rel, dist_to_drain, is_connected]
``data['2d'].y``  : [T, N_2d]   — target WSE (absolute)
``data['2d'].depth``: [T, N_2d] — raw depth (min_elev ref, for deltas)
``data['2d'].elev``: [N_2d]     — **min_elevation** (for pred_wse)

Edge Types
----------
``('1d', 'pipe',  '1d')``  — bidirectional pipe flow
``('2d', 'spread','2d')``  — bidirectional surface mesh adjacency
``('1d', 'link',  '2d')``  — 1D→2D coupling (surcharge direction)
``('2d', 'link',  '1d')``  — 2D→1D coupling (drainage direction)

Dynamic Feature Indices (replaced during AR rollout)
----------------------------------------------------
``DEPTH_IDX = 0``   — depth at t−1 (current state)
``LAG1_IDX  = 2``   — depth at t−2
``LAG2_IDX  = 3``   — depth at t−3
``LAG3_IDX  = 4``   — depth at t−4
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

# Feature index documentation (single source of truth)
_1D_FEATURE_NAMES: List[str] = [
    "depth", "inlet_flow", "lag1", "lag2", "lag3",
    "capacity", "base_area",
    "pipe_diameter", "pipe_length", "pipe_roughness", "pipe_slope",  # Phase B
]
_2D_FEATURE_NAMES: List[str] = [
    "depth", "rainfall", "lag1", "lag2", "lag3",
    "effective_depth",  # P0: water_volume/area (physics-aligned)
    "rain_rolling_mean", "rain_delta", "rain_lag2",  # EDA: best lag=2, rolling, delta
    "elevation", "min_elevation", "slope", "area", "roughness", "aspect",
    "curvature", "flow_accumulation", "elev_rel_neighbors", "dist_to_drain",
    "is_connected",
    "position_x", "position_y",  # Phase A: Member B alignment
]

# Dynamic feature indices replaced during AR rollout in training loop
DEPTH_IDX: int = 0
WATER_VOL_IDX: int = 5  # 2D only; P0: effective_depth = water_volume/area; AR uses pred_depth
EFFECTIVE_DEPTH_IDX: int = 5  # Alias for WATER_VOL_IDX (same slot)
LAG1_IDX: int = 2
LAG2_IDX: int = 3
LAG3_IDX: int = 4


# =====================================================================
#  Internal Helpers
# =====================================================================

def _normalize(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Z-score normalise: ``(x - mean) / max(std, ε)``."""
    return (x - mean) / max(std, _EPS)


def _normalize_or_zero(
    x: torch.Tensor, mean: float, std: float, min_std: float = 1e-6
) -> torch.Tensor:
    """Z-score normalise, or zero if std < min_std (P0: constant features)."""
    if std < min_std:
        return torch.zeros_like(x)
    return _normalize(x, mean, std)


def _pivot_to_tensor(
    df: pd.DataFrame,
    n_nodes: int,
    value_col: str,
    timestep_col: str = "timestep",
    node_col: str = "node_idx",
) -> torch.Tensor:
    """Pivot a long-format dynamic column into a dense ``[T, N]`` tensor.

    Missing entries are filled with ``0.0`` (physically: dry / no flow).
    """
    if df.empty or value_col not in df.columns:
        if not df.empty and value_col not in df.columns:
            warnings.warn(
                f"Column '{value_col}' not in dynamic DataFrame "
                f"(available: {sorted(df.columns)}) — filled with zeros."
            )
        return torch.zeros(1, n_nodes, dtype=torch.float32)

    timesteps = sorted(df[timestep_col].unique())
    t_map = {t: i for i, t in enumerate(timesteps)}
    T = len(timesteps)

    tensor = torch.zeros(T, n_nodes, dtype=torch.float32)
    t_idx = df[timestep_col].map(t_map).values.astype(np.intp)
    n_idx = df[node_col].values.astype(np.intp)
    vals = df[value_col].values.astype(np.float32)
    vals = np.nan_to_num(vals, nan=0.0)
    tensor[t_idx, n_idx] = torch.from_numpy(vals)

    return tensor


def _make_bidirectional(edge_index: torch.Tensor) -> torch.Tensor:
    """Convert directed ``[2, E]`` edge_index to bidirectional (deduped).

    Returns
    -------
    Tensor [2, E_bi]  where E_bi ≤ 2*E (duplicates removed).
    """
    rev = edge_index.flip(0)
    bi = torch.cat([edge_index, rev], dim=1)

    # Canonical key: pack (u, v) into a single int64
    max_id = int(bi.max().item()) + 1
    packed = bi[0].long() * max_id + bi[1].long()
    _, inv = torch.unique(packed, return_inverse=True)

    # Keep first occurrence per unique edge
    keep = torch.zeros(bi.size(1), dtype=torch.bool)
    seen: set[int] = set()
    for i in range(bi.size(1)):
        g = inv[i].item()
        if g not in seen:
            seen.add(g)
            keep[i] = True

    return bi[:, keep]


def _validate_columns(
    df: pd.DataFrame,
    required: List[str],
    name: str,
) -> None:
    """Raise ``ValueError`` if *df* is missing required columns."""
    if df.empty:
        raise ValueError(f"DataFrame '{name}' is empty.")
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"'{name}' is missing columns: {sorted(missing)}. "
            f"Available: {sorted(df.columns)}"
        )


# =====================================================================
#  Derived Feature Computation
# =====================================================================

def compute_elev_rel_neighbors(
    elevation: np.ndarray,
    edge_index: torch.Tensor,
) -> np.ndarray:
    """Elevation of each node relative to the mean of its graph neighbours.

    Positive → hilltop (higher than surroundings).
    Negative → depression where water tends to pool.

    Parameters
    ----------
    elevation : np.ndarray, shape [N]
        Raw elevation values per node.
    edge_index : torch.Tensor, shape [2, E]
        Bidirectional edge index.

    Returns
    -------
    np.ndarray, shape [N]
    """
    N = len(elevation)
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    neighbor_sum = np.zeros(N, dtype=np.float64)
    neighbor_count = np.zeros(N, dtype=np.float64)
    np.add.at(neighbor_sum, src, elevation[dst])
    np.add.at(neighbor_count, src, 1)

    neighbor_count = np.maximum(neighbor_count, 1)
    neighbor_mean = neighbor_sum / neighbor_count
    return (elevation - neighbor_mean).astype(np.float32)


def compute_dist_to_drain(
    coords_2d: np.ndarray,
    coords_1d: np.ndarray,
    conn_df: pd.DataFrame,
    n_2d: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Distance to nearest 1D node + binary is_connected flag.

    Uses simple numpy broadcasting (no scipy dependency needed for
    the small number of 1D nodes).

    Returns
    -------
    dist_to_drain : np.ndarray [N_2d]
    is_connected  : np.ndarray [N_2d]   (1.0 or 0.0)
    """
    if len(coords_1d) > 0:
        # [N_2d, N_1d, 2]
        diffs = coords_2d[:, None, :] - coords_1d[None, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=-1))  # [N_2d, N_1d]
        dist_to_drain = dists.min(axis=1).astype(np.float32)
    else:
        dist_to_drain = np.zeros(n_2d, dtype=np.float32)

    if not conn_df.empty and "node_2d" in conn_df.columns:
        connected = set(conn_df["node_2d"].values)
    else:
        connected = set()
    is_connected = np.array(
        [1.0 if i in connected else 0.0 for i in range(n_2d)],
        dtype=np.float32,
    )
    return dist_to_drain, is_connected


def _compute_rainfall_derivatives(
    rainfall: torch.Tensor,
    window: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute rolling mean and delta for rainfall [T, N].

    Returns
    -------
    rain_rolling_mean : [T, N] mean over last *window* steps
    rain_delta        : [T, N] rainfall[t] - rainfall[t-1], first step = 0
    """
    T, N = rainfall.shape
    rolling = torch.empty_like(rainfall)
    delta = torch.zeros_like(rainfall)
    for t in range(T):
        start = max(0, t - window + 1)
        rolling[t] = rainfall[start : t + 1].mean(dim=0)
        if t > 0:
            delta[t] = rainfall[t] - rainfall[t - 1]
    return rolling, delta


def _compute_depth_lags(
    depth: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute historical depth lags from a [T, N] depth tensor.

    Returns lags that avoid redundancy with the depth feature (index 0).
    Since the training loop replaces feature 0 with depth[t-1],
    the lags represent earlier history:

        lag1[t] = depth[t-2]
        lag2[t] = depth[t-3]
        lag3[t] = depth[t-4]

    Early timesteps are padded with depth[0].
    """
    T, N = depth.shape
    d0 = depth[0]  # [N]

    lags = []
    for offset in [2, 3, 4]:
        lag = torch.empty_like(depth)
        pad_len = min(offset, T)
        lag[:pad_len] = d0
        if T > offset:
            lag[offset:] = depth[:T - offset]
        lags.append(lag)

    return lags[0], lags[1], lags[2]


def compute_mean_slope_per_node(
    ei_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    n_nodes: int,
) -> np.ndarray:
    """Aggregate edge slope to node: mean slope of incident edges (EDA: terrain physics).

    Assumes ei_df rows align with edges_df (row i = edge i). If edges_df has
    edge_idx, uses that; else row index = edge_idx.
    """
    if edges_df.empty or "slope" not in edges_df.columns:
        return np.zeros(n_nodes, dtype=np.float32)
    src = ei_df["from_node"].values.astype(np.intp)
    dst = ei_df["to_node"].values.astype(np.intp)
    n_edges = len(src)
    slopes = edges_df["slope"].values.astype(np.float32)[:n_edges]
    if len(slopes) < n_edges:
        slopes = np.concatenate([slopes, np.zeros(n_edges - len(slopes), dtype=np.float32)])
    slopes = np.nan_to_num(slopes, nan=0.0)
    slope_sum = np.zeros(n_nodes, dtype=np.float64)
    slope_cnt = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(slope_sum, src, slopes)
    np.add.at(slope_sum, dst, slopes)
    np.add.at(slope_cnt, src, 1)
    np.add.at(slope_cnt, dst, 1)
    slope_cnt = np.maximum(slope_cnt, 1)
    return (slope_sum / slope_cnt).astype(np.float32)


def compute_mean_pipe_attr_per_node(
    ei_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    n_nodes: int,
    attr: str,
    default: float = 0.0,
) -> np.ndarray:
    """Aggregate 1D pipe attribute to node: mean over incident edges.

    Used for diameter, length, roughness, slope from static_1d_edges.
    """
    if edges_df.empty or attr not in edges_df.columns:
        return np.full(n_nodes, default, dtype=np.float32)
    src = ei_df["from_node"].values.astype(np.intp)
    dst = ei_df["to_node"].values.astype(np.intp)
    n_edges = len(src)
    vals = edges_df[attr].values.astype(np.float32)[:n_edges]
    if len(vals) < n_edges:
        vals = np.concatenate([vals, np.full(n_edges - len(vals), default, dtype=np.float32)])
    vals = np.nan_to_num(vals, nan=default)
    attr_sum = np.zeros(n_nodes, dtype=np.float64)
    attr_cnt = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(attr_sum, src, vals)
    np.add.at(attr_sum, dst, vals)
    np.add.at(attr_cnt, src, 1)
    np.add.at(attr_cnt, dst, 1)
    attr_cnt = np.maximum(attr_cnt, 1)
    return (attr_sum / attr_cnt).astype(np.float32)


def _safe_col(df: pd.DataFrame, col: str, n: int, default: float = 0.0) -> np.ndarray:
    """Extract column from DataFrame, falling back to constant if missing."""
    if col in df.columns:
        vals = df[col].values.astype(np.float32)
        nan_mask = np.isnan(vals)
        if nan_mask.any():
            vals[nan_mask] = default
        return vals
    return np.full(n, default, dtype=np.float32)


# =====================================================================
#  Public API
# =====================================================================

def build_hetero_graph(
    raw_data: Dict[str, Any],
    norm_stats: Dict[str, Dict[str, Dict[str, float]]],
) -> HeteroData:
    """Convert a ``FloodDataset`` sample into a normalised HeteroData graph.

    Parameters
    ----------
    raw_data : dict
        Sample dictionary from ``FloodDataset.__getitem__()``.
    norm_stats : dict
        Per-model normalisation statistics (v6 structure with rich features).

    Returns
    -------
    HeteroData
        Fully assembled graph ready for ``UnifiedHeteroModel``.
    """

    # ------------------------------------------------------------------
    # 0.  Extract & validate raw DataFrames
    # ------------------------------------------------------------------
    static_1d: pd.DataFrame = raw_data["static_1d_nodes"]
    static_2d: pd.DataFrame = raw_data["static_2d_nodes"]
    dyn_1d: pd.DataFrame = raw_data["dynamic_1d_nodes"]
    dyn_2d: pd.DataFrame = raw_data["dynamic_2d_nodes"]
    ei_1d_df: pd.DataFrame = raw_data["edge_index_1d"]
    ei_2d_df: pd.DataFrame = raw_data["edge_index_2d"]
    conn_df: pd.DataFrame = raw_data["1d2d_conn"]

    _validate_columns(
        static_1d,
        ["node_idx", "invert_elevation", "surface_elevation", "base_area"],
        "static_1d_nodes",
    )
    _validate_columns(
        static_2d,
        ["node_idx", "elevation", "area", "roughness"],
        "static_2d_nodes",
    )

    s1d = static_1d.sort_values("node_idx").reset_index(drop=True)
    s2d = static_2d.sort_values("node_idx").reset_index(drop=True)
    n_1d = len(s1d)
    n_2d = len(s2d)

    # ------------------------------------------------------------------
    # 1.  1D static quantities
    # ------------------------------------------------------------------
    invert_elev = torch.tensor(
        s1d["invert_elevation"].values, dtype=torch.float32
    )
    surface_elev = torch.tensor(
        s1d["surface_elevation"].values, dtype=torch.float32
    )
    capacity = (surface_elev - invert_elev).clamp(min=_EPS)  # [N_1d]
    base_area = torch.tensor(
        s1d["base_area"].values, dtype=torch.float32
    )

    # ------------------------------------------------------------------
    # 2.  2D static quantities — min_elevation as depth reference
    # ------------------------------------------------------------------
    centroid_elev_np = s2d["elevation"].values.astype(np.float32)

    # Prefer min_elevation (water pools at lowest cell point).
    # Fill NaN entries with centroid elevation as fallback.
    if "min_elevation" in s2d.columns:
        min_elev_np = s2d["min_elevation"].values.astype(np.float32)
        nan_mask = np.isnan(min_elev_np)
        if nan_mask.any():
            min_elev_np[nan_mask] = centroid_elev_np[nan_mask]
    else:
        min_elev_np = centroid_elev_np.copy()
        warnings.warn(
            "min_elevation not in 2D static data — "
            "falling back to centroid_elevation as depth reference."
        )

    min_elev = torch.tensor(min_elev_np, dtype=torch.float32)
    centroid_elev = torch.tensor(centroid_elev_np, dtype=torch.float32)

    # Additional 2D static features
    area_t = torch.tensor(_safe_col(s2d, "area", n_2d))
    roughness_t = torch.tensor(_safe_col(s2d, "roughness", n_2d, 0.03))
    aspect_t = torch.tensor(_safe_col(s2d, "aspect", n_2d))
    curvature_t = torch.tensor(_safe_col(s2d, "curvature", n_2d))
    flow_acc_t = torch.tensor(_safe_col(s2d, "flow_accumulation", n_2d, 1.0))

    # ------------------------------------------------------------------
    # 3.  Convert WSE → Depth (dynamic)
    # ------------------------------------------------------------------
    wse_1d = _pivot_to_tensor(dyn_1d, n_1d, "water_level")        # [T1, N_1d]
    depth_1d = wse_1d - invert_elev.unsqueeze(0)                  # [T1, N_1d]
    inlet_flow = _pivot_to_tensor(dyn_1d, n_1d, "inlet_flow")     # [T1, N_1d]

    wse_2d = _pivot_to_tensor(dyn_2d, n_2d, "water_level")        # [T2, N_2d]
    depth_2d = wse_2d - min_elev.unsqueeze(0)                     # KEY: min_elevation!
    rainfall = _pivot_to_tensor(dyn_2d, n_2d, "rainfall")          # [T2, N_2d]
    water_volume = _pivot_to_tensor(dyn_2d, n_2d, "water_volume")  # Phase A; zeros if missing

    # P0: effective_depth = water_volume / area (physics-aligned)
    # Floor area at 1.0 to avoid explosion when Model 1 has area≈0 cells
    area_safe = area_t.unsqueeze(0).clamp(min=1.0)
    effective_depth = (water_volume / area_safe).clamp(min=0.0, max=20.0)

    # Synchronise timestep counts across 1D / 2D
    T = min(depth_1d.size(0), depth_2d.size(0))
    if depth_1d.size(0) != depth_2d.size(0):
        warnings.warn(
            f"Timestep mismatch: 1D={depth_1d.size(0)}, 2D={depth_2d.size(0)}. "
            f"Truncating to T={T}."
        )
    depth_1d = depth_1d[:T]
    inlet_flow = inlet_flow[:T]
    wse_1d = wse_1d[:T]
    depth_2d = depth_2d[:T]
    rainfall = rainfall[:T]
    water_volume = water_volume[:T]
    effective_depth = effective_depth[:T]
    wse_2d = wse_2d[:T]

    # ------------------------------------------------------------------
    # 4.  Depth lags (t-2, t-3, t-4)
    # ------------------------------------------------------------------
    lag1_1d, lag2_1d, lag3_1d = _compute_depth_lags(depth_1d)
    lag1_2d, lag2_2d, lag3_2d = _compute_depth_lags(depth_2d)

    # ------------------------------------------------------------------
    # 4b. Rainfall derivatives + lag (EDA: best lag=2, rolling, delta)
    # ------------------------------------------------------------------
    rain_rolling_mean, rain_delta = _compute_rainfall_derivatives(rainfall, window=3)
    rain_lag2 = torch.zeros_like(rainfall)
    rain_lag2[2:] = rainfall[:-2]
    rain_lag2[:2] = rainfall[0]

    # ------------------------------------------------------------------
    # 5.  Edge indices (built early — 2D needed for elev_rel_neighbors)
    # ------------------------------------------------------------------

    # Pipe edges: 1d ↔ 1d
    _validate_columns(ei_1d_df, ["from_node", "to_node"], "edge_index_1d")
    pipe_src = torch.tensor(ei_1d_df["from_node"].values, dtype=torch.long)
    pipe_dst = torch.tensor(ei_1d_df["to_node"].values, dtype=torch.long)
    pipe_ei = _make_bidirectional(torch.stack([pipe_src, pipe_dst], dim=0))

    # Spread edges: 2d ↔ 2d
    _validate_columns(ei_2d_df, ["from_node", "to_node"], "edge_index_2d")
    surf_src = torch.tensor(ei_2d_df["from_node"].values, dtype=torch.long)
    surf_dst = torch.tensor(ei_2d_df["to_node"].values, dtype=torch.long)
    spread_ei = _make_bidirectional(torch.stack([surf_src, surf_dst], dim=0))

    # Link edges: 1d ↔ 2d
    if (
        not conn_df.empty
        and {"node_1d", "node_2d"}.issubset(conn_df.columns)
    ):
        link_1d = torch.tensor(conn_df["node_1d"].values, dtype=torch.long)
        link_2d = torch.tensor(conn_df["node_2d"].values, dtype=torch.long)
        link_1d_to_2d = torch.stack([link_1d, link_2d], dim=0)
        link_2d_to_1d = torch.stack([link_2d, link_1d], dim=0)
    else:
        warnings.warn("1d2d_connections is empty — no coupling edges.")
        link_1d_to_2d = torch.empty(2, 0, dtype=torch.long)
        link_2d_to_1d = torch.empty(2, 0, dtype=torch.long)

    # ------------------------------------------------------------------
    # 6.  Derived spatial features
    # ------------------------------------------------------------------
    elev_rel = torch.tensor(
        compute_elev_rel_neighbors(centroid_elev_np, spread_ei),
        dtype=torch.float32,
    )

    coords_2d = s2d[["position_x", "position_y"]].values.astype(np.float32)
    coords_1d = s1d[["position_x", "position_y"]].values.astype(np.float32)
    dist_drain_np, is_conn_np = compute_dist_to_drain(
        coords_2d, coords_1d, conn_df, n_2d
    )
    dist_drain = torch.tensor(dist_drain_np, dtype=torch.float32)
    is_connected = torch.tensor(is_conn_np, dtype=torch.float32)

    # Slope (EDA: terrain physics) — aggregate from static_2d_edges
    edges_2d_df: pd.DataFrame = raw_data.get("static_2d_edges", pd.DataFrame())
    slope_np = compute_mean_slope_per_node(ei_2d_df, edges_2d_df, n_2d)
    slope_t = torch.tensor(slope_np, dtype=torch.float32)

    # Phase A: position (Member B alignment)
    pos_x_t = torch.tensor(_safe_col(s2d, "position_x", n_2d), dtype=torch.float32)
    pos_y_t = torch.tensor(_safe_col(s2d, "position_y", n_2d), dtype=torch.float32)

    # Phase B: 1D pipe attributes
    edges_1d_df: pd.DataFrame = raw_data.get("static_1d_edges", pd.DataFrame())
    pipe_diam_np = compute_mean_pipe_attr_per_node(ei_1d_df, edges_1d_df, n_1d, "diameter")
    pipe_len_np = compute_mean_pipe_attr_per_node(ei_1d_df, edges_1d_df, n_1d, "length")
    pipe_rough_np = compute_mean_pipe_attr_per_node(ei_1d_df, edges_1d_df, n_1d, "roughness", 0.03)
    pipe_slope_np = compute_mean_pipe_attr_per_node(ei_1d_df, edges_1d_df, n_1d, "slope")

    # ------------------------------------------------------------------
    # 7.  Z-score normalise all features (Member B: per-model Z-score)
    # ------------------------------------------------------------------
    s1 = norm_stats["1d"]
    s2 = norm_stats["2d"]

    # 1D dynamic
    # ── 1D depth: per-node normalization if available ────────────
    #    Global 1D depth stats have std=24.83 (driven by constant
    #    nodes at extreme depths).  Per-node normalization ensures
    #    each pipe's depth signal is in a meaningful [-3, 3] range.
    if "depth_per_node_mean" in s1 and "depth_per_node_std" in s1:
        pn_mean = torch.tensor(s1["depth_per_node_mean"], dtype=torch.float32)  # [N_1d]
        pn_std = torch.tensor(s1["depth_per_node_std"], dtype=torch.float32)    # [N_1d]
        norm_depth_1d = (depth_1d - pn_mean.unsqueeze(0)) / pn_std.unsqueeze(0)
        norm_lag1_1d = (lag1_1d - pn_mean.unsqueeze(0)) / pn_std.unsqueeze(0)
        norm_lag2_1d = (lag2_1d - pn_mean.unsqueeze(0)) / pn_std.unsqueeze(0)
        norm_lag3_1d = (lag3_1d - pn_mean.unsqueeze(0)) / pn_std.unsqueeze(0)
    else:
        norm_depth_1d = _normalize(depth_1d, s1["depth"]["mean"], s1["depth"]["std"])
        norm_lag1_1d = _normalize(lag1_1d, s1["depth"]["mean"], s1["depth"]["std"])
        norm_lag2_1d = _normalize(lag2_1d, s1["depth"]["mean"], s1["depth"]["std"])
        norm_lag3_1d = _normalize(lag3_1d, s1["depth"]["mean"], s1["depth"]["std"])

    norm_iflow = _normalize(inlet_flow, s1["inlet_flow"]["mean"], s1["inlet_flow"]["std"])

    # 1D static
    norm_cap = _normalize(capacity, s1["capacity"]["mean"], s1["capacity"]["std"])
    norm_ba = _normalize(base_area, s1["base_area"]["mean"], s1["base_area"]["std"])
    # Phase B: pipe attributes (P0: zero constant features when std < 1e-6)
    def _pipe_norm(arr: np.ndarray, key: str) -> torch.Tensor:
        st = s1.get(key, {"mean": 0.0, "std": 1.0})
        return _normalize_or_zero(
            torch.tensor(arr, dtype=torch.float32),
            st["mean"], st["std"],
        )
    norm_pipe_diam = _pipe_norm(pipe_diam_np, "pipe_diameter").unsqueeze(0).expand(T, -1)
    norm_pipe_len = _pipe_norm(pipe_len_np, "pipe_length").unsqueeze(0).expand(T, -1)
    norm_pipe_rough = _pipe_norm(pipe_rough_np, "pipe_roughness").unsqueeze(0).expand(T, -1)
    norm_pipe_slope = _pipe_norm(pipe_slope_np, "pipe_slope").unsqueeze(0).expand(T, -1)

    # ------------------------------------------------------------------
    # 6b. Edge flow features REMOVED (Run 12: data leakage fix)
    #      GT edge flows were visible during training/val but frozen at
    #      test inference → train/test distribution mismatch.
    # ------------------------------------------------------------------

    # 2D dynamic
    norm_depth_2d = _normalize(depth_2d, s2["depth"]["mean"], s2["depth"]["std"])
    norm_rain = _normalize(rainfall, s2["rainfall"]["mean"], s2["rainfall"]["std"])
    norm_lag1_2d = _normalize(lag1_2d, s2["depth"]["mean"], s2["depth"]["std"])
    norm_lag2_2d = _normalize(lag2_2d, s2["depth"]["mean"], s2["depth"]["std"])
    norm_lag3_2d = _normalize(lag3_2d, s2["depth"]["mean"], s2["depth"]["std"])

    # 2D rainfall derivatives (EDA-derived)
    norm_rain_roll = _normalize(
        rain_rolling_mean,
        s2["rain_rolling_mean"]["mean"],
        s2["rain_rolling_mean"]["std"],
    )
    norm_rain_delta = _normalize(
        rain_delta,
        s2["rain_delta"]["mean"],
        s2["rain_delta"]["std"],
    )
    norm_rain_lag2 = _normalize(
        rain_lag2,
        s2["rainfall"]["mean"],
        s2["rainfall"]["std"],
    )

    # 2D static
    norm_elev = _normalize(centroid_elev, s2["elevation"]["mean"], s2["elevation"]["std"])
    norm_min_elev = _normalize(min_elev, s2["min_elevation"]["mean"], s2["min_elevation"]["std"])
    norm_slope = _normalize(slope_t, s2["slope"]["mean"], s2["slope"]["std"])
    norm_area = _normalize(area_t, s2["area"]["mean"], s2["area"]["std"])
    # P0: zero 2D roughness when std < 1e-6 (constant feature)
    norm_rough = _normalize_or_zero(
        roughness_t, s2["roughness"]["mean"], s2["roughness"]["std"]
    )
    norm_aspect = _normalize(aspect_t, s2["aspect"]["mean"], s2["aspect"]["std"])
    norm_curv = _normalize(curvature_t, s2["curvature"]["mean"], s2["curvature"]["std"])
    norm_fa = _normalize(flow_acc_t, s2["flow_accumulation"]["mean"], s2["flow_accumulation"]["std"])
    norm_er = _normalize(elev_rel, s2["elev_rel_neighbors"]["mean"], s2["elev_rel_neighbors"]["std"])
    norm_dd = _normalize(dist_drain, s2["dist_to_drain"]["mean"], s2["dist_to_drain"]["std"])
    # P0: effective_depth = water_volume/area (physics-aligned)
    st_eff = s2.get("effective_depth", {"mean": 0.0, "std": 1.0})
    norm_eff_depth = _normalize(effective_depth, st_eff["mean"], st_eff["std"])
    st_px = s2.get("position_x", {"mean": 0.0, "std": 1.0})
    st_py = s2.get("position_y", {"mean": 0.0, "std": 1.0})
    norm_pos_x = _normalize(pos_x_t, st_px["mean"], st_px["std"])
    norm_pos_y = _normalize(pos_y_t, st_py["mean"], st_py["std"])
    # is_connected is binary (0/1) → no normalisation

    # ------------------------------------------------------------------
    # 8.  Stack into [T, N, F] feature tensors
    # ------------------------------------------------------------------
    x_1d = torch.stack(
        [
            norm_depth_1d,                                    # 0: depth
            norm_iflow,                                       # 1: inlet_flow
            norm_lag1_1d,                                     # 2: lag1 (t-2)
            norm_lag2_1d,                                     # 3: lag2 (t-3)
            norm_lag3_1d,                                     # 4: lag3 (t-4)
            norm_cap.unsqueeze(0).expand(T, -1),              # 5: capacity
            norm_ba.unsqueeze(0).expand(T, -1),               # 6: base_area
            norm_pipe_diam,                                   # 7: pipe_diameter (Phase B)
            norm_pipe_len,                                    # 8: pipe_length
            norm_pipe_rough,                                  # 9: pipe_roughness
            norm_pipe_slope,                                  # 10: pipe_slope
        ],
        dim=-1,
    )  # [T, N_1d, 11]

    x_2d = torch.stack(
        [
            norm_depth_2d,                                    # 0: depth
            norm_rain,                                        # 1: rainfall
            norm_lag1_2d,                                     # 2: lag1 (t-2)
            norm_lag2_2d,                                     # 3: lag2 (t-3)
            norm_lag3_2d,                                     # 4: lag3 (t-4)
            norm_eff_depth,                                    # 5: effective_depth (P0: water_volume/area)
            norm_rain_roll,                                   # 6: rain_rolling_mean
            norm_rain_delta,                                  # 7: rain_delta
            norm_rain_lag2,                                   # 8: rain_lag2 (EDA: best lag=2)
            norm_elev.unsqueeze(0).expand(T, -1),             # 9: elevation
            norm_min_elev.unsqueeze(0).expand(T, -1),          # 10: min_elevation
            norm_slope.unsqueeze(0).expand(T, -1),            # 11: slope (EDA: terrain)
            norm_area.unsqueeze(0).expand(T, -1),             # 12: area
            norm_rough.unsqueeze(0).expand(T, -1),            # 13: roughness
            norm_aspect.unsqueeze(0).expand(T, -1),           # 14: aspect
            norm_curv.unsqueeze(0).expand(T, -1),             # 15: curvature
            norm_fa.unsqueeze(0).expand(T, -1),                # 16: flow_acc
            norm_er.unsqueeze(0).expand(T, -1),               # 17: elev_rel
            norm_dd.unsqueeze(0).expand(T, -1),               # 18: dist_to_drain
            is_connected.unsqueeze(0).expand(T, -1),          # 19: is_connected
            norm_pos_x.unsqueeze(0).expand(T, -1),            # 20: position_x (Phase A)
            norm_pos_y.unsqueeze(0).expand(T, -1),             # 21: position_y
        ],
        dim=-1,
    )  # [T, N_2d, 22]

    # ------------------------------------------------------------------
    # 9.  Assemble HeteroData
    # ------------------------------------------------------------------
    data = HeteroData()

    # ── 1D nodes ──────────────────────────────────────────────────
    data["1d"].x = x_1d                # [T, N_1d, 11]
    data["1d"].y = wse_1d              # [T, N_1d]
    data["1d"].depth = depth_1d        # [T, N_1d]
    data["1d"].elev = invert_elev      # [N_1d]
    data["1d"].num_nodes = n_1d

    # ── 2D nodes ──────────────────────────────────────────────────
    data["2d"].x = x_2d                # [T, N_2d, 22]
    data["2d"].y = wse_2d              # [T, N_2d]
    data["2d"].depth = depth_2d        # [T, N_2d]  (min_elev ref)
    data["2d"].elev = min_elev         # [N_2d]  KEY: min_elevation!
    data["2d"].area = area_t           # [N_2d]  Raw area (used for effective_depth in graph build)
    data["2d"].num_nodes = n_2d

    # ── Edges ─────────────────────────────────────────────────────
    data["1d", "pipe", "1d"].edge_index = pipe_ei
    data["2d", "spread", "2d"].edge_index = spread_ei
    data["1d", "link", "2d"].edge_index = link_1d_to_2d
    data["2d", "link", "1d"].edge_index = link_2d_to_1d

    # ── Metadata ──────────────────────────────────────────────────
    data.model_id = raw_data.get("model_id", "unknown")
    data.event_id = raw_data.get("event_id", "unknown")
    data.num_timesteps = T
    data.feature_names_1d = _1D_FEATURE_NAMES
    data.feature_names_2d = _2D_FEATURE_NAMES

    return data


# =====================================================================
#  Utilities
# =====================================================================

def get_feature_dims(data: HeteroData) -> Dict[str, int]:
    """Extract feature dimensions for model construction."""
    return {
        "in_channels_1d": data["1d"].x.size(-1),
        "in_channels_2d": data["2d"].x.size(-1),
        "n_1d": data["1d"].num_nodes,
        "n_2d": data["2d"].num_nodes,
        "num_timesteps": data.num_timesteps,
    }


def summarise_graph(data: HeteroData) -> str:
    """Human-readable summary for logging and sanity-checking."""
    lines = [
        f"=== HeteroGraph (Model {data.model_id}, Event {data.event_id}) ===",
        f"  Timesteps : {data.num_timesteps}",
        f"  1D Nodes  : {data['1d'].num_nodes:>6d}  x={list(data['1d'].x.shape)}",
        f"  2D Nodes  : {data['2d'].num_nodes:>6d}  x={list(data['2d'].x.shape)}",
    ]
    for et in data.edge_types:
        ei = data[et].edge_index
        label = f"({et[0]}, {et[1]}, {et[2]})"
        lines.append(f"  Edge {label:<30s}  [{ei.size(0)}, {ei.size(1)}]")
    lines.append("=" * 60)
    return "\n".join(lines)

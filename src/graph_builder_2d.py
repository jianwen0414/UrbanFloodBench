"""
graph_builder_2d — Construct a PyG Data object for the 2D surface mesh.

Consumes the sample dict from ``FloodDataset.__getitem__()`` and
produces a homogeneous ``torch_geometric.data.Data`` graph with
bidirectional edges suitable for GNN message passing.

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
- ``data.x``             : Combined static + dynamic features [N, 16]
- ``data.edge_index``    : Mesh adjacency (bidirectional) [2, E]
- ``data.y``             : Target depth at timestep t  [N, 1]
- ``data.min_elevation`` : For converting depth → WSE  [N]
- ``data.num_nodes``     : Number of 2D mesh nodes

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

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from scipy.spatial import KDTree

from src.utils_2d import (
    wse_to_depth,
    depth_to_wse,
    compute_normalization_stats,
    normalize_feature,
    get_min_elevation_filled,
)


# ───────────────────────────────────────────────────────────────────────
#  Timestep extraction helper
# ───────────────────────────────────────────────────────────────────────

def get_values_at_timestep(
    dynamic_df: pd.DataFrame,
    timestep: int,
    column: str,
    num_nodes: int,
) -> np.ndarray:
    """Extract per-node values for a single timestep from long-format data.

    Parameters
    ----------
    dynamic_df : pd.DataFrame
        Long-format dynamic DataFrame with at least ``timestep``,
        ``node_idx``, and *column*.
    timestep : int
        Which timestep to extract.
    column : str
        Column name to read (e.g. ``"rainfall"``, ``"water_level"``).
    num_nodes : int
        Expected number of nodes (used for validation).

    Returns
    -------
    np.ndarray, shape [num_nodes]
        Values sorted by ascending ``node_idx``.
    """
    mask = dynamic_df["timestep"] == timestep
    df_t = dynamic_df.loc[mask].sort_values("node_idx")

    assert len(df_t) == num_nodes, (
        f"Expected {num_nodes} nodes at timestep {timestep}, got {len(df_t)}"
    )

    return df_t[column].values


# ───────────────────────────────────────────────────────────────────────
#  Neighbour-relative elevation
# ───────────────────────────────────────────────────────────────────────

def compute_elev_rel_neighbors(
    elevation: np.ndarray,
    edge_index: torch.Tensor,
) -> np.ndarray:
    """Elevation of each node relative to the mean of its neighbours.

    Positive values indicate a hilltop (higher than surroundings);
    negative values indicate a depression where water tends to pool.

    Parameters
    ----------
    elevation : np.ndarray, shape [N]
        Raw elevation values per node.
    edge_index : torch.Tensor, shape [2, E]
        Bidirectional edge index (must already contain both directions).

    Returns
    -------
    np.ndarray, shape [N]
        ``elevation[i] - mean(elevation[neighbours of i])``.
    """
    num_nodes = len(elevation)

    src = edge_index[0].numpy()  # source nodes
    dst = edge_index[1].numpy()  # destination nodes

    # Accumulate neighbour elevations and counts per source node
    neighbor_sum = np.zeros(num_nodes)
    neighbor_count = np.zeros(num_nodes)

    np.add.at(neighbor_sum, src, elevation[dst])
    np.add.at(neighbor_count, src, 1)

    # Mean neighbour elevation (clamp count to avoid division by zero)
    neighbor_count = np.maximum(neighbor_count, 1)
    neighbor_mean = neighbor_sum / neighbor_count

    return elevation - neighbor_mean


# ───────────────────────────────────────────────────────────────────────
#  Soft coupling: drain distance features
# ───────────────────────────────────────────────────────────────────────

def compute_drain_features(
    sample: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute distance-to-drain features for each 2D node.

    Uses a KD-tree over 1D node positions for efficient nearest-neighbour
    lookup, and the ``1d2d_conn`` table for the binary connectivity flag.

    Parameters
    ----------
    sample : dict
        A single sample from ``FloodDataset.__getitem__()``.
        Expected keys: ``static_2d_nodes``, ``static_1d_nodes``,
        ``1d2d_conn``.

    Returns
    -------
    dist_to_nearest_1d : np.ndarray, shape [N_2d]
        Euclidean distance from each 2D node to the nearest 1D node.
    is_directly_connected : np.ndarray, shape [N_2d]
        ``1.0`` if the 2D node appears in ``1d2d_conn``, else ``0.0``.
    """
    # Get 2D node coordinates
    static_2d = sample["static_2d_nodes"]
    coords_2d = static_2d[["position_x", "position_y"]].values  # [N_2d, 2]

    # Get 1D node coordinates
    static_1d = sample["static_1d_nodes"]
    coords_1d = static_1d[["position_x", "position_y"]].values  # [N_1d, 2]

    # Build KDTree from 1D nodes and query with 2D nodes
    tree = KDTree(coords_1d)
    dist_to_nearest_1d, _ = tree.query(coords_2d)  # (distances, indices)

    # Get directly-connected 2D nodes from 1d2d_conn
    conn = sample["1d2d_conn"]
    connected_2d_set = set(conn["node_2d"].values)

    # Create binary array
    num_2d_nodes = len(static_2d)
    is_directly_connected = np.array(
        [1.0 if i in connected_2d_set else 0.0 for i in range(num_2d_nodes)]
    )

    return dist_to_nearest_1d, is_directly_connected


# ───────────────────────────────────────────────────────────────────────
#  Main graph builder
# ───────────────────────────────────────────────────────────────────────

def build_2d_graph(
    sample: Dict[str, Any],
    norm_stats: Dict[str, float],
    t_index: int,
    num_history: int = 3,
    predicted_depths: list | None = None,
) -> Data:
    """Build a PyTorch Geometric Data object for a single timestep.

    Constructs a bidirectional graph from the directed ``edge_index_2d``
    DataFrame, attaches normalised static features, dynamic features
    (rainfall + depth lags), and the target depth at ``t_index``.

    Supports two modes:

    * **Training** (``predicted_depths=None``): depth lags are read from
      the ground-truth ``dynamic_2d_nodes`` DataFrame.
    * **Inference** (``predicted_depths`` provided): depth lags come from
      the caller (e.g. model predictions from previous rollout steps).

    Parameters
    ----------
    sample : dict
        A single sample from ``FloodDataset.__getitem__()``.
        Expected keys: ``static_2d_nodes``, ``dynamic_2d_nodes``,
        ``edge_index_2d``, ``static_1d_nodes``, ``1d2d_conn``.
    norm_stats : dict
        Normalization statistics from
        :func:`src.utils_2d.compute_normalization_stats`.
    t_index : int
        Current timestep to build features for.
    num_history : int, optional
        Number of depth-lag timesteps to include (default ``3``).
    predicted_depths : list of np.ndarray or None, optional
        If provided, used **instead of** ground-truth for lag features.
        Format: ``[depth_t-1, depth_t-2, depth_t-3]`` where each entry
        is a numpy array of shape ``[N]`` or ``None`` (→ zeros).

    Returns
    -------
    torch_geometric.data.Data
        - ``x``             : [N, 16] combined static + dynamic features
        - ``edge_index``    : [2, E] bidirectional
        - ``y``             : [N, 1] target depth at *t_index*
        - ``min_elevation`` : [N] for converting depth → WSE
        - ``num_nodes``     : int
        - ``t_index``       : int (metadata)
        - ``model_id``      : str (metadata)
        - ``event_id``      : str (metadata)
    """
    # ── Step 1: Get number of nodes ──────────────────────────────────
    static_2d = sample["static_2d_nodes"]
    num_nodes = len(static_2d)

    # ── Step 2: Build edge_index and make bidirectional ──────────────
    edge_df = sample["edge_index_2d"]
    from_nodes = torch.tensor(edge_df["from_node"].values, dtype=torch.long)
    to_nodes = torch.tensor(edge_df["to_node"].values, dtype=torch.long)

    # Stack to [2, E]
    edge_index = torch.stack([from_nodes, to_nodes], dim=0)

    # Make bidirectional by adding reverse edges
    edge_index_reverse = edge_index.flip(0)
    edge_index_bi = torch.cat([edge_index, edge_index_reverse], dim=1)

    # Remove duplicates (in case some edges already have both directions)
    edge_index_bi = torch.unique(edge_index_bi, dim=1)

    # ── Step 3: Validation checks ────────────────────────────────────
    assert edge_index_bi.min() >= 0, "Negative node index found!"
    assert edge_index_bi.max() < num_nodes, (
        f"Node index {edge_index_bi.max()} >= num_nodes {num_nodes}"
    )

    # ── Step 4: Static node features (normalized) ──────────────────
    static_features = []

    # 1. Position (normalized)
    pos_x = static_2d["position_x"].values
    pos_x_norm = normalize_feature(
        pos_x, norm_stats["position_x_mean"], norm_stats["position_x_std"]
    )
    static_features.append(pos_x_norm)

    pos_y = static_2d["position_y"].values
    pos_y_norm = normalize_feature(
        pos_y, norm_stats["position_y_mean"], norm_stats["position_y_std"]
    )
    static_features.append(pos_y_norm)

    # 2. Area (normalized)
    area = static_2d["area"].values
    area_norm = normalize_feature(
        area, norm_stats["area_mean"], norm_stats["area_std"]
    )
    static_features.append(area_norm)

    # 3. Roughness (normalized)
    roughness = static_2d["roughness"].values
    roughness_norm = normalize_feature(
        roughness, norm_stats["roughness_mean"], norm_stats["roughness_std"]
    )
    static_features.append(roughness_norm)

    # 4. Elevation (normalized) — centroid elevation
    elevation = static_2d["elevation"].values
    elevation_norm = normalize_feature(
        elevation, norm_stats["elevation_mean"], norm_stats["elevation_std"]
    )
    static_features.append(elevation_norm)

    # 5. Min elevation (normalized) — NaN filled with centroid elevation
    min_elevation = get_min_elevation_filled(static_2d)
    min_elevation_norm = normalize_feature(
        min_elevation,
        norm_stats["min_elevation_mean"],
        norm_stats["min_elevation_std"],
    )
    static_features.append(min_elevation_norm)

    # 6. Aspect (handle -1 → 0 sentinel, then normalize)
    aspect = static_2d["aspect"].values.copy()
    aspect[aspect == -1] = 0  # Replace undefined flat areas with 0
    aspect_norm = normalize_feature(
        aspect, norm_stats["aspect_mean"], norm_stats["aspect_std"]
    )
    static_features.append(aspect_norm)

    # 7. Curvature (normalized)
    curvature = static_2d["curvature"].values
    curvature_norm = normalize_feature(
        curvature, norm_stats["curvature_mean"], norm_stats["curvature_std"]
    )
    static_features.append(curvature_norm)

    # 8. Flow accumulation (normalized)
    flow_acc = static_2d["flow_accumulation"].values
    flow_acc_norm = normalize_feature(
        flow_acc,
        norm_stats["flow_accumulation_mean"],
        norm_stats["flow_accumulation_std"],
    )
    static_features.append(flow_acc_norm)

    # ── Step 4b: Neighbour-relative elevation ───────────────────────
    # Depends on edge_index_bi, so must come after step 2.
    elevation_raw = static_2d["elevation"].values
    elev_rel_neighbors = compute_elev_rel_neighbors(elevation_raw, edge_index_bi)

    # 9. Neighbour-relative elevation (self-normalised per sample)
    elev_rel_mean = elev_rel_neighbors.mean()
    elev_rel_std = elev_rel_neighbors.std()
    elev_rel_norm = (elev_rel_neighbors - elev_rel_mean) / (elev_rel_std + 1e-8)
    static_features.append(elev_rel_norm)

    # ── Step 4c: Soft coupling features (1D-2D interaction) ────────
    dist_to_drain, is_connected = compute_drain_features(sample)

    # 10. Distance to nearest 1D node (normalized)
    dist_norm = normalize_feature(
        dist_to_drain,
        norm_stats["dist_to_drain_mean"],
        norm_stats["dist_to_drain_std"],
    )
    static_features.append(dist_norm)

    # 11. Binary: is directly connected to a 1D node (no normalization)
    static_features.append(is_connected)

    # Feature order (12 total):
    # 0: pos_x, 1: pos_y, 2: area, 3: roughness, 4: elevation,
    # 5: min_elevation, 6: aspect, 7: curvature, 8: flow_acc,
    # 9: elev_rel_neighbors, 10: dist_to_drain, 11: is_connected

    # Stack into tensor [N, 12]
    x_static = np.stack(static_features, axis=1)
    x_static = torch.tensor(x_static, dtype=torch.float32)

    # ── Step 5: Dynamic features ─────────────────────────────────────
    dynamic_2d = sample["dynamic_2d_nodes"]

    # Get min_elevation for depth conversion (NaN-safe)
    min_elevation = get_min_elevation_filled(static_2d)

    # 12. Current rainfall at t_index (normalized)
    rainfall_t = get_values_at_timestep(
        dynamic_2d, t_index, "rainfall", num_nodes
    )
    if "rainfall_mean" in norm_stats:
        rainfall_mean = norm_stats["rainfall_mean"]
        rainfall_std = norm_stats["rainfall_std"]
    else:
        rainfall_all = dynamic_2d["rainfall"].values
        rainfall_mean = float(rainfall_all.mean())
        rainfall_std = float(rainfall_all.std())
    rainfall_norm = normalize_feature(rainfall_t, rainfall_mean, rainfall_std + 1e-8)

    # 13-15. Depth lags (t-1, t-2, t-3)
    depth_lags: list[np.ndarray] = []

    if predicted_depths is not None:
        # ── INFERENCE MODE: use caller-supplied predicted depths ──
        for lag_idx in range(num_history):
            if lag_idx < len(predicted_depths) and predicted_depths[lag_idx] is not None:
                depth_lag = predicted_depths[lag_idx]
                if isinstance(depth_lag, torch.Tensor):
                    depth_lag = depth_lag.numpy()
                if depth_lag.ndim > 1:
                    depth_lag = depth_lag.squeeze()
            else:
                depth_lag = np.zeros(num_nodes)
            depth_lags.append(depth_lag)
    else:
        # ── TRAINING MODE: read ground-truth depth lags ──────────
        for lag in range(1, num_history + 1):
            t_lag = t_index - lag
            if t_lag >= 0:
                wl_lag = get_values_at_timestep(
                    dynamic_2d, t_lag, "water_level", num_nodes
                )
                depth_lag = wse_to_depth(wl_lag, min_elevation)
            else:
                depth_lag = np.zeros(num_nodes)
            depth_lags.append(depth_lag)

    # Stack dynamic features: [rainfall, depth_t-1, depth_t-2, depth_t-3]
    dynamic_features = [rainfall_norm] + depth_lags
    x_dynamic = np.stack(dynamic_features, axis=1)  # [N, 4]
    x_dynamic = torch.tensor(x_dynamic, dtype=torch.float32)

    # ── Step 6: Combine static + dynamic ─────────────────────────────
    x = torch.cat([x_static, x_dynamic], dim=1)  # [N, 16]

    # ── Step 7: Target — depth at t_index ────────────────────────────
    wl_t = get_values_at_timestep(
        dynamic_2d, t_index, "water_level", num_nodes
    )
    depth_t = wse_to_depth(wl_t, min_elevation)
    y = torch.tensor(depth_t, dtype=torch.float32).unsqueeze(1)  # [N, 1]

    min_elevation_tensor = torch.tensor(min_elevation, dtype=torch.float32)

    # ── Step 8: Return Data object ───────────────────────────────────
    return Data(
        x=x,                                # [N, 16] combined features
        edge_index=edge_index_bi,            # [2, E]  bidirectional
        y=y,                                 # [N, 1]  target depth
        min_elevation=min_elevation_tensor,  # [N]     for WSE recovery
        num_nodes=num_nodes,
        # Metadata
        t_index=t_index,
        model_id=sample["model_id"],
        event_id=sample["event_id"],
    )


# ───────────────────────────────────────────────────────────────────────
#  Depth history manager for autoregressive rollout
# ───────────────────────────────────────────────────────────────────────

class DepthHistory:
    """Sliding window of past depth values for lag features.

    During autoregressive inference the model produces a depth prediction
    at each timestep.  This class keeps the most recent ``num_history``
    predictions so they can be fed back as lag features via the
    ``predicted_depths`` argument of :func:`build_2d_graph`.

    Parameters
    ----------
    num_history : int
        Number of lag timesteps to retain (default ``3``).
    """

    def __init__(self, num_history: int = 3) -> None:
        self.num_history = num_history
        self.history: list[np.ndarray] = []  # most-recent first

    # ── mutators ─────────────────────────────────────────────────────

    def update(self, depth: np.ndarray | torch.Tensor) -> None:
        """Push a new depth array onto the front of the history.

        Parameters
        ----------
        depth : np.ndarray or torch.Tensor, shape [N] or [N, 1]
            Predicted depth for the latest timestep.
        """
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        if depth.ndim > 1:
            depth = depth.squeeze()

        self.history.insert(0, depth.copy())

        # Trim to window size
        if len(self.history) > self.num_history:
            self.history = self.history[: self.num_history]

    def clear(self) -> None:
        """Drop all stored history."""
        self.history = []

    # ── accessors ────────────────────────────────────────────────────

    def get_lags(self) -> list[np.ndarray | None]:
        """Return lag list compatible with ``build_2d_graph``.

        Returns
        -------
        list
            ``[depth_t-1, depth_t-2, ...]`` with ``None`` for missing.
        """
        lags: list[np.ndarray | None] = []
        for i in range(self.num_history):
            if i < len(self.history):
                lags.append(self.history[i])
            else:
                lags.append(None)
        return lags

    # ── convenience ──────────────────────────────────────────────────

    def initialize_from_ground_truth(
        self, sample: Dict[str, Any], t_start: int
    ) -> None:
        """Seed the history from ground-truth data up to *t_start*.

        Fills the buffer with depths at ``t_start-1``, ``t_start-2``, …
        so that the first inference call already has valid lags.

        Parameters
        ----------
        sample : dict
            FloodDataset sample dictionary.
        t_start : int
            The timestep at which inference will begin.
        """
        static_2d = sample["static_2d_nodes"]
        dynamic_2d = sample["dynamic_2d_nodes"]
        min_elev = get_min_elevation_filled(static_2d)
        num_nodes = len(static_2d)

        self.history = []
        for lag in range(1, self.num_history + 1):
            t_lag = t_start - lag
            if t_lag >= 0:
                wl = get_values_at_timestep(
                    dynamic_2d, t_lag, "water_level", num_nodes
                )
                self.history.append(wse_to_depth(wl, min_elev))
            else:
                self.history.append(np.zeros(num_nodes))


# ───────────────────────────────────────────────────────────────────────
#  Quick smoke test
# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.config import RAW_DATA_PATH
    from src.dataset import FloodDataset
    from src.utils_2d import compute_normalization_stats

    ds = FloodDataset(RAW_DATA_PATH, mode="train")
    sample = ds[0]
    norm_stats = compute_normalization_stats(ds, sample["model_id"])

    # ── 1. Training mode (ground-truth lags) ──────────────────────
    print("=" * 60)
    print("Testing build_2d_graph with TRAINING mode (ground truth lags)")
    print("=" * 60)

    t_index = 10
    data_train = build_2d_graph(
        sample, norm_stats, t_index=t_index, predicted_depths=None
    )

    print(f"  x shape: {data_train.x.shape}")
    print(f"  Depth lag features (idx 13-15):")
    print(f"    depth_t-1: mean={data_train.x[:, 13].mean():.4f}")
    print(f"    depth_t-2: mean={data_train.x[:, 14].mean():.4f}")
    print(f"    depth_t-3: mean={data_train.x[:, 15].mean():.4f}")

    # ── 2. Inference mode (predicted lags) ────────────────────────
    print("\n" + "=" * 60)
    print("Testing build_2d_graph with INFERENCE mode (predicted lags)")
    print("=" * 60)

    num_nodes = data_train.num_nodes
    fake_pred_depths = [
        data_train.x[:, 13].numpy() + np.random.randn(num_nodes) * 0.01,
        data_train.x[:, 14].numpy() + np.random.randn(num_nodes) * 0.01,
        data_train.x[:, 15].numpy() + np.random.randn(num_nodes) * 0.01,
    ]

    data_infer = build_2d_graph(
        sample, norm_stats, t_index=t_index, predicted_depths=fake_pred_depths
    )

    print(f"  x shape: {data_infer.x.shape}")
    print(f"  Depth lag features (idx 13-15):")
    print(f"    depth_t-1: mean={data_infer.x[:, 13].mean():.4f}")
    print(f"    depth_t-2: mean={data_infer.x[:, 14].mean():.4f}")
    print(f"    depth_t-3: mean={data_infer.x[:, 15].mean():.4f}")

    diff = (data_train.x[:, 13:16] - data_infer.x[:, 13:16]).abs().mean()
    print(f"\n  Mean difference in depth lags: {diff:.6f} (should be small)")

    # ── 3. DepthHistory helper ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Testing DepthHistory helper class")
    print("=" * 60)

    history = DepthHistory(num_history=3)

    # Seed from ground truth at t=10
    history.initialize_from_ground_truth(sample, t_start=10)
    lags = history.get_lags()
    print(f"  Initialized from ground truth at t=10")
    print(
        f"  Lag shapes: "
        f"{[l.shape if l is not None else None for l in lags]}"
    )

    # Simulate 5 rollout steps
    for step in range(5):
        fake_depth = np.random.rand(num_nodes) * 0.5
        history.update(fake_depth)
        lags = history.get_lags()
        valid = sum(1 for l in lags if l is not None)
        print(f"  After step {step + 1}: {valid} valid lag entries")

    # Build a graph using the history
    data_hist = build_2d_graph(
        sample, norm_stats, t_index=15, predicted_depths=history.get_lags()
    )
    print(f"\n  Built graph via DepthHistory: x shape = {data_hist.x.shape}")

    print("\n✓ Inference mode support added successfully!")

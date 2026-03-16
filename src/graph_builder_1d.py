"""
Graph builder for 1D flood prediction.

Builds PyTorch Geometric `Data` objects from Pandas DataFrame inputs.

Input DataFrames (from FloodDataset sample)
-------------------------------------------
- `static_1d_nodes`  : node_idx, position_x, position_y, depth,
                       invert_elevation, surface_elevation, base_area
- `dynamic_1d_nodes` : timestep, node_idx, water_level, inlet_flow  (flat)
- `edge_index_1d`    : edge_idx, from_node, to_node
- `static_1d_edges`  : edge_idx, relative_position_x, relative_position_y,
                       length, diameter, shape, roughness, slope
- `dynamic_1d_edges` : timestep, edge_idx, flow, velocity  (flat)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List

import torch
import numpy as np
from torch_geometric.data import Data

from src.utils_1d import (
    pivot_dynamic_1d,
    pivot_dynamic_1d_all_features,
    get_edge_index_1d,
    get_static_node_features,
    compute_normalization_stats_1d,
)


def build_1d_graph(
    sample: Dict[str, Any],
    norm_stats: Dict[str, Any],
    t_index: int,
    num_history: int = 5,
    water_level_override: Optional[np.ndarray] = None,
) -> Data:
    """
    Build a PyTorch Geometric graph for 1D nodes at a specific timestep.

    Args:
        sample: Data sample from FloodDataset (contains Pandas DataFrames)
        norm_stats: Normalization statistics from `compute_normalization_stats_1d`
        t_index: Current timestep index (0-based)
        num_history: Number of historical timesteps to include as features
        water_level_override: If provided, use these water levels instead of
            ground truth for autoregressive prediction. Shape: (num_nodes,)

    Returns:
        `torch_geometric.data.Data` with:
        - `x`              : Node features            [N, F_in]
        - `edge_index`     : Graph connectivity       [2, 2E] (bidirectional)
        - `y`              : Target water level next  [N]
        - `current_wl`     : Current water level      [N]
        - `current_wl_norm`: Normalised current wl    [N]
        - `num_nodes`      : Number of nodes
        - `t_index`        : Current timestep index
    """
    # ── Static features ────────────────────────────────────────────────
    static_feats = get_static_node_features(sample)  # (N, 6) or (N, >6 if positions)
    static_feats_t = torch.tensor(static_feats, dtype=torch.float32)
    num_nodes = static_feats_t.shape[0]

    # Normalise static
    static_mean = norm_stats["static_mean"]
    static_std = norm_stats["static_std"]
    static_norm = (static_feats_t - static_mean) / static_std

    # ── Dynamic water levels & inlet flows ─────────────────────────────
    dynamic_df = sample["dynamic_1d_nodes"]
    wl_array = pivot_dynamic_1d(dynamic_df, "water_level", sample=sample)  # (T, N)

    inlet_flow_array: Optional[np.ndarray] = None
    if "inlet_flow" in dynamic_df.columns:
        inlet_flow_array = pivot_dynamic_1d(dynamic_df, "inlet_flow")  # (T, N)

    num_timesteps = wl_array.shape[0]

    # Current water level
    wl_mean = norm_stats.get("water_level_mean", 0.0)
    wl_std = norm_stats.get("water_level_std", 1.0)

    if water_level_override is not None:
        current_wl = torch.tensor(water_level_override, dtype=torch.float32)
    else:
        current_wl = torch.tensor(wl_array[t_index], dtype=torch.float32)

    current_wl_norm = (current_wl - wl_mean) / wl_std

    # Historical water levels (t_index - num_history ... t_index - 1)
    history_wl: List[torch.Tensor] = []
    for h in range(num_history, 0, -1):
        t_hist = t_index - h
        if t_hist < 0:
            wl_h_np = wl_array[0]
        else:
            wl_h_np = wl_array[t_hist]
        wl_h = torch.tensor(wl_h_np, dtype=torch.float32)
        wl_h_norm = (wl_h - wl_mean) / wl_std
        history_wl.append(wl_h_norm.unsqueeze(1))  # (N, 1)

    if history_wl:
        history_wl_tensor = torch.cat(history_wl, dim=1)  # (N, num_history)
    else:
        history_wl_tensor = torch.zeros(num_nodes, 0)

    # Inlet flow features (current only for now)
    flow_features: List[torch.Tensor] = []
    if inlet_flow_array is not None and "inlet_flow_mean" in norm_stats:
        flow_mean = norm_stats["inlet_flow_mean"]
        flow_std = norm_stats["inlet_flow_std"]

        current_flow = torch.tensor(inlet_flow_array[t_index], dtype=torch.float32)
        current_flow_norm = ((current_flow - flow_mean) / flow_std).unsqueeze(1)  # (N, 1)
        flow_features.append(current_flow_norm)

    # Time encoding (simple normalised scalar)
    time_feat = torch.full((num_nodes, 1), float(t_index) / max(num_timesteps - 1, 1))

    # Water level change from previous timestep
    if t_index > 0:
        prev_wl_np = wl_array[t_index - 1]
        prev_wl = torch.tensor(prev_wl_np, dtype=torch.float32)
        delta_wl = ((current_wl - prev_wl) / wl_std).unsqueeze(1)  # (N, 1)
    else:
        delta_wl = torch.zeros(num_nodes, 1)

    # ── Concatenate node features ─────────────────────────────────────
    feature_list: List[torch.Tensor] = [
        static_norm,                    # (N, F_static)
        current_wl_norm.unsqueeze(1),   # (N, 1)
        history_wl_tensor,              # (N, num_history)
        delta_wl,                       # (N, 1)
        time_feat,                      # (N, 1)
    ]

    if flow_features:
        feature_list.extend(flow_features)  # (N, 1) inlet flow

    x = torch.cat(feature_list, dim=1)  # (N, F_in)

    # ── Target water level at next timestep ───────────────────────────
    if t_index + 1 < num_timesteps:
        y_np = wl_array[t_index + 1]
        y = torch.tensor(y_np, dtype=torch.float32)
    else:
        # Last timestep: target = current
        y = current_wl.clone()

    # ── Edge connectivity ─────────────────────────────────────────────
    edge_index = get_edge_index_1d(sample)  # (2, 2E) bidirectional

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        current_wl=current_wl,
        current_wl_norm=current_wl_norm,
        num_nodes=num_nodes,
        t_index=t_index,
    )

    return data


def build_1d_graph_sequence(
    sample: Dict[str, Any],
    norm_stats: Dict[str, Any],
    t_start: int = 0,
    t_end: Optional[int] = None,
    num_history: int = 5,
) -> list[Data]:
    """
    Build a sequence of graphs for multiple timesteps.

    Args:
        sample: Data sample
        norm_stats: Normalization statistics
        t_start: Starting timestep (inclusive)
        t_end: Ending timestep (exclusive). If None, use all except last.
        num_history: Number of historical timesteps per graph.

    Returns:
        List of `Data` objects (one per timestep).
    """
    dynamic_df = sample["dynamic_1d_nodes"]
    wl_array = pivot_dynamic_1d(dynamic_df, "water_level", sample=sample)
    num_timesteps = wl_array.shape[0]

    if t_end is None:
        # We predict t+1, so last usable t is num_timesteps-2
        t_end = num_timesteps - 1

    graphs: list[Data] = []
    for t in range(t_start, t_end):
        g = build_1d_graph(
            sample=sample,
            norm_stats=norm_stats,
            t_index=t,
            num_history=num_history,
        )
        graphs.append(g)

    return graphs


def get_1d_input_dim(sample: Dict[str, Any], norm_stats: Dict[str, Any], num_history: int = 5) -> int:
    """
    Calculate the input feature dimension for the 1D model by building
    a single example graph.
    """
    # Use a safe middle timestep if possible
    dynamic_df = sample["dynamic_1d_nodes"]
    wl_array = pivot_dynamic_1d(dynamic_df, "water_level", sample=sample)
    num_timesteps = wl_array.shape[0]
    t_index = min(10, num_timesteps // 2)

    data = build_1d_graph(sample, norm_stats, t_index=t_index, num_history=num_history)
    return data.x.shape[1]


def get_num_timesteps_1d(sample: Dict[str, Any]) -> int:
    """Get total number of timesteps for a 1D sample."""
    dynamic_df = sample["dynamic_1d_nodes"]
    return int(dynamic_df["timestep"].nunique())


if __name__ == "__main__":
    from src.config import RAW_DATA_PATH
    from src.dataset import FloodDataset

    print("=" * 60)
    print("1D GRAPH BUILDER TEST")
    print("=" * 60)

    ds = FloodDataset(RAW_DATA_PATH, mode="train")

    for model_id in ["1", "2"]:
        print(f"\nModel_{model_id}:")
        print("-" * 40)

        ds_model = ds.filter_by_model(model_id)
        if len(ds_model) == 0:
            print("  No samples for this model.")
            continue

        sample = ds_model[0]

        norm_stats = compute_normalization_stats_1d(ds, model_id)

        # Build single graph
        t_index = 10
        dynamic_df = sample["dynamic_1d_nodes"]
        num_ts = int(dynamic_df["timestep"].nunique())
        if t_index >= num_ts:
            t_index = max(0, num_ts // 2)

        data = build_1d_graph(sample, norm_stats, t_index=t_index)
        print("  Single graph:")
        print(f"    t_index: {t_index}")
        print(f"    x shape: {data.x.shape}")
        print(f"    edge_index shape: {data.edge_index.shape}")
        print(f"    y shape: {data.y.shape}")
        print(f"    current_wl shape: {data.current_wl.shape}")
        print(f"    num_nodes: {data.num_nodes}")
        print(f"    Input dim: {data.x.shape[1]}")

        # Build sequence
        graphs = build_1d_graph_sequence(sample, norm_stats, t_start=5, t_end=15)
        if graphs:
            print("\n  Sequence (t=5..14 or clipped):")
            print(f"    Length: {len(graphs)}")
            print(f"    First graph x shape: {graphs[0].x.shape}")

        # Feature breakdown (approximate, assumes 6 static + positions present)
        in_dim = data.x.shape[1]
        print("\n  Feature breakdown (expected components):")
        print("    Static features: 6 (plus positions if present)")
        print(f"    History WL: {5}")
        print("    Current WL: 1")
        print("    Delta WL: 1")
        print("    Time encoding: 1")
        print("    Inlet flow: 1 (if available)")
        print(f"    Total (approx): {6+5+1+1+1+1} vs actual: {in_dim}")

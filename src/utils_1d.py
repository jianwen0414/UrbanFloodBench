"""
Utility functions for 1D flood prediction.

1D nodes represent the underground drainage network:
- Manholes, junctions, outlets
- Connected by pipes/channels (edges)

Data is stored as Pandas DataFrames:
- dynamic_1d_nodes: flat table (timestep, node_idx, water_level, inlet_flow)
- static_1d_nodes: (num_nodes, 7) with position, depth, elevation, etc.
- edge_index_1d: (num_edges, 3) with from_node, to_node
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from src.dataset import FloodDataset


def clamp_water_level_to_invert(sample: Dict) -> pd.DataFrame:
    """
    Clamp water levels to be >= invert elevation (per node).

    Water levels below invert elevation are numerical artifacts
    from dry conditions. The invert elevation is the physical
    minimum water level.

    Args:
        sample: Data sample from FloodDataset

    Returns:
        Corrected dynamic_1d_nodes DataFrame with clamped water levels.
    """
    dynamic_df = sample["dynamic_1d_nodes"].copy()
    static_df = sample["static_1d_nodes"]

    # Build node_idx -> invert_elevation mapping
    invert_map = static_df.set_index("node_idx")["invert_elevation"].to_dict()

    def clamp_row(row):
        node_idx = row["node_idx"]
        invert_elev = invert_map.get(node_idx, float("-inf"))
        return max(row["water_level"], invert_elev)

    dynamic_df["water_level"] = dynamic_df.apply(clamp_row, axis=1)

    return dynamic_df


def clamp_water_level_array(wl_array: np.ndarray, sample: Dict) -> np.ndarray:
    """
    Clamp a pivoted water level array (T, N) to invert elevations.

    Args:
        wl_array: Water levels shape (T, N)
        sample: Data sample (for static node info)

    Returns:
        Clamped water level array (T, N).
    """
    static_df = sample["static_1d_nodes"].sort_values("node_idx").reset_index(
        drop=True
    )
    invert_elevations = static_df["invert_elevation"].values.astype(np.float32)

    # Broadcast clamp: wl_array (T, N) vs invert_elevations (N,)
    clamped = np.maximum(wl_array, invert_elevations[np.newaxis, :])

    return clamped.astype(np.float32)


def pivot_dynamic_1d(
    dynamic_df: pd.DataFrame,
    feature_col: str = "water_level",
    sample: Optional[Dict] = None,
) -> np.ndarray:
    """
    Convert flat dynamic_1d_nodes DataFrame to 2D array (T, N) for one feature.

    Args:
        dynamic_df: DataFrame with columns [timestep, node_idx, water_level, inlet_flow]
        feature_col: Which column to extract
        sample: If provided and feature_col='water_level', clamp to invert elevation

    Returns:
        Array of shape (num_timesteps, num_nodes) with the requested feature.
    """
    pivot = dynamic_df.pivot_table(
        index="timestep",
        columns="node_idx",
        values=feature_col,
        aggfunc="first",
    )
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    result = pivot.values.astype(np.float32)

    # Clamp water level to invert elevation when sample is provided
    if feature_col == "water_level" and sample is not None:
        result = clamp_water_level_array(result, sample)

    return result


def pivot_dynamic_1d_all_features(dynamic_df: pd.DataFrame) -> np.ndarray:
    """
    Convert flat dynamic_1d_nodes DataFrame to 3D array with ALL features.

    Args:
        dynamic_df: DataFrame with columns [timestep, node_idx, water_level, inlet_flow]

    Returns:
        Array of shape (num_timesteps, num_nodes, num_features)
        Features: [water_level, inlet_flow]
    """
    feature_cols = [c for c in dynamic_df.columns if c not in ["timestep", "node_idx"]]

    arrays: List[np.ndarray] = []
    for col in feature_cols:
        arr = pivot_dynamic_1d(dynamic_df, col)  # (T, N)
        arrays.append(arr)

    return np.stack(arrays, axis=-1)  # (T, N, F)


def pivot_dynamic_1d_edges(dynamic_edges_df: pd.DataFrame) -> np.ndarray:
    """
    Convert flat dynamic_1d_edges DataFrame to 3D array.

    Args:
        dynamic_edges_df: DataFrame with columns [timestep, edge_idx, flow, velocity]

    Returns:
        Array of shape (num_timesteps, num_edges, num_features)
        Features: [flow, velocity]
    """
    feature_cols = [c for c in dynamic_edges_df.columns if c not in ["timestep", "edge_idx"]]

    arrays: List[np.ndarray] = []
    for col in feature_cols:
        pivot = dynamic_edges_df.pivot_table(
            index="timestep",
            columns="edge_idx",
            values=col,
            aggfunc="first",
        ).sort_index(axis=0).sort_index(axis=1)
        arrays.append(pivot.values.astype(np.float32))  # (T, E)

    return np.stack(arrays, axis=-1)  # (T, E, F)


def get_edge_index_1d(sample: Dict) -> torch.Tensor:
    """
    Extract edge connectivity from edge_index_1d DataFrame.

    Args:
        sample: Data sample from FloodDataset

    Returns:
        edge_index: (2, num_edges) tensor for PyTorch Geometric (bidirectional).
    """
    edge_df: pd.DataFrame = sample["edge_index_1d"]

    from_nodes = edge_df["from_node"].values.astype(int)
    to_nodes = edge_df["to_node"].values.astype(int)

    # Bidirectional edges
    sources = np.concatenate([from_nodes, to_nodes])
    targets = np.concatenate([to_nodes, from_nodes])

    edge_index = torch.tensor(np.stack([sources, targets], axis=0), dtype=torch.long)
    return edge_index


def get_static_node_features(sample: Dict) -> np.ndarray:
    """
    Extract static node features as numpy array.

    Features: depth, invert_elevation, surface_elevation, base_area
    (Optionally include positions if present.)

    Args:
        sample: Data sample

    Returns:
        Array of shape (num_nodes, num_features)
    """
    static_df: pd.DataFrame = sample["static_1d_nodes"]

    # Sort by node_idx to ensure consistent ordering
    static_df = static_df.sort_values("node_idx").reset_index(drop=True)

    # Base feature set (exclude node_idx)
    base_feature_cols = ["depth", "invert_elevation", "surface_elevation", "base_area"]

    feature_cols: List[str] = []
    if "position_x" in static_df.columns and "position_y" in static_df.columns:
        feature_cols.extend(["position_x", "position_y"])
    feature_cols.extend(base_feature_cols)

    return static_df[feature_cols].values.astype(np.float32)


def get_static_edge_features(sample: Dict) -> Optional[np.ndarray]:
    """
    Extract static edge features as numpy array.

    Features: length, diameter, shape, roughness, slope

    Args:
        sample: Data sample

    Returns:
        Array of shape (num_edges, num_features) or None
    """
    static_edges_df: Optional[pd.DataFrame] = sample.get("static_1d_edges")
    if static_edges_df is None:
        return None

    static_edges_df = static_edges_df.sort_values("edge_idx").reset_index(drop=True)

    feature_cols = [
        c
        for c in static_edges_df.columns
        if c not in ["edge_idx", "relative_position_x", "relative_position_y"]
    ]

    return static_edges_df[feature_cols].values.astype(np.float32)


def get_1d_node_info(sample: Dict) -> Dict:
    """
    Extract 1D node information from a sample.
    """
    static_1d: Optional[pd.DataFrame] = sample.get("static_1d_nodes")
    dynamic_1d: Optional[pd.DataFrame] = sample.get("dynamic_1d_nodes")
    edges_1d: Optional[pd.DataFrame] = sample.get("edge_index_1d")

    info = {
        "has_1d_data": static_1d is not None and len(static_1d) > 0,
        "num_1d_nodes": 0,
        "num_static_features": 0,
        "num_dynamic_features": 0,
        "num_timesteps": 0,
        "num_edges": 0,
        "has_edges": edges_1d is not None and len(edges_1d) > 0 if edges_1d is not None else False,
    }

    if static_1d is not None and len(static_1d) > 0:
        info["num_1d_nodes"] = len(static_1d)
        # Exclude node_idx from feature count
        info["num_static_features"] = len(static_1d.columns) - 1  # -1 for node_idx

    if dynamic_1d is not None and len(dynamic_1d) > 0:
        info["num_timesteps"] = int(dynamic_1d["timestep"].nunique())
        # Features: everything except timestep, node_idx
        info["num_dynamic_features"] = len(
            [c for c in dynamic_1d.columns if c not in ["timestep", "node_idx"]]
        )

    if edges_1d is not None and len(edges_1d) > 0:
        info["num_edges"] = len(edges_1d)

    return info


def compute_normalization_stats_1d(
    dataset: FloodDataset,
    model_id: str,
) -> Dict:
    """
    Compute normalization statistics for 1D node features.

    Stats include:
    - static_mean, static_std
    - water_level_mean/std (if available)
    - inlet_flow_mean/std (if available)
    """
    ds_model = dataset.filter_by_model(model_id)

    static_list: List[torch.Tensor] = []
    water_levels_list: List[np.ndarray] = []
    inlet_flows_list: List[np.ndarray] = []

    for sample in ds_model:
        static_1d = sample.get("static_1d_nodes")
        dynamic_1d = sample.get("dynamic_1d_nodes")

        if static_1d is None or len(static_1d) == 0:
            continue

        # Static features
        static_feats = get_static_node_features(sample)
        static_list.append(torch.tensor(static_feats, dtype=torch.float32))

        # Dynamic features
        if dynamic_1d is not None and len(dynamic_1d) > 0:
            if "water_level" in dynamic_1d.columns:
                water_levels_list.append(dynamic_1d["water_level"].values.astype(float))
            if "inlet_flow" in dynamic_1d.columns:
                inlet_flows_list.append(dynamic_1d["inlet_flow"].values.astype(float))

    if not static_list:
        raise ValueError(f"No 1D data found for Model_{model_id}")

    all_static = torch.cat(static_list, dim=0)

    # Base stats for static features
    stats: Dict[str, object] = {
        "static_mean": all_static.mean(dim=0),
        "static_std": all_static.std(dim=0).clamp(min=1e-6),
    }

    if water_levels_list:
        all_wl = np.concatenate(water_levels_list)
        stats["water_level_mean"] = float(np.mean(all_wl))
        stats["water_level_std"] = float(np.std(all_wl).clip(min=1e-6))

    if inlet_flows_list:
        all_flow = np.concatenate(inlet_flows_list)
        stats["inlet_flow_mean"] = float(np.mean(all_flow))
        stats["inlet_flow_std"] = float(np.std(all_flow).clip(min=1e-6))

    return stats


def analyze_1d_data(dataset: FloodDataset, model_id: str) -> Dict:
    """
    Analyze 1D data characteristics for a model.
    """
    ds_model = dataset.filter_by_model(model_id)

    num_nodes_list: List[int] = []
    num_edges_list: List[int] = []
    num_timesteps_list: List[int] = []
    water_level_mins: List[float] = []
    water_level_maxs: List[float] = []

    for sample in ds_model:
        info = get_1d_node_info(sample)

        if info["has_1d_data"]:
            num_nodes_list.append(info["num_1d_nodes"])
            num_edges_list.append(info["num_edges"])
            num_timesteps_list.append(info["num_timesteps"])

            dynamic_1d = sample.get("dynamic_1d_nodes")
            if (
                dynamic_1d is not None
                and len(dynamic_1d) > 0
                and "water_level" in dynamic_1d.columns
            ):
                wl = dynamic_1d["water_level"]
                water_level_mins.append(float(wl.min()))
                water_level_maxs.append(float(wl.max()))

    analysis = {
        "model_id": model_id,
        "num_events": len(ds_model),
        "events_with_1d": len(num_nodes_list),
        "num_nodes": f"{min(num_nodes_list)}-{max(num_nodes_list)}" if num_nodes_list else "N/A",
        "num_edges": f"{min(num_edges_list)}-{max(num_edges_list)}" if num_edges_list else "N/A",
        "timesteps_range": f"{min(num_timesteps_list)}-{max(num_timesteps_list)}" if num_timesteps_list else "N/A",
        "water_level_min": f"{min(water_level_mins):.2f}" if water_level_mins else "N/A",
        "water_level_max": f"{max(water_level_maxs):.2f}" if water_level_maxs else "N/A",
    }

    return analysis


if __name__ == "__main__":
    from src.config import RAW_DATA_PATH

    print("=" * 60)
    print("1D DATA ANALYSIS")
    print("=" * 60)

    ds = FloodDataset(RAW_DATA_PATH, mode="train")

    for model_id in ["1", "2"]:
        print(f"\nModel_{model_id}:")
        print("-" * 40)

        # Analysis
        analysis = analyze_1d_data(ds, model_id)
        for key, value in analysis.items():
            print(f"  {key}: {value}")

        # Normalization stats
        try:
            stats = compute_normalization_stats_1d(ds, model_id)
            print(f"\n  Normalization stats:")
            print(f"    static_mean shape: {stats['static_mean'].shape}")
            print(f"    static_std shape:  {stats['static_std'].shape}")
            if "water_level_mean" in stats:
                print(f"    water_level_mean: {stats['water_level_mean']:.2f}")
                print(f"    water_level_std:  {stats['water_level_std']:.2f}")
            if "inlet_flow_mean" in stats:
                print(f"    inlet_flow_mean:  {stats['inlet_flow_mean']:.4f}")
                print(f"    inlet_flow_std:   {stats['inlet_flow_std']:.4f}")
        except Exception as e:
            print(f"  Norm stats error: {e}")

        # Test pivoting on first sample
        ds_model = ds.filter_by_model(model_id)
        if len(ds_model) == 0:
            continue

        sample = ds_model[0]
        print(f"\n  Pivot test (Event_{sample['event_id']}):")

        dynamic_1d = sample.get("dynamic_1d_nodes")
        if dynamic_1d is not None and len(dynamic_1d) > 0:
            wl_array = pivot_dynamic_1d(dynamic_1d, "water_level")
            print(f"    water_level array shape: {wl_array.shape}")  # (T, N)
            print(f"    water_level range: [{wl_array.min():.2f}, {wl_array.max():.2f}]")

            all_features = pivot_dynamic_1d_all_features(dynamic_1d)
            print(f"    all features shape: {all_features.shape}")  # (T, N, F)

        # Test edge index
        if "edge_index_1d" in sample:
            edge_index = get_edge_index_1d(sample)
            print(f"    edge_index shape: {edge_index.shape}")
            print(f"    edge_index node range: [{edge_index.min()}, {edge_index.max()}]")

        # Test static features
        if "static_1d_nodes" in sample:
            static_feats = get_static_node_features(sample)
            print(f"    static features shape: {static_feats.shape}")


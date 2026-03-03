"""
utils_2d — Utility functions for the 2D graph builder.

Provides:
    * WSE ↔ depth conversion (works with numpy arrays and torch tensors).
    * Per-model normalization statistics (z-score parameters).
    * Serialization helpers for normalization stats (JSON).
    * Generic z-score normalization function.

Owner: Member B
See: IMPLEMENTATION_PLAN.md → Task 1.3
"""

from __future__ import annotations

import json
from typing import Any, Dict, Union

import numpy as np
import torch

# Type alias for values that can be either numpy arrays or torch tensors.
ArrayLike = Union[np.ndarray, torch.Tensor]

# Columns for which we compute normalization statistics.
_NORM_COLUMNS = [
    "elevation",
    "area",
    "roughness",
    "curvature",
    "flow_accumulation",
    "position_x",
    "position_y",
    "min_elevation",
    "aspect",
]


# ───────────────────────────────────────────────────────────────────────
#  WSE ↔ Depth conversion
# ───────────────────────────────────────────────────────────────────────

def wse_to_depth(water_level: ArrayLike, min_elevation: ArrayLike) -> ArrayLike:
    """Convert Water Surface Elevation (WSE) to depth above ground.

    Uses ``min_elevation`` (the lowest point of each mesh cell) rather
    than the centroid elevation, because water collects at the lowest
    point.  This produces mostly non-negative depths for valid data.

    Parameters
    ----------
    water_level : np.ndarray or torch.Tensor
        Water surface elevation values (absolute elevation in metres/feet).
    min_elevation : np.ndarray or torch.Tensor
        Minimum elevation of each mesh cell (same units as *water_level*).

    Returns
    -------
    np.ndarray or torch.Tensor
        ``water_level - min_elevation``.  Should be >= 0 for wet nodes.
    """
    return water_level - min_elevation


def depth_to_wse(depth: ArrayLike, min_elevation: ArrayLike) -> ArrayLike:
    """Convert depth above ground back to Water Surface Elevation (WSE).

    Parameters
    ----------
    depth : np.ndarray or torch.Tensor
        Water depth values (>= 0 for wet nodes).
    min_elevation : np.ndarray or torch.Tensor
        Minimum elevation of each mesh cell.

    Returns
    -------
    np.ndarray or torch.Tensor
        ``depth + min_elevation``.
    """
    return depth + min_elevation


# ───────────────────────────────────────────────────────────────────────
#  NaN-safe min_elevation helper
# ───────────────────────────────────────────────────────────────────────

def get_min_elevation_filled(static_2d_nodes: Any) -> np.ndarray:
    """Get ``min_elevation`` with NaN values filled from ``elevation``.

    A small number of boundary/edge nodes have ``NaN`` in
    ``min_elevation``.  We fill those gaps with the centroid
    ``elevation`` as a reasonable fallback so that downstream depth
    calculations never produce NaN.

    Parameters
    ----------
    static_2d_nodes : pd.DataFrame
        DataFrame with at least ``min_elevation`` and ``elevation``
        columns (from ``sample["static_2d_nodes"]``).

    Returns
    -------
    np.ndarray
        Copy of ``min_elevation`` with NaN entries replaced by
        the corresponding ``elevation`` value.
    """
    min_elevation = static_2d_nodes["min_elevation"].values.copy()
    elevation = static_2d_nodes["elevation"].values

    nan_mask = np.isnan(min_elevation)
    min_elevation[nan_mask] = elevation[nan_mask]

    return min_elevation


# ───────────────────────────────────────────────────────────────────────
#  Normalization statistics
# ───────────────────────────────────────────────────────────────────────

def compute_normalization_stats(dataset: Any, model_id: str) -> Dict[str, float]:
    """Compute per-column mean and std for static 2D node features.

    The statistics are computed from the *first* event of the filtered
    dataset because static features are identical across all events within
    the same urban model (they come from the shared ``2d_nodes_static.csv``).

    Parameters
    ----------
    dataset : FloodDataset
        An instance of :class:`src.dataset.FloodDataset`.
    model_id : str
        Model identifier (e.g. ``"1"`` or ``"2"``).

    Returns
    -------
    dict[str, float]
        Keys follow the pattern ``"{column}_mean"`` and ``"{column}_std"``
        for every column in ``_NORM_COLUMNS``.

    Notes
    -----
    * The ``aspect`` column uses ``-1`` as a sentinel for flat/undefined
      areas.  These values are replaced with ``0`` before computing stats
      so that the sentinel does not skew the distribution.
    * Normalization must be per-model because the two urban models live
      in different elevation / coordinate ranges.
    """
    subset = dataset.filter_by_model(model_id)
    if len(subset) == 0:
        raise ValueError(
            f"No events found for model_id='{model_id}'. "
            f"Available models: {dataset.get_model_ids()}"
        )

    sample = subset[0]
    static_df = sample["static_2d_nodes"]

    stats: Dict[str, float] = {}

    for col in _NORM_COLUMNS:
        if col not in static_df.columns:
            raise KeyError(
                f"Column '{col}' not found in static_2d_nodes. "
                f"Available columns: {list(static_df.columns)}"
            )

        values = static_df[col].copy()

        # Handle aspect sentinel: replace -1 with 0
        if col == "aspect":
            values = values.replace(-1.0, 0.0)

        # Drop NaN values before computing stats so they don't propagate
        valid = values.dropna()

        stats[f"{col}_mean"] = float(valid.mean()) if len(valid) > 0 else 0.0
        stats[f"{col}_std"] = float(valid.std()) if len(valid) > 1 else 1.0

    # ── dist_to_drain stats (requires 1D node positions) ─────────────
    from scipy.spatial import KDTree

    static_1d = sample["static_1d_nodes"]
    coords_2d = static_df[["position_x", "position_y"]].values
    coords_1d = static_1d[["position_x", "position_y"]].values

    tree = KDTree(coords_1d)
    dist_to_nearest_1d, _ = tree.query(coords_2d)

    stats["dist_to_drain_mean"] = float(dist_to_nearest_1d.mean())
    stats["dist_to_drain_std"] = float(dist_to_nearest_1d.std())

    return stats


# ───────────────────────────────────────────────────────────────────────
#  Serialization helpers
# ───────────────────────────────────────────────────────────────────────

def save_normalization_stats(stats: Dict[str, float], filepath: str) -> None:
    """Save normalization statistics to a JSON file.

    Parameters
    ----------
    stats : dict[str, float]
        Dictionary produced by :func:`compute_normalization_stats`.
    filepath : str
        Destination path (e.g. ``"data/.cache/norm_stats_model_1.json"``).
    """
    with open(filepath, "w") as f:
        json.dump(stats, f, indent=2)


def load_normalization_stats(filepath: str) -> Dict[str, float]:
    """Load normalization statistics from a JSON file.

    Parameters
    ----------
    filepath : str
        Path to a JSON file previously written by
        :func:`save_normalization_stats`.

    Returns
    -------
    dict[str, float]
        Restored statistics dictionary.
    """
    with open(filepath, "r") as f:
        return json.load(f)


# ───────────────────────────────────────────────────────────────────────
#  Z-score normalization
# ───────────────────────────────────────────────────────────────────────

def normalize_feature(
    values: ArrayLike,
    mean: float,
    std: float,
) -> ArrayLike:
    """Apply z-score normalization: ``(values - mean) / (std + eps)``.

    Parameters
    ----------
    values : np.ndarray or torch.Tensor
        Raw feature values.
    mean : float
        Feature mean (from :func:`compute_normalization_stats`).
    std : float
        Feature standard deviation.

    Returns
    -------
    np.ndarray or torch.Tensor
        Normalized values (same type as input).
    """
    return (values - mean) / (std + 1e-8)


# ───────────────────────────────────────────────────────────────────────
#  Quick smoke test
# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.config import RAW_DATA_PATH
    from src.dataset import FloodDataset
    import numpy as np

    ds = FloodDataset(RAW_DATA_PATH, mode="train")

    # Test for Model_1
    stats = compute_normalization_stats(ds, "1")
    print("Model_1 normalization stats:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

    # Test for Model_2
    stats_2 = compute_normalization_stats(ds, "2")
    print("\nModel_2 normalization stats:")
    for key, value in stats_2.items():
        print(f"  {key}: {value:.4f}")

    # Test depth conversion
    print(f"\n=== Depth Conversion Test ===")
    sample = ds[0]
    static_2d = sample["static_2d_nodes"]
    dynamic = sample["dynamic_2d_nodes"]

    # Get filled min_elevation
    min_elevation = get_min_elevation_filled(static_2d)
    nan_filled = np.isnan(static_2d["min_elevation"].values).sum()
    print(f"  min_elevation NaN filled: {nan_filled}")

    # Get water level at timestep 0
    wl = dynamic[dynamic["timestep"] == 0].sort_values("node_idx")["water_level"].values

    # Convert to depth
    depth = wse_to_depth(wl, min_elevation)

    print(f"  WSE range: [{wl.min():.2f}, {wl.max():.2f}]")
    print(f"  min_elevation range: [{min_elevation.min():.2f}, {min_elevation.max():.2f}]")
    print(f"  Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"  Negative depths: {(depth < 0).sum()} ({(depth < 0).sum()/len(depth)*100:.1f}%)")
    print(f"  Mean depth: {depth.mean():.4f}m")

    # Verify conversion roundtrip
    wl_back = depth_to_wse(depth, min_elevation)
    roundtrip_error = np.abs(wl - wl_back).max()
    print(f"  Roundtrip error: {roundtrip_error:.10f} (should be ~0)")

    print("\n✓ utils_2d.py complete!")
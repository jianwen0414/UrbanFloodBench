"""
train_unified — Dual-Model Training Pipeline for the Unified HeteroGNN (v6).

Implements the **Depth-Based, Delta-Prediction, Multi-Model** strategy:

    Phase 1:  Train on Model 1 → save ``checkpoints/unified_model_1.pt``
    Phase 2:  Train on Model 2 → save ``checkpoints/unified_model_2.pt``

Each phase uses a freshly initialised ``UnifiedHeteroModel`` to prevent
cross-model weight contamination (different elevation scales, different
graph topologies).

Key features (v6 — from v5):
    - **min_elevation** 2D depth reference (0.3% neg vs 93.6% with centroid)
    - **Rich features** — 17 2D / 7 1D features (EDA: rain_rolling, rain_delta, min_elevation)
    - **Depth lags** (t-2, t-3, t-4) with AR replacement
    - ``compute_model_stats()`` — now covers all v6 features
    - **Delta-prediction loss** — physics-compliant WSE recovery
    - **Push-Forward K** — multi-step trajectory loss (K = 2–20)
    - **AR noise injection** — stabilises autoregressive rollout
    - **EMA model** — exponential moving average for stable validation
    - **Gradient clipping** — prevents AR gradient explosions
    - **LR warmup** — 5-epoch linear warmup for stable convergence

Owner : Member C (Lead Architect)
See   : IMPLEMENTATION_PLAN.md → Task 2.4, 3.1, PROJECT_BIBLE.md §7

Usage
-----
    python -m src.train_unified
    python -m src.train_unified --epochs 120 --hidden_channels 192
    python -m src.train_unified --model_ids 1    # train only Model 1
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import RAW_DATA_PATH
from src.dataset import FloodDataset
from src.graph_builder_unified import (
    build_hetero_graph,
    get_feature_dims,
    summarise_graph,
    compute_elev_rel_neighbors,
    compute_dist_to_drain,
    DEPTH_IDX,
    WATER_VOL_IDX,
    LAG1_IDX,
    LAG2_IDX,
    LAG3_IDX,
    _make_bidirectional,
)
from src.loss import (
    standardized_rmse_loss,
    standardized_rmse_metric,
    push_forward_loss,
    per_node_srmse_loss,
)
from src.model_unified import UnifiedHeteroModel

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kw):  # type: ignore[misc]
        return iterable


# =====================================================================
#  Normalisation Pre-Pass
# =====================================================================

def compute_model_stats(
    dataset: FloodDataset,
    model_id: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Iterate through training events ONCE to compute per-feature mean/std.

    v6 changes:
        - Uses **min_elevation** (not centroid) for 2D depth.
        - Adds stats for all v6 features: area, roughness, aspect,
          curvature, flow_accumulation, elev_rel_neighbors,
          dist_to_drain (2D) and base_area (1D).

    Returns
    -------
    dict
        Normalisation statistics (v6 layout).
    """
    subset = dataset.filter_by_model(model_id)

    if len(subset) == 0:
        raise ValueError(f"No events found for model_id='{model_id}'.")

    # ── Extract static data from first event (shared across all) ──
    sample_0 = subset[0]
    s1d = sample_0["static_1d_nodes"].sort_values("node_idx").reset_index(drop=True)
    s2d = sample_0["static_2d_nodes"].sort_values("node_idx").reset_index(drop=True)

    invert_elev = s1d["invert_elevation"].values.astype(np.float64)
    surface_elev = s1d["surface_elevation"].values.astype(np.float64)
    capacity = np.maximum(surface_elev - invert_elev, 1e-8)
    base_area = s1d["base_area"].values.astype(np.float64)

    # ── 2D depth reference: min_elevation ─────────────────────────
    centroid_elev = s2d["elevation"].values.astype(np.float64)
    if "min_elevation" in s2d.columns:
        min_elev = s2d["min_elevation"].values.astype(np.float64)
        nan_mask = np.isnan(min_elev)
        if nan_mask.any():
            min_elev[nan_mask] = centroid_elev[nan_mask]
    else:
        min_elev = centroid_elev.copy()

    # ── 2D static feature arrays ──────────────────────────────────
    def _col_or_default(df, col, n, default=0.0):
        if col in df.columns:
            v = df[col].values.astype(np.float64)
            v[~np.isfinite(v)] = default
            return v
        return np.full(n, default, dtype=np.float64)

    n_2d = len(s2d)
    min_elev_vals = min_elev  # already computed above (NaN-filled)
    area_vals = _col_or_default(s2d, "area", n_2d)
    roughness_vals = _col_or_default(s2d, "roughness", n_2d, 0.03)
    aspect_vals = _col_or_default(s2d, "aspect", n_2d)
    curvature_vals = _col_or_default(s2d, "curvature", n_2d)
    flow_acc_vals = _col_or_default(s2d, "flow_accumulation", n_2d, 1.0)

    # Derived: elev_rel_neighbors (needs 2D edge index)
    ei_2d_df = sample_0["edge_index_2d"]
    surf_src = torch.tensor(ei_2d_df["from_node"].values, dtype=torch.long)
    surf_dst = torch.tensor(ei_2d_df["to_node"].values, dtype=torch.long)
    spread_ei = _make_bidirectional(torch.stack([surf_src, surf_dst], dim=0))
    elev_rel_vals = compute_elev_rel_neighbors(
        centroid_elev.astype(np.float32), spread_ei
    ).astype(np.float64)

    # Derived: dist_to_drain
    coords_2d = s2d[["position_x", "position_y"]].values.astype(np.float32)
    coords_1d = s1d[["position_x", "position_y"]].values.astype(np.float32)
    conn_df = sample_0["1d2d_conn"]
    dist_drain_vals, _ = compute_dist_to_drain(
        coords_2d, coords_1d, conn_df, n_2d
    )
    dist_drain_vals = dist_drain_vals.astype(np.float64)

    # Slope (EDA: terrain physics) — from static_2d_edges via graph_builder
    from src.graph_builder_unified import compute_mean_slope_per_node
    edges_2d_df = sample_0.get("static_2d_edges", pd.DataFrame())
    slope_vals = compute_mean_slope_per_node(
        sample_0["edge_index_2d"], edges_2d_df, n_2d
    ).astype(np.float64)

    # ── Accumulate dynamic feature values across all events ───────
    depth_1d_acc: List[np.ndarray] = []
    iflow_acc: List[np.ndarray] = []
    depth_2d_acc: List[np.ndarray] = []
    water_vol_acc: List[np.ndarray] = []
    effective_depth_acc: List[np.ndarray] = []  # P0: water_volume/area
    rain_acc: List[np.ndarray] = []
    rain_rolling_acc: List[np.ndarray] = []
    rain_delta_acc: List[np.ndarray] = []
    # Edge flow accumulators (per-node aggregated flow from edges)
    edge_inflow_1d_acc: List[np.ndarray] = []
    edge_outflow_1d_acc: List[np.ndarray] = []
    edge_netflow_1d_acc: List[np.ndarray] = []
    edge_inflow_2d_acc: List[np.ndarray] = []
    edge_outflow_2d_acc: List[np.ndarray] = []
    edge_netflow_2d_acc: List[np.ndarray] = []

    # Per-node 1D depth accumulators (for per-node normalization)
    n_1d = len(s1d)
    per_node_1d_depth: Dict[int, List[float]] = {i: [] for i in range(n_1d)}

    print(f"  Computing norm stats for Model {model_id} "
          f"({len(subset)} events)...", end=" ", flush=True)

    for idx in range(len(subset)):
        sample = subset[idx]
        dyn_1d = sample["dynamic_1d_nodes"]
        dyn_2d = sample["dynamic_2d_nodes"]

        # 1D depth = WSE − invert_elevation
        if (
            not dyn_1d.empty
            and "water_level" in dyn_1d.columns
            and "node_idx" in dyn_1d.columns
        ):
            wl = dyn_1d["water_level"].values.astype(np.float64)
            nidx = dyn_1d["node_idx"].values.astype(int)
            depths_1d = wl - invert_elev[nidx]
            depth_1d_acc.append(depths_1d)
            # Per-node accumulation
            for ni, dv in zip(nidx, depths_1d):
                per_node_1d_depth[int(ni)].append(float(dv))

        # 1D inlet_flow
        if not dyn_1d.empty and "inlet_flow" in dyn_1d.columns:
            iflow_acc.append(
                dyn_1d["inlet_flow"].values.astype(np.float64)
            )

        # 2D depth = WSE − min_elevation  (KEY CHANGE from v5)
        if (
            not dyn_2d.empty
            and "water_level" in dyn_2d.columns
            and "node_idx" in dyn_2d.columns
        ):
            wl = dyn_2d["water_level"].values.astype(np.float64)
            nidx = dyn_2d["node_idx"].values.astype(int)
            depth_2d_acc.append(wl - min_elev[nidx])
            # 2D water_volume (mass-balance signal; Member B-style feature)
            if "water_volume" in dyn_2d.columns:
                water_vol_acc.append(dyn_2d["water_volume"].values.astype(np.float64))
            # P0: effective_depth = water_volume / area (physics-aligned)
            # Floor area to avoid explosion when Model 1 has area≈0 cells (min=-1.4e-14)
            if "water_volume" in dyn_2d.columns and len(area_vals) == n_2d:
                wv = dyn_2d["water_volume"].values.astype(np.float64)
                area_n = np.maximum(area_vals[nidx], 1.0)
                effective_depth_acc.append(wv / area_n)

        # 2D rainfall + derivatives (EDA: rolling mean, delta)
        if not dyn_2d.empty and "rainfall" in dyn_2d.columns:
            rain_vals = dyn_2d["rainfall"].values.astype(np.float64)
            rain_acc.append(rain_vals)
            # Pivot to [T, N] for rolling/delta
            pivot = dyn_2d.pivot_table(
                index="timestep", columns="node_idx", values="rainfall",
                aggfunc="mean", fill_value=0.0,
            )
            pivot = pivot.reindex(columns=range(n_2d), fill_value=0.0).sort_index()
            rain_grid = pivot.values.astype(np.float64)  # [T, n_2d]
            window = 3
            rolling = np.array([
                np.mean(rain_grid[max(0, t - window + 1) : t + 1], axis=0)
                for t in range(len(rain_grid))
            ])
            delta = np.zeros_like(rain_grid)
            delta[1:] = rain_grid[1:] - rain_grid[:-1]
            rain_rolling_acc.append(rolling.ravel())
            rain_delta_acc.append(delta.ravel())

        # Edge flow aggregation (simple: just use raw flow values for stats)
        dyn_1d_edges = sample.get("dynamic_1d_edges", pd.DataFrame())
        dyn_2d_edges = sample.get("dynamic_2d_edges", pd.DataFrame())
        ei_1d = sample_0["edge_index_1d"]
        ei_2d = sample_0["edge_index_2d"]

        def _agg_flow_for_stats(edge_dyn, edge_idx_df, n_nodes):
            """Vectorized aggregation: per-node mean absolute flow."""
            if edge_dyn.empty or edge_idx_df.empty:
                return np.zeros(0), np.zeros(0), np.zeros(0)
            src = edge_idx_df["from_node"].values
            dst = edge_idx_df["to_node"].values
            eidxs = edge_dyn["edge_idx"].values.astype(np.int64)
            flows = edge_dyn["flow"].values.astype(np.float64)
            # Filter valid
            valid = eidxs < len(src)
            eidxs = eidxs[valid]
            flows = flows[valid]
            abs_flow = np.abs(flows)
            s = src[eidxs]
            d = dst[eidxs]
            fwd = flows >= 0
            rev = ~fwd
            inflow = np.zeros(n_nodes, dtype=np.float64)
            outflow = np.zeros(n_nodes, dtype=np.float64)
            cnt_in = np.zeros(n_nodes, dtype=np.float64)
            cnt_out = np.zeros(n_nodes, dtype=np.float64)
            if fwd.any():
                np.add.at(outflow, s[fwd], abs_flow[fwd])
                np.add.at(cnt_out, s[fwd], 1)
                np.add.at(inflow, d[fwd], abs_flow[fwd])
                np.add.at(cnt_in, d[fwd], 1)
            if rev.any():
                np.add.at(outflow, d[rev], abs_flow[rev])
                np.add.at(cnt_out, d[rev], 1)
                np.add.at(inflow, s[rev], abs_flow[rev])
                np.add.at(cnt_in, s[rev], 1)
            cnt_in = np.maximum(cnt_in, 1)
            cnt_out = np.maximum(cnt_out, 1)
            inflow /= cnt_in
            outflow /= cnt_out
            net = inflow - outflow
            return inflow, outflow, net

        inf1, outf1, netf1 = _agg_flow_for_stats(dyn_1d_edges, ei_1d, n_1d)
        if len(inf1) > 0:
            edge_inflow_1d_acc.append(inf1)
            edge_outflow_1d_acc.append(outf1)
            edge_netflow_1d_acc.append(netf1)
        inf2, outf2, netf2 = _agg_flow_for_stats(dyn_2d_edges, ei_2d, n_2d)
        if len(inf2) > 0:
            edge_inflow_2d_acc.append(inf2)
            edge_outflow_2d_acc.append(outf2)
            edge_netflow_2d_acc.append(netf2)

    def _stats(
        arrays: List[np.ndarray],
        physical_max: Optional[float] = None,
        robust: bool = False,
    ) -> Dict[str, float]:
        """Compute mean/std from accumulated arrays with outlier clamping."""
        if not arrays:
            return {"mean": 0.0, "std": 1.0}
        vals = np.concatenate(arrays)

        n_nonfinite = int((~np.isfinite(vals)).sum())
        if n_nonfinite > 0:
            warnings.warn(
                f"    Dropping {n_nonfinite:,} NaN/Inf values "
                f"from normalisation data."
            )
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return {"mean": 0.0, "std": 1.0}

        if physical_max is not None:
            n_high = int((vals > physical_max).sum())
            if n_high > 0:
                warnings.warn(
                    f"    Clamping {n_high:,} values > {physical_max}."
                )
            vals = np.clip(vals, None, physical_max)

        if robust:
            center = float(np.median(vals))
            p5, p95 = np.percentile(vals, [5, 95])
            scale = float(max((p95 - p5) / 3.29, 1e-8))
            return {"mean": center, "std": scale}

        return {
            "mean": float(np.mean(vals)),
            "std": float(max(np.std(vals), 1e-8)),
        }

    def _static_stats(arr: np.ndarray) -> Dict[str, float]:
        """Mean/std for a static (per-node) feature array."""
        return {
            "mean": float(np.mean(arr)),
            "std": float(max(np.std(arr), 1e-8)),
        }

    # ── Per-node 1D depth mean/std ─────────────────────────────────
    #    Global 1D depth stats are dominated by constant nodes at
    #    extreme elevations (std=24.83).  Per-node normalization
    #    ensures each pipe's depth feature is in a meaningful range.
    pn_mean_1d = np.zeros(n_1d, dtype=np.float64)
    pn_std_1d = np.ones(n_1d, dtype=np.float64)
    for ni in range(n_1d):
        vals_i = per_node_1d_depth[ni]
        if len(vals_i) > 0:
            arr_i = np.array(vals_i)
            pn_mean_1d[ni] = float(arr_i.mean())
            pn_std_1d[ni] = float(max(arr_i.std(), 1e-4))

    # ── 1D pipe attributes (Phase B) — from static_1d_edges ───────────────
    from src.graph_builder_unified import compute_mean_pipe_attr_per_node
    edges_1d_df = sample_0.get("static_1d_edges", pd.DataFrame())
    ei_1d = sample_0["edge_index_1d"]
    pipe_diam = compute_mean_pipe_attr_per_node(ei_1d, edges_1d_df, n_1d, "diameter")
    pipe_len = compute_mean_pipe_attr_per_node(ei_1d, edges_1d_df, n_1d, "length")
    pipe_rough = compute_mean_pipe_attr_per_node(ei_1d, edges_1d_df, n_1d, "roughness")
    pipe_slope = compute_mean_pipe_attr_per_node(ei_1d, edges_1d_df, n_1d, "slope")

    pos_x_vals = _col_or_default(s2d, "position_x", n_2d)
    pos_y_vals = _col_or_default(s2d, "position_y", n_2d)

    stats = {
        "1d": {
            "depth": _stats(depth_1d_acc),
            "inlet_flow": _stats(iflow_acc),
            "capacity": _static_stats(capacity),
            "base_area": _static_stats(base_area),
            "depth_per_node_mean": pn_mean_1d.tolist(),
            "depth_per_node_std": pn_std_1d.tolist(),
            # Phase B: pipe geometry (capacity, wave speed, resistance)
            "pipe_diameter": _static_stats(pipe_diam.astype(np.float64)),
            "pipe_length": _static_stats(pipe_len.astype(np.float64)),
            "pipe_roughness": _static_stats(pipe_rough.astype(np.float64)),
            "pipe_slope": _static_stats(pipe_slope.astype(np.float64)),
            # Dynamic edge flow (per-node aggregated)
            "edge_mean_inflow": _stats(edge_inflow_1d_acc) if edge_inflow_1d_acc else {"mean": 0.0, "std": 1.0},
            "edge_mean_outflow": _stats(edge_outflow_1d_acc) if edge_outflow_1d_acc else {"mean": 0.0, "std": 1.0},
            "edge_net_flow": _stats(edge_netflow_1d_acc) if edge_netflow_1d_acc else {"mean": 0.0, "std": 1.0},
        },
        "2d": {
            "depth": _stats(depth_2d_acc),
            "rainfall": _stats(rain_acc),
            "rain_rolling_mean": _stats(rain_rolling_acc),
            "rain_delta": _stats(rain_delta_acc),
            "elevation": _static_stats(centroid_elev),
            "min_elevation": _static_stats(min_elev_vals),
            "slope": _static_stats(slope_vals),
            "area": _static_stats(area_vals),
            "roughness": _static_stats(roughness_vals),
            "aspect": _static_stats(aspect_vals),
            "curvature": _static_stats(curvature_vals),
            "flow_accumulation": _static_stats(flow_acc_vals),
            "elev_rel_neighbors": _static_stats(elev_rel_vals),
            "dist_to_drain": _static_stats(dist_drain_vals),
            # Phase A: water_volume (legacy), effective_depth (P0: water_volume/area)
            # Clamp effective_depth to 0–20 ft (physical max flood depth) to avoid outlier-driven stats
            "water_volume": _stats(water_vol_acc) if water_vol_acc else {"mean": 0.0, "std": 1.0},
            "effective_depth": _stats(
                effective_depth_acc, physical_max=20.0
            ) if effective_depth_acc else {"mean": 0.0, "std": 1.0},
            "position_x": _static_stats(pos_x_vals),
            "position_y": _static_stats(pos_y_vals),
            # Dynamic edge flow (per-node aggregated)
            "edge_mean_inflow": _stats(edge_inflow_2d_acc) if edge_inflow_2d_acc else {"mean": 0.0, "std": 1.0},
            "edge_mean_outflow": _stats(edge_outflow_2d_acc) if edge_outflow_2d_acc else {"mean": 0.0, "std": 1.0},
            "edge_net_flow": _stats(edge_netflow_2d_acc) if edge_netflow_2d_acc else {"mean": 0.0, "std": 1.0},
        },
    }

    print("done.")
    return stats


# =====================================================================
#  Scheduled Sampling (Teacher Forcing Scheduler)
# =====================================================================

class TeacherForcingScheduler:
    """Linear decay of teacher forcing ratio.

    Phase 1 (warmup):  ratio = 1.0
    Phase 2 (decay):   linear 1.0 → min_ratio
    Phase 3 (student): ratio = min_ratio
    """

    def __init__(
        self,
        warmup_epochs: int = 3,
        decay_epochs: int = 30,
        min_ratio: float = 0.0,
    ) -> None:
        self.warmup = warmup_epochs
        self.decay = decay_epochs
        self.min_ratio = min_ratio

    def __call__(self, epoch: int) -> float:
        if epoch < self.warmup:
            return 1.0
        progress = (epoch - self.warmup) / max(self.decay, 1)
        if progress >= 1.0:
            return self.min_ratio
        return 1.0 - (1.0 - self.min_ratio) * progress


# =====================================================================
#  Exponential Moving Average (EMA) — Stable Validation
# =====================================================================

class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of the model weights updated as::

        shadow = decay * shadow + (1 - decay) * current

    The EMA model is used exclusively for **validation** — it
    smooths out the noisy per-epoch weight oscillations that cause
    wild val swings (e.g. 16 → 174 → 108).

    Parameters
    ----------
    model : nn.Module
        The model whose weights to track.
    decay : float
        Smoothing factor (default 0.999).  Higher = more smoothing.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: Dict[str, Tensor] = {}
        self.backup: Dict[str, Tensor] = {}
        # Initialise shadow as a copy of current weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights after each optimizer step."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module) -> None:
        """Swap model weights with shadow weights (for validation)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        """Restore original model weights (after validation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# =====================================================================
#  Single-Event Training Step (Push-Forward Delta-Prediction)
# =====================================================================

def _train_one_event(
    model: UnifiedHeteroModel,
    graph: Any,
    norm_stats: Dict,
    stds_1d: Tensor,
    stds_2d: Tensor,
    K: int,
    tf_ratio: float,
    spinup: int,
    device: torch.device,
    clamp_weights: float = 20.0,
    delta_clamp: float = 2.0,
    delta_clamp_1d: float = 5.0,
    phys_weight: float = 0.1,
    ar_noise_std: float = 0.005,
) -> Tuple[Tensor, Dict[str, float]]:
    """Train on a single event with push-forward delta-prediction loss.

    The model outputs raw delta logits.  Recovery via hard clamp:
        delta        = clamp(model_output, -delta_clamp, +delta_clamp)
                        × const_mask
        pred_depth_t = prev_depth_{t-1} + delta  (NO clamp — negative depth = dry)
        pred_wse_t   = pred_depth_t + elevation

    Critical design note
    --------------------
    depth = WSE - elevation can be **naturally negative** for dry cells
    (93% of 2D, 60% of 1D).  An earlier `.clamp(min=0)` created a
    30-47 ft bias that pinned loss at ~892.  Removed.

    Returns
    -------
    (loss, breakdown_dict)
    """
    graph = graph.to(device)
    model.train()

    T = graph.num_timesteps
    n_1d = graph["1d"].num_nodes
    n_2d = graph["2d"].num_nodes

    # Ensure spinup + K fits within the event
    spinup = min(spinup, T - 1)
    K = min(K, T - spinup)
    if K <= 0:
        # Event too short for meaningful training
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {"loss_1d": 0.0, "loss_2d": 0.0, "phys": 0.0, "total": 0.0}

    # Build edge_index_dict once
    edge_index_dict: Dict[Tuple[str, str, str], Tensor] = {}
    for et in graph.edge_types:
        edge_index_dict[et] = graph[et].edge_index

    # Extract depth normalisation stats for feedback construction
    #   1D: per-node normalization (global stats had std=24.83, useless)
    #   2D: global normalization (mean=0.26, std=0.64, reasonable)
    d2_mean = norm_stats["2d"]["depth"]["mean"]
    d2_std = norm_stats["2d"]["depth"]["std"]

    # Per-node 1D depth stats for AR replacement
    if "depth_per_node_mean" in norm_stats["1d"]:
        _pn_mean_1d = torch.tensor(
            norm_stats["1d"]["depth_per_node_mean"], dtype=torch.float32
        ).to(device)  # [N_1d]
        _pn_std_1d = torch.tensor(
            norm_stats["1d"]["depth_per_node_std"], dtype=torch.float32
        ).to(device)   # [N_1d]
    else:
        d1_mean = norm_stats["1d"]["depth"]["mean"]
        d1_std = norm_stats["1d"]["depth"]["std"]
        _pn_mean_1d = torch.full((n_1d,), d1_mean, device=device)
        _pn_std_1d = torch.full((n_1d,), max(d1_std, 1e-4), device=device)

    # ── Constant-node mask ────────────────────────────────────────
    #    Nodes with σ < 0.01 (truly constant WSE) get delta forced
    #    to 0 → zero accumulated drift → zero SRMSE contribution.
    #    All active nodes share the same uniform delta_clamp so the
    #    1/σ² loss weighting correctly prioritises low-σ nodes.
    _stds_1d = stds_1d.to(device)
    _stds_2d = stds_2d.to(device)
    const_mask_1d = (_stds_1d >= 0.01).float()   # 0 for constant, 1 for active
    const_mask_2d = (_stds_2d >= 0.01).float()

    # ── Spinup: run GT features to warm the GRU hidden states ─────
    hidden = model.init_hidden(n_1d, n_2d, device)

    with torch.no_grad():
        for t in range(spinup):
            x_dict = {
                "1d": graph["1d"].x[t].to(device),
                "2d": graph["2d"].x[t].to(device),
            }
            _, hidden = model(x_dict, edge_index_dict, hidden)

    # Detach hidden states at the spinup boundary
    hidden = {k: v.detach() for k, v in hidden.items()}

    # ── Push-Forward Rollout: K steps with delta-prediction ───────
    # Previous depth starts from GT at the last spinup step
    prev_depth_1d = graph["1d"].depth[spinup - 1].to(device)
    prev_depth_2d = graph["2d"].depth[spinup - 1].to(device)

    # ── Build depth history for lag replacement ───────────────────
    #    lag1[t]=depth[t-2], lag2[t]=depth[t-3], lag3[t]=depth[t-4]
    #    Initialise with GT depths before the spinup boundary.
    _hist_1d: List[Tensor] = []
    _hist_2d: List[Tensor] = []
    for _t in range(max(0, spinup - 4), spinup):
        _hist_1d.append(graph["1d"].depth[_t].to(device))
        _hist_2d.append(graph["2d"].depth[_t].to(device))
    # Pad if spinup < 4
    while len(_hist_1d) < 4:
        _hist_1d.insert(0, _hist_1d[0].clone())
        _hist_2d.insert(0, _hist_2d[0].clone())

    def _get_lag(hist: List[Tensor], n: int) -> Tensor:
        """Get depth n steps back in history (1-indexed from end)."""
        idx = len(hist) - n
        return hist[max(0, idx)]

    all_pred_wse_1d: List[Tensor] = []
    all_pred_wse_2d: List[Tensor] = []
    all_target_wse_1d: List[Tensor] = []
    all_target_wse_2d: List[Tensor] = []

    for k in range(K):
        t = spinup + k

        # ── Decide teacher forcing ────────────────────────────────
        use_teacher = (
            k == 0
            or (
                model.training
                and tf_ratio > 0.0
                and torch.rand(1).item() < tf_ratio
            )
        )

        # ── Construct input features ──────────────────────────────
        x_1d_t = graph["1d"].x[t].clone().to(device)  # [N_1d, F]
        x_2d_t = graph["2d"].x[t].clone().to(device)  # [N_2d, F]

        if use_teacher and k > 0:
            prev_depth_1d = graph["1d"].depth[t - 1].to(device)
            prev_depth_2d = graph["2d"].depth[t - 1].to(device)

        # Replace depth feature (index 0) — per-node for 1D, global for 2D
        norm_d1 = (prev_depth_1d - _pn_mean_1d) / _pn_std_1d
        norm_d2 = (prev_depth_2d - d2_mean) / max(d2_std, 1e-8)

        if not use_teacher and ar_noise_std > 0 and model.training:
            norm_d1 = norm_d1 + torch.randn_like(norm_d1) * ar_noise_std
            norm_d2 = norm_d2 + torch.randn_like(norm_d2) * ar_noise_std

        x_1d_t[:, DEPTH_IDX] = norm_d1
        x_2d_t[:, DEPTH_IDX] = norm_d2

        # Replace effective_depth (index 5) during AR: use pred_depth (eff_depth ≈ depth)
        if not use_teacher and x_2d_t.size(-1) > WATER_VOL_IDX:
            st = norm_stats["2d"].get("effective_depth", norm_stats["2d"].get("depth", {}))
            eff_mean = st.get("mean", 0.0)
            eff_std = max(st.get("std", 1.0), 1e-8)
            x_2d_t[:, WATER_VOL_IDX] = (prev_depth_2d - eff_mean) / eff_std

        # Replace lag features (indices 2, 3, 4)
        # lag1=depth[t-2], lag2=depth[t-3], lag3=depth[t-4]
        x_1d_t[:, LAG1_IDX] = (_get_lag(_hist_1d, 2) - _pn_mean_1d) / _pn_std_1d
        x_1d_t[:, LAG2_IDX] = (_get_lag(_hist_1d, 3) - _pn_mean_1d) / _pn_std_1d
        x_1d_t[:, LAG3_IDX] = (_get_lag(_hist_1d, 4) - _pn_mean_1d) / _pn_std_1d
        x_2d_t[:, LAG1_IDX] = (_get_lag(_hist_2d, 2) - d2_mean) / max(d2_std, 1e-8)
        x_2d_t[:, LAG2_IDX] = (_get_lag(_hist_2d, 3) - d2_mean) / max(d2_std, 1e-8)
        x_2d_t[:, LAG3_IDX] = (_get_lag(_hist_2d, 4) - d2_mean) / max(d2_std, 1e-8)

        x_dict = {"1d": x_1d_t, "2d": x_2d_t}

        # ── Forward pass → delta logits ───────────────────────────
        delta_dict, hidden = model(x_dict, edge_index_dict, hidden)

        # ── Recover state: hard clamp + const mask ───────────────
        #    1D uses larger delta_clamp (GT 1D deltas can reach 3.1 ft)
        raw_delta_1d = (
            delta_dict["1d"].clamp(-delta_clamp_1d, delta_clamp_1d)
            * const_mask_1d
        )
        raw_delta_2d = (
            delta_dict["2d"].clamp(-delta_clamp, delta_clamp)
            * const_mask_2d
        )

        # ── WSE-space prediction (no depth clamping!) ─────────────
        pred_depth_1d = prev_depth_1d + raw_delta_1d
        # EDA: 2D depth 99% non-negative → physics clamp (Member B alignment)
        pred_depth_2d = (prev_depth_2d + raw_delta_2d).clamp(min=0.0)

        # Recover WSE for loss computation
        pred_wse_1d = pred_depth_1d + graph["1d"].elev.to(device)
        pred_wse_2d = pred_depth_2d + graph["2d"].elev.to(device)

        all_pred_wse_1d.append(pred_wse_1d)
        all_pred_wse_2d.append(pred_wse_2d)
        all_target_wse_1d.append(graph["1d"].y[t].to(device))
        all_target_wse_2d.append(graph["2d"].y[t].to(device))

        # Update prev_depth + history for next step
        _hist_1d.append(prev_depth_1d)
        _hist_2d.append(prev_depth_2d)

        prev_depth_1d = pred_depth_1d
        prev_depth_2d = pred_depth_2d

        # Early NaN detection
        if torch.isnan(pred_depth_1d).any() or torch.isnan(pred_depth_2d).any():
            warnings.warn(f"NaN detected at k={k}, t={t} — truncating rollout.")
            break

    if not all_pred_wse_1d:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {"loss_1d": 0.0, "loss_2d": 0.0, "phys": 0.0, "total": 0.0}

    # ── Compute push-forward loss with temporal weighting ───────
    #    Linear temporal weighting penalises later steps more,
    #    explicitly teaching the model to fight autoregressive drift.
    preds_1d = torch.stack(all_pred_wse_1d)     # [K', N_1d]
    preds_2d = torch.stack(all_pred_wse_2d)     # [K', N_2d]
    targets_1d = torch.stack(all_target_wse_1d)  # [K', N_1d]
    targets_2d = torch.stack(all_target_wse_2d)  # [K', N_2d]

    # ── Push-forward loss with metric-aligned weighting (v7) ───
    #    Uses min_std=0.01 matching validation metric so low-σ 2D
    #    nodes (91.7% of all 2D) are properly weighted during training.
    #    v6 used clamp_weights=20 (effective min_std=0.224), leaving
    #    3407/3716 2D nodes under-weighted vs validation.
    loss_1d = push_forward_loss(
        preds_1d.float(), targets_1d.float(),
        stds_1d.to(device),
        temporal_scheme="linear",
        min_std=0.01,
    )
    loss_2d = push_forward_loss(
        preds_2d.float(), targets_2d.float(),
        stds_2d.to(device),
        temporal_scheme="linear",
        min_std=0.01,
    )

    # Equal 1D/2D balance (matches competition formula)
    srmse_loss = 0.5 * loss_1d + 0.5 * loss_2d
    total_loss = srmse_loss

    breakdown = {
        "loss_1d": loss_1d.item(),
        "loss_2d": loss_2d.item(),
        "phys": 0.0,
        "total": total_loss.item(),
    }
    return total_loss, breakdown


# =====================================================================
#  Single-Event Validation (full autoregressive rollout)
# =====================================================================

@torch.no_grad()
def _validate_one_event(
    model: UnifiedHeteroModel,
    graph: Any,
    norm_stats: Dict,
    stds_1d: Tensor,
    stds_2d: Tensor,
    device: torch.device,
    spinup: int = 10,
    delta_clamp: float = 2.0,
    delta_clamp_1d: float = 5.0,
) -> Tuple[float, float, float]:
    """Full autoregressive validation on a single event.

    Uses hard-clamped deltas (matching training loop).
    Returns (srmse_1d, srmse_2d, srmse_combined).
    """
    graph = graph.to(device)
    model.eval()

    T = graph.num_timesteps
    n_1d = graph["1d"].num_nodes
    n_2d = graph["2d"].num_nodes
    spinup = min(spinup, T - 1)

    edge_index_dict: Dict[Tuple[str, str, str], Tensor] = {}
    for et in graph.edge_types:
        edge_index_dict[et] = graph[et].edge_index

    d2_mean = norm_stats["2d"]["depth"]["mean"]
    d2_std = norm_stats["2d"]["depth"]["std"]

    # Per-node 1D depth stats for AR replacement
    if "depth_per_node_mean" in norm_stats["1d"]:
        _pn_mean_1d = torch.tensor(
            norm_stats["1d"]["depth_per_node_mean"], dtype=torch.float32
        ).to(device)
        _pn_std_1d = torch.tensor(
            norm_stats["1d"]["depth_per_node_std"], dtype=torch.float32
        ).to(device)
    else:
        d1_mean = norm_stats["1d"]["depth"]["mean"]
        d1_std = norm_stats["1d"]["depth"]["std"]
        _pn_mean_1d = torch.full((n_1d,), d1_mean, device=device)
        _pn_std_1d = torch.full((n_1d,), max(d1_std, 1e-4), device=device)

    # ── Constant-node mask (matching training) ───────────────────
    _stds_1d = stds_1d.to(device)
    _stds_2d = stds_2d.to(device)
    const_mask_1d = (_stds_1d >= 0.01).float()
    const_mask_2d = (_stds_2d >= 0.01).float()

    # Spinup
    hidden = model.init_hidden(n_1d, n_2d, device)
    for t in range(spinup):
        x_dict = {
            "1d": graph["1d"].x[t].to(device),
            "2d": graph["2d"].x[t].to(device),
        }
        _, hidden = model(x_dict, edge_index_dict, hidden)

    # Prediction phase (fully autoregressive, no teacher forcing)
    prev_depth_1d = graph["1d"].depth[spinup - 1].to(device)
    prev_depth_2d = graph["2d"].depth[spinup - 1].to(device)

    # Build depth history for lag replacement (same logic as training)
    _hist_1d: List[Tensor] = []
    _hist_2d: List[Tensor] = []
    for _t in range(max(0, spinup - 4), spinup):
        _hist_1d.append(graph["1d"].depth[_t].to(device))
        _hist_2d.append(graph["2d"].depth[_t].to(device))
    while len(_hist_1d) < 4:
        _hist_1d.insert(0, _hist_1d[0].clone())
        _hist_2d.insert(0, _hist_2d[0].clone())

    def _get_lag(hist: List[Tensor], n: int) -> Tensor:
        idx = len(hist) - n
        return hist[max(0, idx)]

    all_pred_wse_1d: List[Tensor] = []
    all_pred_wse_2d: List[Tensor] = []
    all_target_wse_1d: List[Tensor] = []
    all_target_wse_2d: List[Tensor] = []

    for t in range(spinup, T):
        x_1d_t = graph["1d"].x[t].clone().to(device)
        x_2d_t = graph["2d"].x[t].clone().to(device)

        # Replace depth + lag features (per-node for 1D, global for 2D)
        x_1d_t[:, DEPTH_IDX] = (prev_depth_1d - _pn_mean_1d) / _pn_std_1d
        x_2d_t[:, DEPTH_IDX] = (prev_depth_2d - d2_mean) / max(d2_std, 1e-8)
        # effective_depth during AR: use prev_depth (eff_depth ≈ depth)
        if x_2d_t.size(-1) > WATER_VOL_IDX:
            st = norm_stats["2d"].get("effective_depth", norm_stats["2d"].get("depth", {}))
            eff_mean = st.get("mean", 0.0)
            eff_std = max(st.get("std", 1.0), 1e-8)
            x_2d_t[:, WATER_VOL_IDX] = (prev_depth_2d - eff_mean) / eff_std
        x_1d_t[:, LAG1_IDX] = (_get_lag(_hist_1d, 2) - _pn_mean_1d) / _pn_std_1d
        x_1d_t[:, LAG2_IDX] = (_get_lag(_hist_1d, 3) - _pn_mean_1d) / _pn_std_1d
        x_1d_t[:, LAG3_IDX] = (_get_lag(_hist_1d, 4) - _pn_mean_1d) / _pn_std_1d
        x_2d_t[:, LAG1_IDX] = (_get_lag(_hist_2d, 2) - d2_mean) / max(d2_std, 1e-8)
        x_2d_t[:, LAG2_IDX] = (_get_lag(_hist_2d, 3) - d2_mean) / max(d2_std, 1e-8)
        x_2d_t[:, LAG3_IDX] = (_get_lag(_hist_2d, 4) - d2_mean) / max(d2_std, 1e-8)

        x_dict = {"1d": x_1d_t, "2d": x_2d_t}
        delta_dict, hidden = model(x_dict, edge_index_dict, hidden)

        raw_delta_1d = (
            delta_dict["1d"].clamp(-delta_clamp_1d, delta_clamp_1d)
            * const_mask_1d
        )
        raw_delta_2d = (
            delta_dict["2d"].clamp(-delta_clamp, delta_clamp)
            * const_mask_2d
        )

        pred_depth_1d = prev_depth_1d + raw_delta_1d
        pred_depth_2d = (prev_depth_2d + raw_delta_2d).clamp(min=0.0)

        pred_wse_1d = pred_depth_1d + graph["1d"].elev.to(device)
        pred_wse_2d = pred_depth_2d + graph["2d"].elev.to(device)

        all_pred_wse_1d.append(pred_wse_1d)
        all_pred_wse_2d.append(pred_wse_2d)
        all_target_wse_1d.append(graph["1d"].y[t].to(device))
        all_target_wse_2d.append(graph["2d"].y[t].to(device))

        _hist_1d.append(prev_depth_1d)
        _hist_2d.append(prev_depth_2d)

        prev_depth_1d = pred_depth_1d
        prev_depth_2d = pred_depth_2d

        if torch.isnan(pred_depth_1d).any() or torch.isnan(pred_depth_2d).any():
            break

    if not all_pred_wse_1d:
        return float("inf"), float("inf"), float("inf")

    preds_1d = torch.stack(all_pred_wse_1d)
    preds_2d = torch.stack(all_pred_wse_2d)
    targets_1d = torch.stack(all_target_wse_1d)
    targets_2d = torch.stack(all_target_wse_2d)

    srmse_1d = standardized_rmse_metric(
        preds_1d, targets_1d, stds_1d.to(device)
    ).item()
    srmse_2d = standardized_rmse_metric(
        preds_2d, targets_2d, stds_2d.to(device)
    ).item()

    return srmse_1d, srmse_2d, (srmse_1d + srmse_2d) / 2.0


# =====================================================================
#  Full Training Pipeline for a Single Model
# =====================================================================

def train_model(
    model_id: str,
    dataset: FloodDataset,
    *,
    epochs: int = 120,
    hidden_channels: int = 256,  # EDA: lag-1 autocorr 0.9996
    num_gnn_layers: int = 3,
    lr: float = 0.001,  # EDA+Audit: Member B 0.005; high temporal dep needs faster LR
    weight_decay: float = 1e-5,
    dropout: float = 0.05,
    grad_clip_norm: float = 1.0,
    pushforward_K: int = 20,
    K_start: int = 2,
    K_ramp_epochs: int = 40,
    tf_warmup_epochs: int = 5,
    tf_decay_epochs: int = 40,
    tf_min_ratio: float = 0.0,
    clamp_weights: float = 20.0,
    delta_clamp: float = 2.0,
    delta_clamp_1d: float = 5.0,
    phys_weight: float = 0.1,
    ar_noise_std: float = 0.005,
    spinup_min: int = 3,
    spinup_max: int = 10,
    val_event_ids: Optional[List[str]] = None,
    checkpoint_dir: str = "checkpoints",
    device_str: str = "auto",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train a single UnifiedHeteroModel on one urban model's data.

    Parameters
    ----------
    model_id : str
        Urban model to train on (``'1'`` or ``'2'``).
    dataset : FloodDataset
        Full training dataset (will be filtered by model_id).

    Returns
    -------
    dict
        Training history and metadata.
    """

    # ── Model-specific overrides (Model 2: AR instability at K=20) ─
    #    Use lower K_max, slower ramp, and non-zero tf_min to reduce compounding.
    MODEL_OVERRIDES: Dict[str, Dict[str, Any]] = {
        "2": {
            "pushforward_K": 8,
            "K_ramp_epochs": 20,
            "tf_min_ratio": 0.20,
        },
    }
    overrides = MODEL_OVERRIDES.get(model_id, {})
    effective_K_max = overrides.get("pushforward_K", pushforward_K)
    effective_K_ramp = overrides.get("K_ramp_epochs", K_ramp_epochs)
    effective_tf_min = overrides.get("tf_min_ratio", tf_min_ratio)
    if overrides and verbose:
        print(f"  Model {model_id} overrides: K_max={effective_K_max}, "
              f"K_ramp={effective_K_ramp}, tf_min={effective_tf_min}")

    # ── Resolve device ────────────────────────────────────────────
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    print(f"\n{'='*60}")
    print(f"  TRAINING MODEL {model_id}  (device={device})")
    print(f"{'='*60}")

    # ── 1. Compute normalisation statistics ───────────────────────
    norm_stats = compute_model_stats(dataset, model_id)

    if verbose:
        for domain in ("1d", "2d"):
            for feat, st in norm_stats[domain].items():
                if isinstance(st, dict) and "mean" in st:
                    print(f"    {domain}/{feat}: mean={st['mean']:.4f}, std={st['std']:.4f}")
                elif isinstance(st, list):
                    print(f"    {domain}/{feat}: [{len(st)} values] (per-node)")

    # ── 2. Compute per-node WSE stds for loss ─────────────────────
    print("  Computing per-node WSE stds...", end=" ", flush=True)
    node_stds = dataset.compute_node_stds(model_id=model_id)
    stds_1d = torch.tensor(node_stds[model_id]["1d"], dtype=torch.float32)
    stds_2d = torch.tensor(node_stds[model_id]["2d"], dtype=torch.float32)
    print(f"done. (1D: {len(stds_1d)} nodes, 2D: {len(stds_2d)} nodes)")

    # ── 3. Build train/val graphs ─────────────────────────────────
    if val_event_ids is None:
        val_event_ids = ["3", "9", "15"]

    model_ds = dataset.filter_by_model(model_id)
    val_events = set(val_event_ids)

    print(f"  Building graphs (val events: {val_event_ids})...")
    train_graphs: List[Any] = []
    val_graphs: List[Any] = []

    for idx in range(len(model_ds)):
        sample = model_ds[idx]
        graph = build_hetero_graph(sample, norm_stats)
        if sample["event_id"] in val_events:
            val_graphs.append(graph)
        else:
            train_graphs.append(graph)

    print(f"  Train: {len(train_graphs)} events, Val: {len(val_graphs)} events")

    if not train_graphs:
        raise RuntimeError(f"No training events for Model {model_id}!")

    # ── 4. Construct model ────────────────────────────────────────
    dims = get_feature_dims(train_graphs[0])
    model = UnifiedHeteroModel(
        in_channels_1d=dims["in_channels_1d"],
        in_channels_2d=dims["in_channels_2d"],
        hidden_channels=hidden_channels,
        num_gnn_layers=num_gnn_layers,
        dropout=dropout,
    ).to(device)

    if verbose:
        print(model.summarise())

    # ── 5. Optimizer + Scheduler + EMA ────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # LR warmup (5 epochs linear from lr/10 → lr, then cosine decay)
    warmup_epochs = min(5, epochs)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    tf_scheduler = TeacherForcingScheduler(
        warmup_epochs=tf_warmup_epochs,
        decay_epochs=tf_decay_epochs,
        min_ratio=effective_tf_min,
    )
    ema = EMAModel(model, decay=0.998)

    # ── 6. Training Loop ──────────────────────────────────────────
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_srmse": [], "ema_val_srmse": [],
        "lr": [], "tf_ratio": [],
        "epoch_time": [], "loss_1d": [], "loss_2d": [],
    }
    best_val = float("inf")  # best EMA val
    best_epoch = 0
    best_val_raw = float("inf")  # best raw val (for dual checkpoint)
    best_epoch_raw = 0

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        t0 = time.time()

        # ── Progressive K curriculum ──────────────────────────────
        if epoch < effective_K_ramp:
            effective_K = K_start + int(
                (effective_K_max - K_start) * epoch / max(effective_K_ramp, 1)
            )
        else:
            effective_K = effective_K_max

        tf_ratio = tf_scheduler(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Train ─────────────────────────────────────────────────
        model.train()
        epoch_losses: List[float] = []
        epoch_1d: List[float] = []
        epoch_2d: List[float] = []

        # Randomise event order each epoch
        perm = np.random.permutation(len(train_graphs))

        for g_idx in perm:
            graph = train_graphs[g_idx]

            # Randomised spinup length
            spinup = int(np.random.randint(
                spinup_min, min(spinup_max, graph.num_timesteps - effective_K) + 1
            ))
            spinup = max(1, spinup)

            optimizer.zero_grad()
            loss, breakdown = _train_one_event(
                model, graph, norm_stats,
                stds_1d, stds_2d,
                K=effective_K,
                tf_ratio=tf_ratio,
                spinup=spinup,
                device=device,
                clamp_weights=clamp_weights,
                delta_clamp=delta_clamp,
                delta_clamp_1d=delta_clamp_1d,
                phys_weight=phys_weight,
                ar_noise_std=ar_noise_std,
            )

            if loss.requires_grad and not torch.isnan(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                ema.update(model)

            epoch_losses.append(breakdown["total"])
            epoch_1d.append(breakdown["loss_1d"])
            epoch_2d.append(breakdown["loss_2d"])

        scheduler.step()

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        avg_1d = float(np.mean(epoch_1d)) if epoch_1d else 0.0
        avg_2d = float(np.mean(epoch_2d)) if epoch_2d else 0.0

        # ── Validate (raw model) ──────────────────────────────────
        val_srmse = float("inf")
        if val_graphs:
            val_scores: List[float] = []
            for vg in val_graphs:
                _, _, combined = _validate_one_event(
                    model, vg, norm_stats, stds_1d, stds_2d,
                    device=device, spinup=10, delta_clamp=delta_clamp,
                    delta_clamp_1d=delta_clamp_1d,
                )
                if np.isfinite(combined):
                    val_scores.append(combined)
            if val_scores:
                val_srmse = float(np.mean(val_scores))

        # ── Validate (EMA model — stable) ─────────────────────────
        ema_val_srmse = float("inf")
        if val_graphs:
            ema.apply_shadow(model)
            ema_scores: List[float] = []
            for vg in val_graphs:
                _, _, combined = _validate_one_event(
                    model, vg, norm_stats, stds_1d, stds_2d,
                    device=device, spinup=10, delta_clamp=delta_clamp,
                    delta_clamp_1d=delta_clamp_1d,
                )
                if np.isfinite(combined):
                    ema_scores.append(combined)
            if ema_scores:
                ema_val_srmse = float(np.mean(ema_scores))
            ema.restore(model)

        elapsed = time.time() - t0

        # ── Record history ────────────────────────────────────────
        history["train_loss"].append(avg_loss)
        history["val_srmse"].append(val_srmse)
        history["ema_val_srmse"].append(ema_val_srmse)
        history["lr"].append(current_lr)
        history["tf_ratio"].append(tf_ratio)
        history["epoch_time"].append(elapsed)
        history["loss_1d"].append(avg_1d)
        history["loss_2d"].append(avg_2d)

        # ── Best checkpoint: EMA (primary) ──────────────────────────
        ema_is_best = ema_val_srmse < best_val
        if ema_is_best:
            best_val = ema_val_srmse
            best_epoch = epoch
            ckpt_path = os.path.join(
                checkpoint_dir, f"unified_model_{model_id}.pt"
            )
            ema.apply_shadow(model)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "norm_stats": norm_stats,
                    "val_srmse": best_val,
                    "checkpoint_type": "ema",
                    "model_id": model_id,
                    "hidden_channels": hidden_channels,
                    "num_gnn_layers": num_gnn_layers,
                    "in_channels_1d": dims["in_channels_1d"],
                    "in_channels_2d": dims["in_channels_2d"],
                },
                ckpt_path,
            )
            ema.restore(model)

        # ── Best checkpoint: raw val (dual save) ───────────────────
        #    When raw val beats EMA, save raw weights separately for inference.
        raw_is_best = val_srmse < best_val_raw
        if raw_is_best:
            best_val_raw = val_srmse
            best_epoch_raw = epoch
            ckpt_raw_path = os.path.join(
                checkpoint_dir, f"unified_model_{model_id}_best_val.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "norm_stats": norm_stats,
                    "val_srmse": best_val_raw,
                    "checkpoint_type": "raw_val",
                    "model_id": model_id,
                    "hidden_channels": hidden_channels,
                    "num_gnn_layers": num_gnn_layers,
                    "in_channels_1d": dims["in_channels_1d"],
                    "in_channels_2d": dims["in_channels_2d"],
                },
                ckpt_raw_path,
            )

        # ── Print progress ────────────────────────────────────────
        if verbose:
            star = " *" if ema_is_best else ""
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"loss={avg_loss:.4f} (1d={avg_1d:.4f} 2d={avg_2d:.4f}) | "
                f"val={val_srmse:.4f} ema={ema_val_srmse:.4f} | "
                f"K={effective_K} TF={tf_ratio:.2f} "
                f"LR={current_lr:.1e} | "
                f"{elapsed:.1f}s{star}"
            )

    print(f"\n  Best EMA val SRMSE: {best_val:.6f} (epoch {best_epoch})")
    print(f"  Best raw val SRMSE: {best_val_raw:.6f} (epoch {best_epoch_raw})")
    history["best_val"] = best_val  # type: ignore[assignment]
    history["best_epoch"] = best_epoch  # type: ignore[assignment]
    history["best_val_raw"] = best_val_raw  # type: ignore[assignment]
    history["best_epoch_raw"] = best_epoch_raw  # type: ignore[assignment]

    return history


# =====================================================================
#  Main Entry Point
# =====================================================================

def main() -> None:
    """Phase 1: Train Model 1 → Phase 2: Train Model 2."""
    parser = argparse.ArgumentParser(
        description="Dual-model training for UnifiedHeteroModel (v5)"
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden_channels", type=int, default=256,
                        help="EDA: lag-1 autocorr 0.9996 → larger GRU")
    parser.add_argument("--num_gnn_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="LR (EDA: high autocorr; Member B 0.005)")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--pushforward_K", type=int, default=20)
    parser.add_argument("--K_start", type=int, default=2)
    parser.add_argument("--K_ramp_epochs", type=int, default=40)
    parser.add_argument("--ar_noise_std", type=float, default=0.005)
    parser.add_argument("--delta_clamp", type=float, default=2.0)
    parser.add_argument("--delta_clamp_1d", type=float, default=5.0)
    parser.add_argument("--phys_weight", type=float, default=0.1)
    parser.add_argument("--val_events", type=str, default="3,9,15")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--model_ids", type=str, default="1,2",
        help="Comma-separated model IDs to train (default: '1,2')"
    )
    parser.add_argument(
        "--fold", type=int, default=-1,
        help="CV fold index (0-based). If >= 0, overrides --val_events with fold-based split."
    )
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of CV folds (default: 5)"
    )
    args = parser.parse_args()

    model_ids = [m.strip() for m in args.model_ids.split(",")]
    val_event_ids = [e.strip() for e in args.val_events.split(",")]

    # ── 5-fold CV: override val_events by fold index ────────────
    if args.fold >= 0:
        # Load dataset to discover event IDs per model
        _ds_tmp = FloodDataset(str(RAW_DATA_PATH), mode="train")
        # Use first model's events for fold splitting
        mid0 = model_ids[0]
        all_events = sorted(_ds_tmp.get_event_ids(model_id=mid0), key=int)
        n_per_fold = len(all_events) // args.n_folds
        fold_start = args.fold * n_per_fold
        if args.fold == args.n_folds - 1:
            fold_events = all_events[fold_start:]  # last fold gets remainder
        else:
            fold_events = all_events[fold_start:fold_start + n_per_fold]
        val_event_ids = fold_events
        print(f"  5-Fold CV     : fold {args.fold}/{args.n_folds}, val events: {val_event_ids}")
        del _ds_tmp

    print("=" * 60)
    print("  UNIFIED HETERO-MODEL TRAINING (v8 — Edge Features)")
    print("=" * 60)
    print(f"  Models         : {model_ids}")
    print(f"  Epochs         : {args.epochs}")
    print(f"  Hidden channels: {args.hidden_channels}")
    print(f"  Push-forward K : {args.K_start} → {args.pushforward_K}")
    print(f"  AR noise std   : {args.ar_noise_std}")
    print(f"  Delta max 2D   : ±{args.delta_clamp} ft/step")
    print(f"  Delta max 1D   : ±{args.delta_clamp_1d} ft/step")
    print(f"  Loss           : push_forward (min_std=0.01)")
    print(f"  1D depth norm  : per-node")
    print(f"  Val events     : {val_event_ids}")
    print(f"  Checkpoint dir : {args.checkpoint_dir}")
    print("=" * 60)

    # ── Load dataset ──────────────────────────────────────────────
    dataset = FloodDataset(str(RAW_DATA_PATH), mode="train")

    # ── Train each model with a fresh model (strict separation) ───
    all_histories: Dict[str, Dict] = {}

    for mid in model_ids:
        print(f"\n{'#'*60}")
        print(f"  PHASE: Model {mid}")
        print(f"{'#'*60}")

        history = train_model(
            model_id=mid,
            dataset=dataset,
            epochs=args.epochs,
            hidden_channels=args.hidden_channels,
            num_gnn_layers=args.num_gnn_layers,
            lr=args.lr,
            dropout=args.dropout,
            pushforward_K=args.pushforward_K,
            K_start=args.K_start,
            K_ramp_epochs=args.K_ramp_epochs,
            delta_clamp=args.delta_clamp,
            delta_clamp_1d=args.delta_clamp_1d,
            phys_weight=args.phys_weight,
            ar_noise_std=args.ar_noise_std,
            val_event_ids=val_event_ids,
            checkpoint_dir=args.checkpoint_dir,
            device_str=args.device,
        )
        all_histories[mid] = history

    # ── Save combined training history ────────────────────────────
    log_dir = Path(args.checkpoint_dir).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    history_path = log_dir / "unified_training_history.json"

    # Convert numpy types for JSON serialisation
    serialisable = {}
    for mid, hist in all_histories.items():
        serialisable[mid] = {
            k: [float(v) for v in vs] if isinstance(vs, list) else vs
            for k, vs in hist.items()
        }

    with open(history_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"{'='*60}")
    for mid, hist in all_histories.items():
        print(f"  Model {mid}: best EMA = {hist['best_val']:.6f} "
              f"(ep {hist['best_epoch']}), "
              f"best raw = {hist.get('best_val_raw', hist['best_val']):.6f} "
              f"(ep {hist.get('best_epoch_raw', hist['best_epoch'])})")
    print(f"  Checkpoints : {args.checkpoint_dir}/unified_model_*.pt, "
          f"*_best_val.pt")
    print(f"  History     : {history_path}")


if __name__ == "__main__":
    main()

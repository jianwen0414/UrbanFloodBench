"""
graph_builder_v2 — Physics-Informed HeteroGraph Builder (V2).

**DEPRECATED**: train_v2.py now uses graph_builder_unified.py instead.
This file is kept for reference only.

V2's feature engineering improvements (log-scale, RobustScaler, sinusoidal PE)
were found to be BROKEN because compute_model_stats() does not compute stats
for log-transformed keys (e.g., "log_capacity" doesn't exist in stats).
This caused 7+ features to be un-normalized, degrading SRMSE from 0.87 to 2.07.

See IMPROVEMENT_LOG.md, Run 7 for details.
"""

from __future__ import annotations

import warnings
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

# =====================================================================
#  Constants
# =====================================================================
_EPS: float = 1e-8

# Feature names for documentation/debugging
_1D_FEATURE_NAMES: List[str] = [
    "depth", "inlet_flow", "lag1", "lag2", "lag3",
    "log_capacity", "log_base_area",
    "log_pipe_diameter", "log_pipe_length", "pipe_roughness", "pipe_slope",
    "node_degree", "is_leaf",
]

_2D_FEATURE_NAMES: List[str] = [
    "depth", "rainfall", "lag1", "lag2", "lag3",
    "effective_depth",  # water_volume/area (no log, aligned with V1)
    "rain_rolling_mean", "rain_delta", "rain_lag2",
    "elevation", "min_elevation", "slope", 
    "log_area", "roughness", "aspect", "curvature", 
    "flow_accumulation", "elev_rel_neighbors", "dist_to_drain",
    "is_connected",
    # Sinusoidal PE (4 dims)
    "pos_sin_x", "pos_cos_x", "pos_sin_y", "pos_cos_y", 
]

# Dynamic indices
DEPTH_IDX: int = 0
LAG1_IDX: int = 2
LAG2_IDX: int = 3
LAG3_IDX: int = 4
LOG_EFF_DEPTH_IDX: int = 5 

# NOTE: This module is deprecated. See graph_builder_unified.py for the
# active graph builder used by train_v2.py.

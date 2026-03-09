import os
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

# ── Config ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path("c:/Users/Jian Wen Lee/UrbanFloodBench")
from src.config import RAW_DATA_PATH
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODEL_IDS = ["1", "2"]
SPINUP = 10
DELTA_CLAMP_2D = 2.0
DELTA_CLAMP_1D = 5.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Imports from the v7 pipeline ────────────────────────────────────
from src.dataset import FloodDataset
from src.graph_builder_unified import (
    build_hetero_graph,
    DEPTH_IDX, WATER_VOL_IDX, LAG1_IDX, LAG2_IDX, LAG3_IDX,
)
from src.model_unified import UnifiedHeteroModel
from src.train_unified import compute_model_stats

print(f"Checkpoint dir : {CHECKPOINT_DIR}")
print(f"Models         : {MODEL_IDS}")
print(f"Spinup         : {SPINUP}")
print(f"Delta clamp    : 1D=±{DELTA_CLAMP_1D}, 2D=±{DELTA_CLAMP_2D}")


# =====================================================================
#  AR Inference (mirrors _validate_one_event from train_unified.py)
# =====================================================================

@torch.no_grad()
def predict_event_v7(
    model: UnifiedHeteroModel,
    graph: Any,
    norm_stats: Dict,
    stds_1d: Tensor,
    stds_2d: Tensor,
    device: torch.device,
    spinup: int = 10,
    delta_clamp: float = 2.0,
    delta_clamp_1d: float = 5.0,
) -> Tuple[Tensor, Tensor]:
    """Full AR inference on one event → (pred_wse_1d, pred_wse_2d).

    Returns [T_scored, N_1d] and [T_scored, N_2d] tensors of
    absolute water surface elevation (WSE) for the scored period
    (after spinup).
    """
    graph = graph.to(device)
    model.eval()

    T = graph.num_timesteps
    n_1d = graph["1d"].num_nodes
    n_2d = graph["2d"].num_nodes
    spinup = min(spinup, T - 1)

    # Edge indices
    edge_index_dict = {}
    for et in graph.edge_types:
        edge_index_dict[et] = graph[et].edge_index

    # 2D global depth norm stats
    d2_mean = norm_stats["2d"]["depth"]["mean"]
    d2_std = norm_stats["2d"]["depth"]["std"]

    # Per-node 1D depth stats
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

    # Constant-node masks
    const_mask_1d = (stds_1d.to(device) >= 0.01).float()
    const_mask_2d = (stds_2d.to(device) >= 0.01).float()

    # ── Spinup: warm GRU hidden states with GT features ──────────
    hidden = model.init_hidden(n_1d, n_2d, device)
    for t in range(spinup):
        x_dict = {
            "1d": graph["1d"].x[t].to(device),
            "2d": graph["2d"].x[t].to(device),
        }
        _, hidden = model(x_dict, edge_index_dict, hidden)

    # ── Prediction phase (fully autoregressive) ──────────────────
    prev_depth_1d = graph["1d"].depth[spinup - 1].to(device)
    prev_depth_2d = graph["2d"].depth[spinup - 1].to(device)

    # v9+ models: no edge flow features (removed due to data leakage)
    # Legacy v8 compat: if checkpoint has >11/22 features, freeze last 3 from spinup
    has_edge_flows_1d = graph["1d"].x.size(-1) > 11
    has_edge_flows_2d = graph["2d"].x.size(-1) > 22
    if has_edge_flows_1d:
        last_edge_flows_1d = graph["1d"].x[spinup - 1, :, -3:].to(device)
    if has_edge_flows_2d:
        last_edge_flows_2d = graph["2d"].x[spinup - 1, :, -3:].to(device)

    # Build depth history for lag replacement
    _hist_1d, _hist_2d = [], []
    for _t in range(max(0, spinup - 4), spinup):
        _hist_1d.append(graph["1d"].depth[_t].to(device))
        _hist_2d.append(graph["2d"].depth[_t].to(device))
    while len(_hist_1d) < 4:
        _hist_1d.insert(0, _hist_1d[0].clone())
        _hist_2d.insert(0, _hist_2d[0].clone())

    def _get_lag(hist, n):
        idx = len(hist) - n
        return hist[max(0, idx)]

    all_pred_wse_1d = []
    all_pred_wse_2d = []

    for t in range(spinup, T):
        x_1d_t = graph["1d"].x[t].clone().to(device)
        x_2d_t = graph["2d"].x[t].clone().to(device)

        # Replace depth + lag features with AR predictions
        x_1d_t[:, DEPTH_IDX] = (prev_depth_1d - _pn_mean_1d) / _pn_std_1d
        x_2d_t[:, DEPTH_IDX] = (prev_depth_2d - d2_mean) / max(d2_std, 1e-8)
        # Index 5 during AR: effective_depth (P0) or water_volume (pre-P0 checkpoint compat)
        if x_2d_t.size(-1) > WATER_VOL_IDX:
            if "effective_depth" in norm_stats["2d"]:
                st = norm_stats["2d"]["effective_depth"]
                x_2d_t[:, WATER_VOL_IDX] = (prev_depth_2d - st["mean"]) / max(st["std"], 1e-8)
            elif hasattr(graph["2d"], "area"):
                st = norm_stats["2d"].get("water_volume", {"mean": 0.0, "std": 1.0})
                vol_proxy = prev_depth_2d * graph["2d"].area.to(device)
                x_2d_t[:, WATER_VOL_IDX] = (vol_proxy - st["mean"]) / max(st["std"], 1e-8)
        x_1d_t[:, LAG1_IDX] = (_get_lag(_hist_1d, 2) - _pn_mean_1d) / _pn_std_1d
        x_1d_t[:, LAG2_IDX] = (_get_lag(_hist_1d, 3) - _pn_mean_1d) / _pn_std_1d
        x_1d_t[:, LAG3_IDX] = (_get_lag(_hist_1d, 4) - _pn_mean_1d) / _pn_std_1d
        x_2d_t[:, LAG1_IDX] = (_get_lag(_hist_2d, 2) - d2_mean) / max(d2_std, 1e-8)
        x_2d_t[:, LAG2_IDX] = (_get_lag(_hist_2d, 3) - d2_mean) / max(d2_std, 1e-8)
        x_2d_t[:, LAG3_IDX] = (_get_lag(_hist_2d, 4) - d2_mean) / max(d2_std, 1e-8)

        if has_edge_flows_1d:
            x_1d_t[:, -3:] = last_edge_flows_1d
        if has_edge_flows_2d:
            x_2d_t[:, -3:] = last_edge_flows_2d

        x_dict = {"1d": x_1d_t, "2d": x_2d_t}
        delta_dict, hidden = model(x_dict, edge_index_dict, hidden)

        raw_delta_1d = delta_dict["1d"].clamp(-delta_clamp_1d, delta_clamp_1d) * const_mask_1d
        raw_delta_2d = delta_dict["2d"].clamp(-delta_clamp, delta_clamp) * const_mask_2d

        # Run 13: Physics floor for 2D only (1D naturally negative)
        pred_depth_1d = prev_depth_1d + raw_delta_1d
        pred_depth_2d = (prev_depth_2d + raw_delta_2d).clamp(min=0.0)

        # WSE = depth + elevation reference
        pred_wse_1d = pred_depth_1d + graph["1d"].elev.to(device)
        pred_wse_2d = pred_depth_2d + graph["2d"].elev.to(device)

        all_pred_wse_1d.append(pred_wse_1d)
        all_pred_wse_2d.append(pred_wse_2d)

        # Update history
        _hist_1d.append(prev_depth_1d)
        _hist_2d.append(prev_depth_2d)
        prev_depth_1d = pred_depth_1d
        prev_depth_2d = pred_depth_2d

        if torch.isnan(pred_depth_1d).any() or torch.isnan(pred_depth_2d).any():
            warnings.warn(f"NaN at t={t} — truncating.")
            break

    preds_1d = torch.stack(all_pred_wse_1d)  # [T_scored, N_1d]
    preds_2d = torch.stack(all_pred_wse_2d)  # [T_scored, N_2d]
    return preds_1d, preds_2d


# =====================================================================
#  Build submission rows (vectorized)
# =====================================================================

def build_rows(preds: Tensor, model_id: str, event_id: str,
               node_type: int, spinup: int) -> pd.DataFrame:
    preds_np = preds.cpu().numpy()
    T_scored, N = preds_np.shape

    nan_mask = np.isnan(preds_np)
    if nan_mask.any():
        for t_idx in range(1, T_scored):
            fill = nan_mask[t_idx]
            preds_np[t_idx, fill] = preds_np[t_idx - 1, fill]
        preds_np = np.nan_to_num(preds_np, nan=0.0)

    preds_node_major = preds_np.T  # [N, T_scored]
    wl_flat = preds_node_major.ravel()

    node_ids = np.arange(N)
    n_flat = np.repeat(node_ids, T_scored)

    return pd.DataFrame({
        "row_id": 0,  # placeholder, replaced after concat
        "model_id": int(model_id),
        "event_id": int(event_id),
        "node_type": int(node_type),
        "node_id": n_flat.astype(int),
        "water_level": wl_flat.astype(np.float64),
    })


# =====================================================================
#  Main Generation Loop
# =====================================================================

def main():
    t_start = time.time()
    test_ds = FloodDataset(str(RAW_DATA_PATH), mode="test")
    train_ds = FloodDataset(str(RAW_DATA_PATH), mode="train")

    print(f"\\nTest events: {len(test_ds)}")

    all_dfs = []

    for mid in MODEL_IDS:
        ckpt_path = CHECKPOINT_DIR / f"unified_model_{mid}.pt"  # EMA
        if not ckpt_path.exists():
            ckpt_path = CHECKPOINT_DIR / f"unified_model_{mid}_best_val.pt"
        if not ckpt_path.exists():
            print(f"  [SKIP] {ckpt_path} not found")
            continue

        print(f"\\n{'='*50}")
        print(f"  Model {mid}: loading {ckpt_path.name}")

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        norm_stats = ckpt["norm_stats"]

        model = UnifiedHeteroModel(
            in_channels_1d=ckpt["in_channels_1d"],
            in_channels_2d=ckpt["in_channels_2d"],
            hidden_channels=ckpt["hidden_channels"],
            num_gnn_layers=ckpt["num_gnn_layers"],
        ).to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"  Loaded epoch {ckpt.get('epoch', 'N/A')}, val SRMSE = {ckpt.get('val_srmse', 0.0):.4f}")

        node_stds = train_ds.compute_node_stds(model_id=mid)
        stds_1d = torch.tensor(node_stds[mid]["1d"], dtype=torch.float32)
        stds_2d = torch.tensor(node_stds[mid]["2d"], dtype=torch.float32)
        print(f"  Node stds: 1D={len(stds_1d)}, 2D={len(stds_2d)}")

        test_events = test_ds.filter_by_model(mid)
        print(f"  Test events for Model {mid}: {len(test_events)}")

        for idx in range(len(test_events)):
            sample = test_events[idx]
            event_id = sample["event_id"]

            try:
                graph = build_hetero_graph(sample, norm_stats)

                need_1d = ckpt["in_channels_1d"] - graph["1d"].x.size(-1)
                need_2d = ckpt["in_channels_2d"] - graph["2d"].x.size(-1)
                
                if need_1d > 0:
                    pad = torch.zeros(graph["1d"].x.shape[0], graph["1d"].x.shape[1], need_1d, dtype=graph["1d"].x.dtype, device=graph["1d"].x.device)
                    graph["1d"].x = torch.cat([graph["1d"].x, pad], dim=-1)
                elif need_1d < 0:
                    graph["1d"].x = graph["1d"].x[..., :ckpt["in_channels_1d"]]

                if need_2d > 0:
                    pad = torch.zeros(graph["2d"].x.shape[0], graph["2d"].x.shape[1], need_2d, dtype=graph["2d"].x.dtype, device=graph["2d"].x.device)
                    graph["2d"].x = torch.cat([graph["2d"].x, pad], dim=-1)
                elif need_2d < 0:
                    graph["2d"].x = graph["2d"].x[..., :ckpt["in_channels_2d"]]

                preds_1d, preds_2d = predict_event_v7(
                    model, graph, norm_stats, stds_1d, stds_2d,
                    DEVICE, spinup=SPINUP,
                    delta_clamp=DELTA_CLAMP_2D, delta_clamp_1d=DELTA_CLAMP_1D,
                )

                df_1d = build_rows(preds_1d, mid, event_id, 1, SPINUP)
                df_2d = build_rows(preds_2d, mid, event_id, 2, SPINUP)
                all_dfs.append(df_1d)
                all_dfs.append(df_2d)

                print(f"    Event {event_id}: 1D={preds_1d.shape}, 2D={preds_2d.shape}")

            except Exception as e:
                warnings.warn(f"  Failed Model_{mid} Event_{event_id}: {e}")
                import traceback; traceback.print_exc()
                continue

    submission = pd.concat(all_dfs, ignore_index=True)
    submission["row_id"] = np.arange(len(submission), dtype=np.int64)

    nan_count = submission["water_level"].isna().sum()
    print(f"\\n{'='*50}")
    print(f"  Submission shape   : {submission.shape}")
    print(f"  NaN water_level    : {nan_count}")
    print(f"  row_id range       : [{submission['row_id'].min()}, {submission['row_id'].max()}]")
    print(f"  Models             : {sorted(submission['model_id'].unique())}")
    print(f"  Events             : {sorted(submission['event_id'].unique())}")
    print(f"  node_type values   : {sorted(submission['node_type'].unique())}")
    print(f"  Water level range  : [{submission['water_level'].min():.2f}, "
          f"{submission['water_level'].max():.2f}]")

    submission = submission[["row_id", "model_id", "event_id",
                             "node_type", "node_id", "water_level"]]

    out_path = PROJECT_ROOT / "submission.csv"
    submission.to_csv(out_path, index=False)
    elapsed = time.time() - t_start
    print(f"\\n  Saved to {out_path} ({submission.shape[0]:,} rows)")
    print(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

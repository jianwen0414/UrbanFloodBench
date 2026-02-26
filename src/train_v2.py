"""
train_v2 — V2 Training Pipeline (V1 Graph Builder + V2 Training Loop).

Uses V1's proven graph_builder_unified.py for correct normalization.
Training loop includes V1's proven physics:
- SAGEConv architecture (memory-efficient)
- const_mask for dry node suppression
- min_std=0.01 for loss weighting
- temporal_scheme='linear' for push_forward_loss
- spinup=10 for GRU warm-up
- ar_noise_std=0.005 for regularization
"""

from __future__ import annotations

import sys
import torch
import numpy as np
import warnings
import torch.optim as optim
import time
import copy
import json
import os
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
from torch import Tensor

# Ensure project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Imports
from src.dataset import FloodDataset
from src.graph_builder_unified import (
    build_hetero_graph, DEPTH_IDX, LAG1_IDX, LAG2_IDX, LAG3_IDX,
    EFFECTIVE_DEPTH_IDX
)
from src.model_v2 import UnifiedGATModel
from src.loss import push_forward_loss, standardized_rmse_metric
from src.train_unified import compute_model_stats, TeacherForcingScheduler, EMAModel
from src.config import RAW_DATA_PATH


# =====================================================================
#  Training: Single Event
# =====================================================================

def _train_one_event_v2(
    model: UnifiedGATModel,
    graph,
    norm_stats,
    stds_1d, stds_2d,
    K: int,
    tf_ratio: float,
    spinup: int,
    device,
    delta_clamp_1d: float = 5.0,
    delta_clamp_2d: float = 2.0,
    ar_noise_std: float = 0.005,
) -> Tuple[torch.Tensor, Dict]:
    """Train on a single event with push-forward delta-prediction loss.
    
    Aligned with V1's proven physics: const_mask, min_std, no aux_loss.
    """
    graph = graph.to(device)
    model.train()
    
    T = graph.num_timesteps
    n_1d = graph["1d"].num_nodes
    n_2d = graph["2d"].num_nodes
    
    spinup = min(spinup, T - 1)
    K = min(K, T - spinup)
    
    if K <= 0:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {"total": 0.0, "loss_1d": 0.0, "loss_2d": 0.0}

    edge_index_dict = {k: graph[k].edge_index for k in graph.edge_types}
    
    # ── Per-node 1D depth normalization (V2 improvement over V1) ──
    s2_depth = norm_stats["2d"]["depth"]
    d2_mean = s2_depth["mean"]
    d2_std = s2_depth["std"]
    
    if "depth_per_node_mean" in norm_stats["1d"]:
        pn_mean_1d = torch.tensor(
            norm_stats["1d"]["depth_per_node_mean"], dtype=torch.float32
        ).to(device)
        pn_std_1d = torch.tensor(
            norm_stats["1d"]["depth_per_node_std"], dtype=torch.float32
        ).to(device)
    else:
        d1_mean = norm_stats["1d"]["depth"]["mean"]
        d1_std = norm_stats["1d"]["depth"]["std"]
        pn_mean_1d = torch.full((n_1d,), d1_mean, device=device)
        pn_std_1d = torch.full((n_1d,), max(d1_std, 1e-4), device=device)
    
    # ── Constant-node masks (V1 proven — zero delta on dry nodes) ──
    _stds_1d = stds_1d.to(device)
    _stds_2d = stds_2d.to(device)
    const_mask_1d = (_stds_1d >= 0.01).float()
    const_mask_2d = (_stds_2d >= 0.01).float()
    
    # ── Spinup: warm GRU hidden states with GT features ──
    hidden = model.init_hidden(n_1d, n_2d, device)
    with torch.no_grad():
        for t in range(spinup):
            x = {"1d": graph["1d"].x[t], "2d": graph["2d"].x[t]}
            _, hidden = model(x, edge_index_dict, hidden)
    hidden = {k: v.detach() for k, v in hidden.items()}
    
    # ── Build depth history for lag replacement ──
    prev_d1 = graph["1d"].depth[spinup - 1]
    prev_d2 = graph["2d"].depth[spinup - 1]
    
    _hist_1d: List[Tensor] = []
    _hist_2d: List[Tensor] = []
    for _t in range(max(0, spinup - 4), spinup):
        _hist_1d.append(graph["1d"].depth[_t].clone())
        _hist_2d.append(graph["2d"].depth[_t].clone())
    while len(_hist_1d) < 4:
        _hist_1d.insert(0, _hist_1d[0].clone())
        _hist_2d.insert(0, _hist_2d[0].clone())
    
    def _get_lag(hist: List[Tensor], n: int) -> Tensor:
        idx = len(hist) - n
        return hist[max(0, idx)]
    
    preds_1d_list = []
    preds_2d_list = []
    targets_1d_list = []
    targets_2d_list = []
    
    for k in range(K):
        t = spinup + k
        
        # ── Teacher forcing decision ──
        use_tf = (
            k == 0
            or (model.training and tf_ratio > 0.0 and torch.rand(1).item() < tf_ratio)
        )
        
        x1 = graph["1d"].x[t].clone()
        x2 = graph["2d"].x[t].clone()
        
        if use_tf and k > 0:
            prev_d1 = graph["1d"].depth[t - 1]
            prev_d2 = graph["2d"].depth[t - 1]
        
        # ── Replace depth features with AR predictions ──
        norm_d1 = (prev_d1 - pn_mean_1d) / pn_std_1d
        norm_d2 = (prev_d2 - d2_mean) / max(d2_std, 1e-8)
        
        # AR noise injection (V1 proven regularization)
        if not use_tf and ar_noise_std > 0 and model.training:
            norm_d1 = norm_d1 + torch.randn_like(norm_d1) * ar_noise_std
            norm_d2 = norm_d2 + torch.randn_like(norm_d2) * ar_noise_std
        
        x1[:, DEPTH_IDX] = norm_d1
        x2[:, DEPTH_IDX] = norm_d2
        
        # ── Replace lag features ──
        x1[:, LAG1_IDX] = (_get_lag(_hist_1d, 2) - pn_mean_1d) / pn_std_1d
        x1[:, LAG2_IDX] = (_get_lag(_hist_1d, 3) - pn_mean_1d) / pn_std_1d
        x1[:, LAG3_IDX] = (_get_lag(_hist_1d, 4) - pn_mean_1d) / pn_std_1d
        x2[:, LAG1_IDX] = (_get_lag(_hist_2d, 2) - d2_mean) / max(d2_std, 1e-8)
        x2[:, LAG2_IDX] = (_get_lag(_hist_2d, 3) - d2_mean) / max(d2_std, 1e-8)
        x2[:, LAG3_IDX] = (_get_lag(_hist_2d, 4) - d2_mean) / max(d2_std, 1e-8)
        
        # ── Replace effective_depth (index 5) during AR ──
        if x2.size(-1) > EFFECTIVE_DEPTH_IDX:
            st = norm_stats["2d"].get("effective_depth", norm_stats["2d"].get("depth", {}))
            eff_mean = st.get("mean", 0.0)
            eff_std = max(st.get("std", 1.0), 1e-8)
            x2[:, EFFECTIVE_DEPTH_IDX] = (prev_d2 - eff_mean) / eff_std
        
        # ── Forward pass ──
        delta, hidden = model({"1d": x1, "2d": x2}, edge_index_dict, hidden)
        
        # ── Hard clamp + const mask (V1 proven) ──
        delta_1d = delta["1d"].clamp(-delta_clamp_1d, delta_clamp_1d) * const_mask_1d
        delta_2d = delta["2d"].clamp(-delta_clamp_2d, delta_clamp_2d) * const_mask_2d
        
        # ── State recovery ──
        pred_d1 = prev_d1 + delta_1d
        pred_d2 = (prev_d2 + delta_2d).clamp(min=0.0)  # V1: .clamp(min=0.0), not relu
        
        pred_wse1 = pred_d1 + graph["1d"].elev
        pred_wse2 = pred_d2 + graph["2d"].elev
        
        preds_1d_list.append(pred_wse1)
        preds_2d_list.append(pred_wse2)
        targets_1d_list.append(graph["1d"].y[t])
        targets_2d_list.append(graph["2d"].y[t])
        
        # ── Update history ──
        _hist_1d.append(prev_d1)
        _hist_2d.append(prev_d2)
        prev_d1 = pred_d1 if not use_tf else graph["1d"].depth[t]
        prev_d2 = pred_d2 if not use_tf else graph["2d"].depth[t]
        
        # Early NaN detection
        if torch.isnan(pred_d1).any() or torch.isnan(pred_d2).any():
            warnings.warn(f"NaN detected at k={k}, t={t} — truncating rollout.")
            break
    
    if not preds_1d_list:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, {"total": 0.0, "loss_1d": 0.0, "loss_2d": 0.0}
    
    preds_1d = torch.stack(preds_1d_list)
    targets_1d = torch.stack(targets_1d_list)
    preds_2d = torch.stack(preds_2d_list)
    targets_2d = torch.stack(targets_2d_list)
    
    # ── Push-forward loss with V1's proven settings ──
    l_pf_1d = push_forward_loss(
        preds_1d.float(), targets_1d.float(),
        stds_1d.to(device),
        temporal_scheme="linear",
        min_std=0.01,
    )
    l_pf_2d = push_forward_loss(
        preds_2d.float(), targets_2d.float(),
        stds_2d.to(device),
        temporal_scheme="linear",
        min_std=0.01,
    )
    
    total = 0.5 * l_pf_1d + 0.5 * l_pf_2d
    
    return total, {"total": total.item(), "loss_1d": l_pf_1d.item(), "loss_2d": l_pf_2d.item()}


# =====================================================================
#  Validation: Single Event (Fully Autoregressive)
# =====================================================================

@torch.no_grad()
def _validate_one_event_v2(model, graph, norm_stats, stds_1d, stds_2d, device,
                           spinup: int = 10,
                           delta_clamp_1d: float = 5.0,
                           delta_clamp_2d: float = 2.0):
    """Validate on a single event with fully autoregressive rollout."""
    graph = graph.to(device)
    model.eval()
    T = graph.num_timesteps
    n_1d = graph["1d"].num_nodes
    n_2d = graph["2d"].num_nodes
    
    spinup = min(spinup, T - 1)
    if T <= spinup:
        return float("inf")
    
    edge_index_dict = {k: graph[k].edge_index for k in graph.edge_types}
    hidden = model.init_hidden(n_1d, n_2d, device)
    
    # ── Per-node 1D depth normalization ──
    s2_depth = norm_stats["2d"]["depth"]
    d2_mean = s2_depth["mean"]
    d2_std = s2_depth["std"]
    
    if "depth_per_node_mean" in norm_stats["1d"]:
        pn_mean_1d = torch.tensor(
            norm_stats["1d"]["depth_per_node_mean"], dtype=torch.float32
        ).to(device)
        pn_std_1d = torch.tensor(
            norm_stats["1d"]["depth_per_node_std"], dtype=torch.float32
        ).to(device)
    else:
        d1_mean = norm_stats["1d"]["depth"]["mean"]
        d1_std = norm_stats["1d"]["depth"]["std"]
        pn_mean_1d = torch.full((n_1d,), d1_mean, device=device)
        pn_std_1d = torch.full((n_1d,), max(d1_std, 1e-4), device=device)
    
    # ── Constant-node masks ──
    const_mask_1d = (stds_1d.to(device) >= 0.01).float()
    const_mask_2d = (stds_2d.to(device) >= 0.01).float()
    
    # ── Spinup ──
    for t in range(spinup):
        x = {"1d": graph["1d"].x[t].to(device), "2d": graph["2d"].x[t].to(device)}
        _, hidden = model(x, edge_index_dict, hidden)
    
    prev_d1 = graph["1d"].depth[spinup - 1].to(device)
    prev_d2 = graph["2d"].depth[spinup - 1].to(device)
    
    # ── Build depth history ──
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
    
    preds_1d_list = []
    preds_2d_list = []
    targets_1d_list = []
    targets_2d_list = []
    
    for t in range(spinup, T):
        x1 = graph["1d"].x[t].clone().to(device)
        x2 = graph["2d"].x[t].clone().to(device)
        
        # ── Replace depth + lag features ──
        x1[:, DEPTH_IDX] = (prev_d1 - pn_mean_1d) / pn_std_1d
        x2[:, DEPTH_IDX] = (prev_d2 - d2_mean) / max(d2_std, 1e-8)
        
        if x2.size(-1) > EFFECTIVE_DEPTH_IDX:
            st = norm_stats["2d"].get("effective_depth", norm_stats["2d"].get("depth", {}))
            eff_mean = st.get("mean", 0.0)
            eff_std = max(st.get("std", 1.0), 1e-8)
            x2[:, EFFECTIVE_DEPTH_IDX] = (prev_d2 - eff_mean) / eff_std
        
        x1[:, LAG1_IDX] = (_get_lag(_hist_1d, 2) - pn_mean_1d) / pn_std_1d
        x1[:, LAG2_IDX] = (_get_lag(_hist_1d, 3) - pn_mean_1d) / pn_std_1d
        x1[:, LAG3_IDX] = (_get_lag(_hist_1d, 4) - pn_mean_1d) / pn_std_1d
        x2[:, LAG1_IDX] = (_get_lag(_hist_2d, 2) - d2_mean) / max(d2_std, 1e-8)
        x2[:, LAG2_IDX] = (_get_lag(_hist_2d, 3) - d2_mean) / max(d2_std, 1e-8)
        x2[:, LAG3_IDX] = (_get_lag(_hist_2d, 4) - d2_mean) / max(d2_std, 1e-8)
        
        # ── Forward pass ──
        delta_dict, hidden = model({"1d": x1, "2d": x2}, edge_index_dict, hidden)
        
        # ── Hard clamp + const mask ──
        delta_1d = delta_dict["1d"].clamp(-delta_clamp_1d, delta_clamp_1d) * const_mask_1d
        delta_2d = delta_dict["2d"].clamp(-delta_clamp_2d, delta_clamp_2d) * const_mask_2d
        
        pred_d1 = prev_d1 + delta_1d
        pred_d2 = (prev_d2 + delta_2d).clamp(min=0.0)
        
        preds_1d_list.append(pred_d1 + graph["1d"].elev.to(device))
        preds_2d_list.append(pred_d2 + graph["2d"].elev.to(device))
        targets_1d_list.append(graph["1d"].y[t].to(device))
        targets_2d_list.append(graph["2d"].y[t].to(device))
        
        # ── Update history ──
        _hist_1d.append(prev_d1)
        _hist_2d.append(prev_d2)
        prev_d1 = pred_d1
        prev_d2 = pred_d2
        
        if torch.isnan(pred_d1).any() or torch.isnan(pred_d2).any():
            break
    
    if not preds_1d_list:
        return float("inf")
    
    preds_1d = torch.stack(preds_1d_list)
    preds_2d = torch.stack(preds_2d_list)
    targets_1d = torch.stack(targets_1d_list)
    targets_2d = torch.stack(targets_2d_list)
    
    srmse_1d = standardized_rmse_metric(preds_1d, targets_1d, stds_1d.to(device)).item()
    srmse_2d = standardized_rmse_metric(preds_2d, targets_2d, stds_2d.to(device)).item()
    
    return (srmse_1d + srmse_2d) / 2.0


# =====================================================================
#  Main Training Loop
# =====================================================================

def train_model_v2(
    model_id: str,
    dataset: FloodDataset,
    epochs: int = 100,
    lr: float = 0.001,
    hidden_channels: int = 128,
    num_gnn_layers: int = 4,
    heads: int = 4,
    device_str: str = "auto",
    checkpoint_dir: str = "checkpoints_v2",
    use_amp: bool = False
):
    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
        
    print(f"Training Model {model_id} on {device} (AMP={use_amp})...")
    print(f"Config: H={hidden_channels}, L={num_gnn_layers}")
    
    # 1. Stats
    norm_stats = compute_model_stats(dataset, model_id)
    node_stds = dataset.compute_node_stds(model_id=model_id)
    stds_1d = torch.tensor(node_stds[model_id]["1d"], dtype=torch.float32)
    stds_2d = torch.tensor(node_stds[model_id]["2d"], dtype=torch.float32)
    
    # 2. Graphs
    val_event_ids = ["3", "9", "15"]
    model_ds = dataset.filter_by_model(model_id)
    
    train_graphs = []
    val_graphs = []
    
    for i in range(len(model_ds)):
        sample = model_ds[i]
        g = build_hetero_graph(sample, norm_stats)
        if sample["event_id"] in val_event_ids:
            val_graphs.append(g)
        else:
            train_graphs.append(g)
            
    print(f"  Train events: {len(train_graphs)}, Val events: {len(val_graphs)}")
    print(f"  1D features: {train_graphs[0]['1d'].x.size(-1)}, 2D features: {train_graphs[0]['2d'].x.size(-1)}")
    
    # 3. Model (SAGEConv — memory efficient)
    model = UnifiedGATModel(
        in_channels_1d=train_graphs[0]["1d"].x.size(-1),
        in_channels_2d=train_graphs[0]["2d"].x.size(-1),
        hidden_channels=hidden_channels,
        num_gnn_layers=num_gnn_layers
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * len(train_graphs)
    )
    ema = EMAModel(model, decay=0.999)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    tf_scheduler = TeacherForcingScheduler(warmup_epochs=5, decay_epochs=40, min_ratio=0.1)
    
    # K-Schedule (same as V1)
    K_max = 15
    spinup = 10  # V1 proven: 10 steps warm-up
    
    best_val = float("inf")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        K = min(K_max, 2 + epoch // 2)
        tf_ratio = tf_scheduler(epoch)
        
        losses = []
        np.random.shuffle(train_graphs)
        
        for g in train_graphs:
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, bkd = _train_one_event_v2(
                    model, g, norm_stats, stds_1d, stds_2d,
                    K=K, tf_ratio=tf_ratio, spinup=spinup, device=device
                )
            
            if not torch.isnan(loss):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)
                scheduler.step()
                losses.append(loss.item())
            else:
                print("Warning: NaN loss explicitly detected (skipped step)")
                
        # Val
        val_scores = []
        ema.apply_shadow(model)
        for vg in val_graphs:
            score = _validate_one_event_v2(
                model, vg, norm_stats, stds_1d, stds_2d, device,
                spinup=spinup
            )
            val_scores.append(score)
        ema.restore(model)
        
        avg_val = np.mean(val_scores) if val_scores else float("inf")
        avg_loss = np.mean(losses) if losses else 0
        
        print(f"Ep {epoch} | Loss {avg_loss:.4f} | Val {avg_val:.4f} | K={K} | TF={tf_ratio:.2f}")
        
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), f"{checkpoint_dir}/model_{model_id}_best.pt")
            print(f"  --> Best Val! Saved.")
            
    print(f"Done. Best Val: {best_val}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ids", type=str, default="2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_gnn_layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Enable Mixed Precision")
    args = parser.parse_args()
    
    ds = FloodDataset(str(RAW_DATA_PATH), mode="train")
    
    for mid in args.model_ids.split(","):
        train_model_v2(
            mid, ds, 
            epochs=args.epochs,
            hidden_channels=args.hidden_channels,
            num_gnn_layers=args.num_gnn_layers,
            heads=args.heads,
            use_amp=args.amp
        )

if __name__ == "__main__":
    main()

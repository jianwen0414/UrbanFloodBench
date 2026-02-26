"""
Deep diagnostic probe — find exactly WHY loss=892 is frozen.

Tests:
  1. Graph integrity: features, depth, WSE, edges
  2. Node stds & loss weights
  3. Persistence baseline (true loss, not just SRMSE metric)
  4. Model forward pass: delta magnitudes
  5. Gradient flow: does the loss gradient reach model params?
  6. Single-step overfit: can the model reduce loss at all?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import FloodDataset
from src.graph_builder_unified import build_hetero_graph
from src.model_unified import UnifiedHeteroModel
from src.loss import push_forward_loss, standardized_rmse_loss
from src.train_unified import compute_model_stats

torch.set_printoptions(precision=6, sci_mode=True)
np.set_printoptions(precision=6, suppress=False)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}\n")

# ──────────────────────────────────────────────────────────────────
# 1. Load dataset & build graph
# ──────────────────────────────────────────────────────────────────
print("=" * 70)
print("  SECTION 1: DATA LOADING")
print("=" * 70)

ds = FloodDataset(root_dir="data", mode="train")
model_id = "1"
norm_stats = compute_model_stats(ds, model_id)

model_ds = ds.filter_by_model(model_id)
sample = model_ds[0]
print(f"\nEvent: model={sample['model_id']}, event={sample['event_id']}")

graph = build_hetero_graph(sample, norm_stats)
T = graph.num_timesteps
n_1d = graph["1d"].num_nodes
n_2d = graph["2d"].num_nodes
print(f"T={T}, N_1d={n_1d}, N_2d={n_2d}")

# ──────────────────────────────────────────────────────────────────
# 2. Inspect raw data fields
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 2: RAW DATA INTEGRITY")
print("=" * 70)

depth_1d = graph["1d"].depth  # [T, N_1d]
depth_2d = graph["2d"].depth  # [T, N_2d]
wse_1d = graph["1d"].y        # [T, N_1d]
wse_2d = graph["2d"].y        # [T, N_2d]
elev_1d = graph["1d"].elev    # [N_1d]
elev_2d = graph["2d"].elev    # [N_2d]

print(f"\n1D elevation  : min={elev_1d.min():.2f}, max={elev_1d.max():.2f}, mean={elev_1d.mean():.2f}")
print(f"2D elevation  : min={elev_2d.min():.2f}, max={elev_2d.max():.2f}, mean={elev_2d.mean():.2f}")

print(f"\n1D depth [T,N]: min={depth_1d.min():.4f}, max={depth_1d.max():.4f}, mean={depth_1d.mean():.4f}")
print(f"2D depth [T,N]: min={depth_2d.min():.4f}, max={depth_2d.max():.4f}, mean={depth_2d.mean():.4f}")
print(f"  1D negative depths: {(depth_1d < 0).sum().item()}/{depth_1d.numel()}")
print(f"  2D negative depths: {(depth_2d < 0).sum().item()}/{depth_2d.numel()}")

print(f"\n1D WSE [T,N]  : min={wse_1d.min():.2f}, max={wse_1d.max():.2f}, mean={wse_1d.mean():.2f}")
print(f"2D WSE [T,N]  : min={wse_2d.min():.2f}, max={wse_2d.max():.2f}, mean={wse_2d.mean():.2f}")

# Verify: WSE = depth + elevation
recon_1d = depth_1d + elev_1d.unsqueeze(0)
recon_2d = depth_2d + elev_2d.unsqueeze(0)
max_err_1d = (recon_1d - wse_1d).abs().max().item()
max_err_2d = (recon_2d - wse_2d).abs().max().item()
print(f"\nWSE reconstruction check (should be ~0):")
print(f"  1D max |depth+elev - WSE| = {max_err_1d:.6e}")
print(f"  2D max |depth+elev - WSE| = {max_err_2d:.6e}")

# Check features
x_1d = graph["1d"].x  # [T, N_1d, 3]
x_2d = graph["2d"].x  # [T, N_2d, 3]
print(f"\n1D features [T,N,3]: shape={list(x_1d.shape)}")
for f in range(3):
    v = x_1d[:, :, f]
    print(f"  feat {f}: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}, std={v.std():.4f}")
print(f"  NaN: {torch.isnan(x_1d).sum().item()}, Inf: {torch.isinf(x_1d).sum().item()}")

print(f"\n2D features [T,N,3]: shape={list(x_2d.shape)}")
for f in range(3):
    v = x_2d[:, :, f]
    print(f"  feat {f}: min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}, std={v.std():.4f}")
print(f"  NaN: {torch.isnan(x_2d).sum().item()}, Inf: {torch.isinf(x_2d).sum().item()}")

# ──────────────────────────────────────────────────────────────────
# 3. Check edges
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 3: EDGE INTEGRITY")
print("=" * 70)

for et in graph.edge_types:
    ei = graph[et].edge_index
    src_type, rel, dst_type = et
    n_src = n_1d if src_type == "1d" else n_2d
    n_dst = n_1d if dst_type == "1d" else n_2d
    src_oob = (ei[0] >= n_src).sum().item()
    dst_oob = (ei[1] >= n_dst).sum().item()
    self_loops = (ei[0] == ei[1]).sum().item() if src_type == dst_type else 0
    print(f"  {et}: edges={ei.size(1)}, src_OOB={src_oob}, dst_OOB={dst_oob}, self_loops={self_loops}")

# ──────────────────────────────────────────────────────────────────
# 4. Node stds & loss weights
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 4: NODE STDS & LOSS WEIGHTS")
print("=" * 70)

node_stds = ds.compute_node_stds(model_id=model_id)
stds_1d = torch.tensor(node_stds[model_id]["1d"], dtype=torch.float32)
stds_2d = torch.tensor(node_stds[model_id]["2d"], dtype=torch.float32)

print(f"\n1D node stds ({len(stds_1d)} nodes):")
print(f"  values: {stds_1d.numpy()}")
print(f"  min={stds_1d.min():.6f}, max={stds_1d.max():.6f}, mean={stds_1d.mean():.6f}")
print(f"  zeros (σ=0): {(stds_1d == 0).sum().item()}")
print(f"  near-zero (σ<0.01): {(stds_1d < 0.01).sum().item()}")
print(f"  NaN: {torch.isnan(stds_1d).sum().item()}")

# Check if any NaN in stds
if torch.isnan(stds_1d).any():
    print("  *** CRITICAL: NaN in 1D stds! ***")
    nan_idx = torch.where(torch.isnan(stds_1d))[0]
    print(f"  NaN at indices: {nan_idx.tolist()}")

print(f"\n2D node stds ({len(stds_2d)} nodes):")
print(f"  min={stds_2d.min():.6f}, max={stds_2d.max():.6f}, mean={stds_2d.mean():.6f}")
print(f"  near-zero (σ<0.01): {(stds_2d < 0.01).sum().item()}")
print(f"  NaN: {torch.isnan(stds_2d).sum().item()}")

# Compute actual loss weights
clamp_weights = 20.0
weights_1d = torch.clamp(1.0 / (stds_1d**2 + 1e-6), max=clamp_weights)
weights_2d = torch.clamp(1.0 / (stds_2d**2 + 1e-6), max=clamp_weights)
print(f"\n1D loss weights (clamped to {clamp_weights}):")
print(f"  values: {weights_1d.numpy()}")
print(f"  min={weights_1d.min():.4f}, max={weights_1d.max():.4f}, mean={weights_1d.mean():.4f}")
print(f"  at max: {(weights_1d >= clamp_weights - 0.01).sum().item()} nodes")

print(f"\n2D loss weights (clamped to {clamp_weights}):")
print(f"  min={weights_2d.min():.4f}, max={weights_2d.max():.4f}, mean={weights_2d.mean():.4f}")
print(f"  at max: {(weights_2d >= clamp_weights - 0.01).sum().item()} nodes")

# ──────────────────────────────────────────────────────────────────
# 5. Persistence baseline (actual training loss, not just SRMSE metric)
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 5: PERSISTENCE BASELINE (TRAINING LOSS)")
print("=" * 70)

spinup = 10
K = 2
spinup = min(spinup, T - 1)
K = min(K, T - spinup)

# Persistence: predict wse[spinup-1] for all future steps
persist_wse_1d = wse_1d[spinup - 1].unsqueeze(0).expand(K, -1)  # [K, N_1d]
target_wse_1d = wse_1d[spinup:spinup+K]  # [K, N_1d]
persist_wse_2d = wse_2d[spinup - 1].unsqueeze(0).expand(K, -1)
target_wse_2d = wse_2d[spinup:spinup+K]

persist_err_1d = (persist_wse_1d - target_wse_1d)
persist_err_2d = (persist_wse_2d - target_wse_2d)

print(f"\nPersistence errors (K={K}, spinup={spinup}):")
print(f"  1D: mean_abs_err={persist_err_1d.abs().mean():.6f} ft")
print(f"       max_abs_err={persist_err_1d.abs().max():.6f} ft")
print(f"       per-step:")
for k in range(K):
    e = persist_err_1d[k]
    print(f"    k={k}: max={e.abs().max():.4f}, mean={e.abs().mean():.4f}, rmse={e.pow(2).mean().sqrt():.4f}")

print(f"  2D: mean_abs_err={persist_err_2d.abs().mean():.6f} ft")
print(f"       max_abs_err={persist_err_2d.abs().max():.6f} ft")

# Compute actual training loss for persistence
loss_1d_persist = push_forward_loss(
    persist_wse_1d, target_wse_1d, stds_1d,
    clamp_weights=clamp_weights, temporal_scheme="linear",
)
loss_2d_persist = push_forward_loss(
    persist_wse_2d, target_wse_2d, stds_2d,
    clamp_weights=clamp_weights, temporal_scheme="linear",
)
total_persist = 0.5 * loss_1d_persist + 0.5 * loss_2d_persist

print(f"\nPersistence TRAINING LOSS (push_forward_loss with linear weights):")
print(f"  loss_1d = {loss_1d_persist.item():.4f}")
print(f"  loss_2d = {loss_2d_persist.item():.4f}")
print(f"  total   = {total_persist.item():.4f}")
print(f"  Compare with training output: loss=892, 1d=1732, 2d=52")

# Now try different K values and spinup values
print(f"\n  Persistence loss vs K (spinup={spinup}):")
for test_K in [2, 5, 10, 20]:
    if spinup + test_K > T:
        continue
    p1 = wse_1d[spinup - 1].unsqueeze(0).expand(test_K, -1)
    t1 = wse_1d[spinup:spinup+test_K]
    p2 = wse_2d[spinup - 1].unsqueeze(0).expand(test_K, -1)
    t2 = wse_2d[spinup:spinup+test_K]
    l1 = push_forward_loss(p1, t1, stds_1d, clamp_weights=clamp_weights, temporal_scheme="linear")
    l2 = push_forward_loss(p2, t2, stds_2d, clamp_weights=clamp_weights, temporal_scheme="linear")
    print(f"    K={test_K:2d}: 1d={l1.item():.4f}, 2d={l2.item():.4f}, total={0.5*l1.item()+0.5*l2.item():.4f}")

# Average persistence loss across multiple events
print(f"\n  Average persistence loss across first 10 events (K=2, spinup=10):")
persist_losses = []
for idx in range(min(10, len(model_ds))):
    sample_i = model_ds[idx]
    graph_i = build_hetero_graph(sample_i, norm_stats)
    T_i = graph_i.num_timesteps
    sp = min(10, T_i - 1)
    k_i = min(2, T_i - sp)
    if k_i <= 0:
        continue
    p1 = graph_i["1d"].y[sp-1].unsqueeze(0).expand(k_i, -1)
    t1 = graph_i["1d"].y[sp:sp+k_i]
    p2 = graph_i["2d"].y[sp-1].unsqueeze(0).expand(k_i, -1)
    t2 = graph_i["2d"].y[sp:sp+k_i]
    l1 = push_forward_loss(p1, t1, stds_1d, clamp_weights=clamp_weights, temporal_scheme="linear")
    l2 = push_forward_loss(p2, t2, stds_2d, clamp_weights=clamp_weights, temporal_scheme="linear")
    total_i = 0.5 * l1.item() + 0.5 * l2.item()
    persist_losses.append((total_i, l1.item(), l2.item()))
    print(f"    Event {sample_i.get('event_id','?'):>3s}: 1d={l1.item():.4f}, 2d={l2.item():.4f}, total={total_i:.4f}")

if persist_losses:
    avg_total = np.mean([x[0] for x in persist_losses])
    avg_1d = np.mean([x[1] for x in persist_losses])
    avg_2d = np.mean([x[2] for x in persist_losses])
    print(f"    AVERAGE: 1d={avg_1d:.4f}, 2d={avg_2d:.4f}, total={avg_total:.4f}")

# ──────────────────────────────────────────────────────────────────
# 6. Model forward + gradient check
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 6: MODEL FORWARD & GRADIENT FLOW")
print("=" * 70)

model = UnifiedHeteroModel(
    in_channels_1d=3, in_channels_2d=3,
    hidden_channels=192, num_gnn_layers=3, dropout=0.05,
).to(DEVICE)

# Single forward pass
graph_d = graph.to(DEVICE)
edge_index_dict = {et: graph_d[et].edge_index for et in graph_d.edge_types}
hidden = model.init_hidden(n_1d, n_2d, DEVICE)

x_dict = {"1d": graph_d["1d"].x[0], "2d": graph_d["2d"].x[0]}
delta_dict, hidden = model(x_dict, edge_index_dict, hidden)

print(f"\nModel output (t=0, fresh init):")
print(f"  delta_1d: min={delta_dict['1d'].min():.6f}, max={delta_dict['1d'].max():.6f}, mean={delta_dict['1d'].mean():.6f}")
print(f"  delta_2d: min={delta_dict['2d'].min():.6f}, max={delta_dict['2d'].max():.6f}, mean={delta_dict['2d'].mean():.6f}")
print(f"  delta_1d values: {delta_dict['1d'].detach().cpu().numpy()}")

# Simulate the full training loop for this one event with K=2
print(f"\n--- Simulating training loop (K=2, spinup=10) ---")
model.train()
hidden = model.init_hidden(n_1d, n_2d, DEVICE)

d1_mean = norm_stats["1d"]["depth"]["mean"]
d1_std = norm_stats["1d"]["depth"]["std"]
d2_mean = norm_stats["2d"]["depth"]["mean"]
d2_std = norm_stats["2d"]["depth"]["std"]

_stds_1d = stds_1d.to(DEVICE)
_stds_2d = stds_2d.to(DEVICE)
const_mask_1d = (_stds_1d >= 0.01).float()
const_mask_2d = (_stds_2d >= 0.01).float()

# Spinup
with torch.no_grad():
    for t in range(spinup):
        x_dict = {"1d": graph_d["1d"].x[t], "2d": graph_d["2d"].x[t]}
        _, hidden = model(x_dict, edge_index_dict, hidden)
hidden = {k: v.detach() for k, v in hidden.items()}

prev_depth_1d = graph_d["1d"].depth[spinup - 1]
prev_depth_2d = graph_d["2d"].depth[spinup - 1]

all_pred_wse_1d = []
all_pred_wse_2d = []
all_target_wse_1d = []
all_target_wse_2d = []

delta_clamp = 2.0
for k in range(K):
    t = spinup + k
    x_1d_t = graph_d["1d"].x[t].clone()
    x_2d_t = graph_d["2d"].x[t].clone()

    norm_d1 = (prev_depth_1d - d1_mean) / max(d1_std, 1e-8)
    norm_d2 = (prev_depth_2d - d2_mean) / max(d2_std, 1e-8)
    x_1d_t[:, 0] = norm_d1
    x_2d_t[:, 0] = norm_d2

    delta_dict, hidden = model({"1d": x_1d_t, "2d": x_2d_t}, edge_index_dict, hidden)

    raw_delta_1d = delta_dict["1d"].clamp(-delta_clamp, delta_clamp) * const_mask_1d
    raw_delta_2d = delta_dict["2d"].clamp(-delta_clamp, delta_clamp) * const_mask_2d

    pred_depth_1d_k = (prev_depth_1d + raw_delta_1d).clamp(min=0)
    pred_depth_2d_k = (prev_depth_2d + raw_delta_2d).clamp(min=0)

    pred_wse_1d_k = pred_depth_1d_k + graph_d["1d"].elev
    pred_wse_2d_k = pred_depth_2d_k + graph_d["2d"].elev

    all_pred_wse_1d.append(pred_wse_1d_k)
    all_pred_wse_2d.append(pred_wse_2d_k)
    all_target_wse_1d.append(graph_d["1d"].y[t])
    all_target_wse_2d.append(graph_d["2d"].y[t])

    prev_depth_1d = pred_depth_1d_k
    prev_depth_2d = pred_depth_2d_k

    print(f"\n  Step k={k}, t={t}:")
    print(f"    delta_1d (raw model): {delta_dict['1d'].detach().cpu().numpy()}")
    print(f"    raw_delta_1d (clamped+masked): {raw_delta_1d.detach().cpu().numpy()}")
    print(f"    prev_depth_1d: {prev_depth_1d.detach().cpu().numpy()}")
    print(f"    pred_wse_1d: {pred_wse_1d_k.detach().cpu().numpy()}")
    print(f"    target_wse_1d: {graph_d['1d'].y[t].cpu().numpy()}")
    err_1d_k = (pred_wse_1d_k - graph_d["1d"].y[t]).abs()
    print(f"    |error| 1d: min={err_1d_k.min():.6f}, max={err_1d_k.max():.6f}, mean={err_1d_k.mean():.6f}")

    err_2d_k = (pred_wse_2d_k - graph_d["2d"].y[t]).abs()
    print(f"    |error| 2d: min={err_2d_k.min():.6f}, max={err_2d_k.max():.6f}, mean={err_2d_k.mean():.6f}")

# Compute loss
preds_1d = torch.stack(all_pred_wse_1d)
preds_2d = torch.stack(all_pred_wse_2d)
targets_1d = torch.stack(all_target_wse_1d)
targets_2d = torch.stack(all_target_wse_2d)

loss_1d = push_forward_loss(preds_1d.float(), targets_1d.float(), _stds_1d, clamp_weights=clamp_weights, temporal_scheme="linear")
loss_2d = push_forward_loss(preds_2d.float(), targets_2d.float(), _stds_2d, clamp_weights=clamp_weights, temporal_scheme="linear")
total_loss = 0.5 * loss_1d + 0.5 * loss_2d

print(f"\nComputed loss for this event:")
print(f"  loss_1d = {loss_1d.item():.4f}")
print(f"  loss_2d = {loss_2d.item():.4f}")
print(f"  total   = {total_loss.item():.4f}")

# ──────────────────────────────────────────────────────────────────
# 7. Gradient flow check
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 7: GRADIENT FLOW CHECK")
print("=" * 70)

total_loss.backward()

# Check gradients on key layers
grad_info = []
for name, param in model.named_parameters():
    if param.grad is not None:
        g = param.grad
        grad_info.append((name, g.abs().mean().item(), g.abs().max().item(), g.norm().item()))
    else:
        grad_info.append((name, 0.0, 0.0, 0.0))

print(f"\n{'Parameter':<50s} {'mean|grad|':>12s} {'max|grad|':>12s} {'grad_norm':>12s}")
print("-" * 90)
zero_grad_count = 0
for name, mean_g, max_g, norm_g in grad_info:
    flag = " *** ZERO ***" if mean_g == 0 else ""
    if mean_g == 0:
        zero_grad_count += 1
    print(f"  {name:<48s} {mean_g:12.2e} {max_g:12.2e} {norm_g:12.2e}{flag}")

print(f"\nZero-gradient parameters: {zero_grad_count}/{len(grad_info)}")

# ──────────────────────────────────────────────────────────────────
# 8. Single-step overfit test
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  SECTION 8: CAN THE MODEL REDUCE LOSS? (100 steps on 1 event)")
print("=" * 70)

model2 = UnifiedHeteroModel(
    in_channels_1d=3, in_channels_2d=3,
    hidden_channels=192, num_gnn_layers=3, dropout=0.0,
).to(DEVICE)

optimizer = optim.Adam(model2.parameters(), lr=1e-3)

for step in range(100):
    optimizer.zero_grad()
    model2.train()
    hidden = model2.init_hidden(n_1d, n_2d, DEVICE)

    # Spinup
    with torch.no_grad():
        for t in range(spinup):
            x_dict = {"1d": graph_d["1d"].x[t], "2d": graph_d["2d"].x[t]}
            _, hidden = model2(x_dict, edge_index_dict, hidden)
    hidden = {k: v.detach() for k, v in hidden.items()}

    prev_d1 = graph_d["1d"].depth[spinup - 1]
    prev_d2 = graph_d["2d"].depth[spinup - 1]

    preds_1d_list = []
    preds_2d_list = []
    tgts_1d_list = []
    tgts_2d_list = []

    for k_i in range(K):
        t_i = spinup + k_i
        x1 = graph_d["1d"].x[t_i].clone()
        x2 = graph_d["2d"].x[t_i].clone()
        x1[:, 0] = (prev_d1 - d1_mean) / max(d1_std, 1e-8)
        x2[:, 0] = (prev_d2 - d2_mean) / max(d2_std, 1e-8)

        dd, hidden = model2({"1d": x1, "2d": x2}, edge_index_dict, hidden)
        rd1 = dd["1d"].clamp(-delta_clamp, delta_clamp) * const_mask_1d
        rd2 = dd["2d"].clamp(-delta_clamp, delta_clamp) * const_mask_2d
        pd1 = (prev_d1 + rd1).clamp(min=0)
        pd2 = (prev_d2 + rd2).clamp(min=0)

        preds_1d_list.append(pd1 + graph_d["1d"].elev)
        preds_2d_list.append(pd2 + graph_d["2d"].elev)
        tgts_1d_list.append(graph_d["1d"].y[t_i])
        tgts_2d_list.append(graph_d["2d"].y[t_i])

        prev_d1 = pd1
        prev_d2 = pd2

    p1 = torch.stack(preds_1d_list)
    p2 = torch.stack(preds_2d_list)
    t1 = torch.stack(tgts_1d_list)
    t2 = torch.stack(tgts_2d_list)

    l1 = push_forward_loss(p1.float(), t1.float(), _stds_1d, clamp_weights=clamp_weights, temporal_scheme="linear")
    l2 = push_forward_loss(p2.float(), t2.float(), _stds_2d, clamp_weights=clamp_weights, temporal_scheme="linear")
    loss = 0.5 * l1 + 0.5 * l2

    loss.backward()
    nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
    optimizer.step()

    if step % 10 == 0 or step < 5:
        print(f"  Step {step:3d}: loss={loss.item():.4f} (1d={l1.item():.4f}, 2d={l2.item():.4f})")

print("\n" + "=" * 70)
print("  DIAGNOSTIC COMPLETE")
print("=" * 70)

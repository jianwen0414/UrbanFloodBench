"""
1D Flood Prediction Model using Graph Neural Networks.

Architecture:
- GraphSAGE layers for spatial message passing along the 1D pipe network
- GRU cell for temporal dynamics
- Residual prediction (delta) for training stability
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from src.dataset import FloodDataset
from src.graph_builder_1d import build_1d_graph, get_1d_input_dim
from src.utils_1d import compute_normalization_stats_1d, pivot_dynamic_1d


class DrainageNetwork1D(nn.Module):
    """
    GNN model for 1D drainage network water level prediction.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_sage_layers: int = 2,
        dropout: float = 0.1,
        max_delta: float = 2.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_sage_layers = num_sage_layers
        self.max_delta = max_delta

        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_sage_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # GRU for temporal dynamics
        self.gru = nn.GRUCell(hidden_channels, hidden_channels)

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, data: Data, hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = data.x
        edge_index = data.edge_index
        current_wl = data.current_wl

        # GraphSAGE
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        # GRU
        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_channels, device=x.device)
        hidden = self.gru(x, hidden)

        # Predict delta
        delta = self.head(hidden).squeeze(-1)
        delta = torch.tanh(delta) * self.max_delta

        # Residual connection
        pred_wl = current_wl + delta

        return pred_wl, hidden


def train_model_1d(
    model: DrainageNetwork1D,
    dataset: FloodDataset,
    model_id: str,
    num_epochs: int = 30,
    lr: float = 0.005,
    warmup_epochs: int = 8,
    decay_epochs: int = 17,
    max_timesteps_per_event: int = 50,
    print_every: int = 5,
    validate_every: int = 5,
    validation_events: int = 2,
    early_stopping_patience: int = 12,
    checkpoint_dir: str = "checkpoints",
    save_best: bool = True,
) -> Dict:
    """Train the 1D model with a simple teacher forcing schedule."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ds_model = dataset.filter_by_model(model_id)
    norm_stats = compute_normalization_stats_1d(dataset, model_id)

    # Split train/val (last `num_val` events for validation)
    num_val = min(validation_events, max(1, len(ds_model) // 10))
    train_indices = list(range(len(ds_model) - num_val))
    val_indices = list(range(len(ds_model) - num_val, len(ds_model)))

    print(f"Training on Model_{model_id}")
    print(f"Schedule: warmup={warmup_epochs}, decay={decay_epochs}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print("-" * 60)
    print(f"Train events: {len(train_indices)}, Validation events: {len(val_indices)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history: Dict[str, Any] = {
        "train_loss": [],
        "val_rmse": [],
        "best_val_rmse": float("inf"),
        "best_epoch": 0,
    }

    patience_counter = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()

        # Teacher forcing ratio
        if epoch < warmup_epochs:
            tf_ratio = 1.0
        elif epoch < warmup_epochs + decay_epochs:
            tf_ratio = 1.0 - (epoch - warmup_epochs) / decay_epochs
        else:
            tf_ratio = 0.0

        epoch_losses: List[float] = []
        np.random.shuffle(train_indices)

        for event_idx in train_indices:
            sample = ds_model[event_idx]

            # Pivot water levels once per event (clamped to invert elevation)
            wl_array = pivot_dynamic_1d(
                sample["dynamic_1d_nodes"], "water_level", sample=sample
            )
            num_timesteps = min(wl_array.shape[0] - 1, max_timesteps_per_event)

            # Skip dry initial condition (first 10 timesteps are constant — no learning signal)
            DRY_PERIOD = 10
            t_start_train = DRY_PERIOD

            hidden = None
            event_loss = 0.0
            pred_wl_current: Optional[np.ndarray] = None

            for t in range(t_start_train, num_timesteps):
                # Teacher forcing: use GT for first 5 steps after dry period
                use_teacher = (np.random.random() < tf_ratio) or (
                    t < t_start_train + 5
                )

                if use_teacher or pred_wl_current is None:
                    data = build_1d_graph(sample, norm_stats, t_index=t)
                else:
                    data = build_1d_graph(
                        sample,
                        norm_stats,
                        t_index=t,
                        water_level_override=pred_wl_current,
                    )

                data = data.to(device)

                pred_wl, hidden = model(data, hidden)

                target = data.y.to(device)
                loss = criterion(pred_wl, target)
                event_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                hidden = hidden.detach()
                pred_wl_current = pred_wl.detach().cpu().numpy()

            actual_steps = max(num_timesteps - t_start_train, 1)
            epoch_losses.append(event_loss / actual_steps)

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        history["train_loss"].append(avg_loss)

        # Validation
        if epoch % validate_every == 0:
            val_rmse = validate_model_1d(
                model, ds_model, val_indices, norm_stats, device
            )
            history["val_rmse"].append(val_rmse)

            if val_rmse < history["best_val_rmse"]:
                history["best_val_rmse"] = val_rmse
                history["best_epoch"] = epoch
                patience_counter = 0

                if save_best:
                    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    cp_path = f"{checkpoint_dir}/model_{model_id}_1d_best.pt"
                    save_checkpoint(model, optimizer, epoch, val_rmse, cp_path)
                    print(f"  Checkpoint saved: {cp_path}")
            else:
                patience_counter += 1

            elapsed = time.time() - start_time

            if epoch % print_every == 0:
                print(
                    f"Epoch {epoch:3d} | Train: {avg_loss:.6f} | TF: {tf_ratio:.2f} | "
                    f"Val RMSE: {val_rmse:.4f} | LR: {lr:.6f} | {elapsed:.0f}s"
                )

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"No improvement for {early_stopping_patience} epochs")
            break

    print("-" * 60)
    print("Training complete!")
    print(
        f"Best Val RMSE: {history['best_val_rmse']:.4f} "
        f"at epoch {history['best_epoch']}"
    )

    if save_best:
        cp_path = f"{checkpoint_dir}/model_{model_id}_1d_best.pt"
        if Path(cp_path).exists():
            print("\nLoading best model from checkpoint...")
            load_checkpoint(model, cp_path)

    return history


def validate_model_1d(
    model: DrainageNetwork1D,
    ds_model,
    val_indices: List[int],
    norm_stats: Dict,
    device: torch.device,
    num_warmup: int = 10,
) -> float:
    """Validate model using autoregressive prediction."""
    model.eval()
    rmses: List[float] = []

    with torch.no_grad():
        for event_idx in val_indices:
            sample = ds_model[event_idx]
            wl_array = pivot_dynamic_1d(
                sample["dynamic_1d_nodes"], "water_level", sample=sample
            )
            num_timesteps = wl_array.shape[0]

            hidden = None
            pred_wl_current: Optional[np.ndarray] = None
            all_preds: List[torch.Tensor] = []
            all_gt: List[torch.Tensor] = []

            for t in range(num_timesteps - 1):
                if t < num_warmup or pred_wl_current is None:
                    data = build_1d_graph(sample, norm_stats, t_index=t)
                else:
                    data = build_1d_graph(
                        sample,
                        norm_stats,
                        t_index=t,
                        water_level_override=pred_wl_current,
                    )

                data = data.to(device)
                pred_wl, hidden = model(data, hidden)
                pred_wl_current = pred_wl.cpu().numpy()

                if t >= num_warmup:
                    all_preds.append(pred_wl.cpu())
                    all_gt.append(
                        torch.tensor(wl_array[t + 1], dtype=torch.float32)
                    )

            if all_preds:
                preds = torch.stack(all_preds)
                gts = torch.stack(all_gt)
                rmse = torch.sqrt(torch.mean((preds - gts) ** 2)).item()
                rmses.append(rmse)

    return float(np.mean(rmses)) if rmses else float("inf")


def predict_event_1d(
    model: DrainageNetwork1D,
    sample: Dict,
    norm_stats: Dict,
    num_warmup: int = 10,
    total_timesteps: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predict water levels for an entire 1D event.

    For TEST data: only num_warmup timesteps of ground truth are available.
    The model must predict the remaining timesteps autoregressively.

    Args:
        model: Trained model
        sample: Data sample
        norm_stats: Normalization stats
        num_warmup: Number of warmup timesteps with ground truth
        total_timesteps: Total timesteps to predict. If None, inferred from data.
            For test data, this should be the full event length.

    Returns:
        pred_wl: (total_timesteps, num_nodes)
        gt_wl: (num_gt_timesteps, num_nodes) — only available timesteps
    """
    device = next(model.parameters()).device
    model.eval()

    wl_array = pivot_dynamic_1d(
        sample["dynamic_1d_nodes"], "water_level", sample=sample
    )  # (T_available, N) — clamped to invert elevation
    num_available = wl_array.shape[0]  # Could be just 10 for test data
    num_nodes = wl_array.shape[1]

    # Invert elevations for clamping predictions (physical minimum)
    static_df = sample["static_1d_nodes"].sort_values("node_idx").reset_index(
        drop=True
    )
    invert_elevations = torch.tensor(
        static_df["invert_elevation"].values, dtype=torch.float32
    )

    # Determine total timesteps to predict
    if total_timesteps is None:
        if "timesteps" in sample and sample["timesteps"] is not None:
            ts = sample["timesteps"]
            if hasattr(ts, "__len__"):
                total_timesteps = len(ts)
            else:
                total_timesteps = int(ts)
        else:
            dynamic_2d = sample.get("dynamic_2d_nodes")
            if dynamic_2d is not None:
                if hasattr(dynamic_2d, "shape"):
                    if len(dynamic_2d.shape) == 3:
                        total_timesteps = dynamic_2d.shape[0]
                    else:
                        total_timesteps = (
                            dynamic_2d["timestep"].nunique()
                            if hasattr(dynamic_2d, "columns")
                            and "timestep" in dynamic_2d.columns
                            else num_available
                        )
                elif isinstance(dynamic_2d, pd.DataFrame):
                    total_timesteps = int(dynamic_2d["timestep"].nunique())
                else:
                    total_timesteps = num_available
            else:
                total_timesteps = num_available

    # Get total from dynamic_1d timestep column (may extend beyond pivot rows)
    dynamic_1d_df = sample["dynamic_1d_nodes"]
    if isinstance(dynamic_1d_df, pd.DataFrame) and "timestep" in dynamic_1d_df.columns:
        max_timestep = int(dynamic_1d_df["timestep"].max())
        inferred_total = max_timestep + 1
        if inferred_total > total_timesteps:
            total_timesteps = inferred_total

    if total_timesteps <= num_warmup:
        gt_wl = torch.tensor(wl_array, dtype=torch.float32)
        return gt_wl, gt_wl

    hidden = None
    predictions: List[torch.Tensor] = []

    # Phase 1: Warmup — use ground truth
    predictions.append(torch.tensor(wl_array[0], dtype=torch.float32))

    with torch.no_grad():
        for t in range(min(num_available - 1, num_warmup)):
            data = build_1d_graph(sample, norm_stats, t_index=t)
            data = data.to(device)
            pred_wl, hidden = model(data, hidden)
            pred_wl = torch.max(pred_wl, invert_elevations.to(device))

            if t + 1 < num_available:
                predictions.append(
                    torch.tensor(wl_array[t + 1], dtype=torch.float32)
                )
            else:
                predictions.append(pred_wl.cpu())

        # Phase 2: Autoregressive — use predictions for remaining timesteps
        pred_wl_current = wl_array[min(num_available - 1, num_warmup)].copy()

        for t in range(len(predictions), total_timesteps):
            t_clamped = min(t, num_available - 1)

            data = build_1d_graph(
                sample,
                norm_stats,
                t_index=t_clamped,
                water_level_override=pred_wl_current,
            )
            data = data.to(device)

            pred_wl, hidden = model(data, hidden)
            pred_wl = torch.max(pred_wl, invert_elevations.to(device))
            predictions.append(pred_wl.cpu())

            pred_wl_current = pred_wl.cpu().numpy()

    pred_wl_tensor = torch.stack(predictions)  # (total_timesteps, N)
    gt_wl_tensor = torch.tensor(wl_array, dtype=torch.float32)  # (num_available, N)

    return pred_wl_tensor, gt_wl_tensor


def evaluate_predictions_1d(
    pred_wl: torch.Tensor,
    gt_wl: torch.Tensor,
    num_warmup: int = 10,
) -> Dict:
    """Evaluate 1D predictions against ground truth."""
    pred = pred_wl[num_warmup + 1 :]
    gt = gt_wl[num_warmup + 1 :]

    min_len = min(len(pred), len(gt))
    pred = pred[:min_len]
    gt = gt[:min_len]

    if len(pred) == 0:
        return {
            "mse": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
            "num_timesteps": 0,
            "num_nodes": 0,
        }

    mse = torch.mean((pred - gt) ** 2).item()
    rmse = float(np.sqrt(mse))
    mae = torch.mean(torch.abs(pred - gt)).item()

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "num_timesteps": len(pred),
        "num_nodes": pred.shape[1] if pred.ndim > 1 else 1,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    path: str,
) -> None:
    """Save model checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "loss": loss,
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(
        f"  Loaded checkpoint from epoch {checkpoint['epoch']} "
        f"(loss: {checkpoint['loss']:.6f})"
    )
    return checkpoint


if __name__ == "__main__":
    from src.config import RAW_DATA_PATH

    print("=" * 60)
    print("1D MODEL TEST")
    print("=" * 60)

    ds = FloodDataset(RAW_DATA_PATH, mode="train")

    for model_id in ["1", "2"]:
        print(f"\nModel_{model_id}:")
        print("-" * 40)

        ds_model = ds.filter_by_model(model_id)
        if len(ds_model) == 0:
            print("  No events for this model.")
            continue

        sample = ds_model[0]

        if sample.get("static_1d_nodes") is None or len(sample["static_1d_nodes"]) == 0:
            print("  No 1D data")
            continue

        norm_stats = compute_normalization_stats_1d(ds, model_id)
        in_channels = get_1d_input_dim(sample, norm_stats)

        print(f"  Input features: {in_channels}")

        # Create model
        model = DrainageNetwork1D(
            in_channels=in_channels,
            hidden_channels=64,
            num_sage_layers=2,
            dropout=0.1,
            max_delta=2.0,
        )

        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")

        # Test forward pass
        data = build_1d_graph(sample, norm_stats, t_index=10)
        pred, hidden = model(data, None)
        print(f"  Forward pass: pred shape={pred.shape}, hidden shape={hidden.shape}")

        # Test second forward pass (with hidden state)
        data2 = build_1d_graph(sample, norm_stats, t_index=11)
        pred2, hidden2 = model(data2, hidden)
        print(f"  Second pass:  pred shape={pred2.shape}, hidden shape={hidden2.shape}")

        # Test prediction on single event (untrained model)
        print("\n  Testing predict_event_1d...")
        pred_wl, gt_wl = predict_event_1d(model, sample, norm_stats, num_warmup=10)
        print(f"  pred_wl shape: {pred_wl.shape}")
        print(f"  gt_wl shape:   {gt_wl.shape}")

        metrics = evaluate_predictions_1d(pred_wl, gt_wl, num_warmup=10)
        print(f"  RMSE (untrained): {metrics['rmse']:.4f}")
        print(f"  MAE  (untrained): {metrics['mae']:.4f}")

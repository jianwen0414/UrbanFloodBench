"""
<<<<<<< HEAD
model_2d — SurfaceEngine: GraphSAGE-GRU for 2D surface water depth.

Predicts **delta depth** (change in water depth) at each timestep on
the 2D surface mesh.  The architecture fuses spatial message-passing
(GraphSAGE) with temporal dynamics (GRU) and enforces two physical
=======
model_2d — SurfaceEngine: GraphSAGE- or GAT-GRU for 2D surface water depth.

Predicts **delta depth** (change in water depth) at each timestep on
the 2D surface mesh.  The architecture fuses spatial message-passing
(GraphSAGE or GAT) with temporal dynamics (GRU) and enforces two physical
>>>>>>> feat/2d-pipeline
constraints:

    1. ``|delta| <= max_delta`` — prevents prediction explosions.
    2. ``depth >= 0``          — water depth is non-negative.

Owner: Member B
See: IMPLEMENTATION_PLAN.md → Task 2.2

Architecture
------------
::

    Input  x  [N, in_channels]
         │
<<<<<<< HEAD
    GraphSAGE layers  (spatial neighbour aggregation)
=======
    GraphSAGE or GAT layers  (spatial neighbour aggregation)
>>>>>>> feat/2d-pipeline
         │
    GRU cell          (per-node temporal hidden state)
         │
    Linear head  →  delta_depth  [N, 1]

Activation: LeakyReLU throughout (most 2D nodes are dry →
ReLU would kill gradients on the majority of nodes).
"""

from __future__ import annotations

import random
import time
<<<<<<< HEAD
=======
from typing import Literal
>>>>>>> feat/2d-pipeline

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
<<<<<<< HEAD
from torch_geometric.nn import SAGEConv


class SurfaceEngine(nn.Module):
    """2D Surface Water Level Predictor (GraphSAGE + GRU).
=======
from torch_geometric.nn import GATConv, SAGEConv


class SurfaceEngine(nn.Module):
    """2D Surface Water Level Predictor (GraphSAGE or GAT + GRU).
>>>>>>> feat/2d-pipeline

    Parameters
    ----------
    in_channels : int
        Number of input features per node (e.g. 16 for 12 static + 4
        dynamic).
    hidden_channels : int
<<<<<<< HEAD
        Dimension of hidden representations (e.g. 64).
    num_sage_layers : int, optional
        Number of stacked GraphSAGE convolution layers (default ``2``).
    dropout : float, optional
        Dropout probability applied after each SAGE layer (default ``0.1``).
    max_delta : float, optional
        Maximum absolute depth change per timestep (default ``1.0`` m).
=======
        Dimension of hidden representations (e.g. 64). When
        ``conv_type="gat"``, must be divisible by ``num_heads`` (e.g. 4).
    num_sage_layers : int, optional
        Number of stacked graph convolution layers (default ``2``).
    dropout : float, optional
        Dropout probability applied after each conv layer (default ``0.1``).
    max_delta : float, optional
        Maximum absolute depth change per timestep (default ``1.0`` m).
    conv_type : str, optional
        ``"sage"`` (GraphSAGE) or ``"gat"`` (Graph Attention). Default ``"sage"``.
    num_heads : int, optional
        Number of attention heads when ``conv_type="gat"`` (default ``4``).
        Ignored when ``conv_type="sage"``. Output size is ``hidden_channels``
        so ``hidden_channels`` must be divisible by ``num_heads``.
>>>>>>> feat/2d-pipeline
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_sage_layers: int = 2,
        dropout: float = 0.1,
        max_delta: float = 1.0,
<<<<<<< HEAD
=======
        conv_type: Literal["sage", "gat"] = "sage",
        num_heads: int = 4,
>>>>>>> feat/2d-pipeline
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels
        self.max_delta = max_delta
        self.dropout = dropout
<<<<<<< HEAD

        # ── GraphSAGE layers (spatial message passing) ───────────────
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_sage_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
=======
        self.conv_type = conv_type
        self.num_heads = num_heads

        if conv_type == "gat" and hidden_channels % num_heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by "
                f"num_heads ({num_heads}) when conv_type='gat'"
            )

        # ── Graph conv layers (SAGE or GAT) ─────────────────────────────
        self.convs = nn.ModuleList()
        if conv_type == "sage":
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_sage_layers - 1):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        else:  # gat
            out_per_head = hidden_channels // num_heads
            # NOTE: GATConv dropout is attention-coefficient dropout.
            # We already apply F.dropout after each layer in forward(),
            # so setting both causes double-dropout and inflates the
            # train/eval distribution gap — deadly for autoregressive
            # rollout.  Keep GATConv dropout=0; use post-layer dropout only.
            self.convs.append(
                GATConv(
                    in_channels,
                    out_per_head,
                    heads=num_heads,
                    concat=True,
                    dropout=0.0,
                )
            )
            for _ in range(num_sage_layers - 1):
                self.convs.append(
                    GATConv(
                        hidden_channels,
                        out_per_head,
                        heads=num_heads,
                        concat=True,
                        dropout=0.0,
                    )
                )
>>>>>>> feat/2d-pipeline

        # Batch normalization per layer
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(num_sage_layers)]
        )

        # ── GRU cell (temporal dynamics, per-node) ───────────────────
        self.gru = nn.GRUCell(hidden_channels, hidden_channels)

        # ── Output head → delta depth ────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_channels // 2, 1),
        )

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for **one** timestep.

        Parameters
        ----------
        x : torch.Tensor, shape [N, in_channels]
            Node feature matrix (static + dynamic).
        edge_index : torch.Tensor, shape [2, E]
            Graph connectivity (bidirectional).
        hidden : torch.Tensor or None, shape [N, hidden_channels]
            Previous GRU hidden state.  ``None`` initialises to zeros.

        Returns
        -------
        delta : torch.Tensor, shape [N, 1]
            Predicted depth change, clamped to ``[-max_delta, max_delta]``.
        hidden : torch.Tensor, shape [N, hidden_channels]
            Updated GRU hidden state (pass to the next timestep).
        """
        num_nodes = x.size(0)

        # Initialise hidden state if first timestep
        if hidden is None:
            hidden = self.init_hidden(num_nodes, device=x.device)

<<<<<<< HEAD
        # GraphSAGE layers
=======
        # Graph conv layers (SAGE or GAT)
>>>>>>> feat/2d-pipeline
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.leaky_relu(h, negative_slope=0.01)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # GRU temporal update
        hidden = self.gru(h, hidden)

        # Predict delta depth
        delta = self.head(hidden)

        # Clamp to prevent exploding predictions
        delta = torch.clamp(delta, min=-self.max_delta, max=self.max_delta)

        return delta, hidden

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def init_hidden(
        self, num_nodes: int, device: torch.device | None = None
    ) -> torch.Tensor:
        """Return a zero-initialised hidden state ``[num_nodes, H]``."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(num_nodes, self.hidden_channels, device=device)

    @staticmethod
    def predict_depth(
        current_depth: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Apply *delta* to *current_depth* with a non-negativity clamp.

        Parameters
        ----------
        current_depth : torch.Tensor, shape [N, 1]
            Current water depth.
        delta : torch.Tensor, shape [N, 1]
            Predicted depth change.

        Returns
        -------
        torch.Tensor, shape [N, 1]
            ``max(current_depth + delta, 0)``.
        """
        return torch.clamp(current_depth + delta, min=0.0)


# ───────────────────────────────────────────────────────────────────────
#  Overfit sanity check
# ───────────────────────────────────────────────────────────────────────

def overfit_single_event(
    model: SurfaceEngine,
    dataset: torch.utils.data.Dataset,
    model_id: str = "1",
    event_idx: int = 0,
    num_steps: int = 30,
    num_epochs: int = 50,
    lr: float = 0.005,
    print_every: int = 10,
) -> float:
    """Train *model* on a single event and verify the loss drops.

    This is a **sanity check**: if the model cannot memorise one event,
    something fundamental is broken (wrong target, dead gradients, etc.).

    Parameters
    ----------
    model : SurfaceEngine
        Model to train (will be modified in-place).
    dataset : FloodDataset
        Full training dataset.
    model_id : str
        Which urban model to use (``"1"`` or ``"2"``).
    event_idx : int
        Index of the event within the filtered model subset.
    num_steps : int
        Number of consecutive timesteps to train on.
    num_epochs : int
        Training epochs.
    lr : float
        Learning rate for Adam.
    print_every : int
        Print loss every *print_every* epochs.

    Returns
    -------
    float
        Average loss in the final epoch.
    """
    import time
    from src.graph_builder_2d import build_2d_graph, get_values_at_timestep
    from src.utils_2d import (
        compute_normalization_stats,
        get_min_elevation_filled,
        wse_to_depth,
    )

    # ── Setup ─────────────────────────────────────────────────────────
    ds_model = dataset.filter_by_model(model_id)
    sample = ds_model[event_idx]
    norm_stats = compute_normalization_stats(dataset, model_id)

    print(f"Overfitting on Model_{model_id}, Event_{sample['event_id']}")
    print(f"Training for {num_epochs} epochs on {num_steps} timesteps")
    print("-" * 50)

    static_2d = sample["static_2d_nodes"]
    dynamic_2d = sample["dynamic_2d_nodes"]
    min_elevation = get_min_elevation_filled(static_2d)
    num_nodes = len(static_2d)

    max_timestep = int(dynamic_2d["timestep"].max())
    start_t = 3  # Need at least 3 lags of history
    end_t = min(start_t + num_steps, max_timestep)
    actual_steps = end_t - start_t

    print(f"Available timesteps: 0 to {max_timestep}")
    print(f"Training on timesteps: {start_t} to {end_t - 1} ({actual_steps} steps)")

    # ── Optimiser & loss ──────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    start_time = time.time()
    avg_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        hidden = model.init_hidden(num_nodes)

        for t in range(start_t, end_t):
            # Build graph (features + current depth as target at t)
            data = build_2d_graph(sample, norm_stats, t_index=t)
            current_depth = data.y  # [N, 1] — depth at t

            # Target: depth at t+1
            if t + 1 > max_timestep:
                continue
            wl_next = get_values_at_timestep(
                dynamic_2d, t + 1, "water_level", num_nodes
            )
            depth_next = wse_to_depth(wl_next, min_elevation)
            target_depth = torch.tensor(
                depth_next, dtype=torch.float32
            ).unsqueeze(1)

            # Target delta
            target_delta = target_depth - current_depth

            # Forward
            pred_delta, hidden = model(data.x, data.edge_index, hidden)

            # Loss
            loss = criterion(pred_delta, target_delta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Truncated BPTT — detach hidden between steps
            hidden = hidden.detach()

        avg_loss = epoch_loss / max(actual_steps, 1)

        if epoch % print_every == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.6f}")

    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"Training completed in {elapsed:.1f}s")
    print(f"Final loss: {avg_loss:.6f}")

    if avg_loss < 0.001:
        print("✓ Model successfully overfits! (loss < 0.001)")
    elif avg_loss < 0.01:
        print("✓ Model is learning well (loss < 0.01)")
    elif avg_loss < 0.1:
        print("⚠ Model is learning but slowly (loss < 0.1)")
    else:
        print("✗ Model may have issues — loss still high")

    return avg_loss


# ───────────────────────────────────────────────────────────────────────
#  Full-event autoregressive rollout
# ───────────────────────────────────────────────────────────────────────

def predict_event_2d(
    model: SurfaceEngine,
    sample: dict,
    norm_stats: dict,
    num_warmup: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perform autoregressive rollout for a full event.

    Competition format:

    * First ``num_warmup`` timesteps — ground-truth water levels are
      provided (warmup phase: populate hidden state, store GT as
      predictions).
    * After ``num_warmup`` — only rainfall is known; the model must
      predict water levels using its own prior outputs as lag features.

    Parameters
    ----------
    model : SurfaceEngine
        Trained (or untrained) model.
    sample : dict
        Event dictionary from ``FloodDataset.__getitem__()``.
    norm_stats : dict
        Per-model normalization statistics.
    num_warmup : int, optional
        Number of ground-truth warm-up timesteps (default ``10``).

    Returns
    -------
    pred_wl : torch.Tensor, shape [T, N]
        Predicted water levels (WSE) for every timestep.
    gt_wl : torch.Tensor, shape [T, N]
        Ground-truth water levels (WSE) for every timestep.
    """
    import numpy as np
    from src.graph_builder_2d import (
        DepthHistory,
        build_2d_graph,
        get_values_at_timestep,
    )
    from src.utils_2d import (
        depth_to_wse,
        get_min_elevation_filled,
        wse_to_depth,
    )

    model.eval()

    # ── Setup ─────────────────────────────────────────────────────────
    static_2d = sample["static_2d_nodes"]
    dynamic_2d = sample["dynamic_2d_nodes"]
    min_elevation = get_min_elevation_filled(static_2d)
    num_nodes = len(static_2d)

    max_timestep = int(dynamic_2d["timestep"].max())
    num_timesteps = max_timestep + 1

    pred_wl = torch.zeros(num_timesteps, num_nodes)
    gt_wl = torch.zeros(num_timesteps, num_nodes)

    hidden = model.init_hidden(num_nodes)
    depth_history = DepthHistory(num_history=3)
    current_pred_depth: np.ndarray | None = None

    with torch.no_grad():
        for t in range(num_timesteps):
            # Ground-truth WSE at t
            wl_gt_t = get_values_at_timestep(
                dynamic_2d, t, "water_level", num_nodes
            )
            gt_wl[t] = torch.tensor(wl_gt_t, dtype=torch.float32)
            depth_gt_t = wse_to_depth(wl_gt_t, min_elevation)

            if t < num_warmup:
                # ── WARMUP: use ground truth ─────────────────────
                pred_wl[t] = gt_wl[t]
                depth_history.update(depth_gt_t)
                current_pred_depth = depth_gt_t

                # Feed through model to warm up hidden state
                if t >= 3:
                    data = build_2d_graph(
                        sample, norm_stats, t_index=t, predicted_depths=None
                    )
                    _, hidden = model(data.x, data.edge_index, hidden)
            else:
                # ── PREDICTION: autoregressive rollout ───────────
                data = build_2d_graph(
                    sample,
                    norm_stats,
                    t_index=t,
                    predicted_depths=depth_history.get_lags(),
                )

                if current_pred_depth is not None:
                    current_depth_tensor = torch.tensor(
                        current_pred_depth, dtype=torch.float32
                    ).unsqueeze(1)
                else:
                    current_depth_tensor = torch.tensor(
                        depth_gt_t, dtype=torch.float32
                    ).unsqueeze(1)

                delta, hidden = model(data.x, data.edge_index, hidden)
                pred_depth = model.predict_depth(current_depth_tensor, delta)
                pred_depth_np = pred_depth.squeeze().numpy()

                pred_wl_t = depth_to_wse(pred_depth_np, min_elevation)
                pred_wl[t] = torch.tensor(pred_wl_t, dtype=torch.float32)

                depth_history.update(pred_depth_np)
                current_pred_depth = pred_depth_np

    return pred_wl, gt_wl


def evaluate_predictions(
    pred_wl: torch.Tensor,
    gt_wl: torch.Tensor,
    std_2d: float | None = None,
    num_warmup: int = 10,
) -> dict:
    """Evaluate predicted water levels against ground truth.

    Only timesteps **after** warmup are scored (competition convention).

    Parameters
    ----------
    pred_wl : torch.Tensor, shape [T, N]
        Predicted WSE.
    gt_wl : torch.Tensor, shape [T, N]
        Ground-truth WSE.
    std_2d : float or None
        If given, used to compute a standardised RMSE.
    num_warmup : int
        Number of warm-up timesteps to exclude.

    Returns
    -------
    dict
        Keys: ``mse``, ``rmse``, ``mae``, ``avg_node_rmse``,
        ``standardized_rmse``, ``num_timesteps_evaluated``, ``num_nodes``.
    """
    pred_eval = pred_wl[num_warmup:]
    gt_eval = gt_wl[num_warmup:]

    errors = pred_eval - gt_eval

    mse = (errors**2).mean().item()
    rmse = mse**0.5
    mae = errors.abs().mean().item()

    per_node_mse = (errors**2).mean(dim=0)  # [N]
    per_node_rmse = per_node_mse**0.5
    avg_node_rmse = per_node_rmse.mean().item()

    standardized_rmse = (
        avg_node_rmse / std_2d if std_2d is not None else None
    )

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "avg_node_rmse": avg_node_rmse,
        "standardized_rmse": standardized_rmse,
        "num_timesteps_evaluated": len(pred_eval),
        "num_nodes": pred_eval.shape[1],
    }


# ───────────────────────────────────────────────────────────────────────
#  Checkpointing
# ───────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: SurfaceEngine,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
<<<<<<< HEAD
) -> None:
    """Save a model checkpoint to *path*.

    The checkpoint stores model weights, optimiser state, the current
    epoch, and the validation loss so that training can be resumed
    or the best model reloaded later.
    """
    from pathlib import Path as _Path

    _Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )
=======
    config: dict | None = None,
) -> None:
    """Save a model checkpoint to *path*.

    The checkpoint stores model weights, optimiser state, epoch, and loss.
    If *config* is provided, also stores config, val_rmse, and timestamp
    so training settings are never lost.
    """
    from datetime import datetime
    from pathlib import Path as _Path

    _Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if config is not None:
        state["config"] = config
        state["val_rmse"] = loss
        state["timestamp"] = datetime.now().isoformat()

    torch.save(state, path)
>>>>>>> feat/2d-pipeline
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(
    model: SurfaceEngine,
    path: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Load a model checkpoint from *path*.

<<<<<<< HEAD
    If *optimizer* is provided its state is restored too.
=======
    If *optimizer* is provided and the checkpoint contains optimizer state,
    it is restored too. Supports both legacy checkpoints (epoch, loss,
    model_state_dict, optimizer_state_dict) and new format (+ config,
    val_rmse, timestamp).
>>>>>>> feat/2d-pipeline

    Returns
    -------
    dict
        The raw checkpoint dictionary (keys: ``epoch``, ``loss``, …).
    """
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

<<<<<<< HEAD
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(
        f"  Loaded checkpoint from epoch {checkpoint['epoch']} "
        f"(loss: {checkpoint['loss']:.6f})"
    )
=======
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    loss_val = checkpoint.get("val_rmse", checkpoint.get("loss"))
    print(
        f"  Loaded checkpoint from epoch {checkpoint['epoch']} "
        f"(val_rmse/loss: {loss_val:.6f})"
    )
    if "config" in checkpoint:
        print(f"  Config saved in checkpoint (timestamp: {checkpoint.get('timestamp', 'n/a')})")
>>>>>>> feat/2d-pipeline
    return checkpoint


# ───────────────────────────────────────────────────────────────────────
#  Scheduled sampling helpers
# ───────────────────────────────────────────────────────────────────────

def get_teacher_forcing_ratio(
    epoch: int,
    warmup_epochs: int = 10,
    decay_epochs: int = 30,
<<<<<<< HEAD
) -> float:
    """Compute teacher-forcing ratio for the current epoch.

    Schedule:

    * ``epoch < warmup_epochs``  →  1.0 (always use ground truth)
    * linear decay over ``decay_epochs``  →  1.0 → 0.0
    * ``epoch >= warmup_epochs + decay_epochs``  →  0.0 (always predict)
=======
    num_epochs: int | None = None,
    tf_min_ratio: float | None = None,
) -> float:
    """Compute teacher-forcing ratio for the current epoch.

    Two modes:

    1. Legacy (num_epochs and tf_min_ratio not used):
       * ``epoch < warmup_epochs``  →  1.0 (always use ground truth)
       * linear decay over ``decay_epochs``  →  1.0 → 0.0
       * ``epoch >= warmup_epochs + decay_epochs``  →  0.0 (always predict)

    2. Linear over all epochs (when num_epochs and tf_min_ratio are set):
       * Linear decay from 1.0 to tf_min_ratio over epochs 0 .. num_epochs-1.
>>>>>>> feat/2d-pipeline

    Parameters
    ----------
    epoch : int
        Current epoch.
    warmup_epochs : int
<<<<<<< HEAD
        Epochs with full teacher forcing.
    decay_epochs : int
        Epochs over which the ratio decays linearly.
=======
        Epochs with full teacher forcing (legacy mode).
    decay_epochs : int
        Epochs over which the ratio decays linearly (legacy mode).
    num_epochs : int or None
        Total training epochs (for linear schedule).
    tf_min_ratio : float or None
        Minimum TF ratio at end of training (e.g. 0.3).
>>>>>>> feat/2d-pipeline

    Returns
    -------
    float
        Value in ``[0, 1]``.
    """
<<<<<<< HEAD
=======
    if num_epochs is not None and tf_min_ratio is not None:
        # Linear decay from 1.0 to tf_min_ratio over all epochs
        if num_epochs <= 1:
            return 1.0
        progress = min(1.0, epoch / (num_epochs - 1))
        return 1.0 - (1.0 - tf_min_ratio) * progress

>>>>>>> feat/2d-pipeline
    if epoch < warmup_epochs:
        return 1.0
    if epoch < warmup_epochs + decay_epochs:
        progress = (epoch - warmup_epochs) / decay_epochs
        return 1.0 - progress
    return 0.0


# ───────────────────────────────────────────────────────────────────────
#  Training with scheduled sampling
# ───────────────────────────────────────────────────────────────────────

def train_epoch_scheduled(
    model: SurfaceEngine,
    sample: dict,
    norm_stats: dict,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    warmup_epochs: int = 10,
    decay_epochs: int = 30,
    start_timestep: int = 3,
    max_timesteps: int | None = None,
    grad_clip: float = 1.0,
<<<<<<< HEAD
) -> float:
=======
    num_epochs: int | None = None,
    tf_min_ratio: float | None = None,
) -> tuple[float, float]:
>>>>>>> feat/2d-pipeline
    """Train one epoch on a single event with scheduled sampling.

    At each step, a coin flip (weighted by the teacher-forcing ratio)
    decides whether depth lags come from ground truth or from the
    model's own predictions.

    Parameters
    ----------
    model : SurfaceEngine
    sample : dict
        One event from ``FloodDataset``.
    norm_stats : dict
    optimizer : torch.optim.Optimizer
    epoch : int
    warmup_epochs, decay_epochs : int
<<<<<<< HEAD
        Control the teacher-forcing schedule.
=======
        Control the teacher-forcing schedule (used if tf_min_ratio is None).
>>>>>>> feat/2d-pipeline
    start_timestep : int
        First timestep (must be >= ``num_history`` for valid lags).
    max_timesteps : int or None
        Cap on number of training steps (``None`` → use all).
    grad_clip : float
        Max gradient norm (0 to disable).
<<<<<<< HEAD

    Returns
    -------
    float
        Average loss for this epoch.
=======
    num_epochs : int or None
        Total epochs (for linear TF schedule when tf_min_ratio is set).
    tf_min_ratio : float or None
        Min TF ratio at end of training (linear schedule over num_epochs).

    Returns
    -------
    tuple[float, float]
        (average loss, average gradient norm).
>>>>>>> feat/2d-pipeline
    """
    from src.graph_builder_2d import (
        DepthHistory,
        build_2d_graph,
        get_values_at_timestep,
    )
    from src.utils_2d import get_min_elevation_filled, wse_to_depth

    model.train()

<<<<<<< HEAD
    tf_ratio = get_teacher_forcing_ratio(epoch, warmup_epochs, decay_epochs)
=======
    if num_epochs is not None and tf_min_ratio is not None:
        tf_ratio = get_teacher_forcing_ratio(
            epoch, num_epochs=num_epochs, tf_min_ratio=tf_min_ratio
        )
    else:
        tf_ratio = get_teacher_forcing_ratio(epoch, warmup_epochs, decay_epochs)
>>>>>>> feat/2d-pipeline

    # ── data handles ──────────────────────────────────────────────────
    static_2d = sample["static_2d_nodes"]
    dynamic_2d = sample["dynamic_2d_nodes"]
    min_elevation = get_min_elevation_filled(static_2d)
    num_nodes = len(static_2d)

    max_t = int(dynamic_2d["timestep"].max())
    end_t = (
        max_t
        if max_timesteps is None
        else min(start_timestep + max_timesteps, max_t)
    )

    # ── initialise state ──────────────────────────────────────────────
    hidden = model.init_hidden(num_nodes)
    depth_history = DepthHistory(num_history=3)
    depth_history.initialize_from_ground_truth(sample, t_start=start_timestep)

    wl_prev = get_values_at_timestep(
        dynamic_2d, start_timestep - 1, "water_level", num_nodes
    )
    current_pred_depth = wse_to_depth(wl_prev, min_elevation)

    criterion = nn.MSELoss()
    epoch_loss = 0.0
<<<<<<< HEAD
=======
    epoch_grad_norm_sum = 0.0
>>>>>>> feat/2d-pipeline
    num_steps = 0

    for t in range(start_timestep, end_t):
        # ── teacher-forcing coin flip ─────────────────────────────
        use_tf = random.random() < tf_ratio

        if use_tf:
            data = build_2d_graph(
                sample, norm_stats, t_index=t, predicted_depths=None
            )
            current_depth_tensor = data.y.clone()
        else:
            data = build_2d_graph(
                sample,
                norm_stats,
                t_index=t,
                predicted_depths=depth_history.get_lags(),
            )
            current_depth_tensor = torch.tensor(
                current_pred_depth, dtype=torch.float32
            ).unsqueeze(1)

        # ── target: depth at t+1 ─────────────────────────────────
        if t + 1 > max_t:
            break

        wl_next = get_values_at_timestep(
            dynamic_2d, t + 1, "water_level", num_nodes
        )
        depth_next = wse_to_depth(wl_next, min_elevation)
        target_depth = torch.tensor(
            depth_next, dtype=torch.float32
        ).unsqueeze(1)
        target_delta = target_depth - current_depth_tensor

        # ── forward ───────────────────────────────────────────────
        pred_delta, hidden = model(data.x, data.edge_index, hidden)
        pred_depth = model.predict_depth(current_depth_tensor, pred_delta)

        # Combined loss (absolute depth + delta)
        loss = criterion(pred_depth, target_depth) + 0.5 * criterion(
            pred_delta, target_delta
        )

        optimizer.zero_grad()
        loss.backward()

<<<<<<< HEAD
=======
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float("inf")
        )
>>>>>>> feat/2d-pipeline
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        epoch_loss += loss.item()
<<<<<<< HEAD
=======
        epoch_grad_norm_sum += total_norm.item()
>>>>>>> feat/2d-pipeline
        num_steps += 1

        # ── bookkeeping ───────────────────────────────────────────
        hidden = hidden.detach()
        pred_depth_np = pred_depth.detach().squeeze().numpy()
        depth_history.update(pred_depth_np)
        current_pred_depth = pred_depth_np

<<<<<<< HEAD
    return epoch_loss / max(num_steps, 1)
=======
    n = max(num_steps, 1)
    return epoch_loss / n, epoch_grad_norm_sum / n
>>>>>>> feat/2d-pipeline


def train_model(
    model: SurfaceEngine,
    dataset: torch.utils.data.Dataset,
    model_id: str,
    num_epochs: int = 50,
    lr: float = 0.005,
    warmup_epochs: int = 10,
    decay_epochs: int = 30,
    max_timesteps_per_event: int | None = None,
    print_every: int = 5,
    validate_every: int = 5,
    validation_events: int = 2,
    early_stopping_patience: int = 15,
    checkpoint_dir: str = "checkpoints",
    save_best: bool = True,
<<<<<<< HEAD
=======
    lr_warmup_epochs: int = 0,
    tf_min_ratio: float | None = None,
    grad_clip: float = 1.0,
>>>>>>> feat/2d-pipeline
) -> dict:
    """Full training loop with scheduled sampling, early stopping,
    and checkpointing.

    Parameters
    ----------
    model : SurfaceEngine
    dataset : FloodDataset
        Must be in ``train`` mode.
    model_id : str
    num_epochs : int
    lr : float
    warmup_epochs, decay_epochs : int
<<<<<<< HEAD
        Teacher-forcing schedule.
    max_timesteps_per_event : int or None
        Cap per event (useful for speed during debugging).
=======
        Teacher-forcing schedule (used when tf_min_ratio is None).
    max_timesteps_per_event : int or None
        Cap per event (useful for speed during debugging).
    lr_warmup_epochs : int
        Linear LR warmup: lr starts at lr/10 and ramps to lr over this many epochs (0 = no warmup).
    tf_min_ratio : float or None
        If set, teacher forcing decays linearly from 1.0 to this value over all epochs.
    grad_clip : float
        Max gradient norm for clip_grad_norm_ (0 = no clipping).
>>>>>>> feat/2d-pipeline
    print_every, validate_every : int
        Logging intervals.
    validation_events : int
        Number of events held out for validation.
    early_stopping_patience : int
        Stop training if no validation improvement for this many
        epochs (counted in multiples of *validate_every*).
    checkpoint_dir : str
        Directory to save the best checkpoint.
    save_best : bool
        Whether to persist the best model to disk.

    Returns
    -------
    dict
        Training history with keys ``train_loss``, ``val_loss``,
        ``val_rmse``, ``tf_ratio``, ``lr``, ``best_epoch``,
        ``best_val_rmse``.
    """
    from pathlib import Path as _Path

    from src.utils_2d import compute_normalization_stats

<<<<<<< HEAD
    print(f"Training on Model_{model_id}")
    print(f"Schedule: warmup={warmup_epochs}, decay={decay_epochs}")
=======
    # Build config dict to save in checkpoint so settings are never lost
    training_config = {
        "model_id": model_id,
        "num_epochs": num_epochs,
        "lr": lr,
        "warmup_epochs": warmup_epochs,
        "decay_epochs": decay_epochs,
        "max_timesteps_per_event": max_timesteps_per_event,
        "early_stopping_patience": early_stopping_patience,
        "lr_warmup_epochs": lr_warmup_epochs,
        "tf_min_ratio": tf_min_ratio,
        "grad_clip": grad_clip,
        "validation_events": validation_events,
        "print_every": print_every,
        "validate_every": validate_every,
    }

    print(f"Training on Model_{model_id}")
    if tf_min_ratio is not None:
        print(f"TF schedule: linear 1.0 → {tf_min_ratio} over {num_epochs} epochs")
    else:
        print(f"Schedule: warmup={warmup_epochs}, decay={decay_epochs}")
    if lr_warmup_epochs > 0:
        print(f"LR warmup: {lr_warmup_epochs} epochs (lr/10 → lr)")
    print(f"Grad clip: {grad_clip}")
>>>>>>> feat/2d-pipeline
    print(f"Early stopping patience: {early_stopping_patience}")
    print("-" * 60)

    norm_stats = compute_normalization_stats(dataset, model_id)
    ds_model = dataset.filter_by_model(model_id)
    num_events = len(ds_model)

    # Train / val split (first N events → val, rest → train)
    val_indices = list(range(min(validation_events, num_events)))
    train_indices = list(range(validation_events, num_events))
    if not train_indices:
        train_indices = list(range(num_events))
        val_indices = train_indices[:1]

    print(
        f"Train events: {len(train_indices)}, "
        f"Validation events: {len(val_indices)}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    history: dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "tf_ratio": [],
        "lr": [],
    }

    best_val_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = _Path(checkpoint_dir) / f"model_{model_id}_best.pt"
<<<<<<< HEAD
=======
    base_lr = lr
>>>>>>> feat/2d-pipeline

    t_start = time.time()

    for epoch in range(num_epochs):
<<<<<<< HEAD
        # ── train ─────────────────────────────────────────────────
        model.train()
        epoch_losses: list[float] = []
=======
        # ── LR warmup (linear from lr/10 to lr over first lr_warmup_epochs)
        if lr_warmup_epochs > 0 and epoch < lr_warmup_epochs:
            warmup_factor = 0.1 + 0.9 * (epoch / lr_warmup_epochs)
            for g in optimizer.param_groups:
                g["lr"] = base_lr * warmup_factor

        # ── train ─────────────────────────────────────────────────
        model.train()
        epoch_losses: list[float] = []
        epoch_grad_norms: list[float] = []
>>>>>>> feat/2d-pipeline
        random.shuffle(train_indices)

        for event_idx in train_indices:
            sample = ds_model[event_idx]
<<<<<<< HEAD
            loss = train_epoch_scheduled(
=======
            loss, grad_norm = train_epoch_scheduled(
>>>>>>> feat/2d-pipeline
                model,
                sample,
                norm_stats,
                optimizer,
                epoch,
                warmup_epochs=warmup_epochs,
                decay_epochs=decay_epochs,
                max_timesteps=max_timesteps_per_event,
<<<<<<< HEAD
            )
            epoch_losses.append(loss)

        avg_train_loss = float(np.mean(epoch_losses))
        tf = get_teacher_forcing_ratio(epoch, warmup_epochs, decay_epochs)
=======
                grad_clip=grad_clip,
                num_epochs=num_epochs if tf_min_ratio is not None else None,
                tf_min_ratio=tf_min_ratio,
            )
            epoch_losses.append(loss)
            epoch_grad_norms.append(grad_norm)

        avg_train_loss = float(np.mean(epoch_losses))
        avg_grad_norm = float(np.mean(epoch_grad_norms))
        if tf_min_ratio is not None:
            tf = get_teacher_forcing_ratio(
                epoch, num_epochs=num_epochs, tf_min_ratio=tf_min_ratio
            )
        else:
            tf = get_teacher_forcing_ratio(epoch, warmup_epochs, decay_epochs)
>>>>>>> feat/2d-pipeline
        history["train_loss"].append(avg_train_loss)
        history["tf_ratio"].append(tf)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # ── validate ──────────────────────────────────────────────
        if epoch % validate_every == 0 or epoch == num_epochs - 1:
            model.eval()
            val_rmses: list[float] = []

            with torch.no_grad():
                for vi in val_indices:
                    sample = ds_model[vi]
                    pred_wl, gt_wl = predict_event_2d(
                        model, sample, norm_stats, num_warmup=10
                    )
                    m = evaluate_predictions(pred_wl, gt_wl, num_warmup=10)
                    val_rmses.append(m["rmse"])

            avg_val_rmse = float(np.mean(val_rmses))
            history["val_loss"].append(avg_val_rmse**2)  # MSE
            history["val_rmse"].append(avg_val_rmse)

<<<<<<< HEAD
            scheduler.step(avg_val_rmse)

            # ── improvement check / checkpointing ─────────────
=======
            if epoch >= lr_warmup_epochs:
                scheduler.step(avg_val_rmse)

        # ── improvement check / checkpointing ─────────────
>>>>>>> feat/2d-pipeline
            if avg_val_rmse < best_val_rmse:
                best_val_rmse = avg_val_rmse
                best_epoch = epoch
                epochs_without_improvement = 0

                if save_best:
                    save_checkpoint(
                        model, optimizer, epoch,
                        avg_val_rmse, str(checkpoint_path),
<<<<<<< HEAD
=======
                        config=training_config,
>>>>>>> feat/2d-pipeline
                    )
            else:
                epochs_without_improvement += validate_every

            # ── early stopping ────────────────────────────────
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(
                    f"No improvement for {epochs_without_improvement} epochs"
                )
                break

        # ── log ───────────────────────────────────────────────────
        if epoch % print_every == 0 or epoch == num_epochs - 1:
            elapsed = time.time() - t_start
            current_lr = optimizer.param_groups[0]["lr"]
            msg = (
                f"Epoch {epoch:3d} | Train: {avg_train_loss:.6f} | "
<<<<<<< HEAD
                f"TF: {tf:.2f}"
=======
                f"TF: {tf:.2f} | GradNorm: {avg_grad_norm:.4f}"
>>>>>>> feat/2d-pipeline
            )
            if history["val_rmse"]:
                msg += f" | Val RMSE: {history['val_rmse'][-1]:.4f}"
            msg += f" | LR: {current_lr:.6f} | {elapsed:.0f}s"
            print(msg)

    print("-" * 60)
    print(f"Training complete!")
    print(f"Best Val RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")

    # Reload the best model if a checkpoint was saved
    if save_best and checkpoint_path.exists():
        print("\nLoading best model from checkpoint...")
        load_checkpoint(model, str(checkpoint_path), optimizer)

    history["best_epoch"] = best_epoch  # type: ignore[assignment]
    history["best_val_rmse"] = best_val_rmse  # type: ignore[assignment]

    return history


# ───────────────────────────────────────────────────────────────────────
#  Quick smoke test
# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path as _Path

    from src.config import RAW_DATA_PATH
    from src.dataset import FloodDataset
    from src.graph_builder_2d import build_2d_graph
    from src.utils_2d import compute_normalization_stats

    print("=" * 60)
    print("Testing Training with Early Stopping & Checkpointing")
    print("=" * 60)

    # Load data
    ds = FloodDataset(RAW_DATA_PATH, mode="train")

    # Get dimensions
    sample = ds[0]
    norm_stats = compute_normalization_stats(ds, sample["model_id"])
    data = build_2d_graph(sample, norm_stats, t_index=10)
    in_channels = data.x.shape[1]

    print(f"\nInput features: {in_channels}")

    # Create model
    model = SurfaceEngine(
        in_channels=in_channels,
        hidden_channels=64,
        num_sage_layers=2,
        dropout=0.1,
        max_delta=2.0,
<<<<<<< HEAD
=======
        conv_type="sage",
>>>>>>> feat/2d-pipeline
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Training with early stopping ──────────────────────────────
    print("\n--- Training with Early Stopping (max 20 epochs) ---")

    history = train_model(
        model,
        ds,
        model_id="1",
        num_epochs=20,
        lr=0.005,
        warmup_epochs=5,
        decay_epochs=10,
        max_timesteps_per_event=40,
        print_every=2,
        validate_every=2,
        validation_events=2,
        early_stopping_patience=10,
        checkpoint_dir="checkpoints",
        save_best=True,
    )

    # Verify checkpoint exists
    checkpoint_path = _Path("checkpoints/model_1_best.pt")
    if checkpoint_path.exists():
        print(f"\n✓ Checkpoint saved: {checkpoint_path}")
        print(f"  File size: {checkpoint_path.stat().st_size / 1024:.1f} KB")

    # ── Prediction with best model ────────────────────────────────
    print("\n--- Prediction with Best Model ---")
    sample_test = ds.filter_by_model("1")[0]
    pred_wl, gt_wl = predict_event_2d(
        model, sample_test, norm_stats, num_warmup=10
    )
    metrics = evaluate_predictions(pred_wl, gt_wl, num_warmup=10)

    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  Best epoch: {history['best_epoch']}")
    print(f"  Best Val RMSE: {history['best_val_rmse']:.4f}")

    # ── Test loading checkpoint into fresh model ──────────────────
    print("\n--- Testing Checkpoint Loading ---")
    model_fresh = SurfaceEngine(
        in_channels=in_channels,
        hidden_channels=64,
        num_sage_layers=2,
        dropout=0.1,
        max_delta=2.0,
<<<<<<< HEAD
=======
        conv_type="sage",
>>>>>>> feat/2d-pipeline
    )

    load_checkpoint(model_fresh, str(checkpoint_path))

    pred_wl_fresh, gt_wl = predict_event_2d(
        model_fresh, sample_test, norm_stats, num_warmup=10
    )
    metrics_fresh = evaluate_predictions(pred_wl_fresh, gt_wl, num_warmup=10)

    print(f"  Fresh model RMSE: {metrics_fresh['rmse']:.4f}")

    # Verify predictions match
    diff = (pred_wl - pred_wl_fresh).abs().max().item()
    print(f"  Max diff from original: {diff:.10f} (should be ~0)")

<<<<<<< HEAD
    print("\n✓ Checkpointing and early stopping working!")
=======
    print("\n✓ Checkpointing and early stopping working!")
>>>>>>> feat/2d-pipeline

"""
trainer — Production Training Pipeline with Anti-Drift Protocols.

Implements the full training loop for the Urban Flood Modelling
competition, supporting:

  * **Unified HeteroGNN** (primary, via ``UnifiedFloodModel``)
  * **Decoupled 1D/2D engines** (when Member A/B models are ready)

Core Training Innovations (from PROJECT_BIBLE.md §7)
-----------------------------------------------------
1. **Scheduled Sampling** (Curriculum Learning):
   - Epoch 0–10:  Teacher Forcing = 100%
   - Epoch 11–40: Linear decay from 1.0 → 0.0
   - Epoch 41+:   Student Forcing = 100%

2. **Push-Forward Training**:
   - Loss is computed over K-step rollout trajectories, not single
     steps.  Temporal weighting penalises later steps more heavily
     to combat autoregressive drift.

3. **Variance-Weighted Loss** (Standardized RMSE surrogate):
   - Uses ``FloodLoss`` from ``loss.py`` with clamped inverse-variance
     weights.

4. **Leave-One-Event-Out** cross-validation split.

5. **Mixed-Precision Training** (AMP) for memory efficiency on large
   2D meshes.

6. **Gradient Clipping** to prevent exploding gradients during
   autoregressive training.

See   : IMPLEMENTATION_PLAN.md → Task 3.1, PROJECT_BIBLE.md §7
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.cuda.amp import GradScaler

# Use torch.amp.autocast (new API) with fallback to torch.cuda.amp.autocast (old)
try:
    from torch.amp import autocast as _autocast
    def _amp_context(device_type: str, enabled: bool):
        return _autocast(device_type=device_type, enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast as _cuda_autocast
    def _amp_context(device_type: str, enabled: bool):
        return _cuda_autocast(enabled=enabled)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kw):  # type: ignore[misc]
        return iterable

from src.dataset import FloodDataset
from src.graph_builder_unified import build_unified_graph, get_feature_dims
from src.loss import (
    FloodLoss,
    SRMSEAccumulator,
    standardized_rmse_metric,
)
from src.model_unified import UnifiedFloodModel


# =====================================================================
#  Hyperparameter Configuration
# =====================================================================

@dataclass
class TrainConfig:
    """All training hyperparameters in one place.

    Sensible defaults tuned for the Urban Flood competition.
    Modify via constructor kwargs or dict unpacking.

    Examples
    --------
    >>> cfg = TrainConfig(lr=5e-4, epochs=80, hidden_channels=128)
    >>> cfg = TrainConfig(**yaml.safe_load(open("config.yaml")))
    """

    # ── Data ──────────────────────────────────────────────────────
    data_root: str = "data"
    model_ids: List[str] = field(default_factory=lambda: ["1", "2"])
    # ── Validation ────────────────────────────────────────────────
    val_event_id: str = "3,9,15"  # comma-separated event IDs present in BOTH models
    # These 3 events exist in both Model_1 and Model_2 train splits,
    # giving 6 total val events (3 per model) for a stable SRMSE signal.
    mode: str = "train"

    # ── Architecture ──────────────────────────────────────────────
    hidden_channels: int = 64
    num_gnn_layers: int = 3
    num_gru_layers: int = 1
    dropout: float = 0.1

    # ── Training ──────────────────────────────────────────────────
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0
    batch_accumulation: int = 1  # gradient accumulation steps

    # ── Scheduled Sampling (Curriculum Learning) ──────────────────
    tf_warmup_epochs: int = 10     # 100% teacher forcing
    tf_decay_epochs: int = 30      # linear decay over this many epochs
    tf_min_ratio: float = 0.0      # final teacher forcing ratio

    # ── Push-Forward Training ─────────────────────────────────────
    pushforward_K: int = 30        # target rollout length (or max K for progressive)
    temporal_scheme: Literal["uniform", "linear", "exponential"] = "linear"
    use_push_forward: bool = True

    # ── Progressive K Curriculum (anti-AR-collapse) ───────────────
    # Start training with short rollouts (K_start), then increase
    # linearly to pushforward_K over K_ramp_epochs.  This lets the
    # model master short-horizon prediction before being challenged
    # on longer trajectories.  Without this, K=15 from epoch 0
    # floods the early loss with uninformative late-step errors.
    progressive_K: bool = True
    K_start: int = 3               # initial rollout length
    K_ramp_epochs: int = 30        # epochs to reach full pushforward_K

    # ── Randomized Spinup (exposure bias mitigation) ──────────────
    # During training, randomize the spinup length between
    # spinup_min and spinup_max.  This presents the model with
    # varying qualities of GRU hidden state, preventing it from
    # overfitting to "perfect 10-step spinup" conditions that
    # differ from validation when errors accumulate.
    randomize_spinup: bool = True
    spinup_min: int = 3
    spinup_max: int = 10

    # ── Training Noise Injection (Exposure Bias Mitigation) ───────
    # Disabled by default: previous runs showed it was counterproductive
    # because it corrupts training signal during the critical early
    # learning phase.  Progressive K + randomized spinup achieves
    # the same goal more cleanly.
    training_noise_std: float = 0.0

    # ── Loss ──────────────────────────────────────────────────────
    loss_variant: Literal["mse", "huber"] = "mse"
    huber_delta: float = 1.0
    clamp_weights: float = 10.0   # was 100; lower prevents dry-node weight explosion
    alpha: float = 0.5  # 1D/2D balance

    # ── LR Scheduler ─────────────────────────────────────────────
    scheduler: Literal["cosine", "plateau", "cosine_warm", "onecycle", "none"] = "cosine"
    cosine_T_max: Optional[int] = None  # defaults to epochs
    cosine_eta_min: float = 1e-6
    cosine_warm_T0: int = 15       # period for first restart (CosineWarmRestarts)
    cosine_warm_T_mult: int = 2    # multiply period after each restart
    plateau_patience: int = 5
    plateau_factor: float = 0.5
    # OneCycleLR: single aggressive cycle with warmup + cosine decay.
    # Superior to cosine_warm for our setup because it avoids the
    # destructive warm restarts that reset learning midway.
    onecycle_pct_start: float = 0.3   # fraction of epochs for LR warmup
    onecycle_div_factor: float = 25.0 # initial_lr = max_lr / div_factor

    # ── Early Stopping ────────────────────────────────────────────
    early_stop_patience: int = 15
    early_stop_min_delta: float = 1e-5

    # ── Mixed Precision ──────────────────────────────────────────
    use_amp: bool = True

    # ── Checkpointing ────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5
    save_best: bool = True

    # ── Logging ──────────────────────────────────────────────────
    log_dir: str = "logs"
    log_every_n_steps: int = 1
    verbose: bool = True

    # ── Device ───────────────────────────────────────────────────
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"

    def resolve_device(self) -> torch.device:
        """Resolve the device string to a ``torch.device``."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a JSON-compatible dictionary."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, list):
                d[k] = list(v)
            else:
                d[k] = v
        return d


# =====================================================================
#  Scheduled Sampling Scheduler
# =====================================================================

class TeacherForcingScheduler:
    """Computes the teacher forcing ratio for each epoch.

    Phase 1 (warmup):  ratio = 1.0
    Phase 2 (decay):   linear from 1.0 → tf_min_ratio
    Phase 3 (student): ratio = tf_min_ratio
    """

    def __init__(
        self,
        warmup_epochs: int = 10,
        decay_epochs: int = 30,
        min_ratio: float = 0.0,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.min_ratio = min_ratio

    def get_ratio(self, epoch: int) -> float:
        """Return the teacher forcing ratio for the given epoch."""
        if epoch < self.warmup_epochs:
            return 1.0

        decay_progress = (epoch - self.warmup_epochs) / max(self.decay_epochs, 1)
        if decay_progress >= 1.0:
            return self.min_ratio

        return 1.0 - (1.0 - self.min_ratio) * decay_progress

    def __repr__(self) -> str:
        return (
            f"TeacherForcingScheduler(warmup={self.warmup_epochs}, "
            f"decay={self.decay_epochs}, min={self.min_ratio})"
        )


# =====================================================================
#  Early Stopping
# =====================================================================

class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-5,
        mode: Literal["min", "max"] = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, score: float) -> bool:
        """Check if training should stop.

        Returns ``True`` if early stopping criterion is met.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# =====================================================================
#  Training History Tracker
# =====================================================================

class TrainingHistory:
    """Records per-epoch metrics for training and validation."""

    def __init__(self) -> None:
        self.train_loss: List[float] = []
        self.val_srmse: List[float] = []
        self.lr_history: List[float] = []
        self.tf_ratio: List[float] = []
        self.epoch_times: List[float] = []
        self.best_epoch: int = 0
        self.best_val_srmse: float = float("inf")
        self.extra: Dict[str, List[float]] = {}

    def log(
        self,
        epoch: int,
        train_loss: float,
        val_srmse: float,
        lr: float,
        tf: float,
        elapsed: float,
        **kwargs: float,
    ) -> None:
        """Record one epoch of metrics."""
        self.train_loss.append(train_loss)
        self.val_srmse.append(val_srmse)
        self.lr_history.append(lr)
        self.tf_ratio.append(tf)
        self.epoch_times.append(elapsed)

        if val_srmse < self.best_val_srmse:
            self.best_val_srmse = val_srmse
            self.best_epoch = epoch

        for k, v in kwargs.items():
            if k not in self.extra:
                self.extra[k] = []
            self.extra[k].append(v)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize history to a JSON-friendly dictionary."""
        return {
            "train_loss": self.train_loss,
            "val_srmse": self.val_srmse,
            "lr_history": self.lr_history,
            "tf_ratio": self.tf_ratio,
            "epoch_times": self.epoch_times,
            "best_epoch": self.best_epoch,
            "best_val_srmse": self.best_val_srmse,
            **self.extra,
        }

    def save(self, path: str | Path) -> None:
        """Save history to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary_str(self) -> str:
        """Human-readable training summary."""
        n = len(self.train_loss)
        if n == 0:
            return "No training history recorded."
        total_time = sum(self.epoch_times)
        return (
            f"Training Summary ({n} epochs, {total_time:.0f}s total):\n"
            f"  Best val SRMSE : {self.best_val_srmse:.6f} (epoch {self.best_epoch})\n"
            f"  Final train loss: {self.train_loss[-1]:.6f}\n"
            f"  Final val SRMSE : {self.val_srmse[-1]:.6f}\n"
            f"  Avg epoch time  : {total_time / n:.1f}s\n"
        )


# =====================================================================
#  Checkpoint Management
# =====================================================================

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    epoch: int,
    history: TrainingHistory,
    config: TrainConfig,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "history": history.to_dict(),
        "config": config.to_dict(),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Any = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint.

    Returns the full checkpoint dict (with epoch, history, etc.).
    """
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt


# =====================================================================
#  Core Training Functions
# =====================================================================

def _train_one_event_unified(
    model: UnifiedFloodModel,
    data: Any,  # HeteroData
    criterion: FloodLoss,
    cfg: TrainConfig,
    tf_ratio: float,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    effective_K: Optional[int] = None,
) -> Tuple[float, Dict[str, float]]:
    """Train on a single event using the Unified HeteroGNN.

    Supports both single-step and push-forward training modes.
    Uses progressive K curriculum and randomized spinup to combat
    the autoregressive train-val distribution shift.

    Parameters
    ----------
    effective_K : int or None
        Current push-forward K for this epoch (from progressive
        curriculum).  If None, uses cfg.pushforward_K.

    Returns
    -------
    (loss_value, breakdown_dict)
    """
    data = data.to(device)
    model.train()

    T = data.num_timesteps
    n_1d = data["node_1d"].num_nodes
    n_2d = data["node_2d"].num_nodes

    # ── Training Noise Injection (optional, off by default) ──────
    noise_applied = False
    orig_dyn_1d = orig_dyn_2d = None
    if cfg.training_noise_std > 0 and tf_ratio > 0:
        noise_scale = cfg.training_noise_std * (1.0 - tf_ratio + 0.1)
        orig_dyn_1d = data["node_1d"].dynamic
        orig_dyn_2d = data["node_2d"].dynamic
        data["node_1d"].dynamic = orig_dyn_1d + noise_scale * torch.randn_like(orig_dyn_1d)
        data["node_2d"].dynamic = orig_dyn_2d + noise_scale * torch.randn_like(orig_dyn_2d)
        noise_applied = True

    # Resolve effective K (progressive curriculum)
    K_target = effective_K if effective_K is not None else cfg.pushforward_K

    if cfg.use_push_forward:
        # ── Push-Forward Training ─────────────────────────────────
        # Randomize spinup length to prevent overfitting to "perfect
        # 10-step spinup" hidden states.  In validation, errors
        # contaminate hidden states progressively — this simulates it.
        if cfg.randomize_spinup:
            max_spinup = min(cfg.spinup_max, T - K_target)
            min_spinup = min(cfg.spinup_min, max_spinup)
            spinup_steps = int(np.random.randint(min_spinup, max(max_spinup + 1, min_spinup + 1)))
        else:
            spinup_steps = min(10, T - K_target)
        spinup_steps = max(1, spinup_steps)

        K = min(K_target, T - spinup_steps)
        if K <= 0:
            K = min(K_target, T)
            spinup_steps = 0

        # Forward pass under autocast (model inference in fp16)
        with _amp_context(device.type, cfg.use_amp):
            # Spin-up phase: build hidden state
            h_1d, h_2d = model.init_hidden(n_1d, n_2d, device)
            with torch.no_grad():
                for t in range(spinup_steps):
                    _, _, h_1d, h_2d = model.step(data, t, h_1d, h_2d)

            # Detach hidden states before prediction phase
            h_1d = [h.detach() for h in h_1d]
            h_2d = [h.detach() for h in h_2d]

            # Push-forward rollout
            preds_1d, preds_2d, targets_1d, targets_2d = (
                model.pushforward_rollout(
                    data,
                    start_t=spinup_steps,
                    K=K,
                    teacher_forcing_ratio=tf_ratio,
                    h_1d=h_1d,
                    h_2d=h_2d,
                )
            )

        # CRITICAL: Loss computation in float32 (outside autocast)
        # fp16 max is ~65504 — inverse-variance weights cause overflow.
        total_loss, breakdown = criterion.forward_combined(
            preds_1d.float(), targets_1d.float(),
            preds_2d.float(), targets_2d.float(),
            use_push_forward=True,
        )
    else:
        # ── Full Rollout Training ─────────────────────────────────
        # Forward pass under autocast (model inference in fp16)
        with _amp_context(device.type, cfg.use_amp):
            preds_1d, preds_2d = model.rollout(
                data,
                spinup_steps=min(10, T - 1),
                teacher_forcing_ratio=tf_ratio,
            )

        targets_1d = data["node_1d"].y  # [T, N_1d]
        targets_2d = data["node_2d"].y  # [T, N_2d]

        # CRITICAL: Loss computation in float32 (outside autocast)
        skip = min(10, T - 1)
        total_loss, breakdown = criterion.forward_combined(
            preds_1d[skip:].float(), targets_1d[skip:].float(),
            preds_2d[skip:].float(), targets_2d[skip:].float(),
        )

    # ── Restore original dynamic features after noise injection ──
    if noise_applied:
        data["node_1d"].dynamic = orig_dyn_1d
        data["node_2d"].dynamic = orig_dyn_2d

    return total_loss, {k: v.item() for k, v in breakdown.items()}


@torch.no_grad()
def _validate_one_event_unified(
    model: UnifiedFloodModel,
    data: Any,  # HeteroData
    stds_1d: Tensor,
    stds_2d: Tensor,
    device: torch.device,
    spinup_steps: int = 10,
) -> Tuple[float, float, float]:
    """Validate on a single event with full autoregressive rollout.

    Returns
    -------
    (srmse_1d, srmse_2d, srmse_combined)
        Per-node-type SRMSE and their equal-weight average.
    """
    data = data.to(device)
    model.eval()
    T = data.num_timesteps

    # Full autoregressive rollout (teacher_forcing_ratio=0.0)
    preds_1d, preds_2d = model.rollout(
        data,
        spinup_steps=min(spinup_steps, T - 1),
        teacher_forcing_ratio=0.0,
    )

    targets_1d = data["node_1d"].y.to(device)
    targets_2d = data["node_2d"].y.to(device)

    # Compute SRMSE only on prediction period (skip spinup)
    skip = min(spinup_steps, T - 1)
    srmse_1d = standardized_rmse_metric(
        preds_1d[skip:], targets_1d[skip:], stds_1d.to(device)
    ).item()
    srmse_2d = standardized_rmse_metric(
        preds_2d[skip:], targets_2d[skip:], stds_2d.to(device)
    ).item()

    # Competition: Mean over node types (50/50 balance)
    srmse_combined = (srmse_1d + srmse_2d) / 2.0

    return srmse_1d, srmse_2d, srmse_combined


# =====================================================================
#  Main Trainer Class
# =====================================================================

class UnifiedTrainer:
    """End-to-end trainer for the UnifiedFloodModel.

    Manages the full training lifecycle: data preparation, model
    construction, training with curriculum learning, validation,
    checkpointing, and early stopping.

    Parameters
    ----------
    cfg : TrainConfig
        Training configuration.  See :class:`TrainConfig` for all
        available hyperparameters.

    Examples
    --------
    >>> cfg = TrainConfig(epochs=60, lr=1e-3, hidden_channels=64)
    >>> trainer = UnifiedTrainer(cfg)
    >>> trainer.setup()
    >>> history = trainer.train()
    >>> print(history.summary_str())
    """

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.device = cfg.resolve_device()

        # Populated by setup()
        self.model: Optional[UnifiedFloodModel] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Any = None
        self.criterion: Optional[FloodLoss] = None
        self.scaler: Optional[GradScaler] = None
        self.tf_scheduler: Optional[TeacherForcingScheduler] = None
        self.early_stopper: Optional[EarlyStopping] = None
        self.history: TrainingHistory = TrainingHistory()

        # Data
        self.train_graphs: List[Any] = []  # pre-built HeteroData objects
        self.val_graphs: List[Any] = []
        self.stds_1d: Optional[Tensor] = None
        self.stds_2d: Optional[Tensor] = None

        self._is_setup = False

    # ------------------------------------------------------------------ #
    #  Setup / Data Preparation
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        """Prepare data, model, optimizer, loss, and schedulers.

        This is separate from ``__init__`` so users can inspect the
        config before committing to the (expensive) setup phase.
        """
        if self.cfg.verbose:
            print("=" * 60)
            print("  UNIFIED TRAINER — SETUP")
            print("=" * 60)

        # ── 1. Load dataset and build graphs ──────────────────────
        self._prepare_data()

        # ── 2. Construct model ────────────────────────────────────
        self._build_model()

        # ── 3. Loss function ──────────────────────────────────────
        self.criterion = FloodLoss(
            node_stds_1d=self.stds_1d,
            node_stds_2d=self.stds_2d,
            clamp_weights=self.cfg.clamp_weights,
            alpha=self.cfg.alpha,
            temporal_scheme=self.cfg.temporal_scheme,
            loss_variant=self.cfg.loss_variant,
            huber_delta=self.cfg.huber_delta,
        ).to(self.device)

        # ── 4. Optimizer ──────────────────────────────────────────
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        # ── 5. LR Scheduler ──────────────────────────────────────
        self._build_scheduler()

        # ── 6. AMP Scaler ────────────────────────────────────────
        if self.cfg.use_amp and self.device.type == "cuda":
            try:
                self.scaler = torch.amp.GradScaler("cuda")
            except TypeError:
                self.scaler = GradScaler()  # fallback for older PyTorch
        else:
            self.scaler = None

        # ── 7. Curriculum Learning ────────────────────────────────
        self.tf_scheduler = TeacherForcingScheduler(
            warmup_epochs=self.cfg.tf_warmup_epochs,
            decay_epochs=self.cfg.tf_decay_epochs,
            min_ratio=self.cfg.tf_min_ratio,
        )

        # ── 8. Early Stopping ────────────────────────────────────
        self.early_stopper = EarlyStopping(
            patience=self.cfg.early_stop_patience,
            min_delta=self.cfg.early_stop_min_delta,
            mode="min",
        )

        self._is_setup = True

        if self.cfg.verbose:
            print(f"\nDevice        : {self.device}")
            print(f"Train events  : {len(self.train_graphs)}")
            print(f"Val events    : {len(self.val_graphs)}")
            print(f"Model params  : {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"TF Schedule   : {self.tf_scheduler}")
            print(f"Push-forward  : K={self.cfg.pushforward_K}, scheme={self.cfg.temporal_scheme}")
            print("=" * 60)

    def _prepare_data(self) -> None:
        """Load dataset, split, compute stds, and pre-build graphs."""
        dataset = FloodDataset(self.cfg.data_root, mode=self.cfg.mode)

        # We train per-model but build graphs for all requested models.
        # This mirrors the competition structure: both Model_1 and Model_2
        # must be predicted, so the model should generalise across them.
        all_train_events: List[Dict[str, Any]] = []
        all_val_events: List[Dict[str, Any]] = []
        all_stds: Dict[str, Dict[str, np.ndarray]] = {}

        for mid in self.cfg.model_ids:
            model_ds = dataset.filter_by_model(mid)

            if not model_ds.events:
                print(f"  WARNING: No events for Model_{mid} — skipping.")
                continue

            # Compute node stds for the loss function
            stds = dataset.compute_node_stds(model_id=mid)
            all_stds[mid] = stds.get(mid, {"1d": np.array([]), "2d": np.array([])})

            # Leave-One-Event-Out split (supports comma-separated val IDs)
            available_events = model_ds.get_event_ids()
            val_eids = [v.strip() for v in self.cfg.val_event_id.split(",")]

            # Filter to val events that actually exist for this model
            valid_val_eids = [v for v in val_eids if v in available_events]
            if not valid_val_eids:
                # Fallback: use last 2 available events
                valid_val_eids = available_events[-2:] if len(available_events) >= 2 else available_events[-1:]
                print(
                    f"  Val events {val_eids} not in Model_{mid}. "
                    f"Using {valid_val_eids} instead."
                )

            # Split: val = events matching any valid_val_eid, train = rest
            val_eid_set = set(valid_val_eids)
            train_ds = model_ds._shallow_copy()
            val_ds = model_ds._shallow_copy()
            train_ds.events = [e for e in model_ds.events if e["event_id"] not in val_eid_set]
            val_ds.events = [e for e in model_ds.events if e["event_id"] in val_eid_set]

            if self.cfg.verbose:
                print(f"  Model_{mid}: {len(train_ds)} train, {len(val_ds)} val events")

            # Pre-build graphs (expensive but done once)
            for i in tqdm(
                range(len(train_ds)),
                desc=f"  Building train graphs (Model_{mid})",
                disable=not self.cfg.verbose,
            ):
                sample = train_ds[i]
                graph = build_unified_graph(sample)
                all_train_events.append(graph)

            for i in tqdm(
                range(len(val_ds)),
                desc=f"  Building val graphs (Model_{mid})",
                disable=not self.cfg.verbose,
            ):
                sample = val_ds[i]
                graph = build_unified_graph(sample)
                all_val_events.append(graph)

        self.train_graphs = all_train_events
        self.val_graphs = all_val_events

        # ── Per-model static feature z-score normalization ────────
        # Model_1 has elevations ~293-360m, Model_2 ~23-55m.
        # Without normalization, the GNN sees wildly different feature
        # scales between models, hindering cross-model generalization.
        # Compute per-model mean/std from static features and apply.
        self._normalize_static_features()

        # Aggregate stds across models (use first model's dims as reference)
        # For models with different node counts, we keep model-specific stds
        # and apply them event-by-event during validation.
        # For training, we average the stds per model and use them in the loss.
        self._per_model_stds = all_stds
        self._build_loss_stds()

    def _build_loss_stds(self) -> None:
        """Build σ tensors for the loss from per-model stds.

        Since the unified model trains on both Model_1 and Model_2 which
        may have different node counts, we handle stds per-event during
        training.  For the loss criterion, we store stds from the first
        available model as a placeholder — the actual per-event stds are
        passed during training.
        """
        for mid, stds in self._per_model_stds.items():
            if len(stds.get("1d", [])) > 0 and self.stds_1d is None:
                self.stds_1d = torch.from_numpy(stds["1d"]).float()
            if len(stds.get("2d", [])) > 0 and self.stds_2d is None:
                self.stds_2d = torch.from_numpy(stds["2d"]).float()

        # Fallback: if stds are still None, use ones (unweighted)
        if self.stds_1d is None:
            self.stds_1d = torch.ones(1)
        if self.stds_2d is None:
            self.stds_2d = torch.ones(1)

    def _normalize_static_features(self) -> None:
        """Apply per-model z-score normalization to static node features.

        Model_1 (elevations 293-360m) and Model_2 (elevations 23-55m)
        have very different absolute feature scales.  Without normalization,
        the shared GNN weights must simultaneously handle both scales,
        which is suboptimal.

        For each model, we compute mean/std from the static features of
        the FIRST training graph (all events for a given model share the
        same static features) and normalize all graphs for that model.

        Features that should NOT be normalized (have physical meaning in
        the model forward pass): we normalize all static columns but
        store the raw ``invert_elev``, ``capacity``, and ``baseline``
        separately (these are not part of ``x`` that gets normalized).
        """
        all_graphs = self.train_graphs + self.val_graphs
        if not all_graphs:
            return

        # Group graphs by model_id
        model_groups: Dict[str, List[Any]] = {}
        for g in all_graphs:
            mid = g.model_id
            if mid not in model_groups:
                model_groups[mid] = []
            model_groups[mid].append(g)

        for mid, graphs in model_groups.items():
            ref = graphs[0]  # all share same static features

            # 1D static: z-score normalize
            x_1d = ref["node_1d"].x  # [N_1d, F]
            mean_1d = x_1d.mean(dim=0, keepdim=True)
            std_1d = x_1d.std(dim=0, keepdim=True).clamp(min=1e-6)
            for g in graphs:
                g["node_1d"].x = (g["node_1d"].x - mean_1d) / std_1d

            # 2D static: z-score normalize
            x_2d = ref["node_2d"].x  # [N_2d, F]
            mean_2d = x_2d.mean(dim=0, keepdim=True)
            std_2d = x_2d.std(dim=0, keepdim=True).clamp(min=1e-6)
            for g in graphs:
                g["node_2d"].x = (g["node_2d"].x - mean_2d) / std_2d

            if self.cfg.verbose:
                print(
                    f"  Model_{mid}: static features z-score normalized "
                    f"(1D: {x_1d.shape[1]} feats, 2D: {x_2d.shape[1]} feats)"
                )

            # Store stats for inference-time normalization
            if not hasattr(self, '_static_norm_stats'):
                self._static_norm_stats: Dict[str, Dict[str, Tensor]] = {}
            self._static_norm_stats[mid] = {
                "mean_1d": mean_1d.squeeze(0),
                "std_1d": std_1d.squeeze(0),
                "mean_2d": mean_2d.squeeze(0),
                "std_2d": std_2d.squeeze(0),
            }

    def _build_model(self) -> None:
        """Construct the UnifiedFloodModel from the first training graph."""
        if not self.train_graphs:
            raise RuntimeError("No training graphs available. Check data_root.")

        sample_graph = self.train_graphs[0]
        dims = get_feature_dims(sample_graph)

        self.model = UnifiedFloodModel(
            in_channels_1d_static=dims["in_channels_1d_static"],
            in_channels_1d_dynamic=dims["in_channels_1d_dynamic"],
            in_channels_2d_static=dims["in_channels_2d_static"],
            in_channels_2d_dynamic=dims["in_channels_2d_dynamic"],
            hidden_channels=self.cfg.hidden_channels,
            num_gnn_layers=self.cfg.num_gnn_layers,
            num_gru_layers=self.cfg.num_gru_layers,
            dropout=self.cfg.dropout,
        ).to(self.device)

        if self.cfg.verbose:
            print(self.model.summarise())

    def _build_scheduler(self) -> None:
        """Build the learning rate scheduler."""
        if self.cfg.scheduler == "cosine":
            T_max = self.cfg.cosine_T_max or self.cfg.epochs
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=self.cfg.cosine_eta_min,
            )
        elif self.cfg.scheduler == "cosine_warm":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.cfg.cosine_warm_T0,
                T_mult=self.cfg.cosine_warm_T_mult,
                eta_min=self.cfg.cosine_eta_min,
            )
        elif self.cfg.scheduler == "onecycle":
            # OneCycleLR: single cosine cycle with warmup phase.
            # steps_per_epoch=1 because scheduler.step() is called
            # once per epoch (not per batch) in our training loop.
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.cfg.lr,
                epochs=self.cfg.epochs,
                steps_per_epoch=1,
                pct_start=self.cfg.onecycle_pct_start,
                div_factor=self.cfg.onecycle_div_factor,
                final_div_factor=1e4,
            )
        elif self.cfg.scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=self.cfg.plateau_patience,
                factor=self.cfg.plateau_factor,
            )
        else:
            self.scheduler = None

    # ------------------------------------------------------------------ #
    #  Training Loop
    # ------------------------------------------------------------------ #

    def train(self, resume_from: Optional[str | Path] = None) -> TrainingHistory:
        """Run the full training loop.

        Parameters
        ----------
        resume_from : str or Path or None
            Path to a checkpoint to resume from.

        Returns
        -------
        TrainingHistory
            Complete training history for plotting/analysis.
        """
        if not self._is_setup:
            self.setup()

        start_epoch = 0

        # Resume from checkpoint
        if resume_from and Path(resume_from).exists():
            ckpt = load_checkpoint(
                resume_from, self.model, self.optimizer, self.scheduler, self.device
            )
            start_epoch = ckpt.get("epoch", 0) + 1
            if self.cfg.verbose:
                print(f"Resumed from epoch {start_epoch}")

        # Create output directories
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir = Path(self.cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(log_dir / "train_config.json", "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2)

        best_val = float("inf")

        for epoch in range(start_epoch, self.cfg.epochs):
            epoch_start = time.time()

            # Get curriculum learning ratio
            tf_ratio = self.tf_scheduler.get_ratio(epoch)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Compute effective K for logging
            if self.cfg.progressive_K and self.cfg.use_push_forward:
                _prog = min(epoch / max(self.cfg.K_ramp_epochs, 1), 1.0)
                _eff_K = int(
                    self.cfg.K_start + _prog * (self.cfg.pushforward_K - self.cfg.K_start)
                )
                _eff_K = max(self.cfg.K_start, min(_eff_K, self.cfg.pushforward_K))
            else:
                _eff_K = self.cfg.pushforward_K

            # ── Train one epoch ───────────────────────────────────
            train_loss, train_breakdown = self._train_epoch(epoch, tf_ratio)

            # ── Validate ──────────────────────────────────────────
            val_srmse, val_breakdown = self._validate_epoch()

            elapsed = time.time() - epoch_start

            # ── Log ───────────────────────────────────────────────
            self.history.log(
                epoch=epoch,
                train_loss=train_loss,
                val_srmse=val_srmse,
                lr=current_lr,
                tf=tf_ratio,
                elapsed=elapsed,
                **train_breakdown,
                **{f"val_{k}": v for k, v in val_breakdown.items()},
            )

            if self.cfg.verbose:
                phase = (
                    "WARMUP" if epoch < self.cfg.tf_warmup_epochs
                    else "DECAY" if tf_ratio > self.cfg.tf_min_ratio
                    else "STUDENT"
                )
                print(
                    f"Epoch {epoch:>3d}/{self.cfg.epochs} "
                    f"[{phase:>7s} tf={tf_ratio:.2f} K={_eff_K:>2d}] "
                    f"| loss={train_loss:.5f} "
                    f"| val_srmse={val_srmse:.5f} "
                    f"| lr={current_lr:.2e} "
                    f"| {elapsed:.1f}s"
                    + (f" *BEST*" if val_srmse < best_val else "")
                )

            # ── Checkpointing ─────────────────────────────────────
            if val_srmse < best_val:
                best_val = val_srmse
                if self.cfg.save_best:
                    save_checkpoint(
                        ckpt_dir / "best_model.pt",
                        self.model, self.optimizer, self.scheduler,
                        epoch, self.history, self.cfg,
                        extra={"static_norm_stats": getattr(self, '_static_norm_stats', {})},
                    )

            if (epoch + 1) % self.cfg.save_every_n_epochs == 0:
                save_checkpoint(
                    ckpt_dir / f"epoch_{epoch:03d}.pt",
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.history, self.cfg,
                    extra={"static_norm_stats": getattr(self, '_static_norm_stats', {})},
                )

            # ── LR Scheduler Step ─────────────────────────────────
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_srmse)
                else:
                    self.scheduler.step()

            # ── Early Stopping ────────────────────────────────────
            if self.early_stopper.step(val_srmse):
                if self.cfg.verbose:
                    print(
                        f"\nEarly stopping at epoch {epoch}. "
                        f"Best val SRMSE: {self.early_stopper.best_score:.6f}"
                    )
                break

        # Save final history
        self.history.save(log_dir / "training_history.json")

        if self.cfg.verbose:
            print("\n" + self.history.summary_str())

        return self.history

    def _train_epoch(
        self, epoch: int, tf_ratio: float
    ) -> Tuple[float, Dict[str, float]]:
        """Train one full epoch over all training events.

        Shuffles event order each epoch for stochastic training.
        Computes progressive K for push-forward curriculum.

        Returns
        -------
        (avg_loss, avg_breakdown)
        """
        self.model.train()
        total_loss = 0.0
        breakdown_accum: Dict[str, float] = {}
        n_events = len(self.train_graphs)

        # ── Progressive K Curriculum ──────────────────────────────
        # Start with K_start and ramp linearly to pushforward_K.
        if self.cfg.progressive_K and self.cfg.use_push_forward:
            progress = min(epoch / max(self.cfg.K_ramp_epochs, 1), 1.0)
            effective_K = int(
                self.cfg.K_start + progress * (self.cfg.pushforward_K - self.cfg.K_start)
            )
            effective_K = max(self.cfg.K_start, min(effective_K, self.cfg.pushforward_K))
        else:
            effective_K = self.cfg.pushforward_K

        # Shuffle event order
        event_order = np.random.permutation(n_events)

        self.optimizer.zero_grad()

        for step_idx, event_idx in enumerate(
            tqdm(event_order, desc=f"  Train epoch {epoch}", disable=not self.cfg.verbose, leave=False)
        ):
            data = self.train_graphs[event_idx]

            # Get per-event stds for the loss
            model_id = data.model_id
            stds = self._per_model_stds.get(model_id, {})
            stds_1d = torch.from_numpy(stds.get("1d", np.ones(data["node_1d"].num_nodes))).float()
            stds_2d = torch.from_numpy(stds.get("2d", np.ones(data["node_2d"].num_nodes))).float()

            # Update criterion stds for this event
            event_criterion = FloodLoss(
                node_stds_1d=stds_1d,
                node_stds_2d=stds_2d,
                clamp_weights=self.cfg.clamp_weights,
                alpha=self.cfg.alpha,
                temporal_scheme=self.cfg.temporal_scheme,
                loss_variant=self.cfg.loss_variant,
                huber_delta=self.cfg.huber_delta,
            ).to(self.device)

            try:
                loss, breakdown = _train_one_event_unified(
                    self.model, data, event_criterion, self.cfg,
                    tf_ratio, self.device, self.scaler,
                    effective_K=effective_K,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(f"  OOM on event {data.event_id} — skipping")
                    continue
                raise

            # ── NaN guard ─────────────────────────────────────────
            # If loss is NaN/Inf (numerical instability), skip this
            # event entirely to prevent poisoning model weights.
            if not torch.isfinite(loss):
                self.optimizer.zero_grad()  # discard any partial grads
                if self.cfg.verbose:
                    print(f"  NaN/Inf loss on event {data.event_id} — skipping")
                continue

            # Gradient accumulation
            loss_scaled = loss / self.cfg.batch_accumulation

            if self.scaler is not None:
                self.scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if (step_idx + 1) % self.cfg.batch_accumulation == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss += loss.item()
            for k, v in breakdown.items():
                breakdown_accum[k] = breakdown_accum.get(k, 0.0) + v

        # Handle remaining gradients
        if n_events % self.cfg.batch_accumulation != 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip_norm
                )
                self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / max(n_events, 1)
        avg_breakdown = {k: v / max(n_events, 1) for k, v in breakdown_accum.items()}

        return avg_loss, avg_breakdown

    def _validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validate over all validation events.

        Returns
        -------
        (avg_srmse, breakdown)
        """
        if not self.val_graphs:
            return float("inf"), {}

        self.model.eval()
        accumulator = SRMSEAccumulator()

        for data in self.val_graphs:
            model_id = data.model_id
            event_id = data.event_id

            stds = self._per_model_stds.get(model_id, {})
            stds_1d = torch.from_numpy(
                stds.get("1d", np.ones(data["node_1d"].num_nodes))
            ).float()
            stds_2d = torch.from_numpy(
                stds.get("2d", np.ones(data["node_2d"].num_nodes))
            ).float()

            srmse_1d, srmse_2d, _ = _validate_one_event_unified(
                self.model, data, stds_1d, stds_2d, self.device
            )

            accumulator.update_scalar(model_id, event_id, "1d", srmse_1d)
            accumulator.update_scalar(model_id, event_id, "2d", srmse_2d)

        overall = accumulator.compute()
        breakdown_raw = accumulator.breakdown()

        # Flatten for logging
        breakdown: Dict[str, float] = {"srmse_overall": overall}
        for mid, events in breakdown_raw.items():
            for eid, nts in events.items():
                for nt, val in nts.items():
                    breakdown[f"srmse_m{mid}_e{eid}_{nt}"] = val

        return overall, breakdown

    # ------------------------------------------------------------------ #
    #  Convenience: Load Best Model
    # ------------------------------------------------------------------ #

    def load_best(self) -> None:
        """Load the best checkpoint into the current model."""
        best_path = Path(self.cfg.checkpoint_dir) / "best_model.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"No best checkpoint at {best_path}")
        load_checkpoint(best_path, self.model, device=self.device)
        if self.cfg.verbose:
            print(f"Loaded best model from {best_path}")


# =====================================================================
#  CLI Entrypoint
# =====================================================================

def main() -> None:
    """Command-line training entrypoint.

    Usage::

        python -m src.trainer
        python -m src.trainer --epochs 80 --lr 5e-4
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Train UnifiedFloodModel for the Urban Flood challenge."
    )
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_gnn_layers", type=int, default=3)
    parser.add_argument("--num_gru_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_event_id", type=str, default="4")
    parser.add_argument("--pushforward_K", type=int, default=10)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--no_amp", action="store_true")

    args = parser.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        epochs=args.epochs,
        lr=args.lr,
        hidden_channels=args.hidden_channels,
        num_gnn_layers=args.num_gnn_layers,
        num_gru_layers=args.num_gru_layers,
        dropout=args.dropout,
        val_event_id=args.val_event_id,
        pushforward_K=args.pushforward_K,
        grad_clip_norm=args.grad_clip,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=not args.no_amp,
    )

    trainer = UnifiedTrainer(cfg)
    trainer.setup()
    history = trainer.train(resume_from=args.resume)
    print(history.summary_str())


if __name__ == "__main__":
    main()

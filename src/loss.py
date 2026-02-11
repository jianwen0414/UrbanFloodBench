"""
Loss functions and evaluation metrics for Urban Flood Modelling.

Competition metric (Standardized RMSE):

.. math::

    \\text{Score} = \\text{Mean}_{\\text{models}}\\!\\left(
      \\text{Mean}_{\\text{events}}\\!\\left(
        \\text{Mean}_{\\text{node\\_types}}\\!\\left(
          \\text{Mean}_{\\text{nodes}}\\!\\left(
            \\frac{\\text{RMSE}_i}{\\sigma_i}
          \\right)
        \\right)
      \\right)
    \\right)

This module provides:

  1. :func:`standardized_rmse_loss` — Variance-weighted MSE training
     surrogate.  Differentiable everywhere, clamped for numerical
     safety on low-variance ("always dry") nodes.

  2. :func:`standardized_huber_loss` — Outlier-robust Huber variant.
     Transitions from quadratic to linear beyond *delta*, shielding
     training from extreme flood spikes.

  3. :func:`push_forward_loss` — Multi-step trajectory loss.  Sums
     the per-step SRMSE across a K-step rollout with optional temporal
     weighting to penalise autoregressive drift.

  4. :func:`combined_flood_loss` — Node-type-balanced loss that mirrors
     the competition hierarchy: 1D and 2D contributions are averaged
     *equally* regardless of node counts.

  5. :class:`FloodLoss` — ``nn.Module`` wrapper that stores per-node
     σ as a buffer and exposes a clean ``forward()`` / ``forward_combined()``
     for training loops.

  6. :func:`standardized_rmse_metric` — Exact (non-differentiable)
     leaderboard metric for validation / logging.

  7. :class:`SRMSEAccumulator` — Stateful accumulator that mirrors the
     full hierarchical competition scoring across multiple events,
     node types, and models.

  8. :func:`per_node_loss_breakdown` — Diagnostic tool for finding
     problem nodes that disproportionately drive total error.

  9. :func:`compute_inverse_variance_weights` — Pre-compute clamped
     inverse-variance weights from σ tensors (utility).

Design rationale
~~~~~~~~~~~~~~~~
* **Backward compatibility** — :func:`standardized_rmse_loss` and
  :func:`standardized_rmse_metric` preserve the original positional
  signatures so existing tests/notebooks continue to work unchanged.

* **Physics compliance** — The loss is derived directly from the
  organiser's metric formula with proper handling of the σ ≈ 0 trap
  (clamped inverse-variance weights): dry nodes that never flood have
  near-zero σ, producing weight ≈ 1/ε = 10⁶ without clamping.

* **Autoregressive stability** — :func:`push_forward_loss` with
  temporal weighting teaches the model to fight compounding errors
  at later prediction horizons (Bible §7.4).

* **Node-type balancing** — :func:`combined_flood_loss` weights 1D
  and 2D equally (matching the ``Mean_node_types`` in the competition
  formula), regardless of the vastly different node counts.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 – conventional alias
from torch import Tensor

# ── Numerical stability constant ──────────────────────────────────────
_EPS: float = 1e-6


# =====================================================================
#  Utilities
# =====================================================================

def compute_inverse_variance_weights(
    node_stds: Tensor,
    clamp_max: float = 100.0,
    eps: float = _EPS,
) -> Tensor:
    """Pre-compute clamped 1/(σ² + ε) weights from per-node std devs.

    Use this at dataset-setup time to store the weight tensor once and
    avoid recomputing it every forward pass.

    Parameters
    ----------
    node_stds : Tensor, shape ``(N,)``
        Per-node standard deviations of the target water level over the
        training corpus.
    clamp_max : float
        Upper bound for the weights.  Prevents near-zero-σ nodes
        (always dry) from producing infinite gradients.
    eps : float
        Small constant added to σ² for numerical stability.

    Returns
    -------
    Tensor, shape ``(N,)``
        Clamped inverse-variance weights ready for element-wise
        multiplication with squared errors.
    """
    w = 1.0 / (node_stds.pow(2) + eps)
    return torch.clamp(w, max=clamp_max)


def _temporal_weights(
    T: int,
    scheme: Literal["uniform", "linear", "exponential"] = "uniform",
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Generate a length-*T* weight vector for temporal aggregation.

    Parameters
    ----------
    T : int
        Sequence length (number of prediction steps).
    scheme : {"uniform", "linear", "exponential"}
        * ``"uniform"``     — equal weight at every step (vanilla mean).
        * ``"linear"``      — linearly increasing 1, 2, …, T then
          normalised to sum to 1.  Later steps weighted more heavily to
          fight autoregressive drift.
        * ``"exponential"`` — exponentially increasing weights
          ``2^{0}, 2^{1}, …, 2^{T-1}`` normalised to 1.  Aggressive
          drift penalisation.

    Returns
    -------
    Tensor, shape ``(T,)``
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    if scheme == "uniform":
        w = torch.ones(T, device=device, dtype=dtype)
    elif scheme == "linear":
        w = torch.arange(1, T + 1, device=device, dtype=dtype)
    elif scheme == "exponential":
        w = torch.pow(2.0, torch.arange(T, device=device, dtype=dtype))
    else:
        raise ValueError(
            f"Unknown temporal weight scheme {scheme!r}. "
            f"Choose from 'uniform', 'linear', 'exponential'."
        )

    return w / w.sum()


# =====================================================================
#  Core Training Loss (differentiable surrogate)
# =====================================================================

def standardized_rmse_loss(
    pred: Tensor,
    target: Tensor,
    node_stds: Tensor,
    mask: Optional[Tensor] = None,
    clamp_weights: float = 100.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """Variance-weighted MSE loss that mirrors the competition SRMSE.

    .. math::

        \\mathcal{L}
        = \\frac{1}{|\\mathcal{M}|}
          \\sum_{b,t,n}
          \\frac{(\\hat{y}_{btn} - y_{btn})^2}
               {\\sigma_n^2 + \\varepsilon}

    where :math:`\\sigma_n` is the per-node standard deviation of the
    ground-truth water level and the sum runs over all unmasked entries.

    Parameters
    ----------
    pred : Tensor
        Predicted water levels.  Shape: ``(B, T, N)``, ``(T, N)``, or
        ``(N,)``.
    target : Tensor
        Ground-truth water levels.  Same shape as *pred*.
    node_stds : Tensor
        Per-node standard deviations of the target water level,
        computed over the training set.  Shape ``(N,)`` — automatically
        broadcast to match *pred*.
    mask : Tensor or None, optional
        Boolean tensor of the same shape as *pred*.  ``True`` entries
        are **included** in the loss; ``False`` entries are ignored.
        Useful for variable-length sequences or missing data.
    clamp_weights : float, optional
        Upper bound for the inverse-variance weights.  Prevents
        near-zero-σ nodes (always dry) from producing infinite
        gradients.  Default ``100.0``.
    reduction : {"mean", "sum", "none"}, optional
        How to reduce the weighted squared errors.  Default ``"mean"``.

    Returns
    -------
    Tensor
        Scalar loss (``"mean"`` / ``"sum"``) or element-wise tensor
        (``"none"``).
    """
    # 1. Squared error — shape matches pred
    sq_err = (pred - target).pow(2)

    # 2. Inverse-variance weights from per-node σ
    #    node_stds is (N,) — PyTorch broadcasts across leading dims.
    weights = 1.0 / (node_stds.pow(2) + _EPS)

    # 3. Clamp to prevent "always dry" nodes from dominating gradients
    weights = torch.clamp(weights, max=clamp_weights)

    # 4. Weighted squared error
    weighted_sq_err = sq_err * weights  # broadcast (N,) → (..., N)

    # 5. Apply mask (e.g. padding / missing timesteps)
    if mask is not None:
        weighted_sq_err = weighted_sq_err * mask.float()
        if reduction == "none":
            return weighted_sq_err
        n_valid = mask.float().sum().clamp(min=1.0)
        if reduction == "sum":
            return weighted_sq_err.sum()
        return weighted_sq_err.sum() / n_valid

    if reduction == "none":
        return weighted_sq_err
    if reduction == "sum":
        return weighted_sq_err.sum()
    return weighted_sq_err.mean()


# =====================================================================
#  Huber (Smooth-L1) Variant — outlier-robust training
# =====================================================================

def standardized_huber_loss(
    pred: Tensor,
    target: Tensor,
    node_stds: Tensor,
    mask: Optional[Tensor] = None,
    clamp_weights: float = 100.0,
    delta: float = 1.0,
) -> Tensor:
    """Huber-weighted loss normalised by per-node variance.

    Behaves like :func:`standardized_rmse_loss` near zero error
    (quadratic) but transitions to linear growth beyond *delta*,
    reducing the influence of extreme outlier events on the gradient.

    This is useful for flood modelling because a few catastrophic
    time-steps with large absolute errors should not destabilise
    training.

    Parameters
    ----------
    pred, target : Tensor
        Same semantics as :func:`standardized_rmse_loss`.
    node_stds : Tensor, shape ``(N,)``
    mask : Tensor or None
    clamp_weights : float
    delta : float
        Threshold where the loss transitions from quadratic to linear.
        Larger values make the loss closer to pure MSE.

    Returns
    -------
    Tensor
        Scalar loss (mean reduction).
    """
    huber = F.smooth_l1_loss(pred, target, reduction="none", beta=delta)

    weights = 1.0 / (node_stds.pow(2) + _EPS)
    weights = torch.clamp(weights, max=clamp_weights)

    weighted = huber * weights

    if mask is not None:
        weighted = weighted * mask.float()
        n_valid = mask.float().sum().clamp(min=1.0)
        return weighted.sum() / n_valid

    return weighted.mean()


# =====================================================================
#  Push-Forward Trajectory Loss (anti-drift — Bible §7.4)
# =====================================================================

def push_forward_loss(
    preds: Tensor,
    targets: Tensor,
    node_stds: Tensor,
    mask: Optional[Tensor] = None,
    clamp_weights: float = 100.0,
    temporal_scheme: Literal["uniform", "linear", "exponential"] = "linear",
) -> Tensor:
    """Variance-weighted trajectory loss over a K-step rollout.

    Rather than penalising only the next-step prediction, this loss
    accumulates the SRMSE over an entire rollout window and optionally
    up-weights later time-steps to explicitly combat autoregressive
    drift.

    .. math::

        \\mathcal{L}_{\\text{pf}}
        = \\sum_{k=0}^{K-1} w_k \\cdot
          \\text{SRMSE}(\\hat{y}_{t+k},\\, y_{t+k})

    Parameters
    ----------
    preds : Tensor, shape ``(K, N)``
        Predictions at each step of the rollout.
    targets : Tensor, shape ``(K, N)``
        Ground truth at each step.
    node_stds : Tensor, shape ``(N,)``
    mask : Tensor or None, shape ``(K, N)``
    clamp_weights : float
    temporal_scheme : {"uniform", "linear", "exponential"}
        Weighting scheme over the *K* steps.  ``"linear"`` (default)
        gives later steps progressively more weight, which is the sweet
        spot for autoregressive flood forecasting.

    Returns
    -------
    Tensor
        Scalar loss value.
    """
    K = preds.shape[0]
    tw = _temporal_weights(K, temporal_scheme, device=preds.device, dtype=preds.dtype)

    # Per-step SRMSE (scalar per step)
    step_losses = torch.zeros(K, device=preds.device, dtype=preds.dtype)
    for k in range(K):
        step_mask = mask[k] if mask is not None else None
        step_losses[k] = standardized_rmse_loss(
            preds[k], targets[k], node_stds,
            mask=step_mask, clamp_weights=clamp_weights,
        )

    return (tw * step_losses).sum()


# =====================================================================
#  Combined Node-Type-Balanced Loss
# =====================================================================

def combined_flood_loss(
    pred_1d: Tensor,
    target_1d: Tensor,
    stds_1d: Tensor,
    pred_2d: Tensor,
    target_2d: Tensor,
    stds_2d: Tensor,
    mask_1d: Optional[Tensor] = None,
    mask_2d: Optional[Tensor] = None,
    clamp_weights: float = 100.0,
    alpha: float = 0.5,
    temporal_scheme: Literal["uniform", "linear", "exponential"] = "uniform",
    use_push_forward: bool = False,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Node-type-balanced loss matching the competition hierarchy.

    The competition metric averages over node-types equally:
    ``Mean_node_types(Mean_nodes(RMSE_i / σ_i))``.  This means a model
    with 100 1D nodes and 5,000 2D nodes still weights the two domains
    50/50.  This function replicates that balance in training.

    Parameters
    ----------
    pred_1d, target_1d : Tensor
        1D node predictions/targets.  ``(T, N_1d)`` or ``(K, N_1d)``
        for push-forward.
    stds_1d : Tensor, shape ``(N_1d,)``
    pred_2d, target_2d : Tensor
        2D node predictions/targets.  ``(T, N_2d)`` or ``(K, N_2d)``.
    stds_2d : Tensor, shape ``(N_2d,)``
    mask_1d, mask_2d : Tensor or None
    clamp_weights : float
    alpha : float
        Blend factor.  ``0.5`` = equal weight for 1D/2D (competition
        default).  Increase toward 1.0 to emphasise 1D pipes.
    temporal_scheme : str
        Passed through to :func:`push_forward_loss` when
        ``use_push_forward=True``.
    use_push_forward : bool
        If ``True``, use :func:`push_forward_loss` (expects ``(K, N)``
        input).  Otherwise use the standard per-element loss.

    Returns
    -------
    total_loss : Tensor
        Scalar combined loss.
    breakdown : dict[str, Tensor]
        ``{"loss_1d": ..., "loss_2d": ..., "total": ...}`` for logging.
    """
    if use_push_forward:
        loss_1d = push_forward_loss(
            pred_1d, target_1d, stds_1d,
            mask=mask_1d, clamp_weights=clamp_weights,
            temporal_scheme=temporal_scheme,
        )
        loss_2d = push_forward_loss(
            pred_2d, target_2d, stds_2d,
            mask=mask_2d, clamp_weights=clamp_weights,
            temporal_scheme=temporal_scheme,
        )
    else:
        loss_1d = standardized_rmse_loss(
            pred_1d, target_1d, stds_1d,
            mask=mask_1d, clamp_weights=clamp_weights,
        )
        loss_2d = standardized_rmse_loss(
            pred_2d, target_2d, stds_2d,
            mask=mask_2d, clamp_weights=clamp_weights,
        )

    total = alpha * loss_1d + (1.0 - alpha) * loss_2d

    breakdown = {
        "loss_1d": loss_1d.detach(),
        "loss_2d": loss_2d.detach(),
        "total": total.detach(),
    }
    return total, breakdown


# =====================================================================
#  nn.Module Wrapper
# =====================================================================

class FloodLoss(nn.Module):
    """PyTorch Module wrapping the variance-weighted SRMSE loss.

    Stores per-node σ as a *non-learnable buffer* so it automatically
    follows ``.to(device)`` and ``.half()`` calls.  Supports separate
    1D/2D streams or a single homogeneous stream.

    Parameters
    ----------
    node_stds_1d : Tensor or None, shape ``(N_1d,)``
    node_stds_2d : Tensor or None, shape ``(N_2d,)``
    clamp_weights : float
    alpha : float
        Balance between 1D and 2D losses (used in ``forward_combined``).
    temporal_scheme : str
        Default temporal weighting for push-forward mode.
    loss_variant : {"mse", "huber"}
        Underlying loss function.  ``"mse"`` uses the standard SRMSE
        surrogate; ``"huber"`` uses the outlier-robust Huber variant.
    huber_delta : float
        Threshold for the Huber transition (only when variant="huber").

    Example
    -------
    >>> criterion = FloodLoss(stds_1d, stds_2d, alpha=0.5)
    >>> criterion = criterion.to(device)
    >>> total, info = criterion.forward_combined(
    ...     p1d, t1d, p2d, t2d, use_push_forward=True
    ... )
    >>> total.backward()
    """

    def __init__(
        self,
        node_stds_1d: Optional[Tensor] = None,
        node_stds_2d: Optional[Tensor] = None,
        clamp_weights: float = 100.0,
        alpha: float = 0.5,
        temporal_scheme: Literal["uniform", "linear", "exponential"] = "linear",
        loss_variant: Literal["mse", "huber"] = "mse",
        huber_delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.clamp_weights = clamp_weights
        self.alpha = alpha
        self.temporal_scheme = temporal_scheme
        self.loss_variant = loss_variant
        self.huber_delta = huber_delta

        # Register as buffers so they move with .to(device)
        if node_stds_1d is not None:
            self.register_buffer("stds_1d", node_stds_1d)
        else:
            self.stds_1d: Optional[Tensor] = None

        if node_stds_2d is not None:
            self.register_buffer("stds_2d", node_stds_2d)
        else:
            self.stds_2d: Optional[Tensor] = None

    # ── helpers ──────────────────────────────────────────────────
    def _loss_fn(
        self,
        pred: Tensor,
        target: Tensor,
        stds: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Dispatch to MSE or Huber variant."""
        if self.loss_variant == "huber":
            return standardized_huber_loss(
                pred, target, stds,
                mask=mask,
                clamp_weights=self.clamp_weights,
                delta=self.huber_delta,
            )
        return standardized_rmse_loss(
            pred, target, stds,
            mask=mask,
            clamp_weights=self.clamp_weights,
        )

    # ── single-stream forward (for decoupled 1D-only or 2D-only) ─
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        node_stds: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Single-stream loss (use stored stds_1d or passed stds).

        This is the simplest entry-point, usable for either the 1D or
        2D engine independently.
        """
        if node_stds is None:
            node_stds = self.stds_1d if self.stds_1d is not None else self.stds_2d
        if node_stds is None:
            raise ValueError(
                "No node_stds provided and none registered as buffer."
            )
        return self._loss_fn(pred, target, node_stds, mask)

    # ── combined 1D + 2D forward (for unified model) ─────────────
    def forward_combined(
        self,
        pred_1d: Tensor,
        target_1d: Tensor,
        pred_2d: Tensor,
        target_2d: Tensor,
        mask_1d: Optional[Tensor] = None,
        mask_2d: Optional[Tensor] = None,
        use_push_forward: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Combined 1D + 2D loss with node-type balancing.

        Delegates to :func:`combined_flood_loss`.

        Returns
        -------
        total : Tensor
        breakdown : dict[str, Tensor]
        """
        if self.stds_1d is None or self.stds_2d is None:
            raise ValueError(
                "forward_combined requires both stds_1d and stds_2d "
                "to be registered."
            )
        return combined_flood_loss(
            pred_1d, target_1d, self.stds_1d,
            pred_2d, target_2d, self.stds_2d,
            mask_1d=mask_1d,
            mask_2d=mask_2d,
            clamp_weights=self.clamp_weights,
            alpha=self.alpha,
            temporal_scheme=self.temporal_scheme,
            use_push_forward=use_push_forward,
        )


# =====================================================================
#  Evaluation Metric (exact competition formula)
# =====================================================================

@torch.no_grad()
def standardized_rmse_metric(
    pred: Tensor,
    target: Tensor,
    node_stds: Tensor,
    mask: Optional[Tensor] = None,
    per_node: bool = False,
    min_std: float = 0.01,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Compute the **exact** Standardized RMSE used on the leaderboard.

    .. math::

        \\text{SRMSE}
        = \\frac{1}{N} \\sum_{i=1}^{N}
          \\frac{\\sqrt{\\frac{1}{T}\\sum_{t}(y_{it}-\\hat{y}_{it})^2}}
               {\\sigma_i}

    Unlike :func:`standardized_rmse_loss` this is **non-differentiable**
    (contains a square root per node) and intended only for validation /
    logging.

    Parameters
    ----------
    pred : Tensor, shape ``(T, N)``
        Predictions for one event, all time-steps.
    target : Tensor, shape ``(T, N)``
        Ground truth.
    node_stds : Tensor, shape ``(N,)``
    mask : Tensor or None, shape ``(T, N)``
        ``True`` = include, ``False`` = ignore.
    per_node : bool
        If ``True``, also return the per-node SRMSE vector.
    min_std : float
        Minimum std threshold.  Nodes with σ below this value are
        clamped to prevent near-zero-σ nodes (always dry) from
        dominating the metric with astronomical values.

    Returns
    -------
    srmse : Tensor
        Scalar SRMSE score.
    per_node_srmse : Tensor, shape ``(N,)``  *(only when per_node=True)*
    """
    sq_err = (pred - target).pow(2)  # (T, N)

    if mask is not None:
        sq_err = sq_err * mask.float()
        # mean over valid timesteps per node
        n_valid_t = mask.float().sum(dim=0).clamp(min=1.0)  # (N,)
        mse_per_node = sq_err.sum(dim=0) / n_valid_t
    else:
        mse_per_node = sq_err.mean(dim=0)  # (N,)

    rmse_per_node = mse_per_node.sqrt()                     # (N,)
    # Clamp node_stds to prevent near-zero-σ nodes from exploding
    safe_stds = node_stds.clamp(min=min_std)
    srmse_per_node = rmse_per_node / (safe_stds + _EPS)     # (N,)
    srmse = srmse_per_node.mean()

    if per_node:
        return srmse, srmse_per_node
    return srmse


# =====================================================================
#  Validation Accumulator (full competition hierarchy)
# =====================================================================

class SRMSEAccumulator:
    """Accumulate SRMSE scores across the full competition hierarchy.

    The competition metric is:

        Mean_models( Mean_events( Mean_node_types( Mean_nodes(RMSE/σ) ) ) )

    This accumulator collects per-event, per-node-type SRMSE values
    and computes the final hierarchical score.

    Usage
    -----
    >>> acc = SRMSEAccumulator()
    >>> for event in val_events:
    ...     acc.update("1", event_id, "1d", pred_1d, tgt_1d, std_1d)
    ...     acc.update("1", event_id, "2d", pred_2d, tgt_2d, std_2d)
    >>> final_score = acc.compute()
    >>> print(acc.summary_str())
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated scores."""
        # Structure: {model_id: {event_id: {node_type: srmse_scalar}}}
        self._scores: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )

    @torch.no_grad()
    def update(
        self,
        model_id: str,
        event_id: str,
        node_type: str,
        pred: Tensor,
        target: Tensor,
        node_stds: Tensor,
        mask: Optional[Tensor] = None,
    ) -> float:
        """Compute and store SRMSE for one (model, event, node_type).

        Parameters
        ----------
        model_id, event_id, node_type : str
            Identifiers for the hierarchy.
        pred, target : Tensor, shape ``(T, N)``
        node_stds : Tensor, shape ``(N,)``
        mask : Tensor or None

        Returns
        -------
        float
            The SRMSE value for this entry (for logging convenience).
        """
        srmse = standardized_rmse_metric(pred, target, node_stds, mask=mask)
        val = srmse.item()
        self._scores[model_id][event_id][node_type] = val
        return val

    def update_scalar(
        self,
        model_id: str,
        event_id: str,
        node_type: str,
        value: float,
    ) -> None:
        """Directly store a pre-computed SRMSE scalar."""
        self._scores[model_id][event_id][node_type] = value

    def compute(self) -> float:
        """Compute the full hierarchical competition score.

        Returns
        -------
        float
            Final SRMSE following the official averaging order:
            models → events → node_types → nodes.
        """
        if not self._scores:
            return float("nan")

        model_means: List[float] = []
        for _model_id, events in self._scores.items():
            event_means: List[float] = []
            for _event_id, node_types in events.items():
                if not node_types:
                    continue
                # Mean over node_types (1D and 2D weighted equally)
                nt_mean = sum(node_types.values()) / len(node_types)
                event_means.append(nt_mean)
            if event_means:
                model_means.append(sum(event_means) / len(event_means))

        if not model_means:
            return float("nan")
        return sum(model_means) / len(model_means)

    def breakdown(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Return the raw scores dict for inspection / logging.

        Returns
        -------
        dict
            ``{model_id: {event_id: {node_type: srmse}}}``
        """
        # Convert defaultdicts to plain dicts for clean serialisation
        return {
            mid: {eid: dict(nts) for eid, nts in events.items()}
            for mid, events in self._scores.items()
        }

    def summary_str(self) -> str:
        """Human-readable summary string for logging."""
        lines: List[str] = []
        for mid, events in sorted(self._scores.items()):
            lines.append(f"  Model {mid}:")
            for eid, nts in sorted(events.items()):
                parts = [f"{nt}={v:.4f}" for nt, v in sorted(nts.items())]
                nt_mean = sum(nts.values()) / max(len(nts), 1)
                lines.append(
                    f"    Event {eid}: {', '.join(parts)}  -> {nt_mean:.4f}"
                )
        overall = self.compute()
        lines.append(f"  --- Overall SRMSE: {overall:.6f}")
        return "\n".join(lines)


# =====================================================================
#  Per-Node Diagnostics (for debugging)
# =====================================================================

@torch.no_grad()
def per_node_loss_breakdown(
    pred: Tensor,
    target: Tensor,
    node_stds: Tensor,
    top_k: int = 10,
) -> Dict[str, Tensor]:
    """Identify which nodes contribute most to the total loss.

    Useful during debugging to find "problem nodes" — typically nodes
    with very low σ (always dry) or consistently large errors.

    Parameters
    ----------
    pred, target : Tensor, shape ``(T, N)``
    node_stds : Tensor, shape ``(N,)``
    top_k : int
        Number of worst-performing nodes to highlight.

    Returns
    -------
    dict with keys:
        ``"srmse_per_node"`` : Tensor ``(N,)``
        ``"top_k_indices"``  : Tensor ``(top_k,)``
        ``"top_k_srmse"``    : Tensor ``(top_k,)``
        ``"mean_srmse"``     : Tensor scalar
        ``"median_srmse"``   : Tensor scalar
    """
    _, srmse_per_node = standardized_rmse_metric(
        pred, target, node_stds, per_node=True,
    )

    k = min(top_k, srmse_per_node.shape[0])
    top_vals, top_idx = torch.topk(srmse_per_node, k)

    return {
        "srmse_per_node": srmse_per_node,
        "top_k_indices": top_idx,
        "top_k_srmse": top_vals,
        "mean_srmse": srmse_per_node.mean(),
        "median_srmse": srmse_per_node.median(),
    }

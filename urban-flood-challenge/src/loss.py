"""
Standardized RMSE loss function.

Metric definition (competition scoring):
    For each node *i*, the error is normalised by the standard deviation
    of the ground-truth water level at that node across time.  The final
    score is the mean of these per-node standardised RMSEs.

        SRMSE = (1/N) Σ_i  sqrt( mean_t( (y_it - ŷ_it)² ) ) / σ_i

Training surrogate:
    During back-propagation we use a *squared* variant so that the loss
    is smooth and differentiable everywhere:

        Loss = (1/N) Σ  (pred - target)² / (σ² + ε)

    Weights 1/(σ² + ε) are clamped to a configurable maximum to prevent
    near-zero-variance ("always dry") nodes from dominating the gradient.

    An optional boolean mask allows ignoring padded / missing time-steps.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812 – conventional alias

# ── Numerical stability constant ──────────────────────────────────────
_EPS: float = 1e-6


# ======================================================================
#  Training loss (differentiable surrogate)
# ======================================================================

def standardized_rmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    node_stds: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    clamp_weights: float = 100.0,
) -> torch.Tensor:
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
        Predicted water levels.  Shape: ``(B, T, N)`` **or** ``(T, N)``
        or ``(N,)``.
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

    Returns
    -------
    Tensor
        Scalar loss value (mean reduction).
    """
    # 1. Squared error  (B, T, N) or compatible shape
    sq_err = (pred - target).pow(2)

    # 2. Inverse-variance weights from per-node σ
    #    node_stds is (N,) — PyTorch broadcasts across leading dims.
    weights = 1.0 / (node_stds.pow(2) + _EPS)

    # 3. Stability: clamp weights so "always dry" nodes don't explode
    weights = torch.clamp(weights, max=clamp_weights)

    # 4. Weighted squared error
    weighted_sq_err = sq_err * weights  # broadcast (N,) → (B, T, N)

    # 5. Optional masking (e.g. padding / missing timesteps)
    if mask is not None:
        weighted_sq_err = weighted_sq_err * mask.float()
        # Mean only over valid entries
        n_valid = mask.float().sum().clamp(min=1.0)
        return weighted_sq_err.sum() / n_valid

    return weighted_sq_err.mean()


# ======================================================================
#  Evaluation metric (exact competition formula)
# ======================================================================

@torch.no_grad()
def standardized_rmse_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    node_stds: torch.Tensor,
) -> torch.Tensor:
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
    pred : Tensor
        Shape ``(T, N)`` — one event, all time-steps.
    target : Tensor
        Shape ``(T, N)``.
    node_stds : Tensor
        Shape ``(N,)``.

    Returns
    -------
    Tensor
        Scalar SRMSE score.
    """
    # Per-node RMSE across time: sqrt( mean_t( (y - ŷ)² ) )
    per_node_rmse = (pred - target).pow(2).mean(dim=0).sqrt()  # (N,)

    # Normalise by σ (with ε for safety)
    srmse = per_node_rmse / (node_stds + _EPS)

    return srmse.mean()

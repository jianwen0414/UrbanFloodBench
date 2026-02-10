"""
validate — Production Validation Pipeline (Task 3.2).

Implements the "Honest" Validation Strategy for the Urban Flood
competition.  Simulates the Private Leaderboard *exactly*:

  1. **Leave-One-Event-Out** splits — never random.
  2. **Full autoregressive rollout** with burn-in period (t=1..10).
  3. **Hierarchical SRMSE** matching the competition formula:
     ``Mean_models → Mean_events → Mean_node_types → Mean_nodes``.
  4. **Per-node diagnostics** to identify problem nodes.

This module can operate on:
  * The ``UnifiedFloodModel`` (Tier 2)
  * Decoupled 1D + 2D engines (Tier 1) via ``validate_decoupled()``

See   : IMPLEMENTATION_PLAN.md → Task 3.2, PROJECT_BIBLE.md §7.3
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kw):  # type: ignore[misc]
        return iterable

from src.dataset import FloodDataset
from src.graph_builder_unified import build_unified_graph
from src.loss import (
    SRMSEAccumulator,
    per_node_loss_breakdown,
    standardized_rmse_metric,
)
from src.model_unified import UnifiedFloodModel


# =====================================================================
#  Validation Result Container
# =====================================================================

@dataclass
class ValidationResult:
    """Container for comprehensive validation results.

    Attributes
    ----------
    overall_srmse : float
        Final hierarchical SRMSE score (lower is better).
    breakdown : dict
        Per-model → per-event → per-node-type SRMSE values.
    per_event_scores : list[dict]
        Detailed per-event results including timing.
    diagnostics : dict
        Per-event problem-node analysis (top-K worst nodes).
    elapsed_seconds : float
        Total validation wall-clock time.
    """
    overall_srmse: float = float("inf")
    breakdown: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    per_event_scores: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def summary_str(self) -> str:
        """Human-readable validation summary."""
        lines = [
            "=" * 60,
            "  VALIDATION RESULTS",
            "=" * 60,
            f"  Overall SRMSE: {self.overall_srmse:.6f}",
            f"  Time: {self.elapsed_seconds:.1f}s",
            "",
        ]

        for mid, events in sorted(self.breakdown.items()):
            lines.append(f"  Model {mid}:")
            event_scores = []
            for eid, nts in sorted(events.items()):
                parts = [f"{nt}={v:.4f}" for nt, v in sorted(nts.items())]
                nt_mean = sum(nts.values()) / max(len(nts), 1)
                event_scores.append(nt_mean)
                lines.append(f"    Event {eid}: {', '.join(parts)}  -> {nt_mean:.4f}")
            avg = sum(event_scores) / max(len(event_scores), 1)
            lines.append(f"    --- Model {mid} avg: {avg:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "overall_srmse": self.overall_srmse,
            "breakdown": self.breakdown,
            "per_event_scores": self.per_event_scores,
            "elapsed_seconds": self.elapsed_seconds,
        }

    def save(self, path: str | Path) -> None:
        """Save results to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# =====================================================================
#  Single-Event Autoregressive Validation
# =====================================================================

@torch.no_grad()
def validate_event_unified(
    model: UnifiedFloodModel,
    data: Any,  # HeteroData
    stds_1d: Tensor,
    stds_2d: Tensor,
    device: torch.device,
    spinup_steps: int = 10,
    run_diagnostics: bool = False,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Validate on a single event with exact competition protocol.

    Protocol
    --------
    1. **Spin-up** (t=0..spinup_steps-1): Feed ground truth → build
       GRU hidden states.  Predictions are generated but NOT scored.
    2. **Prediction** (t=spinup_steps..end): Full autoregressive mode
       (teacher_forcing_ratio=0.0).  These predictions are scored.

    Parameters
    ----------
    model : UnifiedFloodModel
    data : HeteroData
        Pre-built heterogeneous graph for one event.
    stds_1d, stds_2d : Tensor
        Per-node standard deviations.
    device : torch.device
    spinup_steps : int
        Number of warm-up steps.
    run_diagnostics : bool
        If True, also compute per-node loss breakdown.
    top_k : int
        Number of worst-performing nodes to report.

    Returns
    -------
    dict with keys:
        srmse_1d, srmse_2d, srmse_combined : float
        preds_1d, preds_2d : Tensor (full rollout predictions)
        diagnostics_1d, diagnostics_2d : dict (if run_diagnostics)
        elapsed : float
    """
    t0 = time.time()
    data = data.to(device)
    model.eval()

    T = data.num_timesteps
    skip = min(spinup_steps, T - 1)

    # Full autoregressive rollout
    preds_1d, preds_2d = model.rollout(
        data,
        spinup_steps=skip,
        teacher_forcing_ratio=0.0,
    )

    targets_1d = data["node_1d"].y.to(device)
    targets_2d = data["node_2d"].y.to(device)
    stds_1d_dev = stds_1d.to(device)
    stds_2d_dev = stds_2d.to(device)

    # Compute SRMSE on prediction period only
    srmse_1d = standardized_rmse_metric(
        preds_1d[skip:], targets_1d[skip:], stds_1d_dev
    ).item()
    srmse_2d = standardized_rmse_metric(
        preds_2d[skip:], targets_2d[skip:], stds_2d_dev
    ).item()
    srmse_combined = (srmse_1d + srmse_2d) / 2.0

    result: Dict[str, Any] = {
        "srmse_1d": srmse_1d,
        "srmse_2d": srmse_2d,
        "srmse_combined": srmse_combined,
        "preds_1d": preds_1d.cpu(),
        "preds_2d": preds_2d.cpu(),
        "targets_1d": targets_1d.cpu(),
        "targets_2d": targets_2d.cpu(),
        "elapsed": time.time() - t0,
    }

    if run_diagnostics:
        diag_1d = per_node_loss_breakdown(
            preds_1d[skip:], targets_1d[skip:], stds_1d_dev, top_k=top_k
        )
        diag_2d = per_node_loss_breakdown(
            preds_2d[skip:], targets_2d[skip:], stds_2d_dev, top_k=top_k
        )
        result["diagnostics_1d"] = {
            k: v.cpu() if isinstance(v, Tensor) else v
            for k, v in diag_1d.items()
        }
        result["diagnostics_2d"] = {
            k: v.cpu() if isinstance(v, Tensor) else v
            for k, v in diag_2d.items()
        }

    return result


# =====================================================================
#  Full Cross-Validation Pipeline
# =====================================================================

class ValidationRunner:
    """Orchestrates full cross-validation across models and events.

    Implements Leave-One-Event-Out validation with the exact
    competition scoring hierarchy.

    Parameters
    ----------
    model : UnifiedFloodModel
        Trained model to validate.
    data_root : str
        Path to the data directory.
    device : torch.device or str
    model_ids : list[str]
        Which urban models to validate on.
    spinup_steps : int
        Number of burn-in steps.
    run_diagnostics : bool
        Whether to compute per-node diagnostics.
    verbose : bool
        Print progress information.

    Examples
    --------
    >>> runner = ValidationRunner(model, "data", device="cuda")
    >>> result = runner.validate_holdout(val_event_id="4")
    >>> print(result.summary_str())
    """

    def __init__(
        self,
        model: UnifiedFloodModel,
        data_root: str = "data",
        device: Union[str, torch.device] = "cpu",
        model_ids: Optional[List[str]] = None,
        spinup_steps: int = 10,
        run_diagnostics: bool = True,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.data_root = data_root
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model_ids = model_ids or ["1", "2"]
        self.spinup_steps = spinup_steps
        self.run_diagnostics = run_diagnostics
        self.verbose = verbose

        # Pre-load dataset for std computation
        self._dataset = FloodDataset(data_root, mode="train")
        self._stds_cache: Dict[str, Dict[str, np.ndarray]] = {}

    def _get_stds(self, model_id: str) -> Dict[str, np.ndarray]:
        """Get or compute per-node stds for a model."""
        if model_id not in self._stds_cache:
            stds = self._dataset.compute_node_stds(model_id=model_id)
            self._stds_cache[model_id] = stds.get(
                model_id, {"1d": np.array([1.0]), "2d": np.array([1.0])}
            )
        return self._stds_cache[model_id]

    def validate_holdout(
        self,
        val_event_id: str = "4",
    ) -> ValidationResult:
        """Validate using Leave-One-Event-Out on a specific hold-out event.

        This is the primary validation entry point.  For each model:
          1. Hold out the specified event (or fallback to last available).
          2. Run the full autoregressive loop.
          3. Compute hierarchical SRMSE.

        Parameters
        ----------
        val_event_id : str
            Event ID to hold out for validation.

        Returns
        -------
        ValidationResult
        """
        overall_start = time.time()
        accumulator = SRMSEAccumulator()
        per_event_scores: List[Dict[str, Any]] = []
        diagnostics: Dict[str, Any] = {}

        for mid in self.model_ids:
            model_ds = self._dataset.filter_by_model(mid)

            if not model_ds.events:
                if self.verbose:
                    print(f"  WARNING: No events for Model_{mid}")
                continue

            available = model_ds.get_event_ids()
            eid = val_event_id if val_event_id in available else available[-1]

            _, val_ds = model_ds.split_by_event(eid)

            stds = self._get_stds(mid)
            stds_1d = torch.from_numpy(stds["1d"]).float()
            stds_2d = torch.from_numpy(stds["2d"]).float()

            if self.verbose:
                print(f"\n  Validating Model_{mid}, Event {eid} "
                      f"({len(val_ds)} event(s))...")

            for i in range(len(val_ds)):
                sample = val_ds[i]
                data = build_unified_graph(sample)

                result = validate_event_unified(
                    self.model, data, stds_1d, stds_2d,
                    self.device, self.spinup_steps,
                    run_diagnostics=self.run_diagnostics,
                )

                accumulator.update_scalar(mid, sample["event_id"], "1d", result["srmse_1d"])
                accumulator.update_scalar(mid, sample["event_id"], "2d", result["srmse_2d"])

                event_result = {
                    "model_id": mid,
                    "event_id": sample["event_id"],
                    "srmse_1d": result["srmse_1d"],
                    "srmse_2d": result["srmse_2d"],
                    "srmse_combined": result["srmse_combined"],
                    "elapsed": result["elapsed"],
                    "num_timesteps": data.num_timesteps,
                }
                per_event_scores.append(event_result)

                if self.run_diagnostics and "diagnostics_1d" in result:
                    diag_key = f"m{mid}_e{sample['event_id']}"
                    diagnostics[diag_key] = {
                        "1d_worst_nodes": result["diagnostics_1d"]["top_k_indices"].tolist(),
                        "1d_worst_srmse": result["diagnostics_1d"]["top_k_srmse"].tolist(),
                        "2d_worst_nodes": result["diagnostics_2d"]["top_k_indices"].tolist(),
                        "2d_worst_srmse": result["diagnostics_2d"]["top_k_srmse"].tolist(),
                    }

                if self.verbose:
                    print(
                        f"    Event {sample['event_id']}: "
                        f"1D={result['srmse_1d']:.4f}, "
                        f"2D={result['srmse_2d']:.4f}, "
                        f"combined={result['srmse_combined']:.4f} "
                        f"({result['elapsed']:.1f}s)"
                    )

        elapsed_total = time.time() - overall_start

        return ValidationResult(
            overall_srmse=accumulator.compute(),
            breakdown=accumulator.breakdown(),
            per_event_scores=per_event_scores,
            diagnostics=diagnostics,
            elapsed_seconds=elapsed_total,
        )

    def cross_validate(
        self,
        n_folds: int = 5,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run Leave-One-Event-Out cross-validation with multiple folds.

        Selects ``n_folds`` diverse events as hold-out sets and runs
        validation on each.  This gives a more robust estimate of
        generalisation performance.

        Parameters
        ----------
        n_folds : int
            Number of cross-validation folds.
        model_id : str or None
            If given, only validate on this model.  Otherwise use
            all models in ``self.model_ids``.

        Returns
        -------
        dict with keys:
            "fold_results" : list[ValidationResult]
            "mean_srmse" : float
            "std_srmse" : float
            "per_fold_srmse" : list[float]
        """
        target_mids = [model_id] if model_id else self.model_ids

        # Collect all available event IDs
        all_events = set()
        for mid in target_mids:
            model_ds = self._dataset.filter_by_model(mid)
            all_events.update(model_ds.get_event_ids())

        # Select n_folds evenly-spaced events
        all_events_sorted = sorted(all_events, key=lambda x: int(x) if x.isdigit() else x)
        step = max(1, len(all_events_sorted) // n_folds)
        fold_events = all_events_sorted[::step][:n_folds]

        if self.verbose:
            print(f"\nCross-Validation: {n_folds} folds")
            print(f"  Hold-out events: {fold_events}")

        fold_results: List[ValidationResult] = []
        fold_srmse: List[float] = []

        for fold_idx, val_eid in enumerate(fold_events):
            if self.verbose:
                print(f"\n--- Fold {fold_idx + 1}/{n_folds} (val_event={val_eid}) ---")

            result = self.validate_holdout(val_event_id=val_eid)
            fold_results.append(result)
            fold_srmse.append(result.overall_srmse)

        mean_srmse = np.mean(fold_srmse)
        std_srmse = np.std(fold_srmse)

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"  CV Results: {mean_srmse:.6f} +/- {std_srmse:.6f}")
            print(f"  Per-fold: {[f'{s:.4f}' for s in fold_srmse]}")
            print(f"{'=' * 60}")

        return {
            "fold_results": fold_results,
            "mean_srmse": mean_srmse,
            "std_srmse": std_srmse,
            "per_fold_srmse": fold_srmse,
            "fold_events": fold_events,
        }


# =====================================================================
#  Prediction Extraction (for visualisation)
# =====================================================================

@torch.no_grad()
def extract_predictions(
    model: UnifiedFloodModel,
    data: Any,  # HeteroData
    device: torch.device,
    spinup_steps: int = 10,
) -> Dict[str, Tensor]:
    """Extract full rollout predictions for visualisation.

    Returns all predictions and targets (including spinup period)
    in a convenient dictionary format.

    Returns
    -------
    dict with keys:
        preds_1d     : Tensor [T, N_1d]
        preds_2d     : Tensor [T, N_2d]
        targets_1d   : Tensor [T, N_1d]
        targets_2d   : Tensor [T, N_2d]
        spinup_steps : int
        num_timesteps: int
    """
    data = data.to(device)
    model.eval()

    T = data.num_timesteps
    skip = min(spinup_steps, T - 1)

    preds_1d, preds_2d = model.rollout(
        data,
        spinup_steps=skip,
        teacher_forcing_ratio=0.0,
    )

    return {
        "preds_1d": preds_1d.cpu(),
        "preds_2d": preds_2d.cpu(),
        "targets_1d": data["node_1d"].y.cpu(),
        "targets_2d": data["node_2d"].y.cpu(),
        "spinup_steps": skip,
        "num_timesteps": T,
        "model_id": data.model_id,
        "event_id": data.event_id,
    }


# =====================================================================
#  Leaderboard Correlation Check
# =====================================================================

def check_leaderboard_correlation(
    local_scores: List[float],
    public_scores: List[float],
) -> Dict[str, float]:
    """Check if local validation scores correlate with public LB.

    A strong positive correlation (r > 0.8) means our local validation
    is trustworthy and we can use it to make submission decisions.

    Parameters
    ----------
    local_scores : list[float]
        SRMSE from local cross-validation.
    public_scores : list[float]
        SRMSE from public leaderboard submissions.

    Returns
    -------
    dict with "pearson_r", "spearman_rho", "trustworthy" (bool)
    """
    from scipy import stats

    if len(local_scores) < 3 or len(public_scores) < 3:
        return {"pearson_r": float("nan"), "spearman_rho": float("nan"),
                "trustworthy": False, "note": "Need at least 3 data points."}

    n = min(len(local_scores), len(public_scores))
    local = local_scores[:n]
    public = public_scores[:n]

    pearson_r, _ = stats.pearsonr(local, public)
    spearman_rho, _ = stats.spearmanr(local, public)

    return {
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "trustworthy": pearson_r > 0.7 and spearman_rho > 0.7,
    }

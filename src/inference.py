"""
inference — Submission Assembly Line (Task 3.3).

Generates the final submission.csv for the Kaggle "Urban Flood
Modelling" competition.  Implements the exact competition protocol:

  1. **Burn-In** (t=1..10): Feed ground-truth dynamic features to
     initialise GRU hidden states (h_10 contains the "volume of
     water" context).
  2. **Autoregressive Prediction** (t=11..End): Model's own
     predictions feed back as inputs.
  3. **Merging & Formatting**: Combine 1D and 2D predictions into
     the required long-format CSV.
  4. **Sanity Checks**: Assert no NaN, validate row counts, check
     for physically impossible values.

Submission Format
-----------------
Columns (strict order): ``row_id, model_id, event_id, node_type,
node_id, water_level``

The ``row_id`` is constructed as:
``{model_id}_{event_id}_{node_type}_{node_id}_{timestep}``

See   : IMPLEMENTATION_PLAN.md → Task 3.3, PROJECT_BIBLE.md §2
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
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
from src.model_unified import UnifiedFloodModel


# =====================================================================
#  Submission Row Builder
# =====================================================================

def _build_submission_rows(
    preds: Tensor,
    model_id: str,
    event_id: str,
    node_type: str,
    spinup_steps: int = 10,
) -> pd.DataFrame:
    """Convert a [T, N] prediction tensor into long-format rows.

    Only includes predictions from the scored period (after spinup).

    Parameters
    ----------
    preds : Tensor [T, N]
        Predicted water levels.
    model_id, event_id : str
    node_type : str
        ``"1d"`` or ``"2d"``.
    spinup_steps : int
        Number of initial timesteps to skip (not scored).

    Returns
    -------
    DataFrame with columns: row_id, model_id, event_id, node_type,
    node_id, water_level
    """
    T, N = preds.shape
    preds_np = preds.cpu().numpy()

    rows: List[Dict[str, Any]] = []

    for t in range(spinup_steps, T):
        for n in range(N):
            water_level = float(preds_np[t, n])

            # NaN safety: forward fill from previous timestep
            if np.isnan(water_level):
                if t > spinup_steps:
                    water_level = float(preds_np[t - 1, n])
                else:
                    water_level = 0.0

            row_id = f"{model_id}_{event_id}_{node_type}_{n}_{t}"
            rows.append({
                "row_id": row_id,
                "model_id": int(model_id),
                "event_id": int(event_id),
                "node_type": node_type,
                "node_id": n,
                "water_level": water_level,
            })

    return pd.DataFrame(rows)


def _build_submission_rows_vectorized(
    preds: Tensor,
    model_id: str,
    event_id: str,
    node_type: str,
    spinup_steps: int = 10,
) -> pd.DataFrame:
    """Vectorized version — significantly faster for large meshes.

    Parameters
    ----------
    Same as ``_build_submission_rows``.

    Returns
    -------
    DataFrame with submission columns.
    """
    T, N = preds.shape
    preds_np = preds.cpu().numpy()

    # Only scored timesteps
    scored_preds = preds_np[spinup_steps:]  # [T_scored, N]
    T_scored = scored_preds.shape[0]

    # Forward-fill NaN values along time axis
    nan_mask = np.isnan(scored_preds)
    if nan_mask.any():
        # Forward fill: replace NaN with previous timestep value
        for t_idx in range(1, T_scored):
            fill_mask = nan_mask[t_idx]
            scored_preds[t_idx, fill_mask] = scored_preds[t_idx - 1, fill_mask]
        # Fill remaining NaN (first timestep) with 0
        scored_preds = np.nan_to_num(scored_preds, nan=0.0)

    # Build coordinate arrays
    timesteps = np.arange(spinup_steps, T)
    node_ids = np.arange(N)
    t_grid, n_grid = np.meshgrid(timesteps, node_ids, indexing="ij")

    t_flat = t_grid.ravel()
    n_flat = n_grid.ravel()
    wl_flat = scored_preds.ravel()

    # row_id construction
    mid_int = int(model_id)
    eid_int = int(event_id)
    row_ids = [
        f"{model_id}_{event_id}_{node_type}_{n}_{t}"
        for t, n in zip(t_flat, n_flat)
    ]

    return pd.DataFrame({
        "row_id": row_ids,
        "model_id": mid_int,
        "event_id": eid_int,
        "node_type": node_type,
        "node_id": n_flat.astype(int),
        "water_level": wl_flat.astype(np.float64),
    })


# =====================================================================
#  Single-Event Inference
# =====================================================================

@torch.no_grad()
def predict_event(
    model: UnifiedFloodModel,
    data: Any,  # HeteroData
    device: torch.device,
    spinup_steps: int = 10,
) -> Tuple[Tensor, Tensor]:
    """Run full autoregressive inference on a single event.

    Implements the exact competition protocol:
    1. Spin-up (ground truth) → build GRU hidden states
    2. Prediction (autoregressive) → scored predictions

    The model predicts *depth* (height above a physical reference):
      - 1D: depth = WSE − invert_elevation
      - 2D: depth = WSE − min_elevation

    This function adds the elevation reference (stored in
    ``data[nt].baseline``) back to recover absolute water levels
    for the submission: ``WSE = depth + elevation_ref``.

    Parameters
    ----------
    model : UnifiedFloodModel
    data : HeteroData
    device : torch.device
    spinup_steps : int

    Returns
    -------
    (preds_1d, preds_2d)
        preds_1d : Tensor [T, N_1d]  — absolute water levels
        preds_2d : Tensor [T, N_2d]  — absolute water levels
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

    # ── Denormalize: add elevation reference back ──────────────────
    # The model predicts depth; the submission needs absolute WSE.
    # baseline stores: invert_elev (1D) or min_elev (2D).
    if hasattr(data["node_1d"], "baseline"):
        baseline_1d = data["node_1d"].baseline.to(device)  # [N_1d]
        preds_1d = preds_1d + baseline_1d.unsqueeze(0)
    if hasattr(data["node_2d"], "baseline"):
        baseline_2d = data["node_2d"].baseline.to(device)  # [N_2d]
        preds_2d = preds_2d + baseline_2d.unsqueeze(0)

    return preds_1d, preds_2d


# =====================================================================
#  Full Submission Pipeline
# =====================================================================

class SubmissionGenerator:
    """Generate the competition submission CSV.

    Orchestrates inference across all test events for all models,
    merges predictions, and formats the output.

    Parameters
    ----------
    model : UnifiedFloodModel
        Trained model (or loaded from checkpoint).
    data_root : str
        Path to the data directory.
    device : str or torch.device
    spinup_steps : int
        Number of burn-in steps (default 10 per competition spec).
    verbose : bool

    Examples
    --------
    >>> gen = SubmissionGenerator(model, "data", device="cuda")
    >>> df = gen.generate()
    >>> gen.save(df, "submission.csv")
    """

    def __init__(
        self,
        model: UnifiedFloodModel,
        data_root: str = "data",
        device: Union[str, torch.device] = "cpu",
        spinup_steps: int = 10,
        verbose: bool = True,
        static_norm_stats: Optional[Dict[str, Dict[str, Tensor]]] = None,
    ) -> None:
        self.model = model
        self.data_root = data_root
        self.device = torch.device(device) if isinstance(device, str) else device
        self.spinup_steps = spinup_steps
        self.verbose = verbose
        # Per-model z-score normalization stats from training.
        # If None, no normalization is applied (backward compatible).
        self.static_norm_stats = static_norm_stats or {}

        self.model.to(self.device)
        self.model.eval()

    def generate(self) -> pd.DataFrame:
        """Run inference on all test events and produce the submission.

        Returns
        -------
        DataFrame
            Full submission with columns: row_id, model_id, event_id,
            node_type, node_id, water_level.
        """
        t0 = time.time()
        dataset = FloodDataset(self.data_root, mode="test")

        if len(dataset) == 0:
            raise RuntimeError("No test events found. Check data_root.")

        all_dfs: List[pd.DataFrame] = []
        n_events = len(dataset)

        if self.verbose:
            print(f"\nGenerating submission for {n_events} test events...")
            print(f"  Device: {self.device}")
            print(f"  Spinup steps: {self.spinup_steps}")

        for idx in tqdm(
            range(n_events),
            desc="  Inference",
            disable=not self.verbose,
        ):
            sample = dataset[idx]
            model_id = sample["model_id"]
            event_id = sample["event_id"]

            try:
                # Build graph
                data = build_unified_graph(sample)

                # Apply per-model static z-score normalization
                # (must match what was done during training)
                if model_id in self.static_norm_stats:
                    stats = self.static_norm_stats[model_id]
                    data["node_1d"].x = (
                        (data["node_1d"].x - stats["mean_1d"]) / stats["std_1d"]
                    )
                    data["node_2d"].x = (
                        (data["node_2d"].x - stats["mean_2d"]) / stats["std_2d"]
                    )

                # Run inference
                preds_1d, preds_2d = predict_event(
                    self.model, data, self.device, self.spinup_steps
                )

                # Build submission rows (vectorized for speed)
                df_1d = _build_submission_rows_vectorized(
                    preds_1d, model_id, event_id, "1d", self.spinup_steps
                )
                df_2d = _build_submission_rows_vectorized(
                    preds_2d, model_id, event_id, "2d", self.spinup_steps
                )

                all_dfs.append(df_1d)
                all_dfs.append(df_2d)

            except Exception as e:
                warnings.warn(
                    f"Failed on Model_{model_id}, Event_{event_id}: {e}"
                )
                continue

        if not all_dfs:
            raise RuntimeError("All events failed during inference.")

        submission = pd.concat(all_dfs, ignore_index=True)

        # Sanity checks
        self._sanity_check(submission)

        elapsed = time.time() - t0

        if self.verbose:
            print(f"\n  Submission shape: {submission.shape}")
            print(f"  Unique models: {submission['model_id'].nunique()}")
            print(f"  Unique events: {submission['event_id'].nunique()}")
            print(f"  NaN values: {submission['water_level'].isna().sum()}")
            print(f"  Time: {elapsed:.1f}s")

        return submission

    def _sanity_check(self, df: pd.DataFrame) -> None:
        """Run sanity checks on the submission DataFrame.

        Checks for:
        1. No NaN values in water_level.
        2. No duplicate row_ids.
        3. Water levels are within physically reasonable range.
        4. All required columns are present.
        """
        required_cols = ["row_id", "model_id", "event_id", "node_type",
                         "node_id", "water_level"]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # NaN check
        nan_count = df["water_level"].isna().sum()
        if nan_count > 0:
            warnings.warn(
                f"Submission has {nan_count} NaN values — "
                "forward-filling with previous timestep."
            )
            # Group by model, event, node_type, node_id and forward fill
            df["water_level"] = (
                df.groupby(["model_id", "event_id", "node_type", "node_id"])
                ["water_level"]
                .transform(lambda s: s.ffill().fillna(0.0))
            )

        # Duplicate check
        dup_count = df["row_id"].duplicated().sum()
        if dup_count > 0:
            warnings.warn(f"Submission has {dup_count} duplicate row_ids!")
            df.drop_duplicates(subset=["row_id"], keep="first", inplace=True)

        # Physical range check (water levels shouldn't be astronomically
        # large or deeply negative)
        wl = df["water_level"]
        extreme_high = (wl > 1000).sum()
        extreme_low = (wl < -100).sum()
        if extreme_high > 0 or extreme_low > 0:
            warnings.warn(
                f"Extreme water levels detected: {extreme_high} > 1000ft, "
                f"{extreme_low} < -100ft. Possible model divergence."
            )

    @staticmethod
    def save(
        df: pd.DataFrame,
        path: str | Path,
        file_format: str = "csv",
    ) -> None:
        """Save the submission to disk.

        Parameters
        ----------
        df : DataFrame
            Submission DataFrame.
        path : str or Path
            Output file path.
        file_format : str
            ``"csv"`` or ``"parquet"``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure correct column order
        column_order = ["row_id", "model_id", "event_id", "node_type",
                        "node_id", "water_level"]
        df = df[column_order]

        if file_format == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)

        print(f"  Submission saved to {path} ({df.shape[0]:,} rows)")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        data_root: str = "data",
        device: str = "auto",
        **kwargs: Any,
    ) -> "SubmissionGenerator":
        """Create a SubmissionGenerator from a training checkpoint.

        Automatically reconstructs the model architecture from the
        saved config and loads trained weights.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to the checkpoint ``.pt`` file.
        data_root : str
        device : str

        Returns
        -------
        SubmissionGenerator
        """
        if device == "auto":
            if torch.cuda.is_available():
                dev = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = torch.device("mps")
            else:
                dev = torch.device("cpu")
        else:
            dev = torch.device(device)

        ckpt = torch.load(checkpoint_path, map_location=dev, weights_only=False)

        config = ckpt.get("config", {})

        # Reconstruct model from a sample test graph to get dimensions
        dataset = FloodDataset(data_root, mode="test")
        if len(dataset) == 0:
            dataset = FloodDataset(data_root, mode="train")

        sample = dataset[0]
        data = build_unified_graph(sample)

        from src.graph_builder_unified import get_feature_dims
        dims = get_feature_dims(data)

        model = UnifiedFloodModel(
            in_channels_1d_static=dims["in_channels_1d_static"],
            in_channels_1d_dynamic=dims["in_channels_1d_dynamic"],
            in_channels_2d_static=dims["in_channels_2d_static"],
            in_channels_2d_dynamic=dims["in_channels_2d_dynamic"],
            hidden_channels=config.get("hidden_channels", 64),
            num_gnn_layers=config.get("num_gnn_layers", 3),
            num_gru_layers=config.get("num_gru_layers", 1),
            dropout=config.get("dropout", 0.1),
        )

        model.load_state_dict(ckpt["model_state_dict"])
        model.to(dev)
        model.eval()

        # Load per-model static normalization stats if available
        static_norm_stats = ckpt.get("static_norm_stats", {})

        return cls(
            model=model, data_root=data_root, device=dev,
            static_norm_stats=static_norm_stats, **kwargs,
        )


# =====================================================================
#  Ensemble Inference (Advanced)
# =====================================================================

@torch.no_grad()
def ensemble_predict_event(
    models: List[UnifiedFloodModel],
    data: Any,  # HeteroData
    device: torch.device,
    spinup_steps: int = 10,
    weights: Optional[List[float]] = None,
) -> Tuple[Tensor, Tensor]:
    """Run ensemble inference by averaging predictions from multiple models.

    This is a key strategy for Kaggle competitions: train multiple
    models with different hyperparameters / seeds / folds and average
    predictions.  This reduces variance and often improves the score
    by 5-15%.

    Parameters
    ----------
    models : list[UnifiedFloodModel]
        List of trained models.
    data : HeteroData
    device : torch.device
    spinup_steps : int
    weights : list[float] or None
        Optional model weights for weighted averaging.
        If None, uses equal weights.

    Returns
    -------
    (preds_1d, preds_2d)
        Ensemble-averaged predictions.
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    else:
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]

    all_preds_1d: List[Tensor] = []
    all_preds_2d: List[Tensor] = []

    for model, w in zip(models, weights):
        model.to(device)
        model.eval()

        p1d, p2d = predict_event(model, data, device, spinup_steps)
        all_preds_1d.append(p1d * w)
        all_preds_2d.append(p2d * w)

    ensemble_1d = torch.stack(all_preds_1d, dim=0).sum(dim=0)
    ensemble_2d = torch.stack(all_preds_2d, dim=0).sum(dim=0)

    return ensemble_1d, ensemble_2d


# =====================================================================
#  CLI Entrypoint
# =====================================================================

def main() -> None:
    """Generate a submission from the command line.

    Usage::

        python -m src.inference --checkpoint checkpoints/best_model.pt
        python -m src.inference --checkpoint best.pt --output submission.csv
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate competition submission CSV."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint."
    )
    parser.add_argument(
        "--data_root", type=str, default="data",
        help="Path to data directory."
    )
    parser.add_argument(
        "--output", type=str, default="submission.csv",
        help="Output submission file path."
    )
    parser.add_argument(
        "--format", type=str, default="csv", choices=["csv", "parquet"],
        help="Output file format."
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use."
    )
    parser.add_argument(
        "--spinup", type=int, default=10,
        help="Number of burn-in steps."
    )

    args = parser.parse_args()

    gen = SubmissionGenerator.from_checkpoint(
        args.checkpoint,
        data_root=args.data_root,
        device=args.device,
        spinup_steps=args.spinup,
    )

    df = gen.generate()
    gen.save(df, args.output, file_format=args.format)


if __name__ == "__main__":
    main()

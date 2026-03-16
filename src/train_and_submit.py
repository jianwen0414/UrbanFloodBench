"""
train_and_submit — Complete training and submission pipeline for 2D
flood prediction.

This script:

1. Trains separate ``SurfaceEngine`` models for Model_1 and Model_2
2. Saves / loads checkpoints for both
3. Evaluates on a handful of validation events
4. Generates a test submission using the correct model per urban area

Usage::

    python -m src.train_and_submit          # full pipeline
    python -m src.train_and_submit --help   # (no argparse yet, but WIP)

Owner: Member B
See: IMPLEMENTATION_PLAN.md → Task 3
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

from src.config import RAW_DATA_PATH
from src.dataset import FloodDataset
from src.graph_builder_2d import build_2d_graph
from src.model_2d import (
    SurfaceEngine,
    evaluate_predictions,
    load_checkpoint,
    predict_event_2d,
    save_checkpoint,
    train_model,
)
from src.submission_2d import generate_test_submission_2d
from src.utils_2d import compute_normalization_stats


# ───────────────────────────────────────────────────────────────────────
#  Step 1 — Train
# ───────────────────────────────────────────────────────────────────────

def train_both_models(
    data_path: str,
    num_epochs: int = 30,
    hidden_channels: int = 64,
    num_sage_layers: int = 2,
    lr: float = 0.005,
    warmup_epochs: int = 8,
    decay_epochs: int = 17,
    max_timesteps_per_event: int | None = None,
    early_stopping_patience: int = 12,
    checkpoint_dir: str = "checkpoints",
    force_retrain: bool = False,
    conv_type: Literal["sage", "gat"] = "sage",
    num_heads: int = 4,
) -> dict:
    """Train separate models for Model_1 and Model_2.

    If a checkpoint already exists for a model and *force_retrain* is
    ``False``, the checkpoint is loaded instead of re-training.

    Parameters
    ----------
    data_path : str
        Root data folder.
    num_epochs, lr, warmup_epochs, decay_epochs, max_timesteps_per_event,
    early_stopping_patience
        Forwarded to :func:`train_model`.
    hidden_channels, num_sage_layers : int
        Architecture hyper-parameters.
    checkpoint_dir : str
        Where to persist ``.pt`` checkpoints.
    force_retrain : bool
        Re-train even when a checkpoint exists.

    Returns
    -------
    dict
        Keys: ``models``, ``norm_stats_dict``, ``histories``,
        ``in_channels``.
    """
    print("=" * 70)
    print("TRAINING PIPELINE: Model_1 and Model_2")
    print("=" * 70)

    ds_train = FloodDataset(data_path, mode="train")

    # ── normalization stats ───────────────────────────────────────
    print("\nComputing normalization statistics...")
    norm_stats_dict = {
        "1": compute_normalization_stats(ds_train, "1"),
        "2": compute_normalization_stats(ds_train, "2"),
    }
    print("  Normalization stats computed for both models")

    # ── input dimensions (same for both) ──────────────────────────
    sample_1 = ds_train.filter_by_model("1")[0]
    data_1 = build_2d_graph(sample_1, norm_stats_dict["1"], t_index=10)
    in_channels = data_1.x.shape[1]
    print(f"  Input features: {in_channels}")

    models: dict[str, SurfaceEngine] = {}
    histories: dict[str, dict | None] = {}

    for model_id in ["1", "2"]:
        print("\n" + "=" * 70)
        print(f"TRAINING MODEL_{model_id}")
        print("=" * 70)

        checkpoint_path = (
            Path(checkpoint_dir) / f"model_{model_id}_{conv_type}.pt"
        )

        # ── skip if already trained ───────────────────────────────
        if checkpoint_path.exists() and not force_retrain:
            print(f"\n  Checkpoint found: {checkpoint_path}")
            print(
                "  Loading existing model "
                "(use force_retrain=True to retrain)"
            )
            model = SurfaceEngine(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                num_sage_layers=num_sage_layers,
                dropout=0.1,
                max_delta=2.0,
                conv_type=conv_type,
                num_heads=num_heads,
            )
            load_checkpoint(model, str(checkpoint_path))
            models[model_id] = model
            histories[model_id] = None
            continue

        # ── create fresh model ────────────────────────────────────
        print(f"\nCreating SurfaceEngine for Model_{model_id}...")
        model = SurfaceEngine(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_sage_layers=num_sage_layers,
            dropout=0.1,
            max_delta=2.0,
            conv_type=conv_type,
            num_heads=num_heads,
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {total_params:,}")

        # ── train ─────────────────────────────────────────────────
        print(f"\nTraining on Model_{model_id} events...")
        t0 = time.time()

        history = train_model(
            model=model,
            dataset=ds_train,
            model_id=model_id,
            num_epochs=num_epochs,
            lr=lr,
            warmup_epochs=warmup_epochs,
            decay_epochs=decay_epochs,
            max_timesteps_per_event=max_timesteps_per_event,
            print_every=5,
            validate_every=5,
            validation_events=2,
            early_stopping_patience=early_stopping_patience,
            checkpoint_dir=checkpoint_dir,
            save_best=True,
        )

        elapsed = time.time() - t0
        print(
            f"\nModel_{model_id} training completed in "
            f"{elapsed / 60:.1f} minutes"
        )
        print(
            f"  Best Val RMSE: {history['best_val_rmse']:.4f} "
            f"at epoch {history['best_epoch']}"
        )

        # train_model saves to model_{id}_best.pt; copy to conv_type path
        best_path = Path(checkpoint_dir) / f"model_{model_id}_best.pt"
        if best_path.exists():
            import shutil
            shutil.copy(best_path, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

        # Ensure the model holds the best weights
        load_checkpoint(model, str(checkpoint_path))

        models[model_id] = model
        histories[model_id] = history

    return {
        "models": models,
        "norm_stats_dict": norm_stats_dict,
        "histories": histories,
        "in_channels": in_channels,
        "conv_type": conv_type,
    }


# ───────────────────────────────────────────────────────────────────────
#  Step 2 — Evaluate
# ───────────────────────────────────────────────────────────────────────

def evaluate_both_models(
    models: dict[str, SurfaceEngine],
    norm_stats_dict: dict[str, dict],
    data_path: str,
    num_events_per_model: int = 3,
) -> None:
    """Evaluate both trained models on a few validation events.

    Parameters
    ----------
    models : dict[str, SurfaceEngine]
    norm_stats_dict : dict[str, dict]
    data_path : str
    num_events_per_model : int
        How many events to score per model.
    """
    print("\n" + "=" * 70)
    print("EVALUATION: Both Models")
    print("=" * 70)

    ds_train = FloodDataset(data_path, mode="train")

    for model_id in ["1", "2"]:
        model = models.get(model_id)
        if model is None:
            print(f"\n  No model for Model_{model_id}")
            continue

        norm_stats = norm_stats_dict[model_id]
        ds_model = ds_train.filter_by_model(model_id)

        print(f"\nModel_{model_id}:")

        rmses: list[float] = []
        maes: list[float] = []

        n_eval = min(num_events_per_model, len(ds_model))
        for event_idx in range(n_eval):
            sample = ds_model[event_idx]
            event_id = sample["event_id"]

            pred_wl, gt_wl = predict_event_2d(
                model, sample, norm_stats, num_warmup=10
            )
            metrics = evaluate_predictions(pred_wl, gt_wl, num_warmup=10)

            rmses.append(metrics["rmse"])
            maes.append(metrics["mae"])

            print(
                f"  Event_{event_id}: "
                f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}"
            )

        print(
            f"  Average: "
            f"RMSE={np.mean(rmses):.4f}, MAE={np.mean(maes):.4f}"
        )


# ───────────────────────────────────────────────────────────────────────
#  Step 3 — Submit
# ───────────────────────────────────────────────────────────────────────

def generate_submission_both_models(
    models: dict[str, SurfaceEngine],
    norm_stats_dict: dict[str, dict],
    data_path: str,
    output_path: str = "submissions/submission_2d.csv",
) -> pd.DataFrame:
    """Generate a test submission using the correct model per area.

    Parameters
    ----------
    models : dict[str, SurfaceEngine]
    norm_stats_dict : dict[str, dict]
    data_path : str
    output_path : str

    Returns
    -------
    pd.DataFrame
        Competition-ready 2D submission.
    """
    print("\n" + "=" * 70)
    print("GENERATING TEST SUBMISSION")
    print("=" * 70)

    return generate_test_submission_2d(
        model=models,
        data_path=data_path,
        norm_stats_dict=norm_stats_dict,
        output_path=output_path,
        num_warmup=10,
        verbose=True,
    )


# ───────────────────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Train → evaluate → generate submission."""
    print("\n" + "=" * 70)
    print("2D FLOOD PREDICTION — COMPLETE PIPELINE")
    print("=" * 70)
    print(f"\nData path: {RAW_DATA_PATH}")

    # === STEP 1: Train both models ================================
    result = train_both_models(
        data_path=RAW_DATA_PATH,
        num_epochs=30,
        hidden_channels=64,
        num_sage_layers=2,
        lr=0.005,
        warmup_epochs=8,
        decay_epochs=17,
        max_timesteps_per_event=50,
        early_stopping_patience=12,
        checkpoint_dir="checkpoints",
        force_retrain=False,
    )

    models = result["models"]
    norm_stats_dict = result["norm_stats_dict"]
    conv_type = result.get("conv_type", "sage")

    # === STEP 2: Evaluate both models =============================
    evaluate_both_models(
        models=models,
        norm_stats_dict=norm_stats_dict,
        data_path=RAW_DATA_PATH,
        num_events_per_model=3,
    )

    # === STEP 3: Generate submission ==============================
    submission = generate_submission_both_models(
        models=models,
        norm_stats_dict=norm_stats_dict,
        data_path=RAW_DATA_PATH,
        output_path="submissions/submission_2d.csv",
    )

    # === STEP 4: Summary ==========================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    conv_type = result.get("conv_type", "sage")
    print("\nCheckpoints saved:")
    for mid in ["1", "2"]:
        cp = Path(f"checkpoints/model_{mid}_{conv_type}.pt")
        if cp.exists():
            print(f"  {cp} ({cp.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  {cp} (NOT FOUND)")

    sub_path = Path("submissions/submission_2d.csv")
    print(f"\nSubmission saved:")
    if sub_path.exists():
        size_mb = sub_path.stat().st_size / (1024 * 1024)
        print(f"  {sub_path} ({size_mb:.2f} MB)")
        if submission is not None:
            print(f"  Total rows: {len(submission):,}")

    print("\n" + "-" * 70)
    print("NEXT STEPS:")
    print("-" * 70)
    print("1. Run:  python -m src.submission_utils")
    print("   -> Adds dummy 1D predictions for Kaggle format testing")
    print()
    print("2. Or combine with real 1D predictions from your teammate:")
    print("   from src.submission_utils import combine_1d_2d_submissions")
    print(
        "   combine_1d_2d_submissions("
        "'submission_1d.csv', 'submission_2d.csv', 'final.csv')"
    )
    print("-" * 70)


if __name__ == "__main__":
    main()

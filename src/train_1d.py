"""
Complete 1D training and submission pipeline.

Usage:
    python -m src.train_1d
    python -m src.train_1d --force_retrain
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.config import RAW_DATA_PATH
from src.dataset import FloodDataset
from src.model_1d import (
    DrainageNetwork1D,
    train_model_1d,
    predict_event_1d,
    evaluate_predictions_1d,
    load_checkpoint,
    save_checkpoint,
)
from src.graph_builder_1d import get_1d_input_dim
from src.utils_1d import compute_normalization_stats_1d, analyze_1d_data
from src.submission_1d import generate_test_submission_1d, combine_1d_2d_submissions


def train_1d_pipeline(force_retrain: bool = False) -> None:
    """Complete 1D training pipeline."""

    print("\n" + "=" * 70)
    print("1D FLOOD PREDICTION — COMPLETE PIPELINE")
    print("=" * 70)
    print(f"\nData path: {RAW_DATA_PATH}")

    # ========================================
    # STEP 1: ANALYZE DATA
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 1: ANALYZING 1D DATA")
    print("=" * 70)

    ds_train = FloodDataset(RAW_DATA_PATH, mode="train")

    for model_id in ["1", "2"]:
        print(f"\nModel_{model_id}:")
        analysis = analyze_1d_data(ds_train, model_id)
        for key, value in analysis.items():
            print(f"  {key}: {value}")

    # ========================================
    # STEP 2: TRAIN MODELS
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING 1D MODELS")
    print("=" * 70)

    models = {}
    norm_stats_dict = {}

    for model_id in ["1", "2"]:
        print("\n" + "=" * 70)
        print(f"MODEL_{model_id} (1D)")
        print("=" * 70)

        # Get data
        ds_model = ds_train.filter_by_model(model_id)
        if len(ds_model) == 0:
            print(f"  No events for Model_{model_id}, skipping")
            continue

        sample = ds_model[0]

        if sample.get("static_1d_nodes") is None or len(sample["static_1d_nodes"]) == 0:
            print(f"  No 1D data for Model_{model_id}, skipping")
            continue

        # Compute normalization stats
        norm_stats = compute_normalization_stats_1d(ds_train, model_id)
        norm_stats_dict[model_id] = norm_stats

        # Get input dimensions
        in_channels = get_1d_input_dim(sample, norm_stats)
        print(f"  Input features: {in_channels}")

        # Check for existing checkpoint
        checkpoint_path = Path(f"checkpoints/model_{model_id}_1d_best.pt")

        if checkpoint_path.exists() and not force_retrain:
            print(f"\n  Checkpoint found: {checkpoint_path}")
            print("  Loading existing model (use --force_retrain to retrain)")

            model = DrainageNetwork1D(
                in_channels=in_channels,
                hidden_channels=64,
                num_sage_layers=2,
                dropout=0.1,
                max_delta=2.0,
            )
            load_checkpoint(model, str(checkpoint_path))
            models[model_id] = model
            continue

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

        # Train
        print("\n  Training...")
        start = time.time()

        history = train_model_1d(
            model=model,
            dataset=ds_train,
            model_id=model_id,
            num_epochs=40,
            lr=0.003,
            warmup_epochs=10,
            decay_epochs=20,
            max_timesteps_per_event=50,
            print_every=5,
            validate_every=5,
            validation_events=3,
            early_stopping_patience=12,
            checkpoint_dir="checkpoints",
            save_best=True,
        )

        elapsed = time.time() - start
        print(f"\n  Training time: {elapsed/60:.1f} minutes")
        print(
            f"  Best Val RMSE: {history['best_val_rmse']:.4f} "
            f"at epoch {history['best_epoch']}"
        )

        models[model_id] = model

    # ========================================
    # STEP 3: EVALUATE
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 3: EVALUATION")
    print("=" * 70)

    for model_id, model in models.items():
        norm_stats = norm_stats_dict[model_id]
        ds_model = ds_train.filter_by_model(model_id)

        print(f"\nModel_{model_id} (1D):")
        print("-" * 40)

        rmses = []
        maes = []

        # Evaluate on first 5 events
        for i in range(min(5, len(ds_model))):
            sample = ds_model[i]

            if sample.get("static_1d_nodes") is None:
                continue

            pred_wl, gt_wl = predict_event_1d(
                model, sample, norm_stats, num_warmup=10
            )
            metrics = evaluate_predictions_1d(pred_wl, gt_wl, num_warmup=10)

            rmses.append(metrics["rmse"])
            maes.append(metrics["mae"])

            print(
                f"  Event_{sample['event_id']}: "
                f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}"
            )

        if rmses:
            print(
                f"\n  Average: RMSE={np.mean(rmses):.4f} "
                f"(±{np.std(rmses):.4f}), MAE={np.mean(maes):.4f}"
            )

    # ========================================
    # STEP 4: GENERATE 1D SUBMISSION
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 4: GENERATING 1D SUBMISSION")
    print("=" * 70)

    generate_test_submission_1d(
        model=models,
        data_path=RAW_DATA_PATH,
        norm_stats_dict=norm_stats_dict,
        output_path="submissions/submission_1d.csv",
        num_warmup=10,
        verbose=True,
    )

    # ========================================
    # STEP 5: COMBINE WITH 2D SUBMISSION
    # ========================================

    # Find best 2D submission
    submission_2d_path = None
    candidates = [
        "submissions/submission_2d_ensemble.csv",
        "submissions/submission_2d_exp1.csv",
        "submissions/submission_2d.csv",
    ]

    for candidate in candidates:
        if Path(candidate).exists():
            submission_2d_path = candidate
            break

    if submission_2d_path:
        print("\n" + "=" * 70)
        print("STEP 5: COMBINING 1D AND 2D SUBMISSIONS")
        print("=" * 70)
        print(f"  Using 2D submission: {submission_2d_path}")

        combine_1d_2d_submissions(
            submission_1d_path="submissions/submission_1d.csv",
            submission_2d_path=submission_2d_path,
            output_path="submissions/submission_full_with_1d.csv",
            verbose=True,
        )

        print("\n  Combined CSV ready (no parquet needed for 1D).")
        print("  Or submit 2D ensemble + 1D separately.")
    else:
        print("\n  No 2D submission found. Generate 2D predictions first.")
        print("  Available files:", [str(p) for p in Path("submissions").glob("*.csv")])

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("1D PIPELINE COMPLETE")
    print("=" * 70)

    print("\nCheckpoints:")
    for model_id in models:
        cp = Path(f"checkpoints/model_{model_id}_1d_best.pt")
        if cp.exists():
            print(f"  ✅ {cp} ({cp.stat().st_size/1024:.1f} KB)")

    print("\nSubmissions:")
    for sub in ["submission_1d.csv", "submission_full_with_1d.csv"]:
        p = Path(f"submissions/{sub}")
        if p.exists():
            size_mb = p.stat().st_size / (1024**2)
            print(f"  ✅ {p} ({size_mb:.1f} MB)")

    print("\n" + "=" * 70)
    print("SUBMIT TO KAGGLE: submissions/submission_full_with_1d.csv")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 1D flood prediction models")
    parser.add_argument(
        "--force_retrain",
        action="store_true",
        help="Force retraining even if checkpoints exist",
    )
    args = parser.parse_args()

    train_1d_pipeline(force_retrain=args.force_retrain)


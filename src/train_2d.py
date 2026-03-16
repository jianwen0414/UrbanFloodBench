"""
Complete 2D training pipeline for Urban Flood Bench.

Usage:
    python -m src.train_2d
    python -m src.train_2d --conv_type gat --force_retrain
    python -m src.train_2d --model_id 1 --conv_type sage
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.config import RAW_DATA_PATH
from src.dataset import FloodDataset
from src.model_2d import SurfaceEngine, train_model, load_checkpoint
from src.graph_builder_2d import build_2d_graph
from src.utils_2d import compute_normalization_stats


def train_2d_pipeline(
    force_retrain: bool = False,
    conv_type: str = "sage",
    target_model: str = "all",
) -> None:
    """Complete 2D training pipeline."""

    print("\n" + "=" * 70)
    print(f"2D FLOOD PREDICTION — COMPLETE PIPELINE ({conv_type.upper()})")
    print("=" * 70)
    print(f"Data path: {RAW_DATA_PATH}")

    ds_train = FloodDataset(RAW_DATA_PATH, mode="train")
    
    models_to_train = ["1", "2"] if target_model == "all" else [target_model]
    
    for model_id in models_to_train:
        print("\n" + "=" * 70)
        print(f"MODEL_{model_id} (2D - {conv_type.upper()})")
        print("=" * 70)

        # Get data
        ds_model = ds_train.filter_by_model(model_id)
        if len(ds_model) == 0:
            print(f"  No events for Model_{model_id}, skipping")
            continue

        sample = ds_model[0]

        # Compute normalization stats
        norm_stats = compute_normalization_stats(ds_train, model_id)

        # Get input dimensions by building a dummy graph
        data_dummy = build_2d_graph(sample, norm_stats, t_index=10)
        in_channels = data_dummy.x.shape[1]
        print(f"  Input features: {in_channels}")

        # Check for existing checkpoint
        suffix = "gat" if conv_type == "gat" else "improved"
        checkpoint_path = Path(f"checkpoints/model_{model_id}_{suffix}.pt")
        ema_checkpoint_path = Path(f"checkpoints/model_{model_id}_{suffix}_ema_best.pt")

        if checkpoint_path.exists() and not force_retrain:
            print(f"\n  Checkpoint found: {checkpoint_path}")
            print("  Loading existing model (use --force_retrain to retrain)")

            model = SurfaceEngine(
                in_channels=in_channels,
                hidden_channels=128,
                num_sage_layers=3,
                dropout=0.15,
                max_delta=2.0,
                conv_type=conv_type,
                num_heads=4 if conv_type == "gat" else 1,
            )
            load_checkpoint(model, str(checkpoint_path))
            continue

        # Create model
        print(f"\n  Initializing new {conv_type.upper()} architecture...")
        model = SurfaceEngine(
            in_channels=in_channels,
            hidden_channels=128,
            num_sage_layers=3,
            dropout=0.15,
            max_delta=2.0,
            conv_type=conv_type,
            num_heads=4 if conv_type == "gat" else 1,
        )

        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")

        # Train
        print("\n  Training...")
        start = time.time()
        
        # We override checkpoint_dir behavior in the model to save exactly where we want
        # but the model hardcodes "model_{model_id}_best.pt" in its checkpoint path inside `train_model`...
        # Wait, if `train_model` saves as `model_{model_id}_best.pt`, we will just move it after training.

        history = train_model(
            model=model,
            dataset=ds_train,
            model_id=model_id,
            num_epochs=80, # Full run
            lr=0.003,
            warmup_epochs=10,
            decay_epochs=30,
            max_timesteps_per_event=60, # Keep manageable for RAM, original was 50-60
            print_every=5,
            validate_every=5,
            validation_events=2,
            early_stopping_patience=15,
            checkpoint_dir="checkpoints",
            save_best=True,
            lr_warmup_epochs=0,
            tf_min_ratio=0.3,
            grad_clip=1.0,
        )

        elapsed = time.time() - start
        
        # Rename standard checkpoints to matching suffix
        default_cp = Path(f"checkpoints/model_{model_id}_best.pt")
        default_ema_cp = Path(f"checkpoints/model_{model_id}_ema_best.pt")
        
        if default_cp.exists() and default_cp != checkpoint_path:
            import shutil
            shutil.move(str(default_cp), str(checkpoint_path))
        if default_ema_cp.exists() and default_ema_cp != ema_checkpoint_path:
            import shutil
            shutil.move(str(default_ema_cp), str(ema_checkpoint_path))

        print(f"\n  Training time: {elapsed/60:.1f} minutes")
        print(f"  Best Val RMSE: {min(history['val_rmse']):.4f}")
        print(f"  Target checkpoints saved: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2D Flood Models")
    parser.add_argument("--force_retrain", action="store_true", help="Force retraining models")
    parser.add_argument("--conv_type", type=str, default="sage", choices=["sage", "gat"], help="GNN layer type")
    parser.add_argument("--model_id", type=str, default="all", choices=["all", "1", "2"], help="Which model to train")
    
    args = parser.parse_args()
    train_2d_pipeline(
        force_retrain=args.force_retrain,
        conv_type=args.conv_type,
        target_model=args.model_id
    )

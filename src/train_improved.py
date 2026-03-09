"""
Per-model training configs for 2D flood prediction.

Model_1: larger architecture (128 hidden, 3 layers, GAT).
Model_2: smaller architecture (64 hidden, 2 layers, GAT) with lower LR.
  Model_2 diverges with the Model_1 config — proven by checkpoint analysis:
  model_2_improved.pt (128h/3L/lr=0.002) best_epoch=0 (diverged),
  model_2_exp1_lower_lr.pt (64h/2L/lr=0.001) best_epoch=35 (converged).
"""

import torch
import numpy as np
from pathlib import Path
import time

from src.config import RAW_DATA_PATH
from src.dataset import FloodDataset
from src.model_2d import (
    SurfaceEngine,
    train_model,
    predict_event_2d,
    evaluate_predictions,
    load_checkpoint,
)
from src.graph_builder_2d import build_2d_graph
from src.utils_2d import compute_normalization_stats
from src.submission_2d import generate_test_submission_2d


# =====================================================================
#  Per-model configs
# =====================================================================

# Model_1: GAT won the SAGE vs GAT comparison (SRMSE 0.019 vs 0.023).
# Architecture: 128 hidden, 3 layers — same as the Feb 14 improved run.
MODEL_1_CONFIG = {
    "hidden_channels": 128,
    "num_sage_layers": 3,
    "dropout": 0.15,
    "max_delta": 2.0,
    "conv_type": "gat",
    "num_heads": 4,

    "num_epochs": 50,
    "lr": 0.002,
    "warmup_epochs": 12,
    "decay_epochs": 28,
    "max_timesteps_per_event": 60,
    "early_stopping_patience": 15,
    "grad_clip": 1.0,
}

# Model_2: smaller architecture + lower LR.  Recovered from the working
# model_2_exp1_lower_lr.pt checkpoint (best_epoch=35, val_rmse=0.504).
# Model_2 diverges immediately with the Model_1 config (128h/3L/lr=0.002).
#
# IMPORTANT: the original working exp1_lower_lr used SAGE, not GAT.
# GAT with 64h/4heads = 16 dims/head was too small and had double-dropout
# (fixed in model_2d.py).  Using num_heads=2 → 32 dims/head for stability.
MODEL_2_CONFIG = {
    "hidden_channels": 64,
    "num_sage_layers": 2,
    "dropout": 0.1,
    "max_delta": 2.0,
    "conv_type": "gat",
    "num_heads": 2,            # 64 / 2 = 32 per head (was 4→16, too small)

    "num_epochs": 40,
    "lr": 0.001,               # half of Model_1 — critical for stability
    "warmup_epochs": 15,       # longer TF warmup than Model_1
    "decay_epochs": 20,
    "max_timesteps_per_event": 50,
    "early_stopping_patience": 15,
    "grad_clip": 1.0,
}

# Fallback: proven-working SAGE config (identical to exp1_lower_lr).
# Use this if GAT still diverges after the dropout + num_heads fixes.
MODEL_2_SAGE_FALLBACK = {
    "hidden_channels": 64,
    "num_sage_layers": 2,
    "dropout": 0.1,
    "max_delta": 2.0,
    "conv_type": "sage",
    "num_heads": 4,

    "num_epochs": 40,
    "lr": 0.001,
    "warmup_epochs": 15,
    "decay_epochs": 20,
    "max_timesteps_per_event": 50,
    "early_stopping_patience": 15,
    "grad_clip": 1.0,
}

# Which model(s) to train: "1", "2", or "both"
TRAIN_MODEL_ID = "2"

CONFIGS = {"1": MODEL_1_CONFIG, "2": MODEL_2_SAGE_FALLBACK}


# =====================================================================
#  Training
# =====================================================================

def train_improved_model(
    model_id: str,
    config: dict,
    force_retrain: bool = False,
) -> tuple:
    """Train an improved model for a specific urban area.

    Returns (model, history, norm_stats).
    """
    print("=" * 70)
    print(f"TRAINING MODEL_{model_id}")
    print("=" * 70)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    conv_type = config.get("conv_type", "sage")
    checkpoint_dir = "checkpoints"
    checkpoint_path = Path(checkpoint_dir) / f"model_{model_id}_{conv_type}.pt"

    print("\nLoading data...")
    ds_train = FloodDataset(RAW_DATA_PATH, mode="train")
    norm_stats = compute_normalization_stats(ds_train, model_id)

    sample = ds_train.filter_by_model(model_id)[0]
    data = build_2d_graph(sample, norm_stats, t_index=10)
    in_channels = data.x.shape[1]
    print(f"  Input features: {in_channels}")

    if checkpoint_path.exists() and not force_retrain:
        print(f"\nCheckpoint found: {checkpoint_path}")
        print("  Loading existing model (use force_retrain=True to retrain)")

        model = SurfaceEngine(
            in_channels=in_channels,
            hidden_channels=config["hidden_channels"],
            num_sage_layers=config["num_sage_layers"],
            dropout=config["dropout"],
            max_delta=config["max_delta"],
            conv_type=conv_type,
            num_heads=config.get("num_heads", 4),
        )
        load_checkpoint(model, str(checkpoint_path))
        return model, None, norm_stats

    print("\nCreating model...")
    model = SurfaceEngine(
        in_channels=in_channels,
        hidden_channels=config["hidden_channels"],
        num_sage_layers=config["num_sage_layers"],
        dropout=config["dropout"],
        max_delta=config["max_delta"],
        conv_type=conv_type,
        num_heads=config.get("num_heads", 4),
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    print(f"\nTraining on Model_{model_id}...")
    start_time = time.time()

    history = train_model(
        model=model,
        dataset=ds_train,
        model_id=model_id,
        num_epochs=config["num_epochs"],
        lr=config["lr"],
        warmup_epochs=config["warmup_epochs"],
        decay_epochs=config["decay_epochs"],
        max_timesteps_per_event=config["max_timesteps_per_event"],
        print_every=5,
        validate_every=5,
        validation_events=3,
        early_stopping_patience=config["early_stopping_patience"],
        checkpoint_dir=checkpoint_dir,
        save_best=True,
        lr_warmup_epochs=config.get("lr_warmup_epochs", 0),
        tf_min_ratio=config.get("tf_min_ratio"),
        grad_clip=config.get("grad_clip", 1.0),
    )

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Best Val RMSE: {history['best_val_rmse']:.4f} at epoch {history['best_epoch']}")

    import shutil
    default_checkpoint = Path(checkpoint_dir) / f"model_{model_id}_best.pt"
    if default_checkpoint.exists():
        shutil.copy(default_checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    load_checkpoint(model, str(checkpoint_path))

    return model, history, norm_stats


def evaluate_improved_model(
    model,
    model_id: str,
    norm_stats: dict,
    num_events: int = 5,
):
    """Evaluate the improved model on more events."""
    print(f"\nEvaluating Model_{model_id} on {num_events} events...")

    ds_train = FloodDataset(RAW_DATA_PATH, mode="train")
    ds_model = ds_train.filter_by_model(model_id)

    rmses = []
    maes = []

    for i in range(min(num_events, len(ds_model))):
        sample = ds_model[i]
        pred_wl, gt_wl = predict_event_2d(model, sample, norm_stats, num_warmup=10)
        metrics = evaluate_predictions(pred_wl, gt_wl, num_warmup=10)
        rmses.append(metrics["rmse"])
        maes.append(metrics["mae"])
        print(f"  Event_{sample['event_id']}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

    print(f"\n  Average RMSE: {np.mean(rmses):.4f} (+-{np.std(rmses):.4f})")
    print(f"  Average MAE:  {np.mean(maes):.4f} (+-{np.std(maes):.4f})")

    return {"rmses": rmses, "maes": maes}


# =====================================================================
#  Main
# =====================================================================

def main():
    """Train models for selected urban area(s)."""
    ids_to_train = ["1", "2"] if TRAIN_MODEL_ID == "both" else [TRAIN_MODEL_ID]

    print("\n" + "=" * 70)
    print("2D TRAINING PIPELINE (per-model configs)")
    print("=" * 70)
    print(f"Training model(s): {', '.join(ids_to_train)} (TRAIN_MODEL_ID={TRAIN_MODEL_ID!r})")
    for mid in ids_to_train:
        cfg = CONFIGS[mid]
        print(f"  Model_{mid}: {cfg['conv_type'].upper()}, "
              f"{cfg['hidden_channels']}h/{cfg['num_sage_layers']}L, "
              f"lr={cfg['lr']}")
    print("=" * 70)

    models = {}
    norm_stats_dict = {}

    for model_id in ids_to_train:
        config = CONFIGS[model_id]
        model, history, norm_stats = train_improved_model(
            model_id=model_id,
            config=config,
            force_retrain=True,
        )
        models[model_id] = model
        norm_stats_dict[model_id] = norm_stats

        evaluate_improved_model(model, model_id, norm_stats, num_events=5)

    if len(ids_to_train) == 2:
        print("\n" + "=" * 70)
        print("GENERATING SUBMISSION")
        print("=" * 70)

        submission = generate_test_submission_2d(
            model=models,
            data_path=RAW_DATA_PATH,
            norm_stats_dict=norm_stats_dict,
            output_path="submissions/submission_2d_improved.csv",
            num_warmup=10,
            verbose=True,
        )
        print("\nSubmission: submissions/submission_2d_improved.csv")
    else:
        print(f"\nSkipping submission (only trained: {ids_to_train})")

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print("\nCheckpoints:")
    for mid in ids_to_train:
        ct = CONFIGS[mid]["conv_type"]
        print(f"  checkpoints/model_{mid}_{ct}.pt")


if __name__ == "__main__":
    main()

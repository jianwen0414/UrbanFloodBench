"""
Experimental training for Model_2 to find optimal configuration.

Experiments:
1. Baseline architecture with lower LR
2. Smaller architecture (32 hidden)
3. More regularization (higher dropout)
4. Much slower teacher forcing decay
5. Freeze early, train late (curriculum)
"""

import torch
import numpy as np
from pathlib import Path
import time
import json

from src.config import RAW_DATA_PATH
from src.dataset import FloodDataset
from src.model_2d import (
    SurfaceEngine,
    train_model,
    predict_event_2d,
    evaluate_predictions,
    save_checkpoint,
    load_checkpoint
)
from src.graph_builder_2d import build_2d_graph
from src.utils_2d import compute_normalization_stats


# Experiment configurations
EXPERIMENTS = {
    "exp1_lower_lr": {
        "description": "Baseline architecture, much lower learning rate",
        "hidden_channels": 64,
        "num_sage_layers": 2,
        "dropout": 0.1,
        "max_delta": 2.0,
        "num_epochs": 40,
        "lr": 0.001,  # Lower than 0.005
        "warmup_epochs": 15,
        "decay_epochs": 20,
        "max_timesteps_per_event": 50,
        "early_stopping_patience": 15,
    },
    "exp2_tiny_lr": {
        "description": "Baseline architecture, very low learning rate",
        "hidden_channels": 64,
        "num_sage_layers": 2,
        "dropout": 0.1,
        "max_delta": 2.0,
        "num_epochs": 50,
        "lr": 0.0005,  # Very low
        "warmup_epochs": 20,
        "decay_epochs": 25,
        "max_timesteps_per_event": 50,
        "early_stopping_patience": 20,
    },
    "exp3_smaller_model": {
        "description": "Smaller model (32 hidden channels)",
        "hidden_channels": 32,
        "num_sage_layers": 2,
        "dropout": 0.1,
        "max_delta": 2.0,
        "num_epochs": 40,
        "lr": 0.002,
        "warmup_epochs": 12,
        "decay_epochs": 20,
        "max_timesteps_per_event": 50,
        "early_stopping_patience": 15,
    },
    "exp4_high_dropout": {
        "description": "Higher dropout for regularization",
        "hidden_channels": 64,
        "num_sage_layers": 2,
        "dropout": 0.3,  # Higher dropout
        "max_delta": 2.0,
        "num_epochs": 40,
        "lr": 0.002,
        "warmup_epochs": 12,
        "decay_epochs": 20,
        "max_timesteps_per_event": 50,
        "early_stopping_patience": 15,
    },
    "exp5_slow_tf_decay": {
        "description": "Very slow teacher forcing decay",
        "hidden_channels": 64,
        "num_sage_layers": 2,
        "dropout": 0.15,
        "max_delta": 2.0,
        "num_epochs": 60,
        "lr": 0.002,
        "warmup_epochs": 25,  # Long warmup
        "decay_epochs": 30,   # Slow decay
        "max_timesteps_per_event": 50,
        "early_stopping_patience": 20,
    },
    "exp6_combined_best": {
        "description": "Combined: smaller model + low LR + high dropout + slow decay",
        "hidden_channels": 48,
        "num_sage_layers": 2,
        "dropout": 0.25,
        "max_delta": 2.0,
        "num_epochs": 50,
        "lr": 0.001,
        "warmup_epochs": 20,
        "decay_epochs": 25,
        "max_timesteps_per_event": 50,
        "early_stopping_patience": 18,
    },
}


def run_experiment(
    exp_name: str,
    config: dict,
    force_run: bool = False
) -> dict:
    """
    Run a single experiment for Model_2.

    Args:
        exp_name: Experiment name
        config: Configuration dict
        force_run: If True, run even if checkpoint exists

    Returns:
        Results dict with metrics
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {exp_name}")
    print(f"Description: {config['description']}")
    print("=" * 70)

    checkpoint_dir = "checkpoints/experiments"
    checkpoint_path = Path(checkpoint_dir) / f"model_2_{exp_name}.pt"
    results_path = Path(checkpoint_dir) / f"model_2_{exp_name}_results.json"

    # Check if already done
    if checkpoint_path.exists() and results_path.exists() and not force_run:
        print(f"\n✓ Already completed. Loading results...")
        with open(results_path, 'r') as f:
            return json.load(f)

    # Create checkpoint dir
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    ds_train = FloodDataset(RAW_DATA_PATH, mode="train")
    norm_stats = compute_normalization_stats(ds_train, "2")

    # Get input dimensions
    sample = ds_train.filter_by_model("2")[0]
    data = build_2d_graph(sample, norm_stats, t_index=10)
    in_channels = data.x.shape[1]

    # Create model
    print(f"\nCreating model...")
    print(f"  hidden_channels: {config['hidden_channels']}")
    print(f"  num_sage_layers: {config['num_sage_layers']}")
    print(f"  dropout: {config['dropout']}")
    print(f"  lr: {config['lr']}")
    print(f"  warmup_epochs: {config['warmup_epochs']}")
    print(f"  decay_epochs: {config['decay_epochs']}")

    model = SurfaceEngine(
        in_channels=in_channels,
        hidden_channels=config["hidden_channels"],
        num_sage_layers=config["num_sage_layers"],
        dropout=config["dropout"],
        max_delta=config["max_delta"],
        conv_type=config.get("conv_type", "sage"),
        num_heads=config.get("num_heads", 4),
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    # Train
    print(f"\nTraining...")
    start_time = time.time()

    history = train_model(
        model=model,
        dataset=ds_train,
        model_id="2",
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
        save_best=True
    )

    elapsed = time.time() - start_time

    # Copy checkpoint to experiment-specific name
    default_cp = Path(checkpoint_dir) / "model_2_best.pt"
    if default_cp.exists():
        import shutil
        shutil.copy(default_cp, checkpoint_path)

    # Load best model and evaluate on more events
    print("\nEvaluating on validation events...")
    load_checkpoint(model, str(checkpoint_path))

    ds_model = ds_train.filter_by_model("2")
    val_rmses = []
    val_maes = []

    # Evaluate on first 5 events
    for i in range(min(5, len(ds_model))):
        sample = ds_model[i]
        pred_wl, gt_wl = predict_event_2d(model, sample, norm_stats, num_warmup=10)
        metrics = evaluate_predictions(pred_wl, gt_wl, num_warmup=10)
        val_rmses.append(metrics["rmse"])
        val_maes.append(metrics["mae"])
        print(f"  Event_{sample['event_id']}: RMSE={metrics['rmse']:.4f}")

    # Compile results
    results = {
        "exp_name": exp_name,
        "description": config["description"],
        "config": config,
        "num_params": num_params,
        "training_time_min": elapsed / 60,
        "best_epoch": history["best_epoch"],
        "best_val_rmse": history["best_val_rmse"],
        "eval_rmse_mean": float(np.mean(val_rmses)),
        "eval_rmse_std": float(np.std(val_rmses)),
        "eval_mae_mean": float(np.mean(val_maes)),
        "checkpoint_path": str(checkpoint_path),
    }

    # Save results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"RESULTS: {exp_name}")
    print(f"{'='*70}")
    print(f"  Best epoch: {results['best_epoch']}")
    print(f"  Best Val RMSE: {results['best_val_rmse']:.4f}")
    print(f"  Eval RMSE: {results['eval_rmse_mean']:.4f} (±{results['eval_rmse_std']:.4f})")
    print(f"  Training time: {results['training_time_min']:.1f} min")

    return results


def run_all_experiments(experiments_to_run: list = None, force_run: bool = False):
    """
    Run all experiments and compare results.

    Args:
        experiments_to_run: List of experiment names to run (None = all)
        force_run: If True, rerun even if completed
    """
    print("\n" + "=" * 70)
    print("MODEL_2 EXPERIMENTS")
    print("=" * 70)

    if experiments_to_run is None:
        experiments_to_run = list(EXPERIMENTS.keys())

    print(f"\nRunning {len(experiments_to_run)} experiments:")
    for exp in experiments_to_run:
        print(f"  - {exp}: {EXPERIMENTS[exp]['description']}")

    all_results = []

    for exp_name in experiments_to_run:
        config = EXPERIMENTS[exp_name]
        results = run_experiment(exp_name, config, force_run=force_run)
        all_results.append(results)

    # Compare all results
    print("\n" + "=" * 70)
    print("COMPARISON OF ALL EXPERIMENTS")
    print("=" * 70)

    # Sort by eval RMSE
    all_results.sort(key=lambda x: x["eval_rmse_mean"])

    print(f"\n{'Experiment':<25} {'Best Epoch':>10} {'Val RMSE':>10} {'Eval RMSE':>12} {'Time':>8}")
    print("-" * 70)

    for r in all_results:
        print(f"{r['exp_name']:<25} {r['best_epoch']:>10} {r['best_val_rmse']:>10.4f} "
              f"{r['eval_rmse_mean']:>10.4f}±{r['eval_rmse_std']:.2f} {r['training_time_min']:>6.1f}m")

    # Best experiment
    best = all_results[0]
    print(f"\n🏆 BEST: {best['exp_name']}")
    print(f"   Eval RMSE: {best['eval_rmse_mean']:.4f}")
    print(f"   Checkpoint: {best['checkpoint_path']}")

    return all_results


def generate_best_submission(best_exp_name: str = None):
    """
    Generate submission using best Model_2 experiment + improved Model_1.
    """
    from src.submission_2d import generate_test_submission_2d

    print("\n" + "=" * 70)
    print("GENERATING SUBMISSION WITH BEST MODEL_2")
    print("=" * 70)

    # Find best experiment if not specified
    if best_exp_name is None:
        results_dir = Path("checkpoints/experiments")
        best_rmse = float('inf')

        for results_file in results_dir.glob("model_2_*_results.json"):
            with open(results_file, 'r') as f:
                r = json.load(f)
                if r["eval_rmse_mean"] < best_rmse:
                    best_rmse = r["eval_rmse_mean"]
                    best_exp_name = r["exp_name"]

        print(f"Best experiment: {best_exp_name} (RMSE: {best_rmse:.4f})")

    # Load config
    config = EXPERIMENTS[best_exp_name]

    # Load data
    ds_train = FloodDataset(RAW_DATA_PATH, mode="train")
    norm_stats_dict = {
        "1": compute_normalization_stats(ds_train, "1"),
        "2": compute_normalization_stats(ds_train, "2")
    }

    sample = ds_train[0]
    data = build_2d_graph(sample, norm_stats_dict[sample["model_id"]], t_index=10)
    in_channels = data.x.shape[1]

    # Load Model_1 (improved)
    print("\nLoading Model_1 (improved)...")
    model_1 = SurfaceEngine(
        in_channels=in_channels,
        hidden_channels=128,
        num_sage_layers=3,
        dropout=0.15,
        max_delta=2.0,
        conv_type="sage",
    )
    load_checkpoint(model_1, "checkpoints/model_1_improved.pt")

    # Load Model_2 (best experiment)
    print(f"Loading Model_2 ({best_exp_name})...")
    model_2 = SurfaceEngine(
        in_channels=in_channels,
        hidden_channels=config["hidden_channels"],
        num_sage_layers=config["num_sage_layers"],
        dropout=config["dropout"],
        max_delta=config["max_delta"],
        conv_type=config.get("conv_type", "sage"),
        num_heads=config.get("num_heads", 4),
    )
    load_checkpoint(model_2, f"checkpoints/experiments/model_2_{best_exp_name}.pt")

    models = {"1": model_1, "2": model_2}

    # Generate submission
    print("\nGenerating submission...")
    submission = generate_test_submission_2d(
        model=models,
        data_path=RAW_DATA_PATH,
        norm_stats_dict=norm_stats_dict,
        output_path=f"submissions/submission_2d_{best_exp_name}.csv",
        num_warmup=10,
        verbose=True
    )

    print(f"\nSubmission saved: submissions/submission_2d_{best_exp_name}.csv")
    return submission


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Model_2 experiments")
    parser.add_argument("--exp", type=str, nargs="+", help="Specific experiments to run")
    parser.add_argument("--force", action="store_true", help="Force rerun experiments")
    parser.add_argument("--submit", action="store_true", help="Generate submission with best model")
    parser.add_argument("--quick", action="store_true", help="Run only 2 quick experiments")

    args = parser.parse_args()

    if args.quick:
        # Run just 2 experiments for quick testing
        experiments = ["exp1_lower_lr", "exp3_smaller_model"]
    elif args.exp:
        experiments = args.exp
    else:
        experiments = None  # Run all

    # Run experiments
    results = run_all_experiments(
        experiments_to_run=experiments,
        force_run=args.force
    )

    # Generate submission if requested
    if args.submit:
        generate_best_submission()

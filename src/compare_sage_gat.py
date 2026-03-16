"""
Compare SAGE vs GAT 2D models on the same validation events.

Loads checkpoints/model_1_sage.pt and checkpoints/model_1_gat.pt,
runs predict_event_2d on 5 validation events, computes per-event SRMSE,
prints a side-by-side table and saves results to outputs/sage_vs_gat_comparison.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import RAW_DATA_PATH
from src.dataset import FloodDataset
from src.graph_builder_2d import build_2d_graph, get_values_at_timestep
from src.model_2d import (
    SurfaceEngine,
    evaluate_predictions,
    load_checkpoint,
    predict_event_2d,
)
from src.utils_2d import compute_normalization_stats

NUM_VAL_EVENTS = 5
NUM_WARMUP = 10
SAGE_CHECKPOINT = Path("checkpoints/model_1_sage.pt")
GAT_CHECKPOINT = Path("checkpoints/model_1_gat.pt")
OUTPUT_PATH = Path("outputs/sage_vs_gat_comparison.json")


def _resolve_checkpoint(path: Path, name: str) -> Path:
    """Resolve checkpoint path; raise clear error if missing."""
    if path.exists():
        return path
    raise FileNotFoundError(
        f"{name} checkpoint not found: {path.resolve()}\n"
        f"  Train it first: set TRAIN_GAT=True in src/train_improved.py and run\n"
        f"  ./venv/bin/python -m src.train_improved\n"
        f"  Then run this script again."
    )


def _ground_truth_wl(sample: dict) -> np.ndarray:
    """Build [T, N] ground-truth water levels from sample (no model)."""
    dynamic_2d = sample["dynamic_2d_nodes"]
    static_2d = sample["static_2d_nodes"]
    num_nodes = len(static_2d)
    max_timestep = int(dynamic_2d["timestep"].max())
    num_timesteps = max_timestep + 1
    gt_wl = np.zeros((num_timesteps, num_nodes), dtype=np.float32)
    for t in range(num_timesteps):
        gt_wl[t] = get_values_at_timestep(
            dynamic_2d, t, "water_level", num_nodes
        )
    return gt_wl


def main() -> None:
    print("=" * 70)
    print("SAGE vs GAT — Model_1 comparison (5 validation events)")
    print("=" * 70)

    # Load data and norm stats (same for both models)
    ds_train = FloodDataset(RAW_DATA_PATH, mode="train")
    ds_model = ds_train.filter_by_model("1")
    norm_stats = compute_normalization_stats(ds_train, "1")

    # Global std for standardizing RMSE (from first 5 events' ground truth)
    all_gt = []
    for i in range(min(NUM_VAL_EVENTS, len(ds_model))):
        sample = ds_model[i]
        gt_wl = _ground_truth_wl(sample)
        all_gt.append(gt_wl)
    if not all_gt:
        raise RuntimeError("No validation events found for model 1")
    global_std_2d = float(np.std(np.concatenate([g.flatten() for g in all_gt])))
    if global_std_2d < 1e-8:
        global_std_2d = 1.0
    print(f"\nGlobal std (2D WSE) for SRMSE: {global_std_2d:.4f}")

    # Resolve checkpoint paths (clear error if missing)
    sage_path = _resolve_checkpoint(SAGE_CHECKPOINT, "SAGE")
    gat_path = _resolve_checkpoint(GAT_CHECKPOINT, "GAT")

    # Build one graph to get in_channels
    sample0 = ds_model[0]
    data0 = build_2d_graph(sample0, norm_stats, t_index=10)
    in_channels = data0.x.shape[1]

    # Load SAGE model
    model_sage = SurfaceEngine(
        in_channels=in_channels,
        hidden_channels=128,
        num_sage_layers=3,
        dropout=0.15,
        max_delta=2.0,
        conv_type="sage",
        num_heads=4,
    )
    load_checkpoint(model_sage, str(sage_path))
    n_sage = sum(p.numel() for p in model_sage.parameters())
    print(f"SAGE parameters: {n_sage:,}")

    # Load GAT model
    model_gat = SurfaceEngine(
        in_channels=in_channels,
        hidden_channels=128,
        num_sage_layers=3,
        dropout=0.15,
        max_delta=2.0,
        conv_type="gat",
        num_heads=4,
    )
    load_checkpoint(model_gat, str(gat_path))
    n_gat = sum(p.numel() for p in model_gat.parameters())
    print(f"GAT parameters: {n_gat:,}")

    # Run predictions and compute per-event SRMSE
    events = []
    results_sage = []
    results_gat = []

    for i in range(min(NUM_VAL_EVENTS, len(ds_model))):
        sample = ds_model[i]
        event_id = sample["event_id"]

        pred_sage, gt_wl = predict_event_2d(
            model_sage, sample, norm_stats, num_warmup=NUM_WARMUP
        )
        metrics_sage = evaluate_predictions(
            pred_sage, gt_wl, std_2d=global_std_2d, num_warmup=NUM_WARMUP
        )
        srmse_sage = (
            metrics_sage["standardized_rmse"]
            if metrics_sage.get("standardized_rmse") is not None
            else metrics_sage["rmse"]
        )

        pred_gat, _ = predict_event_2d(
            model_gat, sample, norm_stats, num_warmup=NUM_WARMUP
        )
        metrics_gat = evaluate_predictions(
            pred_gat, gt_wl, std_2d=global_std_2d, num_warmup=NUM_WARMUP
        )
        srmse_gat = (
            metrics_gat["standardized_rmse"]
            if metrics_gat.get("standardized_rmse") is not None
            else metrics_gat["rmse"]
        )

        events.append(event_id)
        results_sage.append(float(srmse_sage))
        results_gat.append(float(srmse_gat))

    # Side-by-side table
    print("\n" + "=" * 70)
    print("Per-event SRMSE (lower is better)")
    print("=" * 70)
    print(f"{'Event':<10} {'SAGE SRMSE':>12} {'GAT SRMSE':>12} {'Winner':<8}")
    print("-" * 46)

    winners = []
    for ev, rs, rg in zip(events, results_sage, results_gat):
        winner = "SAGE" if rs <= rg else "GAT"
        winners.append(winner)
        print(f"{ev:<10} {rs:>12.6f} {rg:>12.6f} {winner:<8}")

    print("-" * 46)
    mean_sage = np.mean(results_sage)
    mean_gat = np.mean(results_gat)
    overall_winner = "SAGE" if mean_sage <= mean_gat else "GAT"
    print(f"{'Mean':<10} {mean_sage:>12.6f} {mean_gat:>12.6f} {overall_winner:<8}")

    print("\nParameter counts:")
    print(f"  SAGE: {n_sage:,}")
    print(f"  GAT:  {n_gat:,}")

    # Save JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "model_id": "1",
        "num_events": len(events),
        "num_warmup": NUM_WARMUP,
        "global_std_2d": global_std_2d,
        "parameter_count_sage": n_sage,
        "parameter_count_gat": n_gat,
        "per_event": [
            {
                "event_id": int(ev),
                "srmse_sage": rs,
                "srmse_gat": rg,
                "winner": w,
            }
            for ev, rs, rg, w in zip(events, results_sage, results_gat, winners)
        ],
        "mean_srmse_sage": float(mean_sage),
        "mean_srmse_gat": float(mean_gat),
        "overall_winner": overall_winner,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

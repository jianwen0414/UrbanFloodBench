#!/usr/bin/env python3
"""Generate 2D submission CSV.

Three modes controlled by MODE at the top:

  "sage"      →  SAGE ensemble (0.3089 leaderboard)  →  submission_2d_ensemble.csv
  "gat"       →  GAT only                            →  submission_2d_gat.csv
  "blend"     →  SAGE+GAT weighted blend             →  submission_2d_sage_gat_ensemble.csv
"""
import csv

from src.config import RAW_DATA_PATH
from src.dataset import FloodDataset
from src.graph_builder_2d import build_2d_graph
from src.model_2d import SurfaceEngine, load_checkpoint, predict_event_2d
from src.utils_2d import compute_normalization_stats

# ── Mode: "sage", "gat", or "blend" ──────────────────────────────────
MODE = "sage"

# Blend weights (GAT validated ~16% lower SRMSE on Model_1)
GAT_WEIGHT = 0.6
SAGE_WEIGHT = 0.4

print(f"Mode: {MODE}")
if MODE == "blend":
    print(f"Blend weights: GAT={GAT_WEIGHT}, SAGE={SAGE_WEIGHT}")

print("Loading data...")
ds_train = FloodDataset(RAW_DATA_PATH, mode="train")
ds_test = FloodDataset(RAW_DATA_PATH, mode="test")

norm_stats_dict = {
    "1": compute_normalization_stats(ds_train, "1"),
    "2": compute_normalization_stats(ds_train, "2"),
}

sample = ds_train[0]
data = build_2d_graph(sample, norm_stats_dict[sample["model_id"]], t_index=10)
in_channels = data.x.shape[1]

# ── Load models based on mode ────────────────────────────────────────

if MODE in ("sage", "blend"):
    print("Loading Model_1 (SAGE improved)...")
    model_1_sage = SurfaceEngine(
        in_channels=in_channels, hidden_channels=128, num_sage_layers=3,
        dropout=0.15, max_delta=2.0, conv_type="sage",
    )
    load_checkpoint(model_1_sage, "checkpoints/model_1_improved.pt")

    print("Loading Model_2 version A (exp1_lower_lr)...")
    model_2a_sage = SurfaceEngine(
        in_channels=in_channels, hidden_channels=64, num_sage_layers=2,
        dropout=0.1, max_delta=2.0, conv_type="sage",
    )
    load_checkpoint(model_2a_sage, "checkpoints/experiments/model_2_exp1_lower_lr.pt")

    print("Loading Model_2 version B (SAGE improved)...")
    model_2b_sage = SurfaceEngine(
        in_channels=in_channels, hidden_channels=128, num_sage_layers=3,
        dropout=0.15, max_delta=2.0, conv_type="sage",
    )
    load_checkpoint(model_2b_sage, "checkpoints/model_2_improved.pt")

if MODE in ("gat", "blend"):
    print("Loading Model_1 (GAT)...")
    model_1_gat = SurfaceEngine(
        in_channels=in_channels, hidden_channels=128, num_sage_layers=3,
        dropout=0.15, max_delta=2.0, conv_type="gat", num_heads=4,
    )
    load_checkpoint(model_1_gat, "checkpoints/model_1_gat.pt")

    print("Loading Model_2 (GAT)...")
    model_2_gat = SurfaceEngine(
        in_channels=in_channels, hidden_channels=128, num_sage_layers=3,
        dropout=0.15, max_delta=2.0, conv_type="gat", num_heads=4,
    )
    load_checkpoint(model_2_gat, "checkpoints/model_2_gat.pt")

# ── Output path ──────────────────────────────────────────────────────

output_paths = {
    "sage":  "submissions/submission_2d_ensemble.csv",
    "gat":   "submissions/submission_2d_gat.csv",
    "blend": "submissions/submission_2d_sage_gat_ensemble.csv",
}
output_path = output_paths[MODE]

# ── Prediction helpers ───────────────────────────────────────────────

def _predict_sage(model_id, sample, norm_stats, num_warmup):
    if model_id == "1":
        return predict_event_2d(model_1_sage, sample, norm_stats, num_warmup=num_warmup)[0]
    pred_a = predict_event_2d(model_2a_sage, sample, norm_stats, num_warmup=num_warmup)[0]
    pred_b = predict_event_2d(model_2b_sage, sample, norm_stats, num_warmup=num_warmup)[0]
    return (pred_a + pred_b) / 2.0


def _predict_gat(model_id, sample, norm_stats, num_warmup):
    model = model_1_gat if model_id == "1" else model_2_gat
    return predict_event_2d(model, sample, norm_stats, num_warmup=num_warmup)[0]


# ── Generate submission ──────────────────────────────────────────────

print(f"Generating {MODE} submission (streaming)...")
num_warmup = 10

with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["row_id", "model_id", "event_id", "node_type", "node_id", "water_level"])

    row_id = 0

    for model_id in ["1", "2"]:
        ds_model = ds_test.filter_by_model(model_id)
        norm_stats = norm_stats_dict[model_id]

        print(f"Model_{model_id}: {len(ds_model)} events")

        for event_idx in range(len(ds_model)):
            sample = ds_model[event_idx]
            event_id = sample["event_id"]
            num_nodes = len(sample["static_2d_nodes"])

            if MODE == "sage":
                pred_wl = _predict_sage(model_id, sample, norm_stats, num_warmup)
            elif MODE == "gat":
                pred_wl = _predict_gat(model_id, sample, norm_stats, num_warmup)
            else:
                pred_sage = _predict_sage(model_id, sample, norm_stats, num_warmup)
                pred_gat = _predict_gat(model_id, sample, norm_stats, num_warmup)
                pred_wl = SAGE_WEIGHT * pred_sage + GAT_WEIGHT * pred_gat

            num_timesteps = pred_wl.shape[0]

            print(f"  Event_{event_id}: {num_nodes} nodes x {num_timesteps - num_warmup} timesteps")

            for node_id in range(num_nodes):
                for t in range(num_warmup, num_timesteps):
                    writer.writerow([
                        row_id,
                        int(model_id),
                        int(event_id),
                        int(2),
                        node_id,
                        float(pred_wl[t, node_id].item()),
                    ])
                    row_id += 1

print(f"Saved: {output_path}")
print(f"Total rows: {row_id:,}")

#!/usr/bin/env python3
"""
Generate 1D submission, combine with 2D, and verify.
Run in env with PyTorch: python run_1d_2d_submission.py
"""
import csv

from src.config import RAW_DATA_PATH
from src.dataset import FloodDataset
from src.model_1d import (
    DrainageNetwork1D,
    load_checkpoint,
    predict_event_1d,
)
from src.graph_builder_1d import get_1d_input_dim
from src.utils_1d import compute_normalization_stats_1d
from src.submission_1d import (
    combine_1d_2d_submissions,
    generate_test_submission_1d,
)

# Load data
ds_train = FloodDataset(RAW_DATA_PATH, mode="train")

# Load models
models = {}
norm_stats_dict = {}

for model_id in ["1", "2"]:
    norm_stats = compute_normalization_stats_1d(ds_train, model_id)
    norm_stats_dict[model_id] = norm_stats

    ds_model = ds_train.filter_by_model(model_id)
    in_channels = get_1d_input_dim(ds_model[0], norm_stats)

    model = DrainageNetwork1D(
        in_channels=in_channels,
        hidden_channels=64,
        num_sage_layers=2,
        dropout=0.1,
        max_delta=2.0,
    )
    load_checkpoint(model, f"checkpoints/model_{model_id}_1d_best.pt")
    models[model_id] = model

# Step 1: Generate 1D submission
print()
total_1d = generate_test_submission_1d(
    model=models,
    data_path=RAW_DATA_PATH,
    norm_stats_dict=norm_stats_dict,
    output_path="submissions/submission_1d.csv",
    num_warmup=10,
    verbose=True,
)
print(f"\n1D rows written: {total_1d:,}")

if total_1d == 0:
    print("ERROR: Still 0 rows! Something wrong with predict_event_1d")
    raise SystemExit(1)

# Step 2: Combine with 2D
print()
combine_1d_2d_submissions(
    submission_1d_path="submissions/submission_1d.csv",
    submission_2d_path="submissions/submission_2d_ensemble.csv",
    output_path="submissions/submission_full_with_1d.csv",
    verbose=True,
)

# Step 3: Verify node types
print("\n=== VERIFICATION ===")
node_types = set()
count_1d = 0
count_2d = 0
with open("submissions/submission_full_with_1d.csv") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        nt = int(row[3])
        node_types.add(nt)
        if nt == 1:
            count_1d += 1
        elif nt == 2:
            count_2d += 1

print(f"Node types: {sorted(node_types)}")
print(f"1D rows: {count_1d:,}")
print(f"2D rows: {count_2d:,}")
print(f"Total:   {count_1d + count_2d:,}")

if 1 in node_types and 2 in node_types:
    print("\n✅ READY TO SUBMIT: submissions/submission_full_with_1d.csv")
else:
    print("\n❌ MISSING NODE TYPES!")

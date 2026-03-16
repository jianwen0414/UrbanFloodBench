#!/usr/bin/env python3
"""
Produce a Kaggle-ready submission by streaming through the sample_submission
template and looking up water_level from the 1D and 2D prediction files.

Row order is dictated entirely by sample_submission (per-event interleaving
of 1D and 2D), so the output always matches the expected format.

Inputs:
  submissions/submission_1d.csv          (node_type=1)
  submissions/submission_2d_gat.csv      (node_type=2)
  submissions/sample_submission*.csv     (Kaggle template — defines row order)

Output:
  submissions/submission_final_gat.csv
"""
import csv
import glob
import sys
from pathlib import Path

SUBMISSION_1D = "submissions/submission_1d.csv"
SUBMISSION_2D = "submissions/submission_2d_improved.csv"
FINAL_PATH = "submissions/submission_final_improved.csv"

sample_matches = sorted(glob.glob("submissions/sample_submission*.csv")) + \
                 sorted(glob.glob("sample_submission*.csv"))
if not sample_matches:
    sys.exit("No sample_submission*.csv found. Download from Kaggle.")
SAMPLE_PATH = sample_matches[0]

for p in [SUBMISSION_1D, SUBMISSION_2D, SAMPLE_PATH]:
    if not Path(p).exists():
        sys.exit(f"Missing: {p}")

# ── Step 1: Build lookup from 1D predictions ─────────────────────────
# Key: (model_id, event_id, node_type, node_id, timestep_index)
# The sample_submission has one row per (model, event, node_type, node_id)
# per timestep, but node_id rows are repeated for each timestep.
# We need a key that accounts for duplicate (model, event, type, node)
# entries across timesteps.
#
# The sample_submission has a unique row_id per row — so the safest
# approach is to key on (model_id, event_id, node_type, node_id)
# with timestep order preserved.  But since there are multiple timesteps
# per node, we need to handle duplicates.
#
# Strategy: key = (model_id, event_id, node_type, node_id) → list of
# water_level values in the order they appear in the prediction file.
# Then pop them in order as we encounter matching rows in sample.

print("Loading 1D predictions...")
pred_1d: dict[tuple, list[float]] = {}
count_1d = 0
with open(SUBMISSION_1D) as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (int(row["model_id"]), int(row["event_id"]),
               int(row["node_type"]), int(row["node_id"]))
        pred_1d.setdefault(key, []).append(float(row["water_level"]))
        count_1d += 1
print(f"  {count_1d:,} rows, {len(pred_1d):,} unique (model,event,type,node) keys")

print("Loading 2D predictions...")
pred_2d: dict[tuple, list[float]] = {}
count_2d = 0
with open(SUBMISSION_2D) as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (int(row["model_id"]), int(row["event_id"]),
               int(row["node_type"]), int(row["node_id"]))
        pred_2d.setdefault(key, []).append(float(row["water_level"]))
        count_2d += 1
print(f"  {count_2d:,} rows, {len(pred_2d):,} unique keys")

# Merge into one dict; convert lists to iterators for ordered consumption
print("Merging into lookup...")
pred_all: dict[tuple, list[float]] = {}
for k, v in pred_1d.items():
    pred_all[k] = v
for k, v in pred_2d.items():
    pred_all[k] = v
# Track consumption position per key
pred_pos: dict[tuple, int] = {k: 0 for k in pred_all}

del pred_1d, pred_2d

# ── Step 2: Stream through sample, write final submission ────────────

print(f"\nStreaming through sample_submission → {FINAL_PATH}")
Path(FINAL_PATH).parent.mkdir(parents=True, exist_ok=True)

total_rows = 0
missing = 0

with open(SAMPLE_PATH) as fin, open(FINAL_PATH, "w", newline="") as fout:
    reader = csv.DictReader(fin)
    writer = csv.writer(fout)
    writer.writerow(["row_id", "model_id", "event_id", "node_type", "node_id", "water_level"])

    for row in reader:
        rid = row["row_id"]
        key = (int(row["model_id"]), int(row["event_id"]),
               int(row["node_type"]), int(row["node_id"]))

        vals = pred_all.get(key)
        if vals is not None:
            pos = pred_pos[key]
            if pos < len(vals):
                wl = vals[pos]
                pred_pos[key] = pos + 1
            else:
                wl = 0.0
                missing += 1
        else:
            wl = 0.0
            missing += 1

        writer.writerow([rid, key[0], key[1], key[2], key[3], wl])
        total_rows += 1

        if total_rows % 10_000_000 == 0:
            print(f"  {total_rows:,} rows written...")

if missing > 0:
    print(f"\n  Filled {missing:,} missing values with 0.0")
else:
    print("\n  No missing values — all rows matched.")

print(f"\nSaved: {FINAL_PATH} ({total_rows:,} rows)")
print("Ready to submit on Kaggle.")

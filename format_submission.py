#!/usr/bin/env python3
"""
Format competition submission to strictly match the evaluation requirements.

Uses the official sample_submission.csv as the template for row_id mapping
and row order. Merges water_level predictions from our submission file.

Usage:
    python format_submission.py

Requires:
    - submissions/submission_full_with_1d.csv (our predictions)
    - sample_submission.csv (official template from Kaggle; place in project root
      or set SAMPLE_PATH below)
Output:
    - submissions/submission_final.csv
"""

from pathlib import Path

import pandas as pd

# Paths (adjust if your files are elsewhere)
PRED_PATH = Path("submissions/submission_full_with_1d.csv")
SAMPLE_PATH = Path("sample_submission.csv")  # Download from Kaggle if missing
OUTPUT_PATH = Path("submissions/submission_final.csv")

REQUIRED_COLUMNS = ["row_id", "model_id", "event_id", "node_type", "node_id", "water_level"]
MERGE_KEYS = ["model_id", "event_id", "node_type", "node_id"]


def main():
    print("Reading files...")
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Prediction file not found: {PRED_PATH}")
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Official template not found: {SAMPLE_PATH}. "
            "Download sample_submission.csv from Kaggle and place it in the project root."
        )

    pred = pd.read_csv(PRED_PATH)
    sample = pd.read_csv(SAMPLE_PATH)

    print(f"  Predictions: {len(pred):,} rows")
    print(f"  Sample:      {len(sample):,} rows")

    # Rename columns if present (our file may already have model_id / event_id)
    rename = {}
    if "model" in pred.columns and "model_id" not in pred.columns:
        rename["model"] = "model_id"
    if "event" in pred.columns and "event_id" not in pred.columns:
        rename["event"] = "event_id"
    if rename:
        pred = pred.rename(columns=rename)
        print(f"  Renamed: {list(rename.keys())} -> {list(rename.values())}")

    # Drop row_id from predictions (we use sample's row_id and order)
    if "row_id" in pred.columns:
        pred = pred.drop(columns=["row_id"])
    assert set(MERGE_KEYS + ["water_level"]).issubset(pred.columns), (
        f"Predictions must have columns {MERGE_KEYS + ['water_level']}"
    )

    # Left join: keep sample row order and row_id; fill water_level from predictions
    sample_cols = [c for c in sample.columns if c != "water_level"]
    merged = sample[sample_cols].merge(
        pred[MERGE_KEYS + ["water_level"]],
        on=MERGE_KEYS,
        how="left",
    )

    # Strict column order
    final = merged[REQUIRED_COLUMNS]

    # Fill any missing water_level so submission never has NaN (required by evaluation)
    nan_count = int(final["water_level"].isna().sum())
    if nan_count > 0:
        print(f"\n  Filling {nan_count:,} missing water_level rows with 0.0 (placeholder).")
        final["water_level"] = final["water_level"].fillna(0.0)
    else:
        print(f"\n  No missing water_level values.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH} ({len(final):,} rows)")


if __name__ == "__main__":
    main()

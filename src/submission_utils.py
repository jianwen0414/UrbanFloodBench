"""
submission_utils — Combine 1D and 2D predictions into a single submission.

This module provides helpers for assembling a complete Kaggle submission
by merging:

* **2D predictions** — from ``submission_2d.py`` (node_type = 2)
* **1D predictions** — from the 1D team or a dummy fill (node_type = 1)

Owner: Member B
See: IMPLEMENTATION_PLAN.md → Task 3
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ───────────────────────────────────────────────────────────────────────
#  Dummy 1D predictions (format testing only)
# ───────────────────────────────────────────────────────────────────────

def add_dummy_1d_predictions(
    submission_2d_path: str,
    data_path: str,
    output_path: str = "submissions/submission_full_dummy.csv",
    num_warmup: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Append dummy 1D rows to an existing 2D submission.

    .. warning::

       The 1D water-level values are filled with ``surface_elevation``
       (i.e. the node's ground surface height).  This is **not** a real
       prediction — use it only to validate the submission format.

    Parameters
    ----------
    submission_2d_path : str
        Path to the 2D-only CSV produced by
        ``submission_2d.generate_test_submission_2d``.
    data_path : str
        Path to the data folder (must contain ``Model_X/test/``).
    output_path : str
        Where to save the combined CSV.
    num_warmup : int
        Warm-up timesteps to skip (default ``10``).
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Combined 1D + 2D submission.
    """
    from src.dataset import FloodDataset

    print("=" * 60)
    print("Adding DUMMY 1D Predictions (FORMAT TESTING ONLY)")
    print("=" * 60)
    print("  WARNING: 1D values are DUMMY (surface elevation)!")
    print("  Replace with actual 1D predictions for real submission!\n")

    # ── load 2D submission ────────────────────────────────────────
    submission_2d = pd.read_csv(submission_2d_path)
    print(f"Loaded 2D submission: {len(submission_2d):,} rows")

    # ── build 1D rows ─────────────────────────────────────────────
    ds_test = FloodDataset(data_path, mode="test")

    all_1d_rows: list[dict] = []

    for model_id in ["1", "2"]:
        ds_model = ds_test.filter_by_model(model_id)

        if verbose:
            print(f"\nModel_{model_id}: {len(ds_model)} test events")

        for event_idx in range(len(ds_model)):
            sample = ds_model[event_idx]
            event_id = sample["event_id"]

            static_1d = sample["static_1d_nodes"]
            num_1d_nodes = len(static_1d)

            dynamic_1d = sample["dynamic_1d_nodes"]
            max_timestep = int(dynamic_1d["timestep"].max())
            num_predict = max_timestep - num_warmup + 1

            if verbose:
                print(
                    f"  Event_{event_id}: {num_1d_nodes} 1D nodes "
                    f"x {num_predict} timesteps"
                )

            for node_id in range(num_1d_nodes):
                dummy_wl = float(
                    static_1d.iloc[node_id]["surface_elevation"]
                )
                for t in range(num_warmup, max_timestep + 1):
                    all_1d_rows.append(
                        {
                            "model_id": int(model_id),
                            "event_id": int(event_id),
                            "node_type": 1,
                            "node_id": node_id,
                            "water_level": dummy_wl,
                            "_sort_key": (
                                int(model_id),
                                int(event_id),
                                1,
                                node_id,
                                t,
                            ),
                        }
                    )

    df_1d = (
        pd.DataFrame(all_1d_rows)
        .sort_values("_sort_key")
        .drop(columns=["_sort_key"])
        .reset_index(drop=True)
    )

    print(f"\nGenerated {len(df_1d):,} dummy 1D rows")

    # ── combine ───────────────────────────────────────────────────
    submission_2d_clean = submission_2d.drop(columns=["row_id"])

    combined = (
        pd.concat([df_1d, submission_2d_clean], ignore_index=True)
        .sort_values(["model_id", "event_id", "node_type", "node_id"])
        .reset_index(drop=True)
    )

    combined.insert(0, "row_id", range(len(combined)))

    combined = combined[
        [
            "row_id",
            "model_id",
            "event_id",
            "node_type",
            "node_id",
            "water_level",
        ]
    ]

    print(f"\nCombined submission:")
    print(f"  1D rows (dummy): {len(df_1d):,}")
    print(f"  2D rows (model): {len(submission_2d):,}")
    print(f"  Total:           {len(combined):,}")

    # ── save ──────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\nSaved: {output_path}")
    print(f"Size:  {size_mb:.2f} MB")

    return combined


# ───────────────────────────────────────────────────────────────────────
#  Real 1D + 2D combination
# ───────────────────────────────────────────────────────────────────────

def combine_1d_2d_submissions(
    submission_1d_path: str,
    submission_2d_path: str,
    output_path: str = "submissions/submission_final.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """Merge real 1D and 2D submission CSVs into the final submission.

    Parameters
    ----------
    submission_1d_path : str
        CSV with 1D predictions (``node_type == 1``).
    submission_2d_path : str
        CSV with 2D predictions (``node_type == 2``).
    output_path : str
        Destination for the combined CSV.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Final submission.
    """
    print("=" * 60)
    print("Combining 1D and 2D Submissions")
    print("=" * 60)

    df_1d = pd.read_csv(submission_1d_path)
    df_2d = pd.read_csv(submission_2d_path)

    # Coerce node_type to integer (1 or 2 only) in case CSV had floats
    df_1d["node_type"] = df_1d["node_type"].astype(int)
    df_2d["node_type"] = df_2d["node_type"].astype(int)

    print(f"1D submission: {len(df_1d):,} rows")
    print(f"2D submission: {len(df_2d):,} rows")

    # Drop old row_id (will regenerate)
    if "row_id" in df_1d.columns:
        df_1d = df_1d.drop(columns=["row_id"])
    if "row_id" in df_2d.columns:
        df_2d = df_2d.drop(columns=["row_id"])

    assert (
        df_1d["node_type"] == 1
    ).all(), "1D submission should have node_type=1"
    assert (
        df_2d["node_type"] == 2
    ).all(), "2D submission should have node_type=2"

    combined = (
        pd.concat([df_1d, df_2d], ignore_index=True)
        .sort_values(["model_id", "event_id", "node_type", "node_id"])
        .reset_index(drop=True)
    )

    combined.insert(0, "row_id", range(len(combined)))

    # Ensure node_type is integer 1 or 2 only (Kaggle format)
    combined["node_type"] = combined["node_type"].astype(int)
    if set(combined["node_type"].unique()) != {1, 2}:
        raise ValueError(
            f"node_type must be only 1 and 2, got {sorted(combined['node_type'].unique())}"
        )

    combined = combined[
        [
            "row_id",
            "model_id",
            "event_id",
            "node_type",
            "node_id",
            "water_level",
        ]
    ]

    print(f"\nFinal submission: {len(combined):,} rows")

    # Quick data-quality check
    nan_count = int(combined["water_level"].isna().sum())
    inf_count = int(np.isinf(combined["water_level"]).sum())

    if nan_count > 0 or inf_count > 0:
        print(f"  Data issues: {nan_count} NaN, {inf_count} Inf")
    else:
        print("  No NaN or Inf values")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\nSaved: {output_path}")
    print(f"Size:  {size_mb:.2f} MB")

    return combined


# ───────────────────────────────────────────────────────────────────────
#  Quick smoke test
# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.config import RAW_DATA_PATH

    submission_2d_path = Path("submissions/submission_2d.csv")

    if not submission_2d_path.exists():
        print("  No 2D submission found!")
        print(
            "  Run `python -m src.submission_2d` first to generate "
            "2D predictions."
        )
    else:
        print(f"Found 2D submission: {submission_2d_path}")

        combined = add_dummy_1d_predictions(
            submission_2d_path=str(submission_2d_path),
            data_path=RAW_DATA_PATH,
            output_path="submissions/submission_full_dummy.csv",
            num_warmup=10,
            verbose=True,
        )

        print("\n" + "=" * 60)
        print("  REMINDER: This is for FORMAT TESTING ONLY!")
        print("  1D predictions are DUMMY values (surface elevation).")
        print(
            "  For real submission, get actual 1D predictions "
            "from your team."
        )
        print("=" * 60)

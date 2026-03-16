"""
Generate 1D predictions for Kaggle submission.
"""

import csv
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from src.dataset import FloodDataset
from src.model_1d import DrainageNetwork1D, predict_event_1d, load_checkpoint
from src.utils_1d import compute_normalization_stats_1d, pivot_dynamic_1d


def generate_test_submission_1d(
    model: Dict[str, DrainageNetwork1D],
    data_path: str,
    norm_stats_dict: Dict[str, Dict],
    output_path: str = "submissions/submission_1d.csv",
    num_warmup: int = 10,
    verbose: bool = True,
    template_path: Optional[str] = None,
    fill_value: float = 0.0,
) -> int:
    """
    Generate 1D predictions for test set.

    Writes only scored timesteps (num_warmup..T-1) per (model_id, event_id, node_id),
    matching the evaluation row count (same convention as 2D: no warmup rows).
    Order: (node_id, timestep) per event.
    If template_path is set, output order and row set follow the template;
    missing predictions are filled with fill_value (e.g. invert or 0).

    Args:
        model: Dict mapping model_id ("1", "2") to trained DrainageNetwork1D
        data_path: Path to data directory
        norm_stats_dict: Dict mapping model_id to normalization stats
        output_path: Output CSV path
        num_warmup: Number of warmup timesteps
        verbose: Print progress
        template_path: If set, use this CSV (e.g. sample_submission.csv) to drive
            row order and required rows; fill missing with fill_value.
        fill_value: Value for (model_id, event_id, node_id, timestep) not in predictions.

    Returns:
        Total number of rows written
    """
    ds_test = FloodDataset(data_path, mode="test")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("Generating TEST Submission (1D Nodes Only)")
        print("=" * 60)
        print(f"Warmup timesteps: {num_warmup}")
        if template_path:
            print(f"Template (row order): {template_path}")

    # When using template, we fill this grid then write in template order
    preds_grid: Dict[tuple, float] = {}
    row_id = 0

    def write_1d_event_rows(
        writer, model_id, event_id, num_1d_nodes, num_total, pred_wl, num_warmup
    ):
        """Write scored timesteps only (num_warmup..num_total-1) in (node_id, timestep) order.
        Matches evaluation row count: 1D uses same convention as 2D (no warmup rows)."""
        nonlocal row_id
        count = 0
        t_start = num_warmup
        for node_id in range(num_1d_nodes):
            for t in range(t_start, num_total):
                wl = float(pred_wl[t, node_id].item())
                # For template merge: timestep index 0 = first scored step
                key = (int(model_id), int(event_id), node_id, t - t_start)
                preds_grid[key] = wl
                if not template_path:
                    writer.writerow(
                        [
                            row_id,
                            int(model_id),
                            int(event_id),
                            int(1),
                            node_id,
                            wl,
                        ]
                    )
                    row_id += 1
                    count += 1
        return count

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["row_id", "model_id", "event_id", "node_type", "node_id", "water_level"]
        )

        for model_id in ["1", "2"]:
            ds_model = ds_test.filter_by_model(model_id)

            if model_id not in model:
                if verbose:
                    print(f"\nModel_{model_id}: No model provided, skipping")
                continue

            m = model[model_id]
            norm_stats = norm_stats_dict[model_id]

            if verbose:
                print(f"\nModel_{model_id}: {len(ds_model)} test events")

            for event_idx in range(len(ds_model)):
                sample = ds_model[event_idx]
                event_id = sample["event_id"]

                static_1d = sample.get("static_1d_nodes")
                dynamic_1d = sample.get("dynamic_1d_nodes")

                if static_1d is None or len(static_1d) == 0:
                    if verbose:
                        print(f"  Event_{event_id}: No 1D static data, skipping")
                    continue

                if dynamic_1d is None or len(dynamic_1d) == 0:
                    if verbose:
                        print(f"  Event_{event_id}: No 1D dynamic data, skipping")
                    continue

                num_1d_nodes = len(static_1d)

                pred_wl, _ = predict_event_1d(
                    m, sample, norm_stats, num_warmup=num_warmup
                )

                num_total = pred_wl.shape[0]
                if num_total <= 0:
                    if verbose:
                        print(
                            f"  Event_{event_id}: pred shape {pred_wl.shape}, skipping"
                        )
                    continue

                num_scored = max(0, num_total - num_warmup)
                if verbose:
                    print(
                        f"  Event_{event_id}: {num_1d_nodes} nodes x {num_scored} scored timesteps = "
                        f"{num_1d_nodes * num_scored:,} rows"
                    )

                write_1d_event_rows(
                    writer,
                    model_id,
                    event_id,
                    num_1d_nodes,
                    num_total,
                    pred_wl,
                    num_warmup,
                )

    if template_path:
        template_file = Path(template_path)
        if not template_file.exists():
            raise FileNotFoundError(
                f"Template not found: {template_path}. "
                "Generate 1D rows without template or provide a valid path."
            )
        template_df = pd.read_csv(template_file)
        template_1d = template_df[template_df["node_type"] == 1].copy()
        # Keep template row order; assign timestep index 0,1,2,... per (model_id, event_id, node_id)
        template_1d["timestep_idx"] = template_1d.groupby(
            ["model_id", "event_id", "node_id"]
        ).cumcount()
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "row_id",
                    "model_id",
                    "event_id",
                    "node_type",
                    "node_id",
                    "water_level",
                ]
            )
            for row_id_out, row in enumerate(
                template_1d.itertuples(index=False)
            ):
                key = (
                    int(row.model_id),
                    int(row.event_id),
                    int(row.node_id),
                    int(row.timestep_idx),
                )
                wl = preds_grid.get(key, fill_value)
                writer.writerow(
                    [
                        row_id_out,
                        row.model_id,
                        row.event_id,
                        1,
                        row.node_id,
                        float(wl),
                    ]
                )
        total_rows = len(template_1d)
        if verbose:
            filled = sum(
                1
                for _, r in template_1d.iterrows()
                if (int(r["model_id"]), int(r["event_id"]), int(r["node_id"]), int(r["timestep_idx"]))
                not in preds_grid
            )
            if filled > 0:
                print(f"  Filled {filled:,} template rows with fill_value={fill_value}")
    else:
        total_rows = row_id

    if verbose:
        file_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"\n1D Submission Summary:")
        print(f"  Total rows: {total_rows:,}")
        print(f"  File size: {file_size:.2f} MB")
        print(f"  Saved: {output_path}")

    return total_rows


def combine_1d_2d_submissions(
    submission_1d_path: str,
    submission_2d_path: str,
    output_path: str,
    verbose: bool = True,
) -> int:
    """
    Combine 1D and 2D submissions into final submission (streaming).

    Writes 2D rows FIRST, then 1D rows — same order as Kaggle solution file.
    Writes row by row to avoid loading everything into memory.
    Requires both 1D and 2D rows; never overwrites with 2D-only or 1D-only.
    """
    path_out = Path(output_path)
    path_out.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 60)
        print("Combining 2D + 1D Submissions")
        print("=" * 60)

    # Write to a temp file first; only replace target if we have both node types
    import tempfile
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".csv", prefix="submission_combined_", dir=path_out.parent
    )

    row_id = 0
    count_1d = 0
    count_2d = 0

    try:
        with open(tmp_fd, "w", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(
                [
                    "row_id",
                    "model_id",
                    "event_id",
                    "node_type",
                    "node_id",
                    "water_level",
                ]
            )

            # Write 2D rows FIRST (must match Kaggle solution file order)
            with open(submission_2d_path, "r") as f2d:
                reader = csv.reader(f2d)
                next(reader)  # Skip header
                for row in reader:
                    writer.writerow(
                        [row_id, row[1], row[2], row[3], row[4], row[5]]
                    )
                    row_id += 1
                    count_2d += 1

            if verbose:
                print(f"  2D rows: {count_2d:,} (written first)")

            if count_2d == 0:
                raise ValueError(
                    "2D submission has 0 rows. Generate 2D predictions first "
                    "(e.g. run_ensemble_submission.py)."
                )

            # Write 1D rows SECOND
            with open(submission_1d_path, "r") as f1d:
                reader = csv.reader(f1d)
                next(reader)  # Skip header
                for row in reader:
                    writer.writerow(
                        [row_id, row[1], row[2], row[3], row[4], row[5]]
                    )
                    row_id += 1
                    count_1d += 1

            if verbose:
                print(f"  1D rows: {count_1d:,} (written second)")

            if count_1d == 0:
                raise ValueError(
                    "1D submission has 0 rows. Generate 1D predictions first "
                    "(run_1d_2d_submission.py or train_1d pipeline)."
                )

        total = count_2d + count_1d
        if count_1d == 0 or count_2d == 0:
            Path(tmp_path).unlink(missing_ok=True)
            raise ValueError(
                "Combined file must have both 1D and 2D rows. "
                f"Got 1D={count_1d}, 2D={count_2d}. Will not overwrite."
            )
        Path(tmp_path).replace(path_out)

    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    if verbose:
        file_size = path_out.stat().st_size / (1024 * 1024)
        print(f"\nCombined submission:")
        print(f"  Total rows: {total:,} (2D: {count_2d:,}, 1D: {count_1d:,})")
        print(f"  File size: {file_size:.1f} MB")
        print(f"  Saved: {path_out.absolute()}")
        print(f"\n  >>> UPLOAD THIS FILE TO KAGGLE: {path_out.name} <<<")

    return total


if __name__ == "__main__":
    from src.config import RAW_DATA_PATH

    print("=" * 60)
    print("1D SUBMISSION TEST")
    print("=" * 60)

    ds_test = FloodDataset(RAW_DATA_PATH, mode="test")

    for model_id in ["1", "2"]:
        ds_model = ds_test.filter_by_model(model_id)
        print(f"\nModel_{model_id}: {len(ds_model)} test events")

        for i in range(min(2, len(ds_model))):
            sample = ds_model[i]
            static_1d = sample.get("static_1d_nodes")
            dynamic_1d = sample.get("dynamic_1d_nodes")

            if static_1d is not None:
                num_nodes = len(static_1d)
                num_timesteps = (
                    dynamic_1d["timestep"].nunique() if dynamic_1d is not None else 0
                )
                print(
                    f"  Event_{sample['event_id']}: {num_nodes} 1D nodes, "
                    f"{num_timesteps} timesteps"
                )

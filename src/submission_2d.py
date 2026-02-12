"""
submission_2d — 2D Submission utilities for the Urban Flood competition.

Generates 2D predictions in Kaggle competition format.

Format
------
``row_id, model_id, event_id, node_type, node_id, water_level``

* No ``timestep`` column — rows are sorted by timestep ascending.
* ``node_type`` is always ``2`` (2D surface nodes).
* Only timesteps **after** warmup are included.

Owner: Member B
See: IMPLEMENTATION_PLAN.md → Task 3
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import torch


def generate_test_submission_2d(
    model: Union[Any, Dict[str, Any]],
    data_path: str,
    norm_stats_dict: Dict[str, dict],
    output_path: str = "submissions/submission_2d.csv",
    num_warmup: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate 2D predictions for **test** events in competition format.

    Parameters
    ----------
    model : SurfaceEngine or dict[str, SurfaceEngine]
        Trained model.  Pass a dict keyed by ``model_id`` if you have
        separate models per urban catchment.
    data_path : str
        Path to the data folder (must contain ``Model_X/test/``).
    norm_stats_dict : dict[str, dict]
        ``{"1": norm_stats_m1, "2": norm_stats_m2}``.
    output_path : str
        Destination CSV path.
    num_warmup : int
        Ground-truth warm-up timesteps (default ``10``).
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Competition-ready 2D submission.
    """
    from src.dataset import FloodDataset
    from src.model_2d import predict_event_2d

    ds_test = FloodDataset(data_path, mode="test")

    if verbose:
        print("=" * 60)
        print("Generating TEST Submission (2D Nodes Only)")
        print("=" * 60)
        print(f"Warmup timesteps: {num_warmup} (given, not predicted)")

    all_rows: list[dict] = []

    for model_id in ["1", "2"]:
        # ── resolve model ─────────────────────────────────────────
        if isinstance(model, dict):
            current_model = model.get(model_id)
            if current_model is None:
                if verbose:
                    print(
                        f"\n  No model for Model_{model_id}, skipping..."
                    )
                continue
        else:
            current_model = model

        norm_stats = norm_stats_dict.get(model_id)
        if norm_stats is None:
            if verbose:
                print(
                    f"\n  No norm_stats for Model_{model_id}, skipping..."
                )
            continue

        ds_model = ds_test.filter_by_model(model_id)
        num_events = len(ds_model)

        if verbose:
            print(f"\nModel_{model_id}: {num_events} test events")

        for event_idx in range(num_events):
            sample = ds_model[event_idx]
            event_id = sample["event_id"]
            num_nodes = len(sample["static_2d_nodes"])

            if verbose:
                print(f"  Event_{event_id}: ", end="", flush=True)

            pred_wl, _ = predict_event_2d(
                current_model, sample, norm_stats, num_warmup=num_warmup
            )

            num_timesteps = pred_wl.shape[0]
            num_predict = num_timesteps - num_warmup

            if verbose:
                print(
                    f"{num_nodes} nodes x {num_predict} timesteps "
                    f"= {num_nodes * num_predict:,} rows"
                )

            # Rows sorted by (node_id, timestep) within each event.
            # An internal ``_sort_key`` tuple is used to guarantee
            # deterministic ordering before the column is dropped.
            for node_id in range(num_nodes):
                for t in range(num_warmup, num_timesteps):
                    all_rows.append(
                        {
                            "model_id": int(model_id),
                            "event_id": int(event_id),
                            "node_type": 2,
                            "node_id": node_id,
                            "water_level": float(
                                pred_wl[t, node_id].item()
                            ),
                            "_sort_key": (
                                int(model_id),
                                int(event_id),
                                node_id,
                                t,
                            ),
                        }
                    )

    # ── assemble DataFrame ────────────────────────────────────────
    if not all_rows:
        print("  No predictions generated!")
        return pd.DataFrame()

    submission_df = (
        pd.DataFrame(all_rows)
        .sort_values("_sort_key")
        .drop(columns=["_sort_key"])
        .reset_index(drop=True)
    )

    submission_df.insert(0, "row_id", range(len(submission_df)))

    # Enforce competition column order
    submission_df = submission_df[
        [
            "row_id",
            "model_id",
            "event_id",
            "node_type",
            "node_id",
            "water_level",
        ]
    ]

    if verbose:
        print(f"\n{'=' * 60}")
        print("2D Submission Summary:")
        print(f"  Total rows: {len(submission_df):,}")
        for mid in sorted(submission_df["model_id"].unique()):
            n = int((submission_df["model_id"] == mid).sum())
            print(f"  Model_{mid}: {n:,} rows")
        wl = submission_df["water_level"]
        print(f"  Water level range: [{wl.min():.2f}, {wl.max():.2f}]")

    # ── save ──────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    if verbose:
        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"\nSaved: {output_path}")
        print(f"Size: {size_mb:.2f} MB")
        print(f"\nFirst 5 rows:")
        print(submission_df.head().to_string(index=False))

    return submission_df


# ───────────────────────────────────────────────────────────────────────
#  Quick smoke test
# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.config import RAW_DATA_PATH
    from src.dataset import FloodDataset
    from src.graph_builder_2d import build_2d_graph
    from src.model_2d import SurfaceEngine, load_checkpoint
    from src.utils_2d import compute_normalization_stats

    print("=" * 60)
    print("Generating 2D Test Submission")
    print("=" * 60)

    # Norm stats from training data
    ds_train = FloodDataset(RAW_DATA_PATH, mode="train")

    print("\nComputing normalization stats...")
    norm_stats_dict = {
        "1": compute_normalization_stats(ds_train, "1"),
        "2": compute_normalization_stats(ds_train, "2"),
    }

    # Model dimensions
    sample = ds_train[0]
    data = build_2d_graph(
        sample, norm_stats_dict[sample["model_id"]], t_index=10
    )
    in_channels = data.x.shape[1]

    print(f"\nCreating model (in_channels={in_channels})...")
    model = SurfaceEngine(
        in_channels=in_channels,
        hidden_channels=64,
        num_sage_layers=2,
        dropout=0.0,
        max_delta=2.0,
    )

    # Try to load a trained checkpoint
    ckpt = Path("checkpoints/model_1_best.pt")
    if ckpt.exists():
        print(f"Loading trained model from {ckpt}...")
        load_checkpoint(model, str(ckpt))
    else:
        print("  No checkpoint found. Using untrained model.")

    # Generate 2D submission
    try:
        submission_2d = generate_test_submission_2d(
            model=model,
            data_path=RAW_DATA_PATH,
            norm_stats_dict=norm_stats_dict,
            output_path="submissions/submission_2d.csv",
            num_warmup=10,
            verbose=True,
        )
        print("\n  2D submission generated: submissions/submission_2d.csv")

    except Exception as exc:
        print(f"\n  Error: {exc}")
        import traceback

        traceback.print_exc()

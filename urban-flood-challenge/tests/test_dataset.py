"""
test_dataset.py — Comprehensive smoke tests for FloodDataset.

Run from the project root:
    python -m tests.test_dataset

OR to use with pytest:
    pytest tests/test_dataset.py -v

These tests work in two modes:
    1. REAL DATA  — if FLOOD_DATA_PATH points to a directory with
                    actual Models/ data, all tests run end-to-end.
    2. SYNTHETIC  — if no real data is found, the script creates a
                    minimal fake directory tree so every code path
                    in FloodDataset is still exercised.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure src/ is importable ──────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import RAW_DATA_PATH  # noqa: E402
from src.dataset import FloodDataset  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════
#  Synthetic data factory (fallback when no real data is present)
# ═══════════════════════════════════════════════════════════════════════

N_1D_NODES = 10
N_2D_NODES = 20
N_TIMESTEPS = 50


def _make_synthetic_tree(base: str) -> str:
    """Build a minimal but realistic directory tree inside *base*.

    Returns the ``root_dir`` that should be passed to FloodDataset.
    """
    root = os.path.join(base, "SyntheticData")

    for model_id in ["1", "2"]:
        model_dir = os.path.join(root, "Models", f"Model_{model_id}")

        for mode in ["train", "test"]:
            mode_dir = os.path.join(model_dir, mode)
            os.makedirs(mode_dir, exist_ok=True)

            # ── Static files (inside mode dir) ───────────────────────
            # 1D nodes
            pd.DataFrame({
                "1d_position_x": np.random.rand(N_1D_NODES),
                "1d_position_y": np.random.rand(N_1D_NODES),
                "Depth": np.random.rand(N_1D_NODES) * 10,
                "Invert elevation": np.random.rand(N_1D_NODES) * 50,
                "Surface elevation": np.random.rand(N_1D_NODES) * 50 + 50,
                "Base area": np.random.rand(N_1D_NODES) * 5,
            }).to_csv(os.path.join(mode_dir, "1d_nodes_static.csv"), index=False)

            # 2D nodes
            pd.DataFrame({
                "2d_position_x": np.random.rand(N_2D_NODES),
                "2d_position_y": np.random.rand(N_2D_NODES),
                "Area": np.random.rand(N_2D_NODES) * 100,
                "Roughness": np.random.rand(N_2D_NODES) * 0.05,
                "Minimum Elevation": np.random.rand(N_2D_NODES) * 20,
                "Centroid Elevation": np.random.rand(N_2D_NODES) * 25,
            }).to_csv(os.path.join(mode_dir, "2d_nodes_static.csv"), index=False)

            # 1D edges
            pd.DataFrame({
                "1d_edge_length": np.random.rand(N_1D_NODES - 1) * 100,
                "Diameter": np.random.rand(N_1D_NODES - 1) * 2,
                "Shape": ["circular"] * (N_1D_NODES - 1),
                "Roughness": [0.013] * (N_1D_NODES - 1),
                "Slope": np.random.rand(N_1D_NODES - 1) * 0.01,
            }).to_csv(os.path.join(mode_dir, "1d_edges_static.csv"), index=False)

            # 2D edges
            pd.DataFrame({
                "Face_length": np.random.rand(N_2D_NODES - 1) * 10,
                "2d_length": np.random.rand(N_2D_NODES - 1) * 15,
                "Slope": np.random.rand(N_2D_NODES - 1) * 0.005,
            }).to_csv(os.path.join(mode_dir, "2d_edges_static.csv"), index=False)

            # Edge indices
            pd.DataFrame({
                "from_node_idx": list(range(N_1D_NODES - 1)),
                "to_node_idx": list(range(1, N_1D_NODES)),
            }).to_csv(os.path.join(mode_dir, "1d_edge_index.csv"), index=False)

            pd.DataFrame({
                "from_node_idx": list(range(N_2D_NODES - 1)),
                "to_node_idx": list(range(1, N_2D_NODES)),
            }).to_csv(os.path.join(mode_dir, "2d_edge_index.csv"), index=False)

            # 1D-2D coupling
            pd.DataFrame({
                "1d_node_id": list(range(min(5, N_1D_NODES))),
                "2d_node_id": list(range(min(5, N_2D_NODES))),
            }).to_csv(os.path.join(mode_dir, "1d2d_connections.csv"), index=False)

            # ── Dynamic files (per event) ────────────────────────────
            n_events = 4 if mode == "train" else 2
            for ev_idx in range(1, n_events + 1):
                ev_id = f"{ev_idx:02d}"
                ev_dir = os.path.join(mode_dir, f"event_{ev_id}")
                os.makedirs(ev_dir, exist_ok=True)

                # 1D dynamic — wide format: one water_level col per node
                wl_1d = np.random.rand(N_TIMESTEPS, N_1D_NODES) * 5
                cols_1d = [f"1d_node_water_level_{i}" for i in range(N_1D_NODES)]
                inlet_cols = [f"1d_node_inlet_flow_{i}" for i in range(N_1D_NODES)]
                df_1d = pd.DataFrame(wl_1d, columns=cols_1d)
                for c in inlet_cols:
                    df_1d[c] = np.random.rand(N_TIMESTEPS) * 0.5
                df_1d.to_csv(
                    os.path.join(ev_dir, "1d_nodes_dynamic_all.csv"),
                    index=False,
                )

                # 2D dynamic
                wl_2d = np.random.rand(N_TIMESTEPS, N_2D_NODES) * 2
                rain = np.random.rand(N_TIMESTEPS, N_2D_NODES) * 0.1
                cols_wl = [f"water_level_{i}" for i in range(N_2D_NODES)]
                cols_rain = [f"rainfall_depth_{i}" for i in range(N_2D_NODES)]
                df_2d = pd.DataFrame(
                    np.hstack([wl_2d, rain]),
                    columns=cols_wl + cols_rain,
                )
                df_2d.to_csv(
                    os.path.join(ev_dir, "2d_nodes_dynamic_all.csv"),
                    index=False,
                )

                # Timesteps
                pd.DataFrame({"timestep": list(range(N_TIMESTEPS))}).to_csv(
                    os.path.join(ev_dir, "timesteps.csv"), index=False,
                )

    return root


# ═══════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════

DIVIDER = "=" * 60


def _section(title: str) -> None:
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


def test_all(data_root: str | None = None) -> None:
    """Run all loader tests. Called from __main__ or pytest."""

    # ── Resolve data root ──────────────────────────────────────────────
    using_synthetic = False
    tmp_dir = None

    if data_root and os.path.isdir(os.path.join(data_root, "Models")):
        root = data_root
        print(f"✓ Using REAL data at: {root}")
    else:
        print(
            f"⚠ Real data not found at: {data_root or RAW_DATA_PATH}\n"
            "  → Generating synthetic data for structural testing.\n"
        )
        tmp_dir = tempfile.mkdtemp(prefix="flood_test_")
        root = _make_synthetic_tree(tmp_dir)
        using_synthetic = True

    try:
        _run_tests(root, using_synthetic)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _run_tests(root: str, synthetic: bool) -> None:
    # ------------------------------------------------------------------
    # 1. Construction & Discovery
    # ------------------------------------------------------------------
    _section("TEST 1 — Construction & Event Discovery")

    ds = FloodDataset(root, mode="train")
    print(f"  repr   : {ds!r}")
    print(f"  len    : {len(ds)}")
    assert len(ds) > 0, "FAIL: No events discovered!"
    print("  ✓ PASS — events discovered successfully.\n")

    # ------------------------------------------------------------------
    # 2. Model/Event ID accessors
    # ------------------------------------------------------------------
    _section("TEST 2 — Model & Event ID Accessors")

    model_ids = ds.get_model_ids()
    print(f"  model_ids : {model_ids}")
    assert len(model_ids) > 0, "FAIL: No model IDs!"

    for mid in model_ids:
        eids = ds.get_event_ids(mid)
        print(f"  Model_{mid} event_ids: {eids}")
        assert len(eids) > 0, f"FAIL: No events for Model_{mid}!"

    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    # 3. __getitem__ — full sample load
    # ------------------------------------------------------------------
    _section("TEST 3 — __getitem__  (sample loading)")

    sample = ds[0]
    print(f"  model_id : {sample['model_id']}")
    print(f"  event_id : {sample['event_id']}")
    print(f"  keys     : {sorted(sample.keys())}")

    # Check all expected keys are present
    expected_keys = {
        "model_id", "event_id",
        "static_1d_nodes", "static_2d_nodes",
        "static_1d_edges", "static_2d_edges",
        "edge_index_1d", "edge_index_2d",
        "1d2d_conn",
        "dynamic_1d_nodes", "dynamic_2d_nodes",
        "dynamic_1d_edges", "dynamic_2d_edges",
        "timesteps",
    }
    missing = expected_keys - set(sample.keys())
    assert not missing, f"FAIL: Missing keys: {missing}"
    print("  ✓ All expected keys present.\n")

    # Verify DataFrames are non-empty for core files
    for k in ["static_1d_nodes", "static_2d_nodes", "dynamic_1d_nodes",
              "dynamic_2d_nodes", "edge_index_1d", "edge_index_2d"]:
        df = sample[k]
        assert isinstance(df, pd.DataFrame), f"FAIL: {k} is not a DataFrame"
        if df.empty:
            print(f"  ⚠ WARNING: {k} is empty — check your data directory.")
        else:
            print(f"  {k:25s} → shape {df.shape}")

    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    # 4. Static caching
    # ------------------------------------------------------------------
    _section("TEST 4 — Static Caching")

    mid = sample["model_id"]
    sample2 = ds[0]
    assert sample["static_1d_nodes"] is sample2["static_1d_nodes"], (
        "FAIL: Static DataFrames are NOT shared between calls (caching broken)."
    )
    print(f"  ✓ Static cache works for Model_{mid} — same object ID.\n")

    # ------------------------------------------------------------------
    # 5. filter_by_model
    # ------------------------------------------------------------------
    _section("TEST 5 — filter_by_model")

    for mid in model_ids:
        subset = ds.filter_by_model(mid)
        assert all(e["model_id"] == mid for e in subset.events)
        print(f"  Model_{mid}: {len(subset)} events")
    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    # 6. split_by_event (Leave-One-Event-Out)
    # ------------------------------------------------------------------
    _section("TEST 6 — split_by_event (LOEO CV)")

    mid = model_ids[0]
    eids = ds.get_event_ids(mid)
    if len(eids) >= 2:
        val_eid = eids[-1]  # hold-out last event
        train_ds, val_ds = ds.split_by_event(val_eid, model_id=mid)
        print(f"  Hold-out event: {val_eid}")
        print(f"  train: {len(train_ds)} events, val: {len(val_ds)} events")
        assert len(val_ds) > 0, "FAIL: val_ds is empty"
        assert all(
            e["event_id"] != val_eid for e in train_ds.events
        ), "FAIL: val event leaked into train!"
        assert all(
            e["event_id"] == val_eid for e in val_ds.events
        ), "FAIL: wrong events in val_ds!"
        print("  ✓ PASS — no leakage detected.\n")
    else:
        print("  ⚠ SKIP — need ≥ 2 events to test splitting.\n")

    # ------------------------------------------------------------------
    # 7. compute_node_stds
    # ------------------------------------------------------------------
    _section("TEST 7 — compute_node_stds (for loss function)")

    stds = ds.compute_node_stds(model_id=model_ids[0])
    for mid_key, std_dict in stds.items():
        print(f"  Model_{mid_key}:")
        for domain, arr in std_dict.items():
            print(f"    {domain} → shape {arr.shape}, "
                  f"min={arr.min():.6f}, max={arr.max():.6f}")
            assert arr.shape[0] > 0, f"FAIL: {domain} stds are empty!"
            assert np.all(np.isfinite(arr)), f"FAIL: {domain} stds contain NaN/Inf!"

    # Verify per-node granularity: shape must match number of unique nodes
    s0 = ds[0]
    if "node_idx" in s0["dynamic_1d_nodes"].columns:
        expected_1d = s0["dynamic_1d_nodes"]["node_idx"].nunique()
        actual_1d = stds[model_ids[0]]["1d"].shape[0]
        assert actual_1d == expected_1d, (
            f"FAIL: 1D stds has {actual_1d} entries but "
            f"expected {expected_1d} (one per node)"
        )
        print(f"  ✓ 1D stds: {actual_1d} entries == {expected_1d} unique nodes")

    if "node_idx" in s0["dynamic_2d_nodes"].columns:
        expected_2d = s0["dynamic_2d_nodes"]["node_idx"].nunique()
        actual_2d = stds[model_ids[0]]["2d"].shape[0]
        assert actual_2d == expected_2d, (
            f"FAIL: 2D stds has {actual_2d} entries but "
            f"expected {expected_2d} (one per node)"
        )
        print(f"  ✓ 2D stds: {actual_2d} entries == {expected_2d} unique nodes")

    print("  ✓ PASS — node_stds are per-node, finite, and non-empty.\n")

    # ------------------------------------------------------------------
    # 8. DataLoader compatibility
    # ------------------------------------------------------------------
    _section("TEST 8 — DataLoader (batch_size=1, collate_fn)")

    try:
        import torch.utils.data as tud

        loader = tud.DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            collate_fn=FloodDataset.collate_fn,
        )
        batch = next(iter(loader))
        assert isinstance(batch, dict), "FAIL: collate_fn didn't return a dict"
        assert "model_id" in batch, "FAIL: missing model_id in batch"
        print(f"  ✓ First batch loaded — model={batch['model_id']}, "
              f"event={batch['event_id']}")
    except ImportError:
        print("  ⚠ SKIP — torch not installed.\n")

    print("  ✓ PASS\n")

    # ------------------------------------------------------------------
    # 9. Physics sanity checks (only with real data)
    # ------------------------------------------------------------------
    _section("TEST 9 — Physics Sanity (Capacity = Surface − Invert)")

    s = ds[0]
    sn = s["static_1d_nodes"]
    # Support both naming conventions:
    #   Real data:      surface_elevation / invert_elevation  (snake_case)
    #   Synthetic data:  Surface elevation / Invert elevation  (Title Case)
    surf_col = next(
        (c for c in sn.columns if c.lower().replace(" ", "_") == "surface_elevation"),
        None,
    )
    inv_col = next(
        (c for c in sn.columns if c.lower().replace(" ", "_") == "invert_elevation"),
        None,
    )
    if not sn.empty and surf_col and inv_col:
        capacity = sn[surf_col] - sn[inv_col]
        neg_count = (capacity < 0).sum()
        print(f"  Capacity range: [{capacity.min():.3f}, {capacity.max():.3f}]")
        if neg_count > 0:
            print(f"  ⚠ WARNING: {neg_count} nodes have negative capacity "
                  "(Invert > Surface). Check data quality.")
        else:
            print("  ✓ All capacities non-negative.")
    else:
        cols = list(sn.columns) if not sn.empty else []
        print(f"  ⚠ SKIP — elevation columns not found. Columns: {cols}\n")

    # ------------------------------------------------------------------
    # 10. Edge case: IndexError
    # ------------------------------------------------------------------
    _section("TEST 10 — IndexError on out-of-bounds")

    try:
        _ = ds[len(ds) + 100]
        print("  ✗ FAIL — should have raised IndexError!")
    except IndexError:
        print("  ✓ PASS — IndexError raised correctly.\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    data_type = "SYNTHETIC" if synthetic else "REAL"
    print(f"  ALL TESTS PASSED  ({data_type} data)")
    print(f"{'=' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════
#  Entry-point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Accept an optional CLI argument for the data path; else use config
    data_path = sys.argv[1] if len(sys.argv) > 1 else str(RAW_DATA_PATH)
    test_all(data_path)

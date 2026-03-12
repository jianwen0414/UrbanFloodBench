"""
Post-process a 2D submission CSV to reduce autoregressive noise.

Three passes (applied per-node time series):
  1. Temporal smoothing — weighted moving average [0.15, 0.7, 0.15]
  2. Dry-node snapping — if all WL within 0.01 m of ground elevation, snap
  3. Recession limb correction — enforce monotonic decrease on falling segments

Usage:
    python -m src.post_process
"""
import csv
import sys
from pathlib import Path

import numpy as np

from src.config import RAW_DATA_PATH
from src.dataset import FloodDataset
from src.utils_2d import get_min_elevation_filled

# ── Paths ─────────────────────────────────────────────────────────────
INPUT_PATH = "submissions/submission_2d_ensemble.csv"
OUTPUT_PATH = "submissions/submission_2d_postprocessed.csv"

SMOOTH_WEIGHTS = np.array([0.15, 0.70, 0.15])
DRY_THRESHOLD = 0.01  # metres


# ── Post-processing functions ─────────────────────────────────────────

def temporal_smooth(wl: np.ndarray) -> tuple[np.ndarray, int]:
    """Weighted moving average with [0.15, 0.7, 0.15].  Endpoints unchanged."""
    if len(wl) < 3:
        return wl.copy(), 0
    smoothed = wl.copy()
    for i in range(1, len(wl) - 1):
        smoothed[i] = (
            SMOOTH_WEIGHTS[0] * wl[i - 1]
            + SMOOTH_WEIGHTS[1] * wl[i]
            + SMOOTH_WEIGHTS[2] * wl[i + 1]
        )
    changed = int(np.sum(np.abs(smoothed - wl) > 1e-9))
    return smoothed, changed


def snap_dry_nodes(
    wl: np.ndarray, ground_elev: float
) -> tuple[np.ndarray, int]:
    """If all WL within DRY_THRESHOLD of ground, snap to ground."""
    if np.all(np.abs(wl - ground_elev) <= DRY_THRESHOLD):
        changed = int(np.sum(np.abs(wl - ground_elev) > 1e-9))
        return np.full_like(wl, ground_elev), changed
    return wl.copy(), 0


def recession_correction(wl: np.ndarray) -> tuple[np.ndarray, int]:
    """On falling segments, enforce monotonic decrease (cap upward bumps)."""
    out = wl.copy()
    changed = 0
    for i in range(2, len(out)):
        if out[i - 1] < out[i - 2]:
            # We're on a recession limb
            if out[i] > out[i - 1]:
                out[i] = out[i - 1]
                changed += 1
    return out, changed


# ── Build ground-elevation lookup ─────────────────────────────────────

def build_elevation_lookup() -> dict[tuple[str, str], np.ndarray]:
    """Return {(model_id, event_id): min_elevation_array} for test events."""
    ds = FloodDataset(RAW_DATA_PATH, mode="test")
    lookup: dict[tuple[str, str], np.ndarray] = {}
    for i in range(len(ds)):
        sample = ds[i]
        mid = str(sample["model_id"])
        eid = str(sample["event_id"])
        elev = get_min_elevation_filled(sample["static_2d_nodes"])
        lookup[(mid, eid)] = elev
    return lookup


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)
    if not input_path.exists():
        sys.exit(f"Input not found: {input_path}")

    print("Loading ground elevations from test dataset...")
    elev_lookup = build_elevation_lookup()
    print(f"  {len(elev_lookup)} (model, event) entries")

    print(f"\nStreaming: {input_path} → {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "smooth_changed": 0,
        "dry_snapped": 0,
        "recession_fixed": 0,
        "nodes_processed": 0,
        "rows_written": 0,
    }

    fin = open(input_path)
    fout = open(output_path, "w", newline="")
    reader = csv.DictReader(fin)
    writer = csv.writer(fout)
    writer.writerow(["row_id", "model_id", "event_id", "node_type", "node_id", "water_level"])

    # Accumulate rows for the current node, process when the key changes
    cur_key: tuple | None = None
    cur_rows: list[dict] = []

    def flush_node() -> None:
        """Apply post-processing to cur_rows and write to output."""
        if not cur_rows:
            return
        mid = cur_rows[0]["model_id"]
        eid = cur_rows[0]["event_id"]
        nid = int(cur_rows[0]["node_id"])
        wl = np.array([float(r["water_level"]) for r in cur_rows])

        # 1. Temporal smoothing
        wl, n_smooth = temporal_smooth(wl)
        stats["smooth_changed"] += n_smooth

        # 2. Dry-node snapping
        elev_arr = elev_lookup.get((mid, eid))
        if elev_arr is not None and nid < len(elev_arr):
            wl, n_dry = snap_dry_nodes(wl, elev_arr[nid])
            stats["dry_snapped"] += n_dry

        # 3. Recession limb correction
        wl, n_rec = recession_correction(wl)
        stats["recession_fixed"] += n_rec

        # Write
        for i, r in enumerate(cur_rows):
            writer.writerow([
                r["row_id"], r["model_id"], r["event_id"],
                r["node_type"], r["node_id"], wl[i],
            ])
            stats["rows_written"] += 1

        stats["nodes_processed"] += 1

    for row in reader:
        key = (row["model_id"], row["event_id"], row["node_id"])
        if key != cur_key:
            flush_node()
            cur_key = key
            cur_rows = []
        cur_rows.append(row)

    flush_node()  # last node

    fin.close()
    fout.close()

    print(f"\n{'='*60}")
    print("Post-processing summary")
    print(f"{'='*60}")
    print(f"  Nodes processed:          {stats['nodes_processed']:>12,}")
    print(f"  Rows written:             {stats['rows_written']:>12,}")
    print(f"  Temporal smooth modified: {stats['smooth_changed']:>12,}")
    print(f"  Dry-node values snapped:  {stats['dry_snapped']:>12,}")
    print(f"  Recession bumps fixed:    {stats['recession_fixed']:>12,}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()

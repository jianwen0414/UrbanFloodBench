"""
Baseline XGBoost Benchmark — Tabular Regression
================================================

Treats the Urban Flood Modelling challenge as a flat tabular problem:
    * One **global** XGBRegressor for all 1D (pipe) nodes.
    * One **global** XGBRegressor for all 2D (surface) nodes.

Feature vector per (timestep, node) row
----------------------------------------
    [water_level_{t-1}, water_level_{t-2}, water_level_{t-3},
     static_feature_1, static_feature_2, …,
     model_id_encoded]

Target: water_level_t

Evaluation mirrors the competition's Standardized RMSE and uses an
autoregressive loop with a configurable burn-in window.

Usage
-----
    python -m src.baseline_xgb                    # from project root
    python -m src.baseline_xgb --data /path/to/data --burn_in 10
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# Lazy-import heavy libs so --help stays fast
# ---------------------------------------------------------------------------

def _import_xgb():
    from xgboost import XGBRegressor
    return XGBRegressor


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# Ensure ``src`` package is importable when invoked with ``python -m``
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR.parent))

from src.config import RAW_DATA_PATH, RANDOM_SEED, PROJECT_ROOT
from src.dataset import FloodDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_LAGS: int = 3               # number of lag features (t-1, t-2, t-3)
BURN_IN: int = 10              # teacher-forced warm-up steps at inference
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
_EPS: float = 1e-6            # numerical guard for σ ≈ 0


# ======================================================================
#  1.  Feature engineering helpers
# ======================================================================

def _node_columns(df: pd.DataFrame) -> List[str]:
    """Return column names that represent individual nodes (drop time cols)."""
    skip = {"timestep", "time", "Timestep", "Time"}
    return [c for c in df.columns if c not in skip]


def _build_lag_features(
    series: np.ndarray,
    n_lags: int = N_LAGS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create lag features and target from a (T, N) array.

    Returns
    -------
    X : ndarray, shape (T - n_lags, N, n_lags)
        Lag features for every (timestep, node) pair.
    y : ndarray, shape (T - n_lags, N)
        Target water level at each valid timestep.
    """
    T, N = series.shape
    # Use pd.DataFrame.shift logic via manual slicing for efficiency
    X_lags = np.stack(
        [series[n_lags - lag - 1 : T - lag - 1] for lag in range(n_lags)],
        axis=-1,
    )  # (T - n_lags, N, n_lags)
    y = series[n_lags:]  # (T - n_lags, N)
    return X_lags, y


def _merge_static_features(
    X_lags: np.ndarray,
    static_df: pd.DataFrame,
    n_nodes: int,
    n_timesteps: int,
) -> np.ndarray:
    """Tile static features across time and concatenate with lags.

    Parameters
    ----------
    X_lags : (T', N, n_lags)
    static_df : DataFrame with one row per node, numeric columns only.

    Returns
    -------
    X : (T' * N, n_lags + n_static)  — fully flattened feature matrix.
    """
    # Keep only numeric columns from the static df
    static_num = static_df.select_dtypes(include=[np.number])
    if static_num.empty:
        # Nothing to merge — just flatten lags
        return X_lags.reshape(-1, X_lags.shape[-1])

    # Ensure node count matches; truncate or pad if needed
    static_vals = static_num.values.astype(np.float32)
    if len(static_vals) > n_nodes:
        static_vals = static_vals[:n_nodes]
    elif len(static_vals) < n_nodes:
        pad = np.zeros((n_nodes - len(static_vals), static_vals.shape[1]),
                        dtype=np.float32)
        static_vals = np.concatenate([static_vals, pad], axis=0)

    # Tile static (N, S) → (T', N, S)
    static_tiled = np.tile(static_vals[np.newaxis, :, :],
                           (n_timesteps, 1, 1))

    # Concatenate along feature axis → (T', N, n_lags + S)
    X_full = np.concatenate([X_lags, static_tiled], axis=-1)

    # Flatten to (T' * N, F)
    return X_full.reshape(-1, X_full.shape[-1])


# ======================================================================
#  2.  Data preparation (the "Flattening")
# ======================================================================

def prepare_tabular_data(
    dataset: FloodDataset,
    domain: str = "1d",
    n_lags: int = N_LAGS,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Iterate through all events and produce a flat (X, y) matrix.

    Parameters
    ----------
    dataset : FloodDataset
        A dataset in 'train' (or 'test') mode.
    domain : str
        ``'1d'`` or ``'2d'`` — selects the dynamic/static tables.
    n_lags : int
        Number of lag features to generate.
    verbose : bool
        Print progress.

    Returns
    -------
    X : ndarray, shape (total_rows, n_features)
    y : ndarray, shape (total_rows,)
    """
    dyn_key    = f"dynamic_{domain}_nodes"
    static_key = f"static_{domain}_nodes"

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        dyn_df    = sample[dyn_key]
        static_df = sample[static_key]

        if dyn_df.empty:
            if verbose:
                print(f"  [skip] event {idx} — empty dynamic {domain} data")
            continue

        node_cols = _node_columns(dyn_df)
        series = dyn_df[node_cols].values.astype(np.float32)  # (T, N)
        T, N = series.shape

        if T <= n_lags:
            if verbose:
                print(f"  [skip] event {idx} — too few timesteps ({T} ≤ {n_lags})")
            continue

        # Lag features → (T', N, n_lags)  where T' = T - n_lags
        X_lags, y_arr = _build_lag_features(series, n_lags)
        T_prime = X_lags.shape[0]

        # Merge static context → (T' * N, F)
        X_flat = _merge_static_features(X_lags, static_df, N, T_prime)
        y_flat = y_arr.reshape(-1)  # (T' * N,)

        X_parts.append(X_flat)
        y_parts.append(y_flat)

        if verbose and (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} events …")

    if not X_parts:
        raise RuntimeError(f"No valid events found for domain={domain}")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    if verbose:
        print(f"  {domain.upper()} tabular data: X={X.shape}, y={y.shape}")

    return X, y


# ======================================================================
#  3.  Model training
# ======================================================================

def train_xgb(
    X: np.ndarray,
    y: np.ndarray,
    label: str = "1d",
    **xgb_kwargs: Any,
) -> Any:
    """Train a single XGBRegressor and return it.

    Parameters
    ----------
    X, y : arrays
        Feature matrix and target vector (flattened).
    label : str
        Human-readable name for logging.
    **xgb_kwargs
        Overrides for XGBRegressor constructor.

    Returns
    -------
    Fitted XGBRegressor.
    """
    XGBRegressor = _import_xgb()

    defaults = dict(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=1,
    )
    defaults.update(xgb_kwargs)

    print(f"\n{'='*60}")
    print(f"Training XGBRegressor [{label}]  —  {X.shape[0]:,} rows, "
          f"{X.shape[1]} features")
    print(f"{'='*60}")
    t0 = time.time()

    model = XGBRegressor(**defaults)
    model.fit(X, y)

    elapsed = time.time() - t0
    y_hat = model.predict(X[:5000])  # quick in-sample sanity check
    rmse_sample = math.sqrt(mean_squared_error(y[:5000], y_hat))
    print(f"  Trained in {elapsed:.1f}s  |  in-sample RMSE (first 5k): {rmse_sample:.6f}")

    return model


def save_model(model: Any, name: str) -> Path:
    """Persist a model to ``artifacts/{name}.pkl``."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved → {path}")
    return path


def load_model(name: str) -> Any:
    """Load a model from ``artifacts/{name}.pkl``."""
    path = ARTIFACTS_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ======================================================================
#  4.  Standardized RMSE (pure NumPy — no torch dependency)
# ======================================================================

def srmse_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    node_stds: np.ndarray,
) -> float:
    """Compute the **exact** competition Standardized RMSE.

    SRMSE = (1/N) Σ_i  sqrt( mean_t( (y_it - ŷ_it)² ) ) / σ_i

    Parameters
    ----------
    pred, target : (T, N)
    node_stds : (N,)

    Returns
    -------
    float
    """
    per_node_rmse = np.sqrt(np.mean((pred - target) ** 2, axis=0))  # (N,)
    return float(np.mean(per_node_rmse / (node_stds + _EPS)))


# ======================================================================
#  5.  Autoregressive inference & evaluation
# ======================================================================

def _predict_row(
    model: Any,
    lag_buffer: np.ndarray,
    static_row: np.ndarray,
) -> np.ndarray:
    """Predict water level for every node at a single timestep.

    Parameters
    ----------
    model : fitted XGBRegressor
    lag_buffer : (N, n_lags)  — most recent lags per node.
    static_row : (N, S)       — static context per node.

    Returns
    -------
    prediction : (N,)
    """
    if static_row.size > 0:
        X = np.concatenate([lag_buffer, static_row], axis=-1)  # (N, F)
    else:
        X = lag_buffer
    return model.predict(X).astype(np.float32)  # (N,)


def evaluate_baseline(
    dataset: FloodDataset,
    model_1d: Any,
    model_2d: Any,
    burn_in: int = BURN_IN,
    n_lags: int = N_LAGS,
    verbose: bool = True,
) -> Tuple[float, float, pd.DataFrame]:
    """Run autoregressive inference and compute SRMSE.

    Returns
    -------
    srmse_1d, srmse_2d : float
        Per-domain Standardized RMSE averaged across all events.
    submissions : pd.DataFrame
        Competition-format submission (model_id, event_id, timestep, node, prediction).
    """
    scores_1d: List[float] = []
    scores_2d: List[float] = []
    submission_rows: List[Dict[str, Any]] = []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        model_id = sample["model_id"]
        event_id = sample["event_id"]

        for domain, model in [("1d", model_1d), ("2d", model_2d)]:
            dyn_df    = sample[f"dynamic_{domain}_nodes"]
            static_df = sample[f"static_{domain}_nodes"]

            if dyn_df.empty:
                continue

            node_cols = _node_columns(dyn_df)
            gt = dyn_df[node_cols].values.astype(np.float32)  # (T, N)
            T, N = gt.shape

            # Static features (numeric only) — one row per node
            static_num = static_df.select_dtypes(include=[np.number])
            if not static_num.empty:
                static_vals = static_num.values.astype(np.float32)
                if len(static_vals) > N:
                    static_vals = static_vals[:N]
                elif len(static_vals) < N:
                    pad = np.zeros((N - len(static_vals), static_vals.shape[1]),
                                   dtype=np.float32)
                    static_vals = np.concatenate([static_vals, pad], axis=0)
            else:
                static_vals = np.empty((N, 0), dtype=np.float32)

            # Per-node σ from this event's GT (or ideally pre-computed
            # from training data — here per-event is a reasonable proxy).
            node_stds = gt.std(axis=0)  # (N,)

            # ── Autoregressive loop ──────────────────────────────────
            # We maintain a rolling lag buffer of shape (N, n_lags).
            predictions: List[np.ndarray] = []
            lag_buffer = np.zeros((N, n_lags), dtype=np.float32)

            for t in range(T):
                if t < burn_in:
                    # ── BURN-IN: teacher forcing ─────────────────────
                    # Fill lag buffer from ground truth.
                    for lag in range(n_lags):
                        src_t = t - lag
                        if src_t >= 0:
                            lag_buffer[:, lag] = gt[src_t]
                else:
                    # ── FORECAST: autoregressive ─────────────────────
                    pred_t = _predict_row(model, lag_buffer, static_vals)
                    predictions.append(pred_t)

                    # Shift lag buffer: move existing lags right, insert new
                    lag_buffer[:, 1:] = lag_buffer[:, :-1]
                    lag_buffer[:, 0]  = pred_t

            if not predictions:
                continue

            preds = np.stack(predictions, axis=0)   # (T_forecast, N)
            targets = gt[burn_in:]                   # (T_forecast, N)

            # Trim to same length (safety)
            min_len = min(len(preds), len(targets))
            preds   = preds[:min_len]
            targets = targets[:min_len]

            score = srmse_numpy(preds, targets, node_stds)

            if domain == "1d":
                scores_1d.append(score)
            else:
                scores_2d.append(score)

            # Build submission rows
            for ti, t_abs in enumerate(range(burn_in, burn_in + min_len)):
                for ni, nc in enumerate(node_cols):
                    submission_rows.append({
                        "model_id":  model_id,
                        "event_id":  event_id,
                        "domain":    domain,
                        "timestep":  t_abs,
                        "node":      nc,
                        "prediction": float(preds[ti, ni]),
                    })

        if verbose and (idx + 1) % 10 == 0:
            print(f"  Evaluated {idx + 1}/{len(dataset)} events …")

    avg_1d = float(np.mean(scores_1d)) if scores_1d else float("nan")
    avg_2d = float(np.mean(scores_2d)) if scores_2d else float("nan")

    submissions = pd.DataFrame(submission_rows)

    if verbose:
        print(f"\n{'─'*50}")
        print(f"  SRMSE 1D : {avg_1d:.6f}  ({len(scores_1d)} events)")
        print(f"  SRMSE 2D : {avg_2d:.6f}  ({len(scores_2d)} events)")
        combined = (avg_1d + avg_2d) / 2 if scores_1d and scores_2d else float("nan")
        print(f"  Combined : {combined:.6f}")
        print(f"{'─'*50}")

    return avg_1d, avg_2d, submissions


# ======================================================================
#  6.  Main entry point
# ======================================================================

def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()

    data_path = args.data or str(RAW_DATA_PATH)

    # ── Load datasets ────────────────────────────────────────────────
    print("Loading training data …")
    train_ds = FloodDataset(root_dir=data_path, mode="train")

    # ── Flatten to tabular ───────────────────────────────────────────
    print("\nPreparing 1D tabular data …")
    X_1d, y_1d = prepare_tabular_data(train_ds, domain="1d", n_lags=args.n_lags)

    print("\nPreparing 2D tabular data …")
    X_2d, y_2d = prepare_tabular_data(train_ds, domain="2d", n_lags=args.n_lags)

    # ── Train models ─────────────────────────────────────────────────
    model_1d = train_xgb(X_1d, y_1d, label="1D-pipes",
                         n_estimators=args.n_estimators,
                         max_depth=args.max_depth,
                         learning_rate=args.lr)
    model_2d = train_xgb(X_2d, y_2d, label="2D-surface",
                         n_estimators=args.n_estimators,
                         max_depth=args.max_depth,
                         learning_rate=args.lr)

    # ── Save ─────────────────────────────────────────────────────────
    save_model(model_1d, "xgb_baseline_1d")
    save_model(model_2d, "xgb_baseline_2d")

    # ── Evaluate (on train or a held-out split — swap to test later) ─
    eval_mode = "test" if args.eval_test else "train"
    print(f"\nEvaluating on '{eval_mode}' set …")

    eval_ds = (
        FloodDataset(root_dir=data_path, mode="test")
        if args.eval_test
        else train_ds
    )

    srmse_1d, srmse_2d, submissions = evaluate_baseline(
        eval_ds, model_1d, model_2d,
        burn_in=args.burn_in, n_lags=args.n_lags,
    )

    # ── Submission CSV ───────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = ARTIFACTS_DIR / "submission_baseline.csv"
    submissions.to_csv(sub_path, index=False)
    print(f"\nSubmission saved → {sub_path}  ({len(submissions):,} rows)")


# ======================================================================
#  CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train & evaluate an XGBoost baseline for Urban Flood Modelling."
    )
    p.add_argument("--data", type=str, default=None,
                   help="Path to root data directory (default: RAW_DATA_PATH from config).")
    p.add_argument("--n_lags", type=int, default=N_LAGS,
                   help=f"Number of lag features (default: {N_LAGS}).")
    p.add_argument("--burn_in", type=int, default=BURN_IN,
                   help=f"Burn-in (teacher-forcing) steps (default: {BURN_IN}).")
    p.add_argument("--n_estimators", type=int, default=100,
                   help="XGBoost n_estimators.")
    p.add_argument("--max_depth", type=int, default=6,
                   help="XGBoost max_depth.")
    p.add_argument("--lr", type=float, default=0.1,
                   help="XGBoost learning_rate.")
    p.add_argument("--eval_test", action="store_true",
                   help="Evaluate on 'test' split instead of 'train'.")
    return p.parse_args()


if __name__ == "__main__":
    main()

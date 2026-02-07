"""
FloodDataset — Universal Lazy Loader.

Responsible for:
    * Discovering Model / Event folders under RAW_DATA_PATH.
    * Lazy-loading dynamic CSV files per event on __getitem__.
    * Caching static files (e.g. 1d_nodes_static.csv) in memory
      after first read to minimise RAM & I/O overhead.
    * Computing per-node standard deviations (``node_stds``) for
      the variance-weighted loss function.

Caching strategy
----------------
Static CSV files (node/edge attributes, edge indices, 1d-2d coupling)
are *identical* across every rainfall event within the same urban model.
Loading them repeatedly would waste both I/O and RAM.

``self.static_cache`` is a ``dict[str, dict[str, pd.DataFrame]]`` keyed
by ``model_id``.  On the first ``__getitem__`` call for a given model
the helper ``_load_static_data`` reads each static CSV once and stores
the resulting DataFrames.  Subsequent events under the same model reuse
the cached copy — zero extra I/O, zero duplicate memory.
"""

from __future__ import annotations

import os
import re
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data

# ---------------------------------------------------------------------------
# Constants — file names expected in each Model / Event folder
# ---------------------------------------------------------------------------

# Static files that live directly under  Model_{id}/{mode}/
# (fallback: Model_{id}/ root — some dataset layouts put them there)
_STATIC_FILES: Dict[str, str] = {
    "static_1d_nodes": "1d_nodes_static.csv",
    "static_2d_nodes": "2d_nodes_static.csv",
    "static_1d_edges": "1d_edges_static.csv",
    "static_2d_edges": "2d_edges_static.csv",
    "edge_index_1d":   "1d_edge_index.csv",
    "edge_index_2d":   "2d_edge_index.csv",
    "1d2d_conn":       "1d2d_connections.csv",
}

# Dynamic files that live under  Model_{id}/{mode}/event_{id}/
_DYNAMIC_FILES: Dict[str, str] = {
    "dynamic_1d_nodes": "1d_nodes_dynamic_all.csv",
    "dynamic_2d_nodes": "2d_nodes_dynamic_all.csv",
    "dynamic_1d_edges": "1d_edges_dynamic_all.csv",
    "dynamic_2d_edges": "2d_edges_dynamic_all.csv",
    "timesteps":        "timesteps.csv",
}

# Which dynamic keys are *optional* (won't raise if missing)
_OPTIONAL_DYNAMIC: frozenset[str] = frozenset(
    {"dynamic_1d_edges", "dynamic_2d_edges"}
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FloodDataset(torch.utils.data.Dataset):
    """PyTorch-compatible lazy-loading dataset for the Urban Flood data.

    Parameters
    ----------
    root_dir : str | os.PathLike
        Path to the top-level data directory that contains ``Models/``.
    mode : str, optional
        ``'train'`` or ``'test'`` — selects the corresponding subfolder
        inside each model directory.  Default: ``'train'``.
    transform : callable or None, optional
        An optional transform applied to the sample dictionary returned
        by ``__getitem__`` (e.g. the future ``GraphBuilder``).

    Notes
    -----
    This class returns **raw DataFrames** (no tensor conversion).
    It must be used with ``batch_size=1`` in a ``DataLoader`` or with
    the provided ``collate_fn``.

    Examples
    --------
    >>> from src.config import RAW_DATA_PATH
    >>> ds = FloodDataset(RAW_DATA_PATH, mode="train")
    >>> sample = ds[0]
    >>> sample["model_id"], sample["event_id"]
    ('1', '01')
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        root_dir: str | os.PathLike,
        mode: str = "train",
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        super().__init__()

        self.root_dir: str = str(root_dir)
        self.mode: str = mode
        self.transform: Optional[Callable] = transform

        # ---- Event index --------------------------------------------------
        # Each element is a dict:
        #   {"model_id": str, "event_id": str,
        #    "model_path": str,       <- …/Model_{id}/{mode}
        #    "model_root_path": str,  <- …/Model_{id}
        #    "event_path": str}       <- …/Model_{id}/{mode}/event_{id}
        self.events: List[Dict[str, str]] = []

        # ---- Static cache -------------------------------------------------
        # Keyed by model_id → dict of DataFrames (one per static file).
        # Populated lazily on the first __getitem__ for each model.
        self.static_cache: Dict[str, Dict[str, pd.DataFrame]] = {}

        # ---- Discover events ----------------------------------------------
        self._discover_events()

    # ------------------------------------------------------------------ #
    #  Event discovery
    # ------------------------------------------------------------------ #

    def _discover_events(self) -> None:
        """Walk the ``Models/`` tree and register every valid event folder.

        Populates ``self.events`` and prints per-model statistics.
        """
        models_root = os.path.join(self.root_dir, "Models")
        if not os.path.isdir(models_root):
            raise FileNotFoundError(
                f"Expected a 'Models' directory at {models_root}. "
                f"Check that FLOOD_DATA_PATH in src/.env points to the "
                f"correct local directory."
            )

        # Regex for folder names -----------------------------------------
        model_re = re.compile(r"^Model_(\w+)$")
        event_re = re.compile(r"^event_(\w+)$")

        # Counters for summary stats
        counts: Dict[str, int] = defaultdict(int)

        # Sort for deterministic ordering across runs
        for model_name in sorted(os.listdir(models_root)):
            m_match = model_re.match(model_name)
            if m_match is None:
                continue

            model_id = m_match.group(1)
            model_root_path = os.path.join(models_root, model_name)
            mode_dir = os.path.join(model_root_path, self.mode)

            if not os.path.isdir(mode_dir):
                warnings.warn(
                    f"{model_name} has no '{self.mode}/' subfolder — skipping."
                )
                continue

            for event_name in sorted(os.listdir(mode_dir)):
                e_match = event_re.match(event_name)
                if e_match is None:
                    continue

                event_id = e_match.group(1)
                event_path = os.path.join(mode_dir, event_name)

                if not os.path.isdir(event_path):
                    continue

                self.events.append(
                    {
                        "model_id":        model_id,
                        "event_id":        event_id,
                        "model_root_path": model_root_path,
                        "model_path":      mode_dir,
                        "event_path":      event_path,
                    }
                )
                counts[model_id] += 1

        # Summary -----------------------------------------------------------
        total = len(self.events)
        if total == 0:
            warnings.warn(
                f"No events found under {models_root} for mode='{self.mode}'."
            )
        else:
            print(
                f"FloodDataset ({self.mode}) — "
                f"{total} events across {len(counts)} model(s):"
            )
            for mid in sorted(
                counts, key=lambda x: int(x) if x.isdigit() else x
            ):
                print(f"  Model_{mid}: {counts[mid]} events")

    # ------------------------------------------------------------------ #
    #  Static data cache
    # ------------------------------------------------------------------ #

    def _load_static_data(
        self, model_id: str, model_path: str, model_root_path: str
    ) -> Dict[str, pd.DataFrame]:
        """Read all static CSVs for *model_id* and store in the cache.

        Checks two directories for each file (in order):
        1. ``model_path`` — ``Model_{id}/{mode}/``
        2. ``model_root_path`` — ``Model_{id}/``

        If the data is already cached the method returns immediately.

        Parameters
        ----------
        model_id : str
            Unique identifier for the urban model (e.g. ``"1"``).
        model_path : str
            Absolute path to ``Model_{id}/{mode}/``.
        model_root_path : str
            Absolute path to ``Model_{id}/``.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping from logical name → DataFrame for every static file.
        """
        if model_id in self.static_cache:
            return self.static_cache[model_id]

        data: Dict[str, pd.DataFrame] = {}

        for key, filename in _STATIC_FILES.items():
            # Try mode-specific directory first, then model root
            fpath_mode = os.path.join(model_path, filename)
            fpath_root = os.path.join(model_root_path, filename)

            if os.path.isfile(fpath_mode):
                data[key] = pd.read_csv(fpath_mode)
            elif os.path.isfile(fpath_root):
                data[key] = pd.read_csv(fpath_root)
            else:
                warnings.warn(
                    f"Static file missing for Model_{model_id}: "
                    f"checked {fpath_mode} and {fpath_root}"
                )
                data[key] = pd.DataFrame()  # empty sentinel

        self.static_cache[model_id] = data
        return data

    # ------------------------------------------------------------------ #
    #  Dynamic data loading
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_dynamic_data(event_path: str) -> Dict[str, pd.DataFrame]:
        """Read all dynamic CSVs for a single event.

        Optional files that are missing are silently set to empty
        DataFrames; required files that are missing trigger a warning.

        Parameters
        ----------
        event_path : str
            Absolute path to ``event_{id}/``.

        Returns
        -------
        dict[str, pd.DataFrame]
        """
        data: Dict[str, pd.DataFrame] = {}

        for key, filename in _DYNAMIC_FILES.items():
            fpath = os.path.join(event_path, filename)

            if os.path.isfile(fpath):
                data[key] = pd.read_csv(fpath)
            else:
                if key not in _OPTIONAL_DYNAMIC:
                    warnings.warn(
                        f"Expected dynamic file missing: {fpath}"
                    )
                data[key] = pd.DataFrame()  # empty sentinel

        return data

    # ------------------------------------------------------------------ #
    #  torch Dataset interface
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a sample dictionary for the event at *idx*.

        The dictionary contains **raw DataFrames** (no tensor conversion).
        Downstream consumers (e.g. ``GraphBuilder``) are responsible for
        converting to PyG ``Data`` objects.

        Keys
        ----
        model_id, event_id : str
            Identifiers for the urban model and rainfall event.
        static_1d_nodes, static_2d_nodes : pd.DataFrame
            Node-level static attributes for 1D and 2D domains.
        static_1d_edges, static_2d_edges : pd.DataFrame
            Edge-level static attributes.
        edge_index_1d, edge_index_2d : pd.DataFrame
            Connectivity (source → target) for each domain.
        1d2d_conn : pd.DataFrame
            Coupling between the 1D manholes and 2D surface nodes.
        dynamic_1d_nodes, dynamic_2d_nodes : pd.DataFrame
            Time-varying node states (water level, rainfall, …).
        dynamic_1d_edges, dynamic_2d_edges : pd.DataFrame
            Time-varying edge states (flow); may be empty if file is
            absent.
        timesteps : pd.DataFrame
            Simulation time-step metadata.
        """
        if idx < 0 or idx >= len(self.events):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.events)}"
            )

        meta = self.events[idx]

        # --- Static (from cache or first load) --------------------------
        static = self._load_static_data(
            meta["model_id"], meta["model_path"], meta["model_root_path"]
        )

        # --- Dynamic (always loaded fresh per event) --------------------
        dynamic = self._load_dynamic_data(meta["event_path"])

        # --- Assemble sample dict ----------------------------------------
        sample: Dict[str, Any] = {
            "model_id": meta["model_id"],
            "event_id": meta["event_id"],
            # Static data (shared references — do NOT mutate in-place)
            **static,
            # Dynamic data
            **dynamic,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------ #
    #  Node Standard Deviations (for loss function)
    # ------------------------------------------------------------------ #

    def compute_node_stds(
        self,
        model_id: Optional[str] = None,
        node_id_col: str = "node_idx",
        wl_col_1d: str = "water_level",
        wl_col_2d: str = "water_level",
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Compute per-node standard deviations of target water levels.

        These are required by ``standardized_rmse_loss`` (Task 2.3).
        The stds are computed **across all training events** for each
        unique model, separately for 1D and 2D node types.

        The dynamic CSVs are in **long format**:
        ``(timestep, node_idx, water_level, ...)``.  This method pivots
        by ``node_idx`` and computes the std across time and events for
        each node independently.

        Parameters
        ----------
        model_id : str or None
            If given, compute only for this model.  Otherwise compute
            for every model present in the dataset.
        node_id_col : str
            Column name identifying the node index in the dynamic CSV.
        wl_col_1d : str
            Column name for the 1D water-level target.
        wl_col_2d : str
            Column name for the 2D water-level target.

        Returns
        -------
        dict[str, dict[str, np.ndarray]]
            ``{model_id: {"1d": array(N_1d,), "2d": array(N_2d,)}}``
            Arrays are sorted by ascending ``node_idx``.

        Notes
        -----
        This iterates over all events (loading dynamic CSVs) so it is
        expensive.  Call once at startup and cache the result.
        """
        target_models = (
            [model_id] if model_id else self.get_model_ids()
        )

        result: Dict[str, Dict[str, np.ndarray]] = {}

        for mid in target_models:
            events_for_model = [
                e for e in self.events if e["model_id"] == mid
            ]

            accum_1d: List[pd.DataFrame] = []
            accum_2d: List[pd.DataFrame] = []

            for ev in events_for_model:
                dyn = self._load_dynamic_data(ev["event_path"])

                # 1D node water levels (long format)
                df_1d = dyn.get("dynamic_1d_nodes")
                if (
                    df_1d is not None
                    and not df_1d.empty
                    and node_id_col in df_1d.columns
                    and wl_col_1d in df_1d.columns
                ):
                    accum_1d.append(df_1d[[node_id_col, wl_col_1d]])

                # 2D node water levels (long format)
                df_2d = dyn.get("dynamic_2d_nodes")
                if (
                    df_2d is not None
                    and not df_2d.empty
                    and node_id_col in df_2d.columns
                    and wl_col_2d in df_2d.columns
                ):
                    accum_2d.append(df_2d[[node_id_col, wl_col_2d]])

            stds: Dict[str, np.ndarray] = {}

            if accum_1d:
                combined = pd.concat(accum_1d, axis=0, ignore_index=True)
                per_node = (
                    combined
                    .groupby(node_id_col)[wl_col_1d]
                    .std()
                    .sort_index()
                )
                stds["1d"] = per_node.values.astype(np.float32)
            else:
                stds["1d"] = np.array([], dtype=np.float32)

            if accum_2d:
                combined = pd.concat(accum_2d, axis=0, ignore_index=True)
                per_node = (
                    combined
                    .groupby(node_id_col)[wl_col_2d]
                    .std()
                    .sort_index()
                )
                stds["2d"] = per_node.values.astype(np.float32)
            else:
                stds["2d"] = np.array([], dtype=np.float32)

            result[mid] = stds

        return result

    # ------------------------------------------------------------------ #
    #  Event-Based Cross-Validation Splits
    # ------------------------------------------------------------------ #

    def split_by_event(
        self,
        val_event_id: str,
        model_id: Optional[str] = None,
    ) -> Tuple["FloodDataset", "FloodDataset"]:
        """Split into train / val subsets using Leave-One-Event-Out.

        Parameters
        ----------
        val_event_id : str
            The event ID to hold out for validation.
        model_id : str or None
            If given, only include events from this model.

        Returns
        -------
        (train_ds, val_ds) : tuple[FloodDataset, FloodDataset]

        Example
        -------
        >>> ds = FloodDataset(root, mode="train")
        >>> train_ds, val_ds = ds.split_by_event("04", model_id="1")
        """
        train_ds = self._shallow_copy()
        val_ds = self._shallow_copy()

        source = self.events
        if model_id is not None:
            source = [e for e in source if e["model_id"] == model_id]

        train_ds.events = [
            e for e in source if e["event_id"] != val_event_id
        ]
        val_ds.events = [
            e for e in source if e["event_id"] == val_event_id
        ]

        if not val_ds.events:
            avail = sorted({e["event_id"] for e in source})
            raise ValueError(
                f"event_id '{val_event_id}' not found. "
                f"Available: {avail}"
            )

        return train_ds, val_ds

    # ------------------------------------------------------------------ #
    #  Convenience helpers
    # ------------------------------------------------------------------ #

    def _shallow_copy(self) -> "FloodDataset":
        """Create a shallow copy sharing the static cache."""
        clone = FloodDataset.__new__(FloodDataset)
        clone.root_dir = self.root_dir
        clone.mode = self.mode
        clone.transform = self.transform
        clone.static_cache = self.static_cache  # share the cache
        clone.events = list(self.events)  # independent list, shared dicts
        return clone

    def get_model_ids(self) -> List[str]:
        """Return a sorted list of unique model IDs in this dataset."""
        return sorted({e["model_id"] for e in self.events})

    def get_event_ids(self, model_id: Optional[str] = None) -> List[str]:
        """Return a sorted list of unique event IDs.

        Parameters
        ----------
        model_id : str or None
            If given, only return events for this model.
        """
        events = self.events
        if model_id is not None:
            events = [e for e in events if e["model_id"] == model_id]
        return sorted({e["event_id"] for e in events})

    def filter_by_model(self, model_id: str) -> "FloodDataset":
        """Return a shallow copy that only contains events for *model_id*.

        Useful for per-model training loops or cross-validation splits.
        """
        subset = self._shallow_copy()
        subset.events = [e for e in self.events if e["model_id"] == model_id]
        return subset

    @staticmethod
    def collate_fn(
        batch: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Custom collate that simply returns the first (and only) sample.

        This dataset returns heterogeneous dictionaries of DataFrames
        that cannot be stacked by PyTorch's default collator.
        Use with ``DataLoader(ds, batch_size=1, collate_fn=FloodDataset.collate_fn)``.
        """
        if len(batch) != 1:
            raise ValueError(
                "FloodDataset returns DataFrames that cannot be batched. "
                "Use batch_size=1 with collate_fn=FloodDataset.collate_fn."
            )
        return batch[0]

    def __repr__(self) -> str:
        return (
            f"FloodDataset(root_dir='{self.root_dir}', mode='{self.mode}', "
            f"events={len(self.events)})"
        )

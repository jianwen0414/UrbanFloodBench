# Urban Flood Modelling — Kaggle Competition

> **Team Strategy: Twin-Engine Hydra**
> Decoupled 1D pipe-network GNN and 2D surface-mesh GNN, trained and
> inferred independently, following the organiser's guidance:
> *"Separate gets better results"* & *"Simple data is better."*

---

## 1. Competition Overview

Predict **water levels** autoregressively in a coupled urban drainage
system:

| Domain | Physics | Nodes | Edges |
|--------|---------|-------|-------|
| **1D** | Underground pipes — fast, pressurized / gravity flow | Manholes, Junctions, Inlets | Pipes (bidirectional) |
| **2D** | Surface terrain — slow, diffusive spread | Mesh cells (triangles) | Adjacency between cells |

**Dataset:** Two urban models (`Model_1`, `Model_2`) with dozens of
rainfall events each. Test events provide 10 "warm-up" steps (full
ground truth) then require autoregressive prediction for all remaining
steps using only rainfall as input.

**Deadline:** 1 March 2026

---

## 2. Environment Setup

### Prerequisites

- Python ≥ 3.10
- Git

### Step-by-step

```bash
# 1. Clone / open the project
cd urban-flood-challenge

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
#    Windows PowerShell:
.venv\Scripts\activate
#    Linux / macOS / WSL:
source .venv/bin/activate

# 4. Install all dependencies
pip install -r requirements.txt
```

> **PyTorch Geometric note:** If `torch-geometric` fails to install,
> install PyTorch first, then use the PyG wheel index:
> ```bash
> pip install torch
> pip install torch-geometric -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
> ```
> Replace `2.5.0+cpu` with your actual torch version / CUDA variant.

### Data path configuration

Set `FLOOD_DATA_PATH` in **`src/.env`** to the **local filesystem
directory** that contains the `Models/` folder:

```dotenv
# src/.env
FLOOD_DATA_PATH=C:\Users\YourName\data\urban-flood
```

| Platform | Example |
|----------|---------|
| Windows (manual download) | `C:\Users\You\data\urban-flood` |
| Google Drive for Desktop | `G:\My Drive\UrbanFloodProject` |
| Google Colab (default) | `/content/drive/MyDrive/UrbanFloodProject` |

> **Important:** This must be a local path, *not* a Google Drive URL.
> `config.py` will warn you if it detects a URL by mistake.

---

## 3. Project Structure

```
urban-flood-challenge/
├── .venv/                     # Virtual environment (git-ignored)
├── data/                      # Raw data (contains Models/)
│   └── Models/
│       ├── Model_1/
│       │   ├── train/
│       │   │   ├── 1d_nodes_static.csv
│       │   │   ├── 2d_nodes_static.csv
│       │   │   ├── 1d_edges_static.csv
│       │   │   ├── 2d_edges_static.csv
│       │   │   ├── 1d_edge_index.csv
│       │   │   ├── 2d_edge_index.csv
│       │   │   ├── 1d2d_connections.csv
│       │   │   └── event_{id}/
│       │   │       ├── 1d_nodes_dynamic_all.csv
│       │   │       ├── 2d_nodes_dynamic_all.csv
│       │   │       ├── 1d_edges_dynamic_all.csv
│       │   │       ├── 2d_edges_dynamic_all.csv
│       │   │       └── timesteps.csv
│       │   └── test/   (same structure)
│       └── Model_2/    (same structure)
├── src/                       # Core Python package
│   ├── __init__.py
│   ├── .env                   # FLOOD_DATA_PATH (local filesystem path)
│   ├── config.py              # Loads .env, exposes RAW_DATA_PATH
│   ├── dataset.py             # FloodDataset — Universal Lazy Loader
│   ├── graph_builder.py       # CSV → PyG graph construction
│   ├── model_1d.py            # Engine A (GCN-GRU for pipes)
│   ├── model_2d.py            # Engine B (GraphSAGE-GRU for surface)
│   ├── loss.py                # Standardized RMSE loss
│   └── baseline_xgb.py       # Tier 0 XGBoost benchmark
├── tests/
│   └── test_dataset.py        # Smoke tests for the data loader
├── experiments/               # Jupyter notebooks
│   ├── member_a_playground.ipynb
│   ├── member_b_playground.ipynb
│   └── validation_run.ipynb
├── requirements.txt
├── PROJECT_BIBLE.md           # Physics, strategy, dataset encyclopedia
├── IMPLEMENTATION_PLAN.md     # Execution plan & task distribution
└── README.md
```

---

## 4. Dataset Schema (Actual Column Names)

All dynamic CSVs are in **long format**: one row per `(timestep, node_idx)` pair.

### Static files (one per model, shared across events)

| File | Key columns |
|------|------------|
| `1d_nodes_static.csv` | `node_idx`, `position_x`, `position_y`, `depth`, `invert_elevation`, `surface_elevation`, `base_area` |
| `2d_nodes_static.csv` | `node_idx`, `position_x`, `position_y`, `area`, `roughness`, `min_elevation`, `elevation`, `aspect`, `curvature`, `flow_accumulation` |
| `1d_edges_static.csv` | `edge_idx`, `relative_position_x`, `relative_position_y`, `length`, `diameter`, `shape`, `roughness`, `slope` |
| `2d_edges_static.csv` | `edge_idx`, `relative_position_x`, `relative_position_y`, `face_length`, `length`, `slope` |
| `1d_edge_index.csv` | `edge_idx`, `from_node`, `to_node` |
| `2d_edge_index.csv` | `edge_idx`, `from_node`, `to_node` |
| `1d2d_connections.csv` | `connection_idx`, `node_1d`, `node_2d` |

### Dynamic files (one per event)

| File | Key columns | Notes |
|------|------------|-------|
| `1d_nodes_dynamic_all.csv` | `timestep`, `node_idx`, **`water_level`** (target), `inlet_flow` | Long format |
| `2d_nodes_dynamic_all.csv` | `timestep`, `node_idx`, `rainfall`, **`water_level`** (target), `water_volume` | Long format |
| `1d_edges_dynamic_all.csv` | `timestep`, `edge_idx`, `flow`, `velocity` | Auxiliary |
| `2d_edges_dynamic_all.csv` | `timestep`, `edge_idx`, `flow`, `velocity` | Auxiliary |
| `timesteps.csv` | `timestep_idx`, `timestamp` | Metadata |

### Critical physics derivations

```
Capacity       = surface_elevation − invert_elevation
Relative Depth = water_level − invert_elevation
```

---

## 5. The Data Pipeline — `FloodDataset`

`src/dataset.py` implements the **Universal Lazy Loader** — the single
entry-point for all downstream modules to access the competition data.

### Architecture

```
                    ┌─────────────────────────────────────┐
                    │          FloodDataset                │
  src/.env ──────►  │  root_dir, mode="train"/"test"      │
  (FLOOD_DATA_PATH) │                                     │
                    │  _discover_events()                  │
                    │    → self.events = [{model_id,       │
                    │       event_id, model_path, ...}]    │
                    │                                     │
                    │  __getitem__(idx)                    │
                    │    → static (cached per model)       │
                    │    → dynamic (loaded fresh per event)│
                    │    → returns Dict[str, DataFrame]    │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │        graph_builder.py              │
                    │  build_1d_graph(sample) → PyG Data   │
                    │  build_2d_graph(sample) → PyG Data   │
                    │  build_unified_graph(sample)         │
                    │          → HeteroData                │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  model_1d.py / model_2d.py           │
                    │  (Training & Inference)              │
                    └─────────────────────────────────────┘
```

### Key design decisions

| Decision | Why |
|----------|-----|
| **Lazy loading** | Dynamic CSVs loaded on `__getitem__`, not at init — keeps peak RAM low across 137+ events |
| **Static caching** | Static files read once per model and reused — zero duplicate I/O or memory |
| **Separation of concerns** | `dataset.py` handles I/O only; `graph_builder.py` handles topology + feature engineering |
| **`compute_node_stds()`** | Pre-computes per-node $\sigma_i$ across all training events — feeds directly into `standardized_rmse_loss` |
| **`split_by_event()`** | Leave-One-Event-Out CV — prevents temporal data leakage |
| **`collate_fn`** | Custom collator enforcing `batch_size=1` — dictionaries of DataFrames can't be stacked |

### Quick usage

```python
from src.config import RAW_DATA_PATH
from src.dataset import FloodDataset

# ── 1. Load the dataset ─────────────────────────────
ds = FloodDataset(RAW_DATA_PATH, mode="train")
print(ds)  # FloodDataset(..., events=137)

# ── 2. Grab a single event ──────────────────────────
sample = ds[0]
# sample keys: model_id, event_id, static_1d_nodes,
#              dynamic_1d_nodes, edge_index_1d, ...

# ── 3. Filter by model ──────────────────────────────
ds_m1 = ds.filter_by_model("1")     # 68 events
ds_m2 = ds.filter_by_model("2")     # 69 events

# ── 4. Leave-One-Event-Out split ────────────────────
train_ds, val_ds = ds.split_by_event("96", model_id="1")
# train: 67 events | val: 1 event — zero leakage

# ── 5. Compute per-node stds for the loss function ──
stds = ds.compute_node_stds(model_id="1")
# stds["1"]["1d"].shape == (17,)  — one σ per 1D node
# stds["1"]["2d"].shape == (3716,) — one σ per 2D node

# ── 6. Use with DataLoader ──────────────────────────
from torch.utils.data import DataLoader

loader = DataLoader(
    ds,
    batch_size=1,                          # MUST be 1
    collate_fn=FloodDataset.collate_fn,    # REQUIRED
    shuffle=True,
)
for sample in loader:
    # sample is a plain dict — pass to graph_builder
    pass
```

---

## 6. Evaluation Metric — Standardized RMSE

$$
\text{SRMSE} = \frac{1}{N} \sum_{i=1}^{N}
\frac{\sqrt{\frac{1}{T}\sum_{t=1}^{T}(y_{it} - \hat{y}_{it})^2}}
{\sigma_i}
$$

Because the metric weights every node equally *after* normalising by
its $\sigma_i$, dry nodes (σ ≈ 0) can dominate the score. The training
loss in `src/loss.py` handles this via clamped inverse-variance weights:

```python
from src.loss import standardized_rmse_loss

# node_stds: Tensor of shape (N,) from compute_node_stds()
loss = standardized_rmse_loss(pred, target, node_stds, clamp_weights=100.0)
```

---

## 7. Testing the Loader

```bash
# From the project root, with venv activated:
python -m tests.test_dataset
```

The test suite runs **10 checks** covering:

| # | Test | Validates |
|---|------|-----------|
| 1 | Event Discovery | Constructor finds all Model/Event folders |
| 2 | ID Accessors | `get_model_ids()`, `get_event_ids()` return correct values |
| 3 | `__getitem__` | All 14 expected keys present with valid DataFrames |
| 4 | Static Caching | Repeated access returns the *same object* (no duplication) |
| 5 | `filter_by_model()` | Correctly isolates events for one model |
| 6 | `split_by_event()` | Train/Val split with zero leakage |
| 7 | `compute_node_stds()` | Returns one σ per node (not a scalar), all finite |
| 8 | DataLoader | `collate_fn` works with `batch_size=1` |
| 9 | Physics Sanity | `Capacity = surface_elevation − invert_elevation ≥ 0` |
| 10 | Edge Cases | `IndexError` on out-of-bounds access |

If no real data is present, the test auto-generates a synthetic
directory tree to exercise every code path.

---

## 8. Next Steps — Graph Building (Phase 1, Tasks 1.2–1.4)

With the data loader verified, the next milestone is **`src/graph_builder.py`** — converting the raw DataFrames from `FloodDataset` into PyTorch Geometric `Data` objects.

### What needs to happen

| Task | Owner | Function | Key requirements |
|------|-------|----------|-----------------|
| **1.2** | Member A | `build_1d_graph(sample)` | Bidirectional edges (`from_node ↔ to_node`); features: `Relative Depth`, `Capacity`, `Rain` |
| **1.3** | Member B | `build_2d_graph(sample)` | Soft coupling: compute Euclidean distance from every 2D node to nearest 1D node (`1d2d_connections.csv`); features: Z-scored elevation, roughness, rainfall, `dist_to_drain` |
| **1.4** | A & B | `build_unified_graph(sample)` | `HeteroData` with explicit 1D↔2D edges for the fallback Unified Engine |

### How the graph builder consumes FloodDataset

```python
from src.dataset import FloodDataset
from src.graph_builder import build_1d_graph, build_2d_graph

ds = FloodDataset(RAW_DATA_PATH, mode="train")
sample = ds[0]

# Graph builder receives the full sample dict:
graph_1d = build_1d_graph(sample)
# graph_1d.x          → node features [N_1d, F]
# graph_1d.edge_index → [2, E_1d] (bidirectional)
# graph_1d.y          → water_level targets [T, N_1d]

graph_2d = build_2d_graph(sample)
# graph_2d.x          → node features [N_2d, F]
# graph_2d.edge_index → [2, E_2d]
# graph_2d.y          → water_level targets [T, N_2d]
```

### Data the graph builder has access to (from `sample` dict)

**For 1D graph:**
- `sample["static_1d_nodes"]` — `node_idx, depth, invert_elevation, surface_elevation, base_area`
- `sample["dynamic_1d_nodes"]` — `timestep, node_idx, water_level, inlet_flow` (long format)
- `sample["edge_index_1d"]` — `from_node, to_node` (must be made bidirectional)
- `sample["static_1d_edges"]` — `diameter, roughness, slope, length`

**For 2D graph:**
- `sample["static_2d_nodes"]` — `node_idx, area, roughness, min_elevation, elevation, aspect, curvature, flow_accumulation`
- `sample["dynamic_2d_nodes"]` — `timestep, node_idx, rainfall, water_level, water_volume` (long format)
- `sample["edge_index_2d"]` — `from_node, to_node`
- `sample["1d2d_conn"]` — `node_1d, node_2d` (for computing `dist_to_drain`)

---

## 9. Strategy Summary — Twin-Engine Hydra

| Tier | Model | Architecture | Purpose |
|------|-------|-------------|---------|
| 0 | XGBoost | Tabular regression | Baseline "Score to Beat" |
| 1A | Pipe Engine | GCN-GRU | 1D underground pipe network |
| 1B | Surface Engine | GraphSAGE-GRU | 2D surface terrain mesh |
| 2 | Unified Engine | HeteroGNN-GRU | Fallback for surcharge events |

---

## 10. Team Responsibilities

| Member | Role | Modules |
|--------|------|---------|
| A | 1D Model Engineer | `model_1d.py`, `graph_builder.py` (1D parts) |
| B | 2D Model Engineer | `model_2d.py`, `graph_builder.py` (2D parts) |
| C (Lead Architect) | Infrastructure & Pipeline | `dataset.py`, `loss.py`, `config.py`, validation, submission |

---

*Last updated: 2026-02-07*

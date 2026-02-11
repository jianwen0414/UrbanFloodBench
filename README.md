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
# 1. Clone the repo
git clone https://github.com/jianwen0414/UrbanFloodBench.git
cd UrbanFloodBench

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
UrbanFloodBench/                   # ← repo root
├── .venv/                         # Virtual environment (git-ignored)
├── data/                          # Raw data (git-ignored, contains Models/)
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
├── src/                           # Core Python package
│   ├── __init__.py
│   ├── .env                       # FLOOD_DATA_PATH (git-ignored)
│   ├── config.py                  # Loads .env, exposes RAW_DATA_PATH
│   ├── dataset.py                 # FloodDataset — Universal Lazy Loader
│   ├── graph_builder.py           # Re-export shim (backward compat)
│   ├── graph_builder_1d.py        # 1D pipe graph (Member A)
│   ├── graph_builder_2d.py        # 2D surface mesh graph (Member B)
│   ├── graph_builder_unified.py   # Coupled HeteroData graph (Member C)
│   ├── model_1d.py                # Engine A — GCN-GRU (Member A)
│   ├── model_2d.py                # Engine B — GraphSAGE-GRU (Member B)
│   ├── model_unified.py           # Engine C — HeteroGNN-GRU (Member C)
│   ├── loss.py                    # SRMSE loss suite (§6)
│   ├── trainer.py                 # Training pipeline (§7)
│   ├── validate.py                # Validation pipeline (§8)
│   ├── inference.py               # Submission generator (§9)
│   └── baseline_xgb.py           # Tier 0 XGBoost benchmark
├── tests/
│   ├── test_dataset.py            # Smoke tests for the data loader
│   ├── test_loss.py               # Loss functions & accumulator
│   ├── test_unified_graph.py      # Graph builder validation
│   └── test_unified_model.py      # Model forward/rollout tests
├── experiments/                   # Jupyter notebooks
│   ├── member_a_playground.ipynb
│   ├── member_b_playground.ipynb
│   ├── member_c_playground.ipynb  # Full pipeline walkthrough
│   └── validation_run.ipynb       # LOEO cross-validation
├── train_production.py            # Standalone production training (§7)
├── .gitignore
├── requirements.txt
├── PROJECT_BIBLE.md               # Physics, strategy, dataset encyclopedia
├── IMPLEMENTATION_PLAN.md         # Execution plan & task distribution
├── DOCUMENTATION.md               # Detailed API docs for each module
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

## 5. Core Modules

Detailed API documentation, architecture diagrams, code examples, and
parameter references for each module are in
[DOCUMENTATION.md](DOCUMENTATION.md).

| Module | Purpose | Key entry-point |
|--------|---------|------------------|
| `dataset.py` | Universal lazy loader for all competition data | `FloodDataset(root, mode)` |
| `loss.py` | SRMSE loss suite matching the leaderboard metric | `FloodLoss`, `SRMSEAccumulator` |
| `trainer.py` | Full training lifecycle with curriculum learning | `UnifiedTrainer(cfg).train()` |
| `validate.py` | Leaderboard-mirroring validation pipeline | `ValidationRunner.validate_holdout()` |
| `inference.py` | Submission CSV generation & sanity checks | `SubmissionGenerator.from_checkpoint()` |
| `graph_builder_*.py` | Per-engine graph construction from raw CSVs | `build_1d_graph()`, `build_2d_graph()`, `build_unified_graph()` |

---

## 6. Testing

```bash
# From the project root, with venv activated:
python -m tests.test_dataset
python -m tests.test_loss
python -m tests.test_unified_graph
python -m tests.test_unified_model
```

### Test suite summary

| File | Tests | Validates |
|------|-------|-----------|
| `test_dataset.py` | 10 | Event discovery, `__getitem__`, caching, filtering, splitting, stds, DataLoader, physics sanity, edge cases |
| `test_loss.py` | — | Loss functions, metrics, accumulator, push-forward loss, node-type balancing |
| `test_unified_graph.py` | — | Graph builder produces valid `HeteroData`, correct edge types, feature dimensions |
| `test_unified_model.py` | — | Model forward pass, rollout, push-forward rollout, hidden state shapes |

If no real data is present, the test auto-generates a synthetic
directory tree to exercise every code path.

---

## 7. Strategy Summary — Twin-Engine Hydra

| Tier | Model | Architecture | Purpose |
|------|-------|-------------|---------|
| 0 | XGBoost | Tabular regression | Baseline "Score to Beat" |
| 1A | Pipe Engine | GCN-GRU | 1D underground pipe network |
| 1B | Surface Engine | GraphSAGE-GRU | 2D surface terrain mesh |
| 2 | Unified Engine | HeteroGNN-GRU | Fallback for surcharge events |

---

## 8. Team Responsibilities

| Member | Role | Modules | Branch |
|--------|------|---------|--------|
| A | 1D Model Engineer | `graph_builder_1d.py`, `model_1d.py` | `feat/1d-pipeline` |
| B | 2D Model Engineer | `graph_builder_2d.py`, `model_2d.py` | `feat/2d-pipeline` |
| C (Lead Architect) | Infrastructure & Pipeline | `dataset.py`, `loss.py`, `config.py`, `graph_builder_unified.py`, `model_unified.py`, `trainer.py`, `validate.py`, `inference.py` | `feat/unified` |

### Shared infrastructure you can import directly

Members A & B: these modules are **ready for use** in your decoupled engines. You do **not** need to rewrite any of them.

| Module | What you get | Import example |
|--------|-------------|----------------|
| `loss.py` | `FloodLoss`, `standardized_rmse_metric`, `SRMSEAccumulator`, `per_node_loss_breakdown` | `from src.loss import FloodLoss` |
| `trainer.py` | `TrainConfig`, `TeacherForcingScheduler`, `EarlyStopping`, `TrainingHistory`, `save_checkpoint`, `load_checkpoint` | `from src.trainer import TrainConfig, EarlyStopping` |
| `validate.py` | `ValidationResult`, `extract_predictions`, `check_leaderboard_correlation` | `from src.validate import extract_predictions` |
| `inference.py` | `_build_submission_rows_vectorized`, `ensemble_predict_event` | `from src.inference import _build_submission_rows_vectorized` |
| `dataset.py` | `FloodDataset` (data loading, splitting, σ computation) | `from src.dataset import FloodDataset` |
| `config.py` | `RAW_DATA_PATH`, `PROJECT_ROOT` | `from src.config import RAW_DATA_PATH` |

### Git workflow

```
main                      ← protected, always passes tests
├── feat/1d-pipeline      ← Member A
├── feat/2d-pipeline      ← Member B
└── feat/unified          ← Member C
```

**Rules:**
1. Never edit another member's owned files on your branch.
2. Rebase from `main` often: `git pull --rebase origin main`.
3. PR into `main` — each PR must pass `python -m tests.test_dataset`.
4. Shared infrastructure (`dataset.py`, `config.py`, `loss.py`, `trainer.py`, `validate.py`, `inference.py`) lives on `main` — only Member C merges to it.

---

*Last updated: 2026-02-11*

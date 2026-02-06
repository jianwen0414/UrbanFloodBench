# Urban Flood Modelling — Kaggle Competition

> **Team Approach: Twin-Engine Strategy**
> Decoupled 1D pipe-network and 2D surface-mesh models, trained and
> inferred independently, following the organiser's guidance to separate
> physics domains.

---

## 1. Competition Overview

The goal is to predict **water levels** in a coupled urban drainage
system comprising:

| Domain | Description |
|--------|-------------|
| **1D** | Underground pipe network (manholes + conduits) |
| **2D** | Surface terrain mesh (triangular elements) |

Raw data is hosted on Google Drive:
<https://drive.google.com/drive/folders/1keFslckgCkEq-OuySFbWPCYJrjnW-iwB>

### Folder layout

```
Models/
└── Model_{id}/
    └── train/
        ├── 1d_nodes_static.csv      (static, per-model)
        ├── 2d_nodes_static.csv      (static, per-model)
        └── event_{id}/
            ├── 1d_nodes_dynamic.csv  (time-series)
            ├── 2d_nodes_dynamic.csv  (time-series)
            └── rainfall.csv
```

---

## 2. Strategy — "Twin-Engine"

We decouple the problem into **two independent graph-learning engines**
and combine their predictions at submission time.

### Engine A — 1D Pipe Network

* **Architecture:** Bidirectional GRU encoder + GCN message-passing
  layers operating on a **bidirectional graph** (each pipe is
  represented as two directed edges).
* **Input features:** Dynamic hydraulic states (flow, velocity, depth)
  + static node attributes from `1d_nodes_static.csv`.
* **Target:** Water level at each manhole node.

### Engine B — 2D Surface Mesh

* **Architecture:** GraphSAGE with neighbourhood sampling on the
  triangular surface mesh.
* **Key static feature:** **Distance to Drain** — Euclidean distance
  from each 2D node to the nearest manhole, providing the model with a
  physics-informed spatial prior for drainage behaviour.
* **Input features:** Dynamic surface water depths + static mesh
  properties from `2d_nodes_static.csv`.
* **Target:** Water level at each surface node.

---

## 3. Evaluation Metric — Standardized RMSE

The competition uses a **variance-weighted RMSE** (Standardized RMSE):

$$
\text{SRMSE} = \frac{1}{N} \sum_{i=1}^{N}
\frac{\sqrt{\frac{1}{T}\sum_{t=1}^{T}(y_{it} - \hat{y}_{it})^2}}
{\sigma_i}
$$

where $\sigma_i$ is the standard deviation of the ground-truth water
level at node $i$ across all time steps.  Nodes with higher variance
contribute equally to those with lower variance, preventing the score
from being dominated by a few high-variance locations.

The implementation lives in **`src/loss.py`**.

---

## 4. Data Pipeline

```
Raw CSVs (Google Drive)
        │
        ▼
┌────────────────────┐
│  FloodDataset      │  src/dataset.py
│  (Lazy Loader)     │
│  • Discovers Model/│
│    Event folders   │
│  • Lazy-loads      │
│    dynamic CSVs    │
│  • Caches static   │
│    files in RAM    │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  graph_builder     │  src/graph_builder.py
│  • Builds PyG Data │
│    objects          │
│  • 1D → bidir.     │
│    pipe graph       │
│  • 2D → mesh graph │
│    + dist-to-drain  │
└────────────────────┘
```

### Key design decisions

1. **Lazy loading** — Dynamic CSV files are read on `__getitem__`, not
   at init time, keeping peak RAM usage low.
2. **Static caching** — Static node files are read once per model and
   held in a dictionary keyed by `model_id`; subsequent events under the
   same model re-use the cached DataFrame.
3. **Separation of concerns** — `dataset.py` handles I/O only;
   `graph_builder.py` handles topology + feature engineering.

---

## 5. Project Structure

```
urban-flood-challenge/
├── data/                  # Raw input data (symlink or mount)
├── src/                   # Core Python package
│   ├── __init__.py
│   ├── config.py          # RAW_DATA_PATH + constants
│   ├── dataset.py         # FloodDataset — Universal Lazy Loader
│   ├── graph_builder.py   # CSV → PyG graph construction
│   ├── model_1d.py        # Engine A (GRU + GCN)
│   ├── model_2d.py        # Engine B (GraphSAGE)
│   └── loss.py            # Standardized RMSE
├── experiments/           # Jupyter notebooks
│   ├── member_a_playground.ipynb
│   ├── member_b_playground.ipynb
│   └── validation_run.ipynb
└── README.md
```

---

## 6. Quick Start

```bash
# 1. Clone / open the project
cd urban-flood-challenge

# 2. (Colab) Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Override data path if needed
export FLOOD_DATA_PATH=/path/to/local/data   # or set in src/config.py

# 4. Install dependencies
pip install torch torch-geometric pandas numpy scikit-learn

# 5. Run a validation notebook
jupyter notebook experiments/validation_run.ipynb
```

---

## 7. Team Responsibilities

| Member | Role | Module |
|--------|------|--------|
| A | 1D Model Engineer | `model_1d.py`, `graph_builder.py` (1D) |
| B | 2D Model Engineer | `model_2d.py`, `graph_builder.py` (2D) |
| C (Lead ML Eng.) | Infrastructure & Pipeline | `dataset.py`, `loss.py`, `config.py` |

---

*Last updated: 2026-02-06*

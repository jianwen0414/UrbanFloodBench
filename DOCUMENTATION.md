# UrbanFloodBench — Detailed Module Documentation

> This file contains the in-depth API references and usage guides for
> each core module. For project overview, setup, and structure, see
> [README.md](README.md).

---

## Table of Contents

1. [FloodDataset — Data Pipeline (`dataset.py`)](#1-flooddataset--data-pipeline)
2. [Loss Functions (`loss.py`)](#2-evaluation-metric--loss-functions)
3. [Training Pipeline (`trainer.py`)](#3-training-pipeline)
4. [Validation Pipeline (`validate.py`)](#4-validation-pipeline)
5. [Submission Generation (`inference.py`)](#5-submission-generation)
6. [Graph Building (`graph_builder_*.py`)](#6-graph-building)

---

## 1. FloodDataset — Data Pipeline

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
                    │  graph_builder_1d.py    (Member A)    │
                    │    build_1d_graph(sample) → Data      │
                    │  graph_builder_2d.py    (Member B)    │
                    │    build_2d_graph(sample) → Data      │
                    │  graph_builder_unified.py (Member C)  │
                    │    build_unified_graph() → HeteroData │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  model_1d.py      (Engine A)         │
                    │  model_2d.py      (Engine B)         │
                    │  model_unified.py (Engine C / Tier 2)│
                    └─────────────────────────────────────┘
```

### Key design decisions

| Decision | Why |
|----------|-----|
| **Lazy loading** | Dynamic CSVs loaded on `__getitem__`, not at init — keeps peak RAM low across 137+ events |
| **Static caching** | Static files read once per model and reused — zero duplicate I/O or memory |
| **Separation of concerns** | `dataset.py` handles I/O only; `graph_builder_*.py` modules handle topology + feature engineering |
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

## 2. Evaluation Metric & Loss Functions

`src/loss.py` provides the full loss-function suite engineered to
mirror the competition's Standardized RMSE while remaining differentiable
for training.

### The Competition Metric

$$
\text{SRMSE} = \text{Mean}_{\text{models}}\!\left(
  \text{Mean}_{\text{events}}\!\left(
    \text{Mean}_{\text{node\_types}}\!\left(
      \text{Mean}_{\text{nodes}}\!\left(
        \frac{\text{RMSE}_i}{\sigma_i}
      \right)
    \right)
  \right)
\right)
$$

Because the metric weights every node equally *after* normalising by
its $\sigma_i$, dry nodes ($\sigma \approx 0$) can dominate the score.

### Architecture overview

```
src/loss.py
├── standardized_rmse_loss()       ← Training surrogate (differentiable)
├── standardized_huber_loss()      ← Outlier-robust variant
├── push_forward_loss()            ← K-step trajectory loss (anti-drift)
├── combined_flood_loss()          ← 1D/2D node-type balancer (50/50)
├── FloodLoss (nn.Module)          ← Stores σ as buffer, clean .forward()
├── standardized_rmse_metric()     ← Exact leaderboard metric (non-diff)
├── SRMSEAccumulator               ← Stateful hierarchical scorer
├── per_node_loss_breakdown()      ← Diagnostic: find problem nodes
└── compute_inverse_variance_weights()
```

### Quick usage

```python
import torch
from src.loss import (
    standardized_rmse_loss,
    standardized_rmse_metric,
    push_forward_loss,
    FloodLoss,
    SRMSEAccumulator,
    per_node_loss_breakdown,
)

# ── 1. Basic training loss ──────────────────────────
# node_stds: Tensor of shape (N,) from FloodDataset.compute_node_stds()
loss = standardized_rmse_loss(pred, target, node_stds, clamp_weights=100.0)
loss.backward()

# ── 2. Validation metric (exact leaderboard formula) ─
srmse = standardized_rmse_metric(pred, target, node_stds)
# Returns a scalar — mean over nodes of (RMSE_i / σ_i)

# ── 3. Push-forward loss (K-step rollout) ────────────
# preds/targets shape: (K, N) — K steps in the rollout
pf_loss = push_forward_loss(
    preds, targets, node_stds,
    temporal_scheme="linear",  # later steps weighted more
)

# ── 4. FloodLoss nn.Module (recommended for training) ─
criterion = FloodLoss(
    node_stds_1d=stds_1d,     # shape (N_1d,)
    node_stds_2d=stds_2d,     # shape (N_2d,)
    alpha=0.5,                # 50/50 balance 1D/2D
    temporal_scheme="linear",
    loss_variant="mse",       # or "huber" for outlier-robustness
).to(device)

# Single-stream (for your decoupled 1D or 2D engine):
loss_1d = criterion(pred_1d, target_1d, node_stds=stds_1d)

# Combined 1D + 2D (for unified model):
total_loss, breakdown = criterion.forward_combined(
    pred_1d, target_1d, pred_2d, target_2d,
    use_push_forward=True,
)
# breakdown = {"loss_1d": ..., "loss_2d": ..., "total": ...}

# ── 5. Hierarchical accumulator (cross-event scoring) ─
acc = SRMSEAccumulator()
for event_id, pred, tgt, stds in event_results:
    acc.update("1", event_id, "1d", pred_1d, tgt_1d, stds_1d)
    acc.update("1", event_id, "2d", pred_2d, tgt_2d, stds_2d)
final_score = acc.compute()
print(acc.summary_str())

# ── 6. Find problem nodes ───────────────────────────
diag = per_node_loss_breakdown(pred, target, node_stds, top_k=10)
# diag["top_k_indices"]  — which nodes hurt the score most
# diag["top_k_srmse"]    — their per-node SRMSE values
```

### Key API reference

| Function / Class | Input | Output | When to use |
|-----------------|-------|--------|-------------|
| `standardized_rmse_loss(pred, target, node_stds)` | `(T, N)`, `(T, N)`, `(N,)` | Scalar loss | Training — differentiable surrogate |
| `standardized_huber_loss(pred, target, node_stds, delta=1.0)` | Same | Scalar loss | Training — outlier-robust events |
| `push_forward_loss(preds, targets, node_stds)` | `(K, N)`, `(K, N)`, `(N,)` | Scalar loss | Push-forward rollout training |
| `combined_flood_loss(p1d, t1d, s1d, p2d, t2d, s2d)` | Two streams | `(loss, breakdown)` | Node-type balanced training |
| `FloodLoss(stds_1d, stds_2d)` | Tensors | nn.Module | Recommended training wrapper |
| `standardized_rmse_metric(pred, target, node_stds)` | `(T, N)` | Scalar SRMSE | Validation / logging |
| `SRMSEAccumulator()` | — | Stateful object | Cross-event hierarchical scoring |
| `per_node_loss_breakdown(pred, target, stds, top_k)` | `(T, N)` | Dict | Debugging — find worst nodes |

### For Members A & B (decoupled engines)

When building your 1D-only or 2D-only training loop, use the
**single-stream** API:

```python
from src.loss import FloodLoss, standardized_rmse_metric

# Setup (once)
stds = ds.compute_node_stds(model_id="1")
stds_1d = torch.from_numpy(stds["1"]["1d"]).float()
criterion = FloodLoss(node_stds_1d=stds_1d).to(device)

# In your training loop:
loss = criterion(pred_1d, target_1d)
loss.backward()

# In your validation loop:
srmse = standardized_rmse_metric(pred_1d, target_1d, stds_1d)
```

---

## 3. Training Pipeline

`src/trainer.py` manages the full training lifecycle.

### Architecture overview

```
src/trainer.py
├── TrainConfig (dataclass)          ← All hyperparameters in one place
├── TeacherForcingScheduler          ← Curriculum learning schedule
│     warmup (100% TF) → decay → student (0% TF)
├── EarlyStopping                    ← Patience-based stopper
├── TrainingHistory                  ← Per-epoch metric recorder
├── save_checkpoint() / load_checkpoint()
├── _train_one_event_unified()       ← Single-event forward/backward
├── _validate_one_event_unified()    ← Single-event eval
└── UnifiedTrainer                   ← Full lifecycle manager
      .setup()  → data + model + optimizer + loss + schedulers
      .train()  → epoch loop with curriculum, checkpointing, early stop
```

### Training innovations

| Feature | What it does | Config key |
|---------|-------------|------------|
| **Scheduled Sampling** | Gradually removes teacher forcing: epochs 0–10 = 100% TF, 11–40 = linear decay, 41+ = 0% TF | `tf_warmup_epochs`, `tf_decay_epochs`, `tf_min_ratio` |
| **Push-Forward Training** | Loss over K-step rollout, not single step. Later steps weighted more heavily to fight autoregressive drift | `pushforward_K`, `temporal_scheme`, `use_push_forward` |
| **Mixed Precision (AMP)** | Halves memory on GPU for large 2D meshes | `use_amp` |
| **Gradient Clipping** | Prevents exploding gradients during autoregressive training | `grad_clip_norm` |
| **Cosine LR Schedule** | Smooth decay from `lr` → `cosine_eta_min` | `scheduler="cosine"` |
| **Early Stopping** | Halts when val SRMSE stalls for `patience` epochs | `early_stop_patience` |
| **Checkpointing** | Saves best + periodic checkpoints | `save_best`, `save_every_n_epochs` |

### Quick usage

```python
from src.trainer import TrainConfig, UnifiedTrainer

# ── 1. Configure ────────────────────────────────────
cfg = TrainConfig(
    data_root="data",
    model_ids=["1", "2"],
    val_event_id="4",           # Leave-One-Event-Out
    epochs=60,
    lr=1e-3,
    hidden_channels=128,
    num_gnn_layers=3,
    num_gru_layers=2,
    dropout=0.1,
    use_push_forward=True,
    pushforward_K=10,
    temporal_scheme="linear",
    scheduler="cosine",
    use_amp=True,               # Set False for CPU
    device="auto",
)

# ── 2. Build everything ─────────────────────────────
trainer = UnifiedTrainer(cfg)
trainer.setup()
# Prints: device, train/val counts, model params, TF schedule

# ── 3. Train ────────────────────────────────────────
history = trainer.train()
print(history.summary_str())
# Training Summary (60 epochs, 1842s total):
#   Best val SRMSE : 0.423156 (epoch 47)
#   Avg epoch time : 30.7s

# ── 4. Resume from checkpoint ──────────────────────
history = trainer.train(resume_from="checkpoints/checkpoint_epoch_20.pt")

# ── 5. Access trained model ─────────────────────────
model = trainer.model       # UnifiedFloodModel (on device)
stds_1d = trainer.stds_1d   # Tensor (N_1d,)
stds_2d = trainer.stds_2d   # Tensor (N_2d,)
```

### `TrainConfig` — full parameter reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_root` | `"data"` | Path to data directory |
| `model_ids` | `["1", "2"]` | Urban models to train on |
| `val_event_id` | `"4"` | Event held out for validation |
| `hidden_channels` | `64` | GNN/GRU hidden dimension |
| `num_gnn_layers` | `3` | Number of message-passing layers |
| `num_gru_layers` | `1` | GRU depth |
| `dropout` | `0.1` | Dropout rate |
| `epochs` | `60` | Maximum training epochs |
| `lr` | `1e-3` | AdamW learning rate |
| `weight_decay` | `1e-5` | L2 regularisation |
| `grad_clip_norm` | `1.0` | Max gradient norm |
| `tf_warmup_epochs` | `10` | Epochs of 100% teacher forcing |
| `tf_decay_epochs` | `30` | Epochs over which TF decays to `tf_min_ratio` |
| `tf_min_ratio` | `0.0` | Final teacher forcing ratio |
| `pushforward_K` | `10` | Rollout window length |
| `temporal_scheme` | `"linear"` | `"uniform"` / `"linear"` / `"exponential"` |
| `use_push_forward` | `True` | Enable push-forward training |
| `loss_variant` | `"mse"` | `"mse"` or `"huber"` |
| `alpha` | `0.5` | 1D/2D loss balance |
| `scheduler` | `"cosine"` | LR scheduler: `"cosine"` / `"plateau"` / `"none"` |
| `early_stop_patience` | `15` | Epochs without improvement before stopping |
| `use_amp` | `True` | Mixed precision (GPU only) |
| `checkpoint_dir` | `"checkpoints"` | Where to save `.pt` files |
| `device` | `"auto"` | `"auto"` / `"cuda"` / `"cpu"` / `"mps"` |

### CLI (command-line training)

```bash
# Basic training
python -m src.trainer --epochs 60 --lr 1e-3

# Resume from checkpoint
python -m src.trainer --resume checkpoints/checkpoint_epoch_20.pt
```

### Production training script

For long training runs (avoid notebook kernel timeouts):

```bash
python train_production.py
python train_production.py --epochs 80 --hidden_channels 128
python train_production.py --device cuda --resume checkpoints/checkpoint_epoch_40.pt
python train_production.py --no_amp  # For CPU-only training
```

### For Members A & B (decoupled engines)

You can **reuse** `TrainConfig`, `TeacherForcingScheduler`,
`EarlyStopping`, `TrainingHistory`, `save_checkpoint`, and
`load_checkpoint` directly in your own training loops:

```python
from src.trainer import (
    TrainConfig,
    TeacherForcingScheduler,
    EarlyStopping,
    TrainingHistory,
    save_checkpoint,
    load_checkpoint,
)

# Create your own training loop using these building blocks:
tf = TeacherForcingScheduler(warmup_epochs=10, decay_epochs=30)
stopper = EarlyStopping(patience=15)
history = TrainingHistory()

for epoch in range(60):
    ratio = tf.get_ratio(epoch)   # 1.0 → 0.0 over epochs
    # ... your training code ...
    history.log(epoch, train_loss, val_srmse, lr, ratio, elapsed)
    if stopper.step(val_srmse):
        break
```

---

## 4. Validation Pipeline

`src/validate.py` orchestrates model evaluation matching the Private Leaderboard protocol.

### Architecture overview

```
src/validate.py
├── ValidationResult (dataclass)     ← Container for all results
│     .summary_str()                 ← Human-readable report
│     .save(path)                    ← Dump to JSON
├── validate_event_unified()         ← Single-event autoregressive eval
├── ValidationRunner                 ← Full cross-validation orchestrator
│     .validate_holdout(val_eid)     ← LOEO on one hold-out event
│     .cross_validate(n_folds)       ← Multi-fold LOEO
├── extract_predictions()            ← For plotting pred vs ground truth
└── check_leaderboard_correlation()  ← Local ↔ Public LB correlation
```

### Validation protocol

The validation pipeline exactly mirrors the **Private Leaderboard** evaluation:

1. **Spin-up** (t = 0..9): Feed ground truth → build GRU hidden states. Predictions generated but **not scored**.
2. **Prediction** (t = 10..end): Full autoregressive rollout (teacher forcing = 0%). These predictions **are scored**.
3. **Hierarchical SRMSE**: `Mean_models → Mean_events → Mean_node_types → Mean_nodes(RMSE_i / σ_i)`

### Quick usage

```python
from src.validate import ValidationRunner, validate_event_unified, extract_predictions

# ── 1. Simple hold-out validation ────────────────────
runner = ValidationRunner(
    model=trained_model,
    data_root="data",
    device="cuda",
    model_ids=["1", "2"],
    spinup_steps=10,
    run_diagnostics=True,
)

result = runner.validate_holdout(val_event_id="4")
print(result.summary_str())
# ============================================================
#   VALIDATION RESULTS
# ============================================================
#   Overall SRMSE: 0.423156
#   Time: 45.2s
#
#   Model 1:
#     Event 4: 1d=0.3821, 2d=0.4512  -> 0.4167
#   Model 2:
#     Event 4: 1d=0.4102, 2d=0.4493  -> 0.4298
# ============================================================

# Save results to JSON
result.save("logs/val_results.json")

# ── 2. Multi-fold cross-validation ──────────────────
cv = runner.cross_validate(n_folds=5)
print(f"CV SRMSE: {cv['mean_srmse']:.4f} ± {cv['std_srmse']:.4f}")
# Per-fold scores: cv["per_fold_srmse"]

# ── 3. Single-event validation (low-level) ──────────
from src.graph_builder_unified import build_unified_graph

sample = ds[0]
data = build_unified_graph(sample)
result = validate_event_unified(
    model, data, stds_1d, stds_2d,
    device=device, spinup_steps=10,
    run_diagnostics=True, top_k=10,
)
print(f"1D: {result['srmse_1d']:.4f}, 2D: {result['srmse_2d']:.4f}")
# result["diagnostics_1d"]["top_k_indices"]  — worst 1D nodes
# result["diagnostics_2d"]["top_k_srmse"]    — their error values

# ── 4. Extract predictions for plotting ─────────────
preds = extract_predictions(model, data, device, spinup_steps=10)
# preds["preds_1d"]   : Tensor [T, N_1d]
# preds["targets_1d"] : Tensor [T, N_1d]
# preds["preds_2d"]   : Tensor [T, N_2d]
# preds["targets_2d"] : Tensor [T, N_2d]

import matplotlib.pyplot as plt
node_idx = 5
plt.plot(preds["targets_1d"][:, node_idx], label="Ground Truth")
plt.plot(preds["preds_1d"][:, node_idx], label="Predicted")
plt.axvline(preds["spinup_steps"], color="red", linestyle="--", label="Scoring start")
plt.legend()
```

### `ValidationResult` — output fields

| Field | Type | Description |
|-------|------|-------------|
| `overall_srmse` | `float` | Final hierarchical SRMSE score |
| `breakdown` | `dict` | `{model_id: {event_id: {node_type: srmse}}}` |
| `per_event_scores` | `list[dict]` | Per-event timing + scores |
| `diagnostics` | `dict` | Per-event top-K worst nodes |
| `elapsed_seconds` | `float` | Total wall-clock time |

### For Members A & B (decoupled engines)

You can write your own `validate_event_1d()` / `validate_event_2d()` following the same protocol. Use `SRMSEAccumulator` from `loss.py` for hierarchical scoring and `extract_predictions()` as a template for your rollout logic.

---

## 5. Submission Generation

`src/inference.py` handles autoregressive inference over test events and
produces the competition-ready CSV.

### Architecture overview

```
src/inference.py
├── predict_event()                        ← Single-event autoregressive inference
├── _build_submission_rows_vectorized()    ← Fast long-format row builder
├── SubmissionGenerator                    ← Full submission pipeline
│     .generate() → DataFrame             ← Run inference on all test events
│     .save(df, path)                      ← Write CSV/Parquet
│     .from_checkpoint(path) [classmethod] ← Load model + generate
├── ensemble_predict_event()               ← Average predictions from N models
└── main()                                 ← CLI entrypoint
```

### Submission format

The competition requires this exact CSV structure:

| Column | Example | Notes |
|--------|---------|-------|
| `row_id` | `1_5_1d_3_15` | `{model_id}_{event_id}_{node_type}_{node_id}_{timestep}` |
| `model_id` | `1` | Integer |
| `event_id` | `5` | Integer |
| `node_type` | `1d` | `"1d"` or `"2d"` |
| `node_id` | `3` | Integer |
| `water_level` | `0.1542` | Float — predicted water level |

Rows from the **spin-up period** (t = 0..9) are excluded (not scored).

### Quick usage

```python
from src.inference import SubmissionGenerator, predict_event, ensemble_predict_event

# ── 1. Full submission from checkpoint ───────────────
gen = SubmissionGenerator.from_checkpoint(
    "checkpoints/best_model.pt",
    data_root="data",
    device="cuda",
)
df = gen.generate()
gen.save(df, "submission.csv")
#   Submission saved to submission.csv (1,234,567 rows)

# ── 2. Manual submission ────────────────────────────
gen = SubmissionGenerator(
    model=trained_model,
    data_root="data",
    device="cuda",
    spinup_steps=10,
)
df = gen.generate()
gen.save(df, "submission.csv")

# ── 3. From the command line ─────────────────────────
# python -m src.inference --checkpoint checkpoints/best_model.pt
# python -m src.inference --checkpoint best.pt --output sub.csv --device cuda

# ── 4. Ensemble inference (5-15% score improvement) ──
models = [model_fold1, model_fold2, model_fold3]
preds_1d, preds_2d = ensemble_predict_event(
    models, data, device,
    weights=[0.4, 0.35, 0.25],  # Optional per-model weights
)
```

### Sanity checks (automatic)

`SubmissionGenerator.generate()` runs these checks automatically:

| Check | Action |
|-------|--------|
| Missing columns | Raises `ValueError` |
| NaN in `water_level` | Forward-fill from previous timestep, then zero-fill |
| Duplicate `row_id` | Warns + deduplicates (keeps first) |
| Extreme values (> 1000 or < -100) | Warns about possible model divergence |

### For Members A & B (decoupled engines)

Use `_build_submission_rows_vectorized()` to convert your model's
predictions into submission rows:

```python
from src.inference import _build_submission_rows_vectorized

# After running your 1D engine:
df_1d = _build_submission_rows_vectorized(
    preds_1d,       # Tensor [T, N_1d]
    model_id="1",
    event_id="5",
    node_type="1d",
    spinup_steps=10,
)

# Combine with 2D predictions:
import pandas as pd
submission = pd.concat([df_1d, df_2d], ignore_index=True)
submission.to_csv("submission.csv", index=False)
```

---

## 6. Graph Building

Each graph builder lives in its own file to avoid merge conflicts.

| Task | Owner | File | Function | Key requirements |
|------|-------|------|----------|------------------|
| **1.2** | Member A | `graph_builder_1d.py` | `build_1d_graph(sample)` | Bidirectional edges (`from_node ↔ to_node`); features: `Relative Depth`, `Capacity`, `Rain` |
| **1.3** | Member B | `graph_builder_2d.py` | `build_2d_graph(sample)` | Soft coupling: Euclidean distance to nearest 1D node; features: Z-scored elevation, roughness, rainfall, `dist_to_drain` |
| **1.4** | Member C | `graph_builder_unified.py` | `build_unified_graph(sample)` | `HeteroData` with explicit 1D↔2D edges for the Unified Engine |

### How the graph builders consume FloodDataset

```python
from src.dataset import FloodDataset
from src.graph_builder_1d import build_1d_graph
from src.graph_builder_2d import build_2d_graph
# or use the legacy shim:  from src.graph_builder import build_1d_graph, build_2d_graph

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

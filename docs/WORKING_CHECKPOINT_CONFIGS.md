# Working checkpoint configs (0.3089 leaderboard)

Recovered training configuration for the checkpoints used in `submission_2d_ensemble.csv` + `submission_1d.csv` that produced the **0.3089** leaderboard score.

---

## 1. Checkpoint contents (no hyperparameters saved)

All three checkpoint files store **only**:

- `epoch`, `model_state_dict`, `optimizer_state_dict`, `loss`

**No** config/hyperparameters are saved inside the `.pt` files.

| Checkpoint | Epoch saved | Loss |
|------------|-------------|------|
| `checkpoints/model_1_improved.pt` | 15 | 0.4955 |
| `checkpoints/model_2_improved.pt` | 0 | 0.6813 |
| `checkpoints/experiments/model_2_exp1_lower_lr.pt` | 35 | 0.5038 |

*(Inferred with: `torch.load(..., weights_only=False)` and inspecting `ckpt.keys()` and non-state-dict values.)*

---

## 2. model_1_improved.pt and model_2_improved.pt

**Script / save path:** The codebase **never writes** to `model_1_improved.pt` or `model_2_improved.pt`. Current code saves to `model_{id}_best.pt` (or `model_{id}_sage.pt` in `train_improved.py`). So these were almost certainly produced by:

1. Training with a script that wrote to `model_1_best.pt` / `model_2_best.pt`, then  
2. **Copying** those files to `model_1_improved.pt` and `model_2_improved.pt` (by hand or a one-off script).

**Architecture (from `run_ensemble_submission.py`):**

- `hidden_channels=128`, `num_sage_layers=3`, `dropout=0.15`, `max_delta=2.0`, `conv_type='sage'`

**Training config (inferred):**  
The version of the improved pipeline **before** the recent anti-divergence changes (no LR warmup, no linear TF to 0.3, no grad clipping, original lr and timesteps). That corresponds to the **original** `SAGE_CONFIG` in `src/train_improved.py` before you changed it:

| Parameter | Recovered value |
|-----------|-----------------|
| **lr** | **0.002** |
| **num_epochs** | 50 |
| **warmup_epochs** | 12 (TF = 1.0) |
| **decay_epochs** | 28 (TF linear 1.0 → 0 over 28 epochs after warmup) |
| **max_timesteps_per_event** | **60** |
| **early_stopping_patience** | 15 |
| **validation_events** | 3 |
| **print_every / validate_every** | 5 |
| **LR scheduler** | ReduceLROnPlateau(factor=0.5, patience=5) |
| **Grad clipping** | None |
| **LR warmup** | None |
| **tf_min_ratio** | Not used (classic warmup + decay) |

So: **lr=0.002**, teacher forcing **warmup 12 epochs at 1.0**, then **linear decay over 28 epochs to 0**, **max_timesteps_per_event=60**, no grad clipping, no LR warmup.

**Git:** No commit in the repo history introduces or changes a save path containing `model_1_improved` or `model_2_improved`.

---

## 3. model_2_exp1_lower_lr.pt — exact config

This checkpoint was produced by **`src/train_model2_experiments.py`** (experiment **`exp1_lower_lr`**). The exact config is stored in:

**`checkpoints/experiments/model_2_exp1_lower_lr_results.json`**

and in **`EXPERIMENTS["exp1_lower_lr"]`** in `src/train_model2_experiments.py`.

| Parameter | Value |
|-----------|--------|
| **description** | Baseline architecture, much lower learning rate |
| **hidden_channels** | 64 |
| **num_sage_layers** | 2 |
| **dropout** | 0.1 |
| **max_delta** | 2.0 |
| **num_epochs** | 40 |
| **lr** | **0.001** |
| **warmup_epochs** | 15 |
| **decay_epochs** | 20 |
| **max_timesteps_per_event** | **50** |
| **early_stopping_patience** | 15 |
| **conv_type** | sage (default) |

Teacher forcing: **1.0 for 15 epochs**, then **linear decay to 0 over 20 epochs**. No grad clipping, no LR warmup in that script.

Recorded results: **best_epoch=35**, **best_val_rmse≈0.504**, **eval_rmse_mean≈0.535**.

---

## 4. Git history (Feb 14 and training)

- **64fd9ae** (2026-02-13, “2D Done”): Touched `src/train_and_submit.py`, `src/model_2d.py`, etc.  
- **d2d3e09** (2026-02-12): Introduced 2D pipeline; `train_model` defaults: `num_epochs=50`, `lr=0.005`, `warmup_epochs=10`, `decay_epochs=30`, `max_timesteps_per_event=None`, `early_stopping_patience=15`.

No commits around that time add or change a path like `model_*_improved.pt`; the “improved” checkpoints were almost certainly created by training (to `*_best.pt`) then copying.

---

## 5. Summary table (working configs)

| Source | lr | num_epochs | warmup_epochs | decay_epochs | max_timesteps | TF schedule | Grad clip |
|--------|-----|------------|---------------|--------------|---------------|-------------|-----------|
| **model_1_improved / model_2_improved** (inferred) | 0.002 | 50 | 12 | 28 | 60 | 1.0 for 12, then linear → 0 over 28 | None |
| **model_2_exp1_lower_lr** (from JSON + code) | **0.001** | 40 | 15 | 20 | 50 | 1.0 for 15, then linear → 0 over 20 | None |

To **reproduce** the improved models you can:

- Add a “legacy” or “working” config in `train_improved.py` (or a small script) that uses the inferred improved config above and saves to `model_{id}_best.pt`, then copy to `model_{id}_improved.pt` if you want to keep that naming.
- Keep using **exp1_lower_lr** as-is in `train_model2_experiments.py` for the Model_2 experiment checkpoint; its config is already defined and recorded in the results JSON.

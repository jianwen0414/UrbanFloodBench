# Feature Engineering & Model Audit — Path to SRMSE < 0.3

**Date:** 2025-02-16  
**Context:** Training stagnant (Model 1 ~0.46, Model 2 ~0.87 raw / 1.76 EMA). Root-cause audit of features, physics, and hyperparameters.

---

## Executive Summary

| Finding | Severity | Status |
|---------|----------|--------|
| **effective_depth explosion (Model 1)** | CRITICAL | **FIXED** |
| Constant features (pipe_roughness, 2d roughness) | Done | Zeroed |
| Model 2 AR collapse at K≥12 | High | Tuned (K_max=15) |
| Small validation set (3 events) | Medium | Open |
| Position / dist_to_drain redundancy | Low | Open |
| Hyperparameter tuning | Medium | Recommendations below |

---

## 1. Critical Bug Fixed: effective_depth for Model 1

**Evidence from diagnostic:**
- Model 1: `effective_depth` mean=**95,287,863**, std=**1,674,129,394** (absurd)
- Model 2: `effective_depth` mean=0.23, std=0.49 (reasonable)

**Root cause:**
- Model 1 static 2D `area` has **min = -1.6e-14** (numerical noise / degenerate cell).
- `effective_depth = water_volume / (area + 1e-8)` → division by ~1e-8 for that cell → values up to ~3e10.
- Percentiles: p50=0.0006, p95=0.15, p99=4.5 → median is fine; mean/std destroyed by outliers.

**Fix applied:**
1. **train_unified.py**: `area_n = np.maximum(area_vals[nidx], 1.0)` — floor area at 1.0 sq ft.
2. **train_unified.py**: `_stats(effective_depth_acc, physical_max=20.0)` — clamp to 20 ft before computing mean/std.
3. **graph_builder_unified.py**: `area_safe = area_t.clamp(min=1.0)`, `effective_depth.clamp(0, 20)`.

**Expected impact:** Model 1 index 5 feature now has physically plausible scale. Retraining should show improvement.

---

## 2. Feature Engineering — Status & Gaps

### 2.1 Physics alignment

| Feature | Physics | Status |
|---------|---------|--------|
| **2D depth** | WSE − min_elevation | Correct (min_elev for pooling) |
| **1D depth** | WSE − invert_elevation | Correct |
| **effective_depth** | water_volume / area | Fixed (was broken for Model 1) |
| **capacity** | surface − invert | Correct |
| **elev_rel_neighbors** | node elev − mean(neighbor elev) | Correct (depression detection) |
| **dist_to_drain** | Euclidean to nearest 1D node | Correct |
| **Manning roughness** | pipe_roughness, 2d roughness | Constant for some models → zeroed |

### 2.2 Feature quality issues

| Issue | Detail | Action |
|-------|--------|--------|
| **Constant features** | pipe_roughness std=0 (both), 2d roughness std=0 (Model 2) | Zeroed via `_normalize_or_zero` |
| **Position vs dist_to_drain** | Both derived from coords; possible redundancy | Consider dropping position or using PCA |
| **Area scale** | Model 2 area ~25× Model 1 | Per-model z-score ok; `log(1+area)` not yet tried |
| **Depth lags** | t-2, t-3, t-4 | Good (EDA: lag 2 best for rainfall) |

### 2.3 Missing physics signals

- **Flow / velocity**: `dynamic_*_edges` has `flow`, `velocity` — unused.
- **1D–2D mass balance**: effective_depth is a proxy; explicit flux at 1d2d links could help.
- **Rainfall intensity**: We have rain, rain_rolling, rain_delta, rain_lag2; consider `rain × area` as forcing.

---

## 3. Parameter Tuning — Recommendations

### 3.1 Current defaults (train_unified.py)

```
hidden_channels=256, num_gnn_layers=3
lr=0.001, weight_decay=1e-5
pushforward_K=20, K_ramp_epochs=40
tf_warmup=5, tf_decay=40, tf_min_ratio=0.0
delta_clamp_2d=2.0, delta_clamp_1d=5.0
ar_noise_std=0.005
Model 2 override: K_max=15, K_ramp=50
```

### 3.2 Recommended changes

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| **lr** | 0.001 | **0.002–0.003** | Member B uses 0.005; higher LR may escape plateau |
| **tf_min_ratio** (Model 2) | 0.0 | **0.10–0.15** | Non-zero TF reduces AR compounding |
| **ar_noise_std** | 0.005 | **0.01** | Slightly more AR noise for stability |
| **K_ramp** (Model 2) | 50 | **60–70** | Slower K growth delays collapse |
| **epochs** | 60 | **80–100** | More time to converge |
| **auxiliary single-step loss** | None | **0.2 × MSE(K=1)** | Anchor per-step accuracy |

### 3.3 Validation strategy

- **Current:** 3 fixed val events (3, 9, 15).
- **Recommendation:** 5-fold leave-events-out to reduce val variance.
- **Fallback:** Use EMA of last 5 epochs for checkpoint selection.

---

## 4. Model 2 AR Instability

**Observed:** Val SRMSE improves to ~0.87 at K=8, then degrades to 4+ as K ramps to 15.

**Actions in place:**
- K_max=15, K_ramp=50.

**Additional options:**
1. `tf_min_ratio=0.15` so some GT is always fed.
2. Lower K_max to 12 for Model 2.
3. Curriculum: train on shorter events first (T ≤ 100).
4. Auxiliary loss: add `0.2 × MSE(pred_t, target_t)` at each step.

---

## 5. Checklist for Next Training Run

- [x] effective_depth bug fix (area floor, physical_max clamp)
- [ ] Retrain with fixed effective_depth
- [ ] Consider lr=0.002
- [ ] Consider tf_min_ratio=0.1 for Model 2
- [ ] Monitor Model 1 effective_depth stats (should be ~0.1–0.5 mean, ~0.5–1 std)
- [ ] If still stagnant: add auxiliary single-step loss

---

## 6. Quick Verification

After fix, run:

```python
from src.train_unified import compute_model_stats
from src.dataset import FloodDataset
from src.config import RAW_DATA_PATH

ds = FloodDataset(str(RAW_DATA_PATH), mode="train")
s1 = compute_model_stats(ds, "1")
s2 = compute_model_stats(ds, "2")
print("Model 1 effective_depth:", s1["2d"]["effective_depth"])  # expect mean ~0.1–0.5
print("Model 2 effective_depth:", s2["2d"]["effective_depth"])  # expect mean ~0.2–0.3
```

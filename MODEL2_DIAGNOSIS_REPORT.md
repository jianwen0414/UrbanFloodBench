# Model 2 Performance Diagnosis & Improvement Plan

**Date:** 2025-02-15  
**Context:** Model 1 achieved val SRMSE 0.50; Model 2 stagnated at ~1.0 (best EMA), raw val 0.75 at epoch 43. Training was cut at epoch 58/60 due to oscillation.

---

## 1. Root Cause Analysis

### 1.1 Data / Domain Differences

| Aspect | Model 1 | Model 2 |
|--------|---------|---------|
| **1D nodes** | 17 | 198 |
| **2D nodes** | 3,716 | 4,299 |
| **1D depth std** | 24.83 | 4.94 |
| **2D depth std** | 0.64 | 0.92 |
| **2D roughness std** | 0.019 | **0.000** (constant) |
| **2D slope std** | 0.044 | 0.0056 |
| **2D area mean** | 609 | 15,126 |
| **2D area std** | 116 | 8,465 |
| **Elevation** | ~323 ft | ~44 ft |

**Key findings:**
- **Constant roughness (Model 2):** `2d/roughness` has zero variance. After z-score, the feature is zero for all nodes → wasted input dimension. Model 1 has useful roughness signal.
- **Different spatial scale:** Model 2 has ~25× larger cells (area) and different elevation range → different physical domain.
- **Flatter slope:** Model 2 slope std=0.0056 vs 0.044 → less terrain variability signal.

### 1.2 AR Rollout Instability (Primary Cause)

Model 2's loss and val collapse when K ramps to 20:

| Phase | K | Model 1 loss | Model 2 loss | Model 2 val |
|-------|---|--------------|---------------|--------------|
| Early | 2–10 | 3.9 → 0.5 | 0.75 → 0.02 | 56 → 1.0 |
| Late | 11–20 | 0.5–0.67 | **0.02 → 0.15** | **1.0 → 1.6** |

- Model 2 trains well at K=2–10 (val improved to ~1.0).
- At K=16+ with TF≈0.3, errors compound over 16–20 AR steps.
- Model 1’s dynamics are more stable; Model 2’s larger graph and different physics amplify rollout errors.

### 1.3 Checkpoint Logic

- Only **best EMA** checkpoint is saved; best **raw val** (0.75 at epoch 43) was never stored.
- Dual checkpoint (best EMA + best raw val) would preserve both.

---

## 2. Improvement Plan

### 2.1 High Impact (Implemented)

1. **Model-specific K curriculum (Model 2)**  
   - Use lower `K_max` (e.g. 12–15) and/or slower K ramp.  
   - Prevents the collapse seen at K=20.

2. **Dual checkpoint saving**  
   - Save both `unified_model_{id}.pt` (best EMA) and `unified_model_{id}_best_val.pt` (best raw val).  
   - Inference can then use the better-performing checkpoint per model.

3. **PyTorch scheduler deprecation**  
   - Replace `scheduler.step(epoch)` with `scheduler.step()` to remove warnings.

### 2.2 Medium Impact (Recommended)

4. **Slower K ramp for Model 2**  
   - Extend K ramp over more epochs so K=20 is reached later (or never).

5. **Higher teacher forcing for Model 2**  
   - Keep `min_ratio` > 0 (e.g. 0.2) so rollout always gets some GT injection.

6. **Per-model hyperparameter overrides**  
   - Add config for `K_max`, `K_ramp_epochs`, `tf_min_ratio` per model_id.

### 2.3 Feature Engineering (Future)

7. **Constant-feature handling**  
   - Detect features with std < 1e-6 and either drop or zero them explicitly.  
   - Roughness is already zero after norm; no functional change, just clarity.

8. **Roughness fallback**  
   - If 2D roughness is constant, consider aggregating from 1D pipe roughness for connected cells. Non-trivial to implement.

9. **log(1+area)**  
   - Optional feature for scale-heavy models. Z-score on raw area may already suffice; test if helpful.

10. **Model-specific feature selection**  
    - Option to omit constant features to reduce input dim and noise.

---

## 3. Implementation Summary

| Change | File | Status |
|--------|------|--------|
| Dual checkpoint (EMA + raw val) | `train_unified.py` | Done |
| Model-specific K_max, K_ramp | `train_unified.py` | Done |
| Scheduler deprecation fix | `train_unified.py` | Done |

---

## 4. Usage

**Model 2 with conservative K:**
```bash
python -m src.train_unified --model_ids 2 --pushforward_K 15 --K_ramp_epochs 40
```

**Checkpoints produced:**
- `checkpoints/unified_model_2.pt` — best EMA val
- `checkpoints/unified_model_2_best_val.pt` — best raw val

Use `*_best_val.pt` for Model 2 if raw val (e.g. 0.75) is better than EMA val (1.0).

**Inference with best raw val checkpoint:**
```bash
# When running inference, point to the _best_val checkpoint for Model 2:
# e.g. checkpoint_path="checkpoints/unified_model_2_best_val.pt"
```

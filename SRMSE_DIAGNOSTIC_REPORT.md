# SRMSE Diagnostic Report — Path to < 0.3

**Date:** 2025-02-15  
**Context:** Model 1 best raw val 0.45, Model 2 val oscillates 1.0 → 2.5+ (cut at epoch 51). Neither achieved target SRMSE < 0.3.

---

## 1. Training Log Summary

| Model | Best raw val | Best EMA | Plateau / collapse |
|-------|--------------|----------|-------------------|
| Model 1 | **0.449** (epoch 42) | 0.48 (epoch 58) | Plateau ~0.47–0.71 from epoch 36 onward |
| Model 2 | ~**0.91** (epoch 24) | ~2.77 | K=12+ val explodes 1.0 → 2.5+; EMA climbs |

**Key observations:**
- Model 1: Val oscillates early (146 → 14 → 159 → 21) then stabilises; best at epoch 42.
- Model 2: Val improves to ~1.0 at K≤9, then collapses as K reaches 12–15. Loss rises 0.02 → 0.20.
- Both models: 3 val events (`['3', '9', '15']`) — small validation set → high variance in reported SRMSE.

---

## 2. Six Root Causes & Recommended Actions

### Finding 1: Constant features waste model capacity

**Evidence:**
- `1d/pipe_roughness`: std=**0.0000** (both models)
- `2d/roughness`: std=**0.0000** (Model 2)

**Impact:** After z-score, these features are zero for all nodes. The model wastes 2 input dimensions on noise. Roughness is physically critical for Manning’s equation (flow resistance). Constant roughness means the model cannot learn roughness-dependent behaviour.

**Action:**
1. **Drop or zero** features with std < 1e-6 in the graph builder.
2. **Roughness fallback:** When 2D roughness is constant, derive a proxy from connected 1D pipe roughness (distance-weighted mean over 1d2d links).
3. **Model-specific feature selection:** Maintain a per-model feature mask; exclude constant features for Model 2.

---

### Finding 2: Position features have incompatible coordinate scales

**Evidence:**
- Model 1: `position_x` mean=802,776, std=533; `position_y` mean=349,535, std=400
- Model 2: `position_x` mean=6,633,396, std=3,012; `position_y` mean=1,964,009, std=1,764

**Impact:** Position is per-model z-scored, so within-model scaling is fine. But the absolute magnitudes (6M vs 0.8M) suggest different coordinate systems (e.g. UTM zones). As raw features, they carry little cross-model meaning. More importantly: position may be **highly correlated with dist_to_drain** (derived from the same coords). Redundancy could dilute useful signal.

**Action:**
1. **Diagnostic:** Compute correlation between `(position_x, position_y)` and `dist_to_drain`. If |r| > 0.8, consider dropping position or using a combined spatial encoding.
2. **Alternative:** Replace raw position with **relative** features: e.g. `(x - x_centroid) / domain_extent`, or PCA of coordinates to decorrelate.
3. **Spatial encoding:** Use sinusoidal positional encoding (e.g. `sin(2π·x/period)`) for scale-invariance.

---

### Finding 3: water_volume has extreme scale mismatch with depth

**Evidence:**
- Model 1: `water_volume` mean=52.2, std=**335.9** (std >> mean)
- Model 2: `water_volume` mean=2,765.7, std=**6,509.1** (std >> mean)

**Impact:** water_volume is highly right-skewed (long tail). Z-score normalisation can produce extreme values for rare large floods. Physics: volume ≈ depth × area; Model 2 area is ~25× larger, so volume scales with area. Using raw water_volume mixes area and depth effects.

**Action:**
1. **Derived feature:** Use `effective_depth = water_volume / (area + ε)` instead of raw water_volume — aligns with depth physics.
2. **Log transform:** `log(1 + water_volume)` to reduce skew and make scale more manageable.
3. **Diagnostic:** Plot water_volume vs (depth × area) per node; confirm linearity. If not, volume may include storage or geometry effects worth modelling.

---

### Finding 4: Validation strategy is unstable (3 events)

**Evidence:** Val SRMSE jumps 1.25 → 7.42 → 1.54 (Model 1 epochs 6–9); 1.6 → 13.67 → 1.73 (Model 2 epochs 5–10). Val events = `['3', '9', '15']` fixed.

**Impact:**
1. **Small sample:** 3 events → high variance in reported val SRMSE.
2. **Possible train/val mismatch:** Fixed events may not cover the full rainfall/geometry distribution.
3. **Optimisation noise:** Early-stopping and checkpoint selection become unreliable when val oscillates.

**Action:**
1. **K-fold validation:** Use 5-fold leave-events-out (e.g. 5 folds of ~13–14 events each). Report mean ± std SRMSE.
2. **Stratified split:** Ensure val events span rainfall intensity (check rainfall stats per event). Avoid val = only heavy or only light events.
3. **Rolling validation:** Average val over the last N epochs (e.g. 5) to smooth selection.

---

### Finding 5: Model 2 AR instability is tied to K, not just features

**Evidence:** Model 2 val improves to ~0.9–1.0 at K=6–9, then deteriorates to 2.5+ as K ramps to 12–15. Loss rises from 0.02 to 0.20.

**Impact:** Errors compound over AR steps. With TF decaying to 0, the model feeds its own predictions back. Small per-step errors (e.g. 2D depth) accumulate, especially on a larger graph (198 1D + 4,299 2D nodes).

**Action:**
1. **Lower K_max for Model 2:** Keep K_max=12 (not 15) and extend K_ramp to 70+ epochs so K grows very slowly.
2. **Non-zero teacher forcing:** Set `tf_min_ratio=0.15` for Model 2 so some GT is always injected — reduces compounding.
3. **Curriculum by sequence length:** Train on shorter events first (e.g. T ≤ 100); then fine-tune on long events.
4. **Auxiliary loss:** Add single-step (K=1) MSE loss with weight 0.2–0.3 to anchor per-step accuracy.

---

### Finding 6: 2D area scale difference hurts generalisation

**Evidence:**
- Model 1: area mean=609, std=116
- Model 2: area mean=15,125, std=8,465 (~25× larger)

**Impact:** Raw area after z-score has different interpretation: Model 2 cells are much larger. Depth × area → volume scales differently. The model may struggle to generalise volume/depth relationship across models when area spans two orders of magnitude.

**Action:**
1. **Scale-invariant area:** Use `log(1 + area)` or `area / median(area)` per model to reduce raw scale sensitivity.
2. **Volume-derived depth:** Replace raw water_volume with `water_volume / area` (effective depth) as the primary volume-related feature.
3. **Per-cell normalisation:** For 2D depth, consider `depth / sqrt(area)` as a scale-normalised depth (depth per unit length scale).

---

## 3. Prioritised Implementation Roadmap

| Priority | Action | Expected gain | Effort |
|----------|--------|---------------|--------|
| **P0** | Drop constant features (pipe_roughness, 2d roughness when std=0) | Reduce noise, free capacity | Low |
| **P0** | Use `effective_depth = water_volume / area` instead of raw water_volume | Better physics alignment | Low |
| **P1** | K-fold or larger val set (5+ events) | Stable val metric | Medium |
| **P1** | Model 2: K_max=12, tf_min_ratio=0.15 | Reduce AR collapse | Low |
| **P1** | Add single-step auxiliary loss (weight 0.2) | Better per-step accuracy | Medium |
| **P2** | log(1+area), log(1+water_volume) | Scale-invariant features | Low |
| **P2** | Correlation check: position vs dist_to_drain | Remove redundancy | Low |
| **P3** | Roughness fallback from 1D pipes for Model 2 | Restore roughness signal | High |

---

## 4. Quick Diagnostic Script (Suggested)

Run the following to verify findings:

```python
# In experiments/ or a notebook
from src.dataset import FloodDataset
from src.config import RAW_DATA_PATH
from src.train_unified import compute_model_stats
import numpy as np

ds = FloodDataset(str(RAW_DATA_PATH), mode="train")
for mid in ["1", "2"]:
    stats = compute_model_stats(ds, mid)
    s1, s2 = stats["1d"], stats["2d"]
    # Constant-feature check
    for k, v in {**s1, **s2}.items():
        if isinstance(v, dict) and "std" in v and v["std"] < 1e-6:
            print(f"  Model {mid} {k}: CONSTANT (std={v['std']})")
    # water_volume / depth scale
    wv_mean, wv_std = s2["water_volume"]["mean"], s2["water_volume"]["std"]
    d_mean, d_std = s2["depth"]["mean"], s2["depth"]["std"]
    print(f"  Model {mid} water_volume: mean={wv_mean:.1f} std={wv_std:.1f} (std/mean={wv_std/max(wv_mean,1e-6):.2f})")
```

---

## 5. Summary

The main gaps blocking SRMSE < 0.3 are:

1. **Feature quality:** Constant roughness and raw water_volume add noise.
2. **Validation reliability:** 3 fixed events cause high val variance.
3. **Model 2 AR regime:** K=12+ triggers error compounding; TF and K need tuning.
4. **Scale mismatches:** Area and water_volume vary greatly between models; use scale-invariant or derived features.

Implementing P0 and P1 items should yield noticeable gains. P2/P3 can follow once P0–P1 are in place.

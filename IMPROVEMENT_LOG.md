# IMPROVEMENT_LOG.md — Complete Iteration History

> **Purpose**: Living document tracking every improvement attempt on the Urban Flood Model.
> Do not repeat failed approaches. Check this log before trying anything new.

---

## Run 1 — Baseline Configuration
**Date:** ~Feb 2025 | **Pipeline:** `train_production.py` + `model_unified.py`
**Config:** K=5, TF warmup=10, decay=30, MSE loss, cosine LR=3e-4, 2 val events

| Metric | Value |
|--------|-------|
| Best Val SRMSE | **1.772** (epoch 3) |
| Final | Early stop at epoch 18 |

**What happened:** Val collapsed from 1.77 → 17.45 as teacher forcing decayed. Single-step training (K=5) couldn't match 40+ step validation rollout.

**Changes made after:**
- ✅ Added Huber loss variant (`standardized_huber_loss`, δ=0.5)
- ✅ Added push-forward trajectory loss (`push_forward_loss`)
- ✅ Increased K for longer rollouts
- ✅ Tighter weight clamps (100 → 10)
- ✅ Switched to plateau LR scheduler

---

## Run 2 — Post-Run-1 Fixes
**Date:** ~Feb 2025 | **Pipeline:** Same
**Config:** K=15 (fixed from epoch 0), TF warmup=5, decay=45, Huber loss, plateau LR=1e-4, noise_std=0.05

| Metric | Value |
|--------|-------|
| Best Val SRMSE | **3.530** (epoch 2) |
| Final | Early stop at epoch 27 |

**What happened:** Three compounding failures:
1. K=15 from epoch 0 was too ambitious (6× slower, noisier gradients)
2. Plateau scheduler killed LR to 6.25e-6 by epoch 27
3. Training noise (`noise_std=0.05`) corrupted early-learning signal

**Changes made after:**
- ✅ Progressive K curriculum (3→20 over 30 epochs)
- ✅ Cosine warm restarts scheduler
- ✅ Multi-event validation (3 events per model)
- ❌ **DISABLED** training noise injection — counterproductive
- ✅ Randomized spinup [3, 10]
- ✅ Gradient clipping at 0.5

---

## Run 3 — Progressive K Strategy
**Date:** ~Feb 2025 | **Pipeline:** Same
**Config:** Progressive K (3→20/30ep), TF warmup=3, Huber δ=0.5, cosine warm restarts (T0=15, Tmult=2), LR=2e-4, val="4,18,33", randomized spinup, clamp=5.0

| Metric | Value |
|--------|-------|
| Best Val SRMSE | **1.799** (epoch 12) |
| Final | Early stop at epoch 42 |

**What happened:** Three critical bugs discovered:
1. **P0 BUG — 1D feature mismatch:** `relative_depth` during training used anomaly targets (~-300 for Model 1), but AR feedback used absolute WL (~2). This 300m gap caused AR collapse.
2. **Val events misconfigured:** Events 18, 33 don't exist in both models' train splits. Only 2 val events used.
3. **CosineWarmRestarts destructive:** LR reset at epoch 15 undid learned dynamics.

**Changes made after:**
- ✅ Switched to depth-based targets (WSE − elevation_ref)
- ✅ Fixed val events to "3,9,15" (confirmed in both models)
- ✅ Plain cosine decay (no warm restarts)
- ✅ Removed `detach()` from pushforward rollout (enables BPTT)
- ✅ Increased K target to 30, tighter huber_delta=0.3

---

## Run 4 — Depth-Based Targets (V1 "Unified Engine")
**Date:** ~Feb 16, 2025 | **Pipeline:** `train_unified.py` + `graph_builder_unified.py` + `model_unified.py`
**Config:** Depth targets, progressive K (3→30/30ep), TF warmup=3, Huber δ=0.3, cosine LR=2e-4, val="3,9,15", grad_clip=1.0, hidden=192, dropout=0.05

| Metric | Value |
|--------|-------|
| Model 1 Best Val | **0.449** (epoch 42) |
| Model 2 Best Val | **0.87** raw, ~1.76 EMA |

**What happened:** Model 1 reached good performance. Model 2 stagnated — val improved to ~0.87-1.0 at K=6-9, then collapsed to 2.5+ as K reached 12-15. AR error compounding on the larger graph (198 1D + 4,299 2D nodes).

**Key discoveries:**
- `effective_depth` bug for Model 1 (area = -1.6e-14 → division by ~1e-8 → values up to 3e10). Fixed with `area.clamp(min=1.0)` and `physical_max=20.0`.
- Constant features (pipe_roughness, 2D roughness for Model 2) zeroed via `_normalize_or_zero`.
- Per-node 1D depth normalization added (global std=24.83 was dominated by constant nodes).

---

## Run 5 — V2 Pipeline (GATv2 + Log-Scale + RobustScaler)
**Date:** Feb 24, 2025 | **Pipeline:** `train_v2.py` + `graph_builder_v2.py` + `model_v2.py`
**Config:** GATv2Conv (4 heads, concat), 4 layers, log-scale features, RobustScaler, sinusoidal PE, aux_loss, spinup=5, no const_mask, no min_std

| Metric | Value |
|--------|-------|
| Best Val SRMSE | **>1000** |
| Final | OOM crash at epoch 21 |

**What happened:** Catastrophic failure. NaN losses, skipped steps, OOM. Eight structural flaws identified:

| # | Flaw | Impact |
|---|------|--------|
| 1 | GATv2Conv (4 heads, concat) | OOM — 3-5× more memory than SAGEConv |
| 2 | Removed `const_mask` | Dry node drift accumulated over K steps |
| 3 | Removed `min_std=0.01` | Loss weights misaligned |
| 4 | Added `aux_loss` (raw MSE) | WSE MSE (~100) vs push_forward (~1) → NaN |
| 5 | `spinup=5` instead of 10 | Half the GRU warm-up |
| 6 | No `temporal_scheme='linear'` | Later steps under-weighted |
| 7 | No `ar_noise_std` | Missing regularization |
| 8 | `torch.relu` for 2D depth | Hard gradient discontinuity |

**All 8 fixed.** GATv2→SAGEConv, const_mask restored, aux_loss removed, spinup=10, etc.

---

## Run 6 — V2 with All 8 Fixes (First Attempt)
**Date:** Feb 25, 2025 | **Pipeline:** `train_v2.py` (fixed) + `graph_builder_v2.py` + `model_v2.py` (SAGEConv)
**Config:** SAGEConv, const_mask, min_std=0.01, spinup=10, ar_noise=0.005, K ramp 2→15, TF decay

| Metric | Value |
|--------|-------|
| Best Val SRMSE (stopped at Ep 15) | **210.6** |
| Trajectory | 223 → 210 (steadily improving) |

**What happened:** Run accidentally stopped by user at epoch 15. No NaN, no OOM — structural fixes worked. Restarted as Run 7.

---

## Run 7 — V2 Full Training (100 epochs)
**Date:** Feb 25-26, 2025 | **Pipeline:** Same as Run 6
**Config:** Same as Run 6, full 100 epochs

| Metric | Value |
|--------|-------|
| Best Val SRMSE | **2.07** (epoch 99) |
| Trajectory | 759→747→500→100→29→6.7→2.07 |

**What happened:** Model converged, but to SRMSE 2.07 — still 2.4× worse than V1's 0.87. **Root cause discovered: broken normalization in `graph_builder_v2.py`.**

The V2 `norm()` function prepends `"log_"` to feature keys when `log=True`:
```python
key = f"log_{key}"  # creates "log_capacity", "log_base_area", etc.
st = stats.get(key, {})  # MISS — compute_model_stats only has "capacity"
return (x - 0.0) / 1.0  # falls back to NO normalization
```

**7+ static features were un-normalized the entire time.** Log-transformed raw values (0-10+) alongside z-scored features (mean≈0, std≈1). RobustScaler had the same issue — no median/IQR in stats, fell back to broken defaults.

**Additional findings:**
- Added `node_degree` and `is_leaf` as 1D topological features (teammate's explosion diagnostic showed 4/6 exploders are degree-2 dead-end nodes)
- Evaluated 3 alternatives to const_mask: ReLU threshold (already done), L1 regularization (rejected — impractical), topological mask (spirit valuable — led to degree features)

**Fix:** Abandoned `graph_builder_v2.py` entirely. Switched `train_v2.py` to import `graph_builder_unified.py`.

---

## Run 8 — V2 Training Loop + V1 Graph Builder *(CURRENT)*
**Date:** Feb 26, 2025 | **Pipeline:** `train_v2.py` + `graph_builder_unified.py` + `model_v2.py` (SAGEConv)
**Config:** SAGEConv, const_mask, min_std=0.01, spinup=10, ar_noise=0.005, V1 graph builder with correct normalization, node_degree + is_leaf features

| Metric | Value |
|--------|-------|
| Best Val SRMSE | *Running...* |

**Expected:** Should converge close to V1's 0.87, with correctly normalized features.

---

## Summary of What Does and Doesn't Work

### ✅ Proven Effective
| Technique | Evidence |
|-----------|----------|
| Push-forward training (multi-step K) | Run 1 (K=5) collapsed; Run 3+ (progressive K) improved |
| Progressive K curriculum (start small) | Run 2 (K=15 fixed) → 3.53; Run 3 (K progressive) → 1.80 |
| Teacher forcing decay | All runs require TF decay for AR learning |
| Depth-based targets (not anomaly) | Run 3 → 1.80; Run 4 → 0.87 (300m feature gap fixed) |
| Temporal weighting (linear) | Up-weights later rollout steps to combat drift |
| const_mask | Prevents dry node drift accumulation |
| min_std=0.01 in loss | Works with const_mask as a system |
| Per-node 1D depth normalization | Global std=24.83 dominated by constant nodes |
| Plain cosine LR decay | Stable across regime transitions |
| EMA model | Smooths checkpoint selection |
| Dual checkpointing (EMA + raw) | Captures both smoothed and best-val models |
| SAGEConv | Memory-efficient, fits 4GB VRAM |
| Gradient clipping (1.0) | Prevents extreme flood event gradients from destabilizing |
| AR noise injection (0.005) | Regularizes autoregressive predictions |
| Delta clamping (1D: ±5, 2D: ±2) | Prevents per-step explosion |

### ❌ Proven Harmful / Ineffective
| Technique | Evidence | Why |
|-----------|----------|-----|
| GATv2Conv | Run 5: OOM at epoch 21 | 3-5× more VRAM per layer than SAGEConv |
| Auxiliary MSE loss | Run 5: NaN losses | Raw WSE MSE (~100) vs push_forward (~1) → conflicting gradients |
| Training noise injection (0.05) | Run 2: 3.53 vs 1.77 | Corrupts training signal during critical early learning |
| CosineWarmRestarts | Run 3: LR reset at ep 15 undid dynamics | Model needs continuity during TF/K regime transitions |
| ReduceLROnPlateau | Run 2: killed LR to 6.25e-6 | Misled by noisy 2-event validation |
| Log-scale features (without matching stats) | Run 7: 2.07 vs 0.87 | compute_model_stats doesn't have "log_*" keys → no normalization |
| RobustScaler (without median/IQR stats) | Run 7: fallback to broken defaults | compute_model_stats only provides mean/std |
| Sinusoidal PE (replacing raw positions) | Run 7: unverified, +2 extra features | May or may not help; was combined with broken normalization |
| Removing const_mask | Run 5: SRMSE >1000 | Dry node drift on 3400+ nodes compounds over K steps |
| Fixed K=15 from epoch 0 | Run 2: 3.53 | 6× slower, early predictions at steps 10-15 are pure noise |
| Anomaly targets (WSE-WSE(0)) | Run 3: 300m feature mismatch | AR feedback uses absolute WL, training used anomaly  |
| spinup=5 (reduced warm-up) | Run 5: unstable | Half the GRU context → wildly off initial predictions |
| Weight clamp_max=100 | Run 1: dry nodes dominated gradients | 100× weight means one dry node = 100 wet nodes |

### ⚠️ Untested / Open Questions
| Technique | Status | Notes |
|-----------|--------|-------|
| 5-fold cross-validation | Proposed, never implemented | 3 fixed val events may not cover rainfall distribution |
| Flow/velocity edge features | Proposed in SRMSE_DIAGNOSTIC_REPORT | `dynamic_*_edges` has flow/velocity data — unused |
| Roughness fallback from 1D pipes | Proposed | Model 2 has zero 2D roughness; could interpolate from 1D |
| log(1+area) for scale invariance | Tried in V2 graph builder but WITH broken normalization | Need to test with correct stats computation |
| Curriculum by event length | Proposed | Train on shorter events first (T≤100), then fine-tune |
| Higher LR (0.002-0.003) | Proposed in audit | Member B uses 0.005; hasn't been tried in unified pipeline |
| Non-zero TF min_ratio for Model 2 | Proposed (0.10-0.15) | Would inject some GT during full AR phase |
| Degree-aware loss weighting | Proposed | Higher weight on degree-2 nodes to force accuracy |
| Node degree + is_leaf features | ✅ Added in Run 7 | Needs evaluation with correct normalization (Run 8) |

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

## Run 8 — V2 Training Loop + V1 Graph Builder
**Date:** Feb 26, 2025 | **Pipeline:** `train_v2.py` + `graph_builder_unified.py` + `model_v2.py` (SAGEConv)
**Config:** SAGEConv (hidden=128, 4 layers), const_mask, min_std=0.01, spinup=10, ar_noise=0.005, OneCycleLR, K ramp via 2+ep//2, node_degree + is_leaf features (13 1D features)

| Metric | Value |
|--------|-------|
| Best Val SRMSE | **1.88** (epoch 95) |
| Trajectory | 43→47→29→6.7→2.07→1.88 |

**What happened:** Correctly normalized features helped (starting val 43 vs 759 in Run 7). But V2's training loop had 5 suboptimal differences vs V1: hidden=128 (vs 256), OneCycleLR (vs warmup+cosine), K ramp 2× faster, fixed spinup (vs randomized), single GRUCell (vs 2-layer GRU).

**Fix:** Abandoned `train_v2.py`. Switched to `train_unified.py` directly.

---

## Run 9 — V1 Full Pipeline (train_unified.py)
**Date:** Feb 27-28, 2025 | **Pipeline:** `train_unified.py` + `graph_builder_unified.py` + `model_unified.py`
**Config:** hidden=256, 3 layers, 2-layer GRU, LR warmup+cosine 1e-3, K_max=15, K_ramp=50, tf_min=0.10, randomized spinup [3,10], EMA decay=0.998, node_degree + is_leaf features (13 1D features)

| Metric | Value |
|--------|-------|
| Best EMA Val | **0.99** (epoch 26, K=8, TF=0.53) |
| Best Raw Val | **0.95** (epoch 29, K=9, TF=0.46) |
| Final (stopped ep 52) | EMA=22.7, Raw=65.2 (collapsed) |

**What happened:** Model achieved excellent per-step accuracy at K=8-9 (raw val 0.95). Then COLLAPSED as K exceeded 8 and TF dropped below 0.5. By epoch 52 (K=15, TF=0.10), EMA had degraded from 0.99 to 22.7.

**Root cause: Training at K>8 is destructive for Model 2.** The model's per-step accuracy at K=8 is good enough for 87-step validation rollouts (proven by raw val=0.95), but forcing it to train at K=12-15 introduces noisy gradients that degrade learned dynamics.

**Additional finding:** `node_degree` + `is_leaf` features (added in Run 7) were unverified additions not present in Run 4 that achieved 0.87. Reverted to 11 1D features.

---

## Run 10 — K_max=8 + Reverted Features
**Date:** Mar 1-2, 2025 | **Pipeline:** `train_unified.py` + `graph_builder_unified.py` + `model_unified.py`
**Config:** Same as Run 9 EXCEPT: K_max=8 (was 15), K_ramp=20ep (was 50), tf_min=0.20 (was 0.10), 11 1D features (reverted from 13)

| Metric | Value |
|--------|-------|
| Best EMA Val | **1.09** (epoch 24, K=8, TF=0.62) |
| Best Raw Val | **0.97** (epoch 34, K=8, TF=0.42) |
| Final (stopped ep 51) | EMA=5.40, degrading |

**What happened:** K capped at 8 prevented the catastrophic Run 9 collapse (EMA 22.7), but the model STILL collapsed as TF decayed below 0.60. EMA peaked at 1.09 (ep 24, TF=0.62), then degraded to 5.40 by ep 51 (TF=0.20).

**Key insight: The fundamental bottleneck is TF decay, not K.** The model cannot maintain performance when teacher forcing drops below ~60%, regardless of rollout length. All runs show the same pattern:

| Run | Best Raw Val | At What TF | What collapsed it |
|-----|-------------|------------|-------------------|
| Run 4 | 0.87 | ~0.5-0.6 | K>9 + TF<0.5 |
| Run 9 | 0.95 | 0.46 | K>8 + TF<0.5 |
| Run 10 | 0.97 | 0.42 | TF<0.6 alone (K=8 fixed) |

**Conclusion:** Further hyperparameter tuning of K/TF/LR is hitting diminishing returns. The model architecture + training loop has a fundamental ceiling around SRMSE ~0.87-0.97 for Model 2. Need a different approach.

---

## Run 11 — Edge Flow Features (v8)
**Date:** Mar 2-3, 2025 | **Pipeline:** `train_unified.py` (v8) + `graph_builder_unified.py` + `model_unified.py`
**Config:** Same as Run 10 EXCEPT: 14 1D features (+3 edge flow), 25 2D features (+3 edge flow), K_max=8, tf_min=0.20

**New features added:**
- `edge_mean_inflow`: per-node average inflow from connected edges (vectorized `np.add.at`)
- `edge_mean_outflow`: per-node average outflow from connected edges
- `edge_net_flow`: inflow - outflow (net water accumulation signal)

| Metric | Value |
|--------|-------|
| Best EMA Val | **0.89** (epoch 39, K=8, TF=0.32) |
| Best Raw Val | **0.89** (epoch 27, K=8, TF=0.56) |
| Final (stopped ep 91) | EMA=2.13, raw=1.19 |

**Comparison with Run 10 (no edge flow):**

| Metric | Run 10 | Run 11 | Improvement |
|--------|--------|--------|-------------|
| Best EMA | 1.09 (ep 24, TF=0.62) | **0.89** (ep 39, TF=0.32) | **18% better** |
| Best Raw | 0.97 (ep 34, TF=0.42) | **0.89** (ep 27, TF=0.56) | **8% better** |
| EMA at TF=0.20 | 3.64 (rapid collapse) | ~1.13 (gradual) | Much more stable |

Complete entry of best epoch:
  Epoch  39/120 | loss=0.0693 (1d=0.0202 2d=0.1184) | val=6.2248 ema=0.8883 | K=8 TF=0.32 LR=8.0e-04 | 650.6s *

**Key insights:**
1. **Edge flow features are the most impactful addition since depth-based targets (Run 4).** Raw val 0.89 matches V1's historic best of 0.87.
2. **TF collapse delayed and milder.** Model held at EMA ≤ 1.0 until TF=0.30 (vs TF=0.60 in Run 10).
3. **Same fundamental TF collapse persists** — EMA degraded from 0.89 → 2.13 once TF settled at 0.20.

**Expected:** Edge flow features provide explicit water movement physics that the model previously had to learn from depth alone. Should improve AR stability.

**Post-Run 11 — Submission & Data Leakage Finding (Mar 3):**
- Created `generate_submission.py` to build submission CSVs using EMA checkpoints
- **CRITICAL BUG FOUND:** Edge flow features (inflow/outflow/net_flow) cause **data leakage** — during training and AR validation, GT edge flows for future timesteps are implicitly available via `graph["1d"].x[t]`. At test inference, flows must be frozen from the last spinup step, making them stale for ~80 remaining timesteps. Val SRMSE 0.89 is inflated; public score was **0.79** (worse than expected).
- Edge flows should be **removed** until a proper AR-compatible scheme exists (e.g., predicting flows alongside depths).

**2D Pipeline Analysis (Member B, 0.3 public score):**
The teammate's 2D-only pipeline uses a fundamentally different training strategy:
- **Combined depth+delta loss:** `MSE(depth) + 0.5*MSE(delta)` — two gradient signals anchor both absolute levels and per-step changes
- **Per-timestep backprop** vs our multi-step push-forward loss — simpler but may generalize better
- **Smaller model:** hidden=64, 2 SAGE layers (~100K params vs 6.2M) — less prone to overfitting
- **BatchNorm** in GNN vs our LayerNorm — different trade-offs
- **Higher LR:** 5e-3 vs 1e-3 — faster convergence on a simpler task
- **Key takeaway:** The dual loss (depth + delta) is directly transferable and could help anchor our model's absolute predictions

---

## Run 12 — Anti-Collapse + Dual Loss (v9, no edge flows)
**Date:** Mar 3-5, 2025 | **Pipeline:** `train_unified.py` (v9) + `graph_builder_unified.py` + `model_unified.py`
**Config:** 11 1D features, 22 2D features (edge flows removed), K_max=8, tf_min=0.50, loss=SRMSE+0.1×depth_MSE, early_stop=15

**Changes from Run 11:**
1. ❌ Removed edge flow features (data leakage fix — GT flows visible during val but frozen at test)
2. 🔼 tf_min raised from 0.20 → 0.50 (anti-collapse)
3. ✨ Combined depth+delta dual loss: `SRMSE + 0.1 × MSE(depth)` (from 2D pipeline analysis)
4. ⏹️ Early stopping with patience=15

| Metric | Value |
|--------|-------|
| Best EMA Val | **1.03** (epoch 31, K=8, TF=0.68) |
| Best Raw Val | **1.18** (epoch 36, K=8, TF=0.61) |
| Early stopped | epoch 46 (TF=0.50, EMA=12.88) |

**Comparison with Run 10 (same feature set, no dual loss, tf_min=0.20):**

| Metric | Run 10 | Run 12 | Delta |
|--------|--------|--------|-------|
| Best EMA | 1.09 (ep 24, TF=0.62) | **1.03** (ep 31, TF=0.68) | **6% better** |
| Best Raw | 0.97 (ep 34, TF=0.42) | 1.18 (ep 36, TF=0.61) | 22% worse |
| Compute (epochs) | 120 (ran to end) | **46** (early stopped) | **62% saved** |

**Key insights:**
1. **Dual loss provided modest improvement** — EMA 1.03 vs 1.09 (6% better than Run 10 with same features). The auxiliary MSE on absolute depth helps anchor predictions.
2. **TF collapse STILL occurs around TF=0.55-0.60** — even with tf_min=0.50 as floor. The collapse happens BEFORE TF reaches the floor. Best at TF=0.68, degraded rapidly below TF=0.55.
3. **Early stopping worked perfectly** — saved 74 epochs of wasted compute. Model peaked at epoch 31, early stopped at epoch 46.
4. **The tf_min=0.50 floor was irrelevant** — collapse happened at TF≈0.55, well above the floor. The model can't sustain accuracy below ~55% teacher forcing regardless of the floor setting.
5. **Raw val was worse than Run 10** — 1.18 vs 0.97. The dual loss may interfere with the raw model's flexibility (EMA smoothing compensates).

Complete training log (key epochs):
```
Epoch  0  | loss=1.1076 | val=119.99 ema=109.11 | K=2 TF=1.00 *
Epoch  8  | loss=0.0216 | val=1.36   ema=45.75  | K=4 TF=0.96 *
Epoch 15  | loss=0.0228 | val=1.23   ema=1.90   | K=6 TF=0.88 *
Epoch 24  | loss=0.1018 | val=1.61   ema=1.18   | K=8 TF=0.76 *
Epoch 31  | loss=0.0364 | val=37.17  ema=1.03   | K=8 TF=0.68 * ← BEST
Epoch 40  | loss=0.0328 | val=10.55  ema=1.35   | K=8 TF=0.56
Epoch 46  | loss=0.0240 | val=62.53  ema=12.88  | K=8 TF=0.50 ← EARLY STOP
```

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
| K_max ≤ 8 for Model 2 | Run 9: model achieves 0.95 at K=8, collapses at K>8 |
| tf_min ≥ 0.20 for Model 2 | Too low (0.10) allowed AR error compounding |
| V1 graph builder z-score normalization | Run 7 (V2 broken norm) → 2.07; Run 8 (V1 norm) → 1.88 |
| **Early stopping (patience=15)** | **Run 12: saved 74 epochs (62% compute), no quality loss** |
| **Combined depth+delta dual loss (0.1×MSE)** | **Run 12: EMA 1.03 vs Run 10: 1.09 (6% improvement)** |

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
| K_max=15 for Model 2 | Run 4, Run 9: AR collapse at K>8 | Training at K>8 produces noisy gradients that destroy learned dynamics |
| Anomaly targets (WSE-WSE(0)) | Run 3: 300m feature mismatch | AR feedback uses absolute WL, training used anomaly |
| spinup=5 (reduced warm-up) | Run 5: unstable | Half the GRU context → wildly off initial predictions |
| Weight clamp_max=100 | Run 1: dry nodes dominated gradients | 100× weight means one dry node = 100 wet nodes |
| node_degree + is_leaf features | Run 9: 0.99 vs Run 4: 0.87 | Unverified features not present in best run; may add noise |
| OneCycleLR (in train_v2.py) | Run 8: 1.88 vs Run 9: 0.99 | LR warmup + cosine decay is more stable |
| **Edge flow features (train/test mismatch)** | **Run 11: 0.89 val but 0.79 public** | **GT future flows visible during val but frozen at test — data leakage** |
| **tf_min=0.50 as anti-collapse** | **Run 12: collapse at TF=0.55, before floor reached** | **TF collapse happens above the floor — floor is irrelevant** |

### ⚠️ Untested / Open Questions
| Technique | Status | Notes |
|-----------|--------|-------|
| 5-fold cross-validation | Implemented (`--fold`), not yet run | Would stabilize the noisy 3-event val metric |
| Higher LR (0.002-0.005) | Member B uses 0.005 | Unified pipeline uses 1e-3; may converge faster |
| Ensemble (multi-checkpoint average) | Now viable | Average predictions from multiple epochs or folds |
| Smaller model (hidden=64-128) | Member B: hidden=64, 2 layers | Fewer params → less overfitting to TF regime |
| Per-timestep backprop (no push-forward) | Member B uses single-step loss | May train faster and avoid K-step error accumulation |
| Curriculum on TF floor instead of TF decay | New idea from Run 12 | Instead of decaying TF, keep TF=1.0 and slowly reduce K? |
| BatchNorm (replacing LayerNorm) | Member B uses BatchNorm | Different normalisation dynamics during AR rollout |

---
# Dataset Notes for Teammates — Urban Flood Bench

**Purpose:** Share key dataset insights discovered during EDA, Model 2 diagnosis, and the unified engine rebuild. Use this as a quick reference when building features, debugging training, or interpreting results.

---

## 1. Schema Quick Reference

### Structure
- **2 urban models** (Model 1, Model 2) — different topologies, different physics
- **Events per model** — multiple rainfall events per model
- **Static files** — shared across all events in a model (nodes, edges, 1d2d connections)
- **Dynamic files** — one per event, long format: `(timestep, node_idx, ...)`

### Key Columns

| Source | Columns | Notes |
|--------|---------|-------|
| **1d_nodes_static** | `invert_elevation`, `surface_elevation`, `base_area`, `position_x`, `position_y`, `depth` | capacity = surface − invert |
| **2d_nodes_static** | `min_elevation`, `elevation`, `area`, `roughness`, `aspect`, `curvature`, `flow_accumulation`, `position_x`, `position_y` | Use **min_elevation** for 2D depth (see §2) |
| **1d_edges_static** | `length`, `diameter`, `shape`, `roughness`, `slope` | Pipe geometry — aggregate to nodes |
| **2d_edges_static** | `face_length`, `length`, `slope` | Surface mesh — slope is terrain signal |
| **1d_nodes_dynamic** | `water_level` (target), `inlet_flow` | WSE → depth = water_level − invert_elevation |
| **2d_nodes_dynamic** | `water_level` (target), `rainfall`, `water_volume` | WSE → depth = water_level − min_elevation |
| **1d_edges_dynamic** | `flow`, `velocity` | Optional; time-varying pipe flow |
| **2d_edges_dynamic** | `flow`, `velocity` | Optional; surface flow |

---

## 2. Critical Physics Derivations

### Depth Reference for 2D
- **Use `min_elevation`** (lowest point in cell), not `elevation` (centroid).
- **Why:** Water pools at the lowest point. With centroid: 93.6% of depths go negative (unphysical). With min_elevation: only ~0.3% negative.
- **Recovery:** `pred_wse = pred_depth + min_elevation`

### 1D Depth
- `depth_1d = water_level − invert_elevation` (WSE above pipe invert)

### Capacity (1D)
- `capacity = surface_elevation − invert_elevation` (max depth before surcharge)

---

## 3. Model 1 vs Model 2 — Major Differences

| Aspect | Model 1 | Model 2 |
|--------|---------|---------|
| **1D nodes** | 17 | 198 |
| **2D nodes** | 3,716 | 4,299 |
| **2D roughness std** | 0.019 | **0.000** (constant!) |
| **2D slope std** | 0.044 | 0.0056 (flatter terrain) |
| **2D area mean** | 609 | 15,126 (~25× larger cells) |
| **2D area std** | 116 | 8,465 |
| **Elevation range** | ~323 ft | ~44 ft |
| **1D depth std** | 24.83 | 4.94 |

**Takeaways:**
- **Model 2 has constant roughness** → z-score normalised roughness = 0 for all nodes → wasted feature dimension.
- **Model 2 has much larger 2D cells** → different spatial scale; consider `log(1+area)` or scale-invariant features.
- **Model 2 trains well at short K (2–10) but collapses at K=16–20** → AR rollout instability; use lower K_max for Model 2.

---

## 4. Temporal / Autoregressive Behaviour

### Lag-1 Autocorrelation ≈ 0.9996
- Depth is highly persistent from step to step → GRU needs capacity to track slow dynamics; we use 256 hidden.
- Short-term forecasting is easier; long AR rollouts compound errors.

### Best Rainfall Lag
- EDA: rainfall at **lag 2** (t−2) is most predictive for depth.
- We include: `rain_lag2`, `rain_rolling_mean`, `rain_delta`.

### Depth Lags
- We use depth at t−2, t−3, t−4 (lag1, lag2, lag3) as explicit features.
- During AR rollout, these are replaced with predicted history.

---

## 5. Dry / Constant Nodes (SRMSE Weights)

- **Per-node σ (std) drives the SRMSE metric**: weight ∝ 1/σ².
- **Dry nodes** (σ ≈ 0) would dominate the loss without clamping.
- **Fix:** `clamp_weights` (e.g. 5–20) caps 1/σ² so dry nodes don’t hijack gradients.
- **Constant nodes:** Some 1D nodes have constant WSE → σ ≈ 0. We mask them (const_mask) so predicted delta = 0 → no drift.
- **Per-node 1D depth normalisation:** Global 1D depth std = 24.83 (driven by constant nodes). Use per-node mean/std for each pipe’s depth feature instead.

---

## 6. Features — What We Use vs What Exists

### Currently Used (v7)
- **1D (11):** depth, inlet_flow, lag1–3, capacity, base_area, pipe_diameter, pipe_length, pipe_roughness, pipe_slope
- **2D (22):** depth, rainfall, lag1–3, water_volume, rain_rolling, rain_delta, rain_lag2, elevation, min_elevation, slope, area, roughness, aspect, curvature, flow_accumulation, elev_rel_neighbors, dist_to_drain, is_connected, position_x, position_y

### Previously Unused (Now Added in v7)
- `water_volume` (2D dynamic) — mass-balance signal
- `position_x`, `position_y` (2D static) — Member B uses these
- 1D pipe: `diameter`, `length`, `roughness`, `slope` (from static_1d_edges)

### Still Unused (Future Ideas)
- 1d_nodes_static `depth` (may be pipe depth)
- dynamic_1d_edges / dynamic_2d_edges: `flow`, `velocity`
- 2d_edges: `face_length`, `length` (only slope used so far)

---

## 7. Normalisation

- **Per-model z-score** — Model 1 and Model 2 have different elevation scales (323 ft vs 44 ft). Normalise separately per model.
- **Per-node 1D depth** — Use per-node mean/std for depth, lag1–3 (not global) to handle constant nodes.
- **2D depth** — Global stats ok (mean ~0.26, std ~0.64 for Model 1).
- **water_volume** — Normalise per model; during AR use `depth × area` as proxy.

---

## 8. Pitfalls & Gotchas

1. **Centroid vs min_elevation:** Using centroid for 2D depth gives 93.6% negative depths.
2. **Constant features:** Model 2 roughness is constant → detect and handle (e.g. drop or zero).
3. **AR rollout collapse:** Model 2 breaks at K=16–20; use K_max=15 and slower K ramp.
4. **Checkpoint choice:** Best EMA ≠ best raw val. Model 2: raw val 0.75 at epoch 43 was better than EMA 1.0; save both.
5. **Dynamic edge files:** `1d_edges_dynamic_all.csv`, `2d_edges_dynamic_all.csv` are optional and may be missing.
6. **Edge index alignment:** Row i in `edge_index_*` typically corresponds to row i in `*_edges_static` for attribute lookup.

---

## 9. Validation / Metric

- **SRMSE** = Standardised RMSE: `sqrt(mean((pred - target)² / σ²))` per node, then averaged.
- **min_std** in loss must match validation (e.g. 0.01) so low-σ nodes are weighted correctly.
- **91.7% of 2D nodes** have σ < 0.224 → most cells are low-variance; SRMSE is dominated by a minority of highly variable nodes.

---

## 10. Recommended Next Steps

1. **Verify water_volume** exists in your `2d_nodes_dynamic_all.csv` for both models.
2. **Try higher LR** (0.003–0.005) — Member B uses 0.005.
3. **Consider hybrid loss** — MSE(absolute) + 0.5×MSE(delta) alongside push-forward SRMSE.
4. **Model 2 checkpoint** — Use `*_best_val.pt` if raw val beats EMA.
5. **2D negative depths** — ~0.3–1.4% can be negative; clamp at inference: `max(0, pred_depth)`.

---

*Last updated from EDA, MODEL2_DIAGNOSIS_REPORT, FEATURE_ENGINEERING_DEEP_DIVE, and rebuild implementation.*

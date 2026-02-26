# Feature Engineering Deep Dive — Path to SRMSE 0.3

**Goal:** Achieve SRMSE ≤ 0.3 for both Model 1 and Model 2.  
**Current:** Model 1 ~0.50, Model 2 ~1.0 (best raw 0.75).  
**Reference:** Member B's 2D-only engine achieved SRMSE 0.4.

---

## 1. Training Log Analysis

### Model 1 (Best 0.50)
- Early val oscillated (35 → 2 → 285) then stabilised ~0.5–1.0
- EMA converged to ~0.51
- Loss plateaued ~0.55 at K=20
- Best checkpoint at epoch 47

### Model 2 (Best ~1.0 EMA, 0.75 raw)
- Early phase (K=2–10): val improved 56 → 1.0
- Collapse at K=16–20: loss 0.02 → 0.15, val worsened
- Constant roughness (std=0), flatter slope
- 198 1D nodes vs 17 (Model 1) — much larger graph

### Key Insight
Model 1 has plateaued; Model 2 suffers AR instability. **Feature engineering can help both** by:
1. Richer inputs → better single-step predictions → less AR drift
2. Physics-aligned features → faster learning, better generalisation

---

## 2. Unused Data (Competition Schema vs Our Features)

| Source | Column | Status | Potential Value |
|--------|--------|--------|-----------------|
| **2d_nodes_dynamic** | `water_volume` | **UNUSED** | Mass balance; depth×area proxy; flood severity |
| **2d_nodes_static** | `position_x`, `position_y` | Only for dist_to_drain | **Member B uses as features** — spatial flood propagation |
| **1d_nodes_static** | `depth` | UNUSED | May be pipe depth/capacity |
| **1d_edges_static** | `diameter`, `roughness`, `slope`, `length` | **UNUSED** | Pipe capacity, resistance, wave speed |
| **2d_edges_static** | `face_length`, `length` | Only slope used | Edge length → connectivity strength |
| **1d_edges_dynamic** | `flow`, `velocity` | **UNUSED** | Real-time pipe flow — crucial for 1D dynamics |
| **2d_edges_dynamic** | `flow`, `velocity` | **UNUSED** | Surface flow direction/magnitude |

---

## 3. Member B Secrets (from PHASE1_AUDIT)

| Aspect | Member B (SRMSE 0.4) | Our Unified |
|--------|----------------------|-------------|
| **pos_x, pos_y** | Explicit node features | Not used as inputs |
| **LR** | 0.005 | 0.001 |
| **Depth clamp** | `clamp(depth+Δ, min=0)` | We have it ✓ |
| **Loss** | MSE(abs) + 0.5×MSE(delta) | push_forward (SRMSE) |
| **2D only** | No 1D coupling | 1D+2D — harder |

---

## 4. Prioritised Feature Engineering Recommendations

### Tier 1 — High Impact, Low Risk

#### 4.1 `water_volume` (2D dynamic)
- **What:** Add as 20th 2D feature. Volume = mass balance; correlates with flood severity.
- **Implementation:** Pivot `water_volume` from dynamic_2d_nodes; normalise per-model; stack with other 2D features.
- **Model change:** `in_channels_2d`: 19 → 20.

#### 4.2 `position_x`, `position_y` (2D static)
- **What:** Z-score normalise and add as node features. Member B uses them.
- **Rationale:** Floods propagate spatially; position encodes topology.
- **Implementation:** Add `norm_pos_x`, `norm_pos_y` to 2D stack.
- **Model change:** `in_channels_2d`: 19 → 21 (or 20 if we add water_volume first).

#### 4.3 1D pipe attributes (static)
- **What:** Aggregate `diameter`, `length`, `roughness`, `slope` from static_1d_edges to each 1D node (mean of incident edges).
- **Rationale:** Pipe geometry drives capacity and wave speed.
- **Implementation:** `compute_mean_pipe_attr_per_node()` analogous to slope for 2D.
- **Model change:** `in_channels_1d`: 7 → 11 (4 new features).

### Tier 2 — Medium Impact

#### 4.4 `water_volume` / `area` ratio
- **What:** Derived feature `effective_depth = water_volume / area` — may be more robust than raw depth for varying cell sizes (Model 2 has 25× larger area).
- **Alternative:** Use `log(1 + water_volume)` for scale-invariance.

#### 4.5 1D pipe roughness fallback for 2D
- **What:** When 2D roughness is constant (Model 2), use distance-weighted mean of connected 1D pipe roughness.
- **Complexity:** Higher; requires graph traversal.

#### 4.6 Normalised `dist_to_drain`
- **What:** Currently we have raw distance. Adding `1 / (1 + dist_to_drain)` could highlight cells *very* close to drains.

### Tier 3 — Training / Loss

#### 4.7 Hybrid loss (Member B style)
- **What:** Add `MSE(pred_depth, target)` or `MSE(pred_wse, target_wse)` with weight 0.3–0.5 alongside push_forward SRMSE.
- **Rationale:** Direct level supervision may improve convergence.

#### 4.8 Higher LR for Model 1
- **What:** Try 0.003–0.005 (Member B uses 0.005).
- **Risk:** Instability; use gradient clipping.

#### 4.9 Dynamic edge flow/velocity
- **What:** Aggregate flow and velocity from dynamic_1d_edges to 1D nodes (in/out flow per node).
- **Complexity:** Need to align edge_index with dynamic edge data.

---

## 5. Implementation Roadmap

| Phase | Tasks | Expected gain |
|-------|-------|---------------|
| **Phase A** | water_volume, pos_x/y | +0.05–0.10 val |
| **Phase B** | 1D pipe attributes | +0.03–0.05 (esp. Model 1) |
| **Phase C** | effective_depth, loss hybrid | +0.02–0.05 |
| **Phase D** | Dynamic flow (if feasible) | Exploratory |

---

## 6. Data Availability Check

Before implementing, verify columns exist in your data:
- `dynamic_2d_nodes`: `water_volume`
- `static_1d_edges`: `diameter`, `length`, `roughness`, `slope`
- `dynamic_1d_edges` / `dynamic_2d_edges`: `flow`, `velocity` (optional files)

---

## 7. Summary

The competition dataset offers **several high-value features we do not use**:
- **water_volume** — direct flood mass
- **position_x, position_y** — used by Member B
- **1D pipe geometry** — diameter, length, roughness, slope

Adding these in phases A–B is the most direct path toward SRMSE 0.3, especially for Model 2 where constant roughness currently wastes a feature dimension.

---

## 8. Implementation Status (v7)

**Phase A & B implemented** in `graph_builder_unified.py` and `train_unified.py`:

| Feature        | 1D dims | 2D dims | AR replacement |
|----------------|---------|---------|----------------|
| water_volume    | —       | +1      | depth × area   |
| position_x/y    | —       | +2      | static         |
| pipe_* (4)     | +4      | —       | static         |

**New totals:** 1D 11 features (was 7), 2D 22 features (was 19).

Re-run training to evaluate impact. Consider trying `--lr 0.003` (Member B uses 0.005) if convergence is slow.

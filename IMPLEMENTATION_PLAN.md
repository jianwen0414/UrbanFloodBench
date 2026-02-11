Workflow

> **Document Version:** v4.0 — Updated after Run 3 forensic analysis.
> All "Revision Notes" below document changes discovered during the
> iterative training cycle that were **not** in the original plan.

---

Task 0.1: The Baseline Benchmark (XGBoost)
Owner: Member C (Lead Architect) 
Status: COMPLETED.
Objective: Establish a "Score to Beat." If our complex GNNs cannot beat this simple model, our GNNs are flawed.
Direction: Treat the problem as a Tabular Regression problem, ignoring the graph topology.
Implementation Strategy:
Data Structure: Flatten the data.
Input: [Rain_t, WaterLevel_t-1, WaterLevel_t-2, WaterLevel_t-3, Month, Hour].
Target: WaterLevel_t.
Model: Use XGBRegressor (from xgboost) or LGBMRegressor (from lightgbm).
Training: Train one model per Node Type (one model for all 1D nodes, one model for all 2D nodes). Do not train 5,000 separate models. Add Node_ID or Node_Statistics (mean/std) as features so the model distinguishes between nodes.
Organizer Insight Compliance:
"Sometimes simple data get better results" — This is the ultimate test of that hint.
Deliverable:
A script src/baseline_xgb.py that loads data, trains the booster, and outputs a submission.csv.
Benchmark Score: A specific RMSE number on the Public Leaderboard to serve as the team's "Minimum Viable Product."


Phase 1: Data Infrastructure (The Foundation)
Objective: Transform raw CSVs into "Physics-Ready" PyTorch Geometric (PyG) objects.
Organizer Constraint: "Keep data simple." Do not over-engineer features.
Task 1.1: The Universal Lazy Loader (Owner: Member C)
Status: COMPLETED.
Direction: The dataset is huge. If we load everything into RAM, we crash. Build a FloodDataset class that loads one event at a time on demand.
Organizer Insight: Static features are stored once per model. Do not reload 1d_nodes_static.csv for every event. Cache it.
Deliverable: A Python Class FloodDataset(root_dir, mode='train') that yields a dictionary of DataFrames for a specific event when indexed.
Task 1.2: 1D Graph Engineering (Owner: Member A)
Status: COMPLETED.
Direction: Construct the graph for the pipe network.
Critical Insight: Pipes are not just lines; they are containers.
Feature 1: Capacity. Calculate Surface_Elevation - Invert_Elevation. This is the physical ceiling. If water > Capacity, it's a flood.
Feature 2: Relative Depth. The model must see Water_Level - Invert_Elevation. Absolute sea-level elevation is useless for flow physics.
Connectivity: Pipes flow both ways (backwater effect). Create Bidirectional Edges in PyG (u → v and v → u).
Deliverable: A function build_1d_graph(static_df, dynamic_df) returning a PyG Data object.
Task 1.3: 2D Graph Engineering (Owner: Member B)
Status: COMPLETED.
Direction: Construct the graph for the surface mesh.
Critical Insight: The 2D model is blind to the pipes if we separate them. We must give it a "hint" about where the drains are.
Feature 1: The "Sink" Hint. Calculate the Euclidean Distance from every 2D node to the nearest 1D node. Add this as a static feature.
Feature 2: Local Topology. Instead of raw elevation, use Z-Scored Elevation relative to the immediate neighbors. This highlights "depressions" where water pools.
Deliverable: A function build_2d_graph(static_df, dynamic_df) returning a PyG Data object.
Task 1.4: Multi-Fidelity Graph Engineering (The Factory)
Owner: Member A & B (Co-developed)
Status: COMPLETED.
Objective: Create a centralized module (src/graph_builder.py) that converts raw CSV data into PyTorch Geometric graph objects. This module must support all three modeling approaches.
Functions to Implement:
build_1d_graph: Constructs a homogeneous graph for pipes.
Crucial: Edges must be Bidirectional ($u \to v$ and $v \to u$) to allow backwater flow.
Features: Relative Depth, Capacity, Rain.
build_2d_graph: Constructs a homogeneous graph for the surface mesh.
Crucial: Implements "Soft Coupling". Uses 1d2d_connections.csv to calculate a static feature: "Euclidean Distance to Nearest 1D Node".
Features: Z-Scored Elevation, Roughness, Rain, Dist_to_Drain.
build_unified_graph: Constructs a Heterogeneous Graph (HeteroData) for the Unified Engine.
Crucial: Implements "Hard Coupling". Uses 1d2d_connections.csv to create explicit physical edges (node_1d to node_2d).
Deliverable: A robust src/graph_builder.py containing these three functions.

Data Flow Checkpoint
At the end of Phase 1, the team should have a pipeline where:
Member C's loader fetches an event CSV.
Member A's function turns the pipe data into a Graph.
Member B's function turns the surface data into a Graph.


Phase 2: The Modeling Engines (1D & 2D)
Strategic Pivot: The organizer explicitly stated, "Simple data get better results" and "Separate will get better results."
Implication: We will not build a complex "Meta-Model" that learns 1D and 2D simultaneously.
Action: We build two robust, independent models. This simplifies debugging significantly—if the pipes are predicting well but the street is wrong, we know exactly which "Engine" to fix.
Task 2.1: The 1D "Pipe Engine" (Owner: Member A)
Objective: Build the specialized neural network for the underground pipe graph.
Architecture: GCN-GRU (Graph Convolutional Network + Gated Recurrent Unit).
Why: GCN passes information spatially between connected manholes; GRU maintains the temporal memory of how full the pipe was at $t-1$.
Physics-Fallback Protocol (Updated Requirement):
Stage 1 (Default): The model must be configured to predict ONLY water_level ($Output \in \mathbb{R}^1$). This reduces noise and variance.
Stage 2 (Emergency): The code structure must allow expanding the output head to predict [water_level, flow, velocity] ($Output \in \mathbb{R}^3$) later. This is only triggered if the Stage 1 model fails to converge or produces physically impossible results (e.g., water moving uphill).
Deliverable: src/model_1d.py containing a flexible PipeEngine class.

Task 2.2: The 2D "Surface Engine" (Owner: Member B)
Objective: Build the specialized neural network for the surface terrain graph.
Architecture: GraphSAGE-GRU (Graph Sample and Aggregate + Recurrent Unit).
Why: GraphSAGE is designed for inductive learning on large graphs and excels at modeling "diffusion" processes (like water spreading on a map).
Coupling Protocol:
Standard (Soft Coupling): The model relies on the dist_to_drain static feature (implemented in Task 1.4) to "know" where the sinks are. It does not have physical edges to the pipes.
Fallback Strategy: If the Soft Coupling underperforms, we do not complicate this model. We switch to Task 2.4 (Unified Engine) which explicitly handles the physical 1D-2D edges.
Key Implementation Detail: Use LeakyReLU activations. Most surface nodes are dry (value=0.0) for most of the time. Standard ReLU kills gradients on zeros ("Dead Neurons"). LeakyReLU allows small gradients to flow even when dry.
Deliverable: src/model_2d.py containing the SurfaceEngine class.

Task 2.3: The Physics-Informed Loss Function (Owner: Member C)
Status: COMPLETED + REVISED (see below).
Objective: "Hack" the metric. The model must learn to minimize the Organizer's Specific Error, not standard mathematical error.
The Equation:

Critical Implementation Detail:
You cannot just use MSELoss. You must write a custom WeightedMSE.
Data Flow: The Universal Loader (from Task 1.1) must pass the node_std tensor to the training loop.
Handling "Explosion": If a node has σ ≈ 0 (always dry), the weight 1/σ² becomes huge ($1,000,000+$).
The Fix: Clamp the weights. weight = torch.clamp(1.0 / (std**2 + 1e-6), max=100.0). This prevents one dry node from hijacking the entire gradient descent.

> **REVISION NOTE — Clamp Weight Tightened (Run 1 Analysis):**
> The original plan specified `clamp_max=100.0`.  Run 1 forensics revealed
> that even 100.0 is too permissive — dry nodes (σ ≈ 0) still dominate
> gradients with weights around 100.  Reduced to `clamp_max=10.0` in
> Run 1, then further to `clamp_max=5.0` in Run 3.
> **Justification:** Lower clamp prevents a handful of always-dry nodes
> from consuming >50% of the total gradient magnitude.  The 5.0 cap means
> the worst-case dry-node weight is 5× a typical wet node (instead of
> 100×), which is a healthier gradient balance.

> **REVISION NOTE — Huber Loss Variant Added (Run 1 Analysis):**
> The original plan only mentioned MSE-based loss.  A Huber (Smooth L1)
> variant `standardized_huber_loss()` was added to `loss.py` and is now
> the **production default** (`loss_variant="huber"`, `huber_delta=0.5`).
> **Justification:** Extreme flood events produce catastrophic spikes
> in squared error that destabilise gradient descent.  Huber transitions
> from quadratic to linear beyond δ=0.5, capping the influence of any
> single outlier timestep on the gradient.  This is critical for the
> push-forward training regime (Task 3.1b) where late-rollout predictions
> can have very large errors.

> **REVISION NOTE — Push-Forward Trajectory Loss Added (Run 1 Analysis):**
> A new loss function `push_forward_loss()` was added that computes SRMSE
> over an entire K-step rollout trajectory with temporal weighting
> (uniform / linear / exponential).  Later steps receive higher weight
> ("linear" scheme) to explicitly penalise autoregressive drift.
> This is used inside `combined_flood_loss()` when `use_push_forward=True`.
> **Justification:** Single-step loss (predicting t+1 from perfect t)
> teaches the model nothing about error compounding.  Multi-step
> trajectory loss forces the model to fight its own drift, directly
> addressing the "autoregressive instability" problem described in Task 3.1.

Deliverable:
A function standardized_rmse_loss(pred, target, stds) validated against the organizer's sample evaluation script.
Additional deliverables (new):
- standardized_huber_loss() — outlier-robust variant.
- push_forward_loss() — multi-step trajectory loss.
- FloodLoss nn.Module — wraps both variants with stored per-node σ buffers.
- SRMSEAccumulator — hierarchical validation metric accumulator.

Task 2.4: The Unified Engine (1D-2D Coupled GNN) — **PRIMARY ENGINE**
Status: COMPLETED + ACTIVELY ITERATED.
Objective: Build a Heterogeneous GNN that explicitly models the interaction between pipes and the surface. This addresses the "Explicit Coupling" requirement.
Architecture: HeteroGNN-GRU (128 hidden channels, 3 GNN layers, 2 GRU layers).
Input: The HeteroData object from graph_builder.build_unified_graph.
Mechanism: Uses HeteroConv to perform message passing across four directed edge types:
1. Pipe-to-Pipe (GCNConv) — bidirectional pipe flow.
2. Surface-to-Surface (SAGEConv) — surface mesh adjacency.
3. Surcharge: 1D → 2D (SAGEConv) — pipe overflow onto street.
4. Drainage: 2D → 1D (SAGEConv) — street water draining into pipes.

> **REVISION NOTE — Four Edge Types (Implementation):**
> The original plan described three edge types.  In implementation,
> the surcharge (1D→2D) and drainage (2D→1D) directions were separated
> into distinct edge types with independent learned message functions.
> **Justification:** Pressure-driven surcharge overflow and gravity-driven
> inlet drainage are fundamentally different physics.  Shared weights
> would force the model to use the same learned function for both,
> which is physically incorrect.

> **REVISION NOTE — Prediction Output Clamping (Run 1+3 Analysis):**
> Autoregressive rollout predictions are clamped per node type:
>   - 1D: `clamp(-2.0, 30.0)` — depth can be slightly negative (below
>     invert), upper bounded by max capacity (~25m) + surcharge margin.
>   - 2D: `clamp(-0.5, 15.0)` — surface depth is ~non-negative.
> Originally ±20.0 (Run 1), then ±10.0 (Run 2), now depth-specific (Run 4).
> **Justification:** With depth-based targets, the prediction space has
> physical meaning.  Per-type clamps are tighter and more informative
> than a symmetric ±10 bound that doesn't distinguish pipes from surfaces.

Deliverable: src/model_unified.py containing the UnifiedFloodModel class.


Interconnection & Data Flow (Phase 2)
The primary workflow now centres on the Unified Engine:
- Member C runs train_production.py using both Model_1 and Model_2 data simultaneously.
- The Unified Engine sees all four edge types (pipe, surface, surcharge, drainage).
- Loss function is FloodLoss (Huber variant, α=0.5 for 1D/2D balance).
- Member C monitors Training Curves and val SRMSE.


Phase 3: The Winning Edge (Training, Validation & Submission)
Strategic Goal: Solve the "Autoregressive Instability."
The Problem: Your model predicts t_11 with a 1% error. It uses that wrong prediction to predict t_12. The error becomes 2%. By t_100, the model predicts a tsunami where there is a puddle.
The Solution: Curriculum Learning & Rigorous Validation.

Task 3.1: The "Anti-Drift" Training Protocol (All Members)
Status: COMPLETED + EXTENSIVELY REVISED (see sub-tasks below).
Objective: Teach the model to recover from its own mistakes.

### Task 3.1a: Scheduled Sampling (Teacher Forcing Curriculum)
Action: Both Member A (1D) and Member B (2D) must implement Scheduled Sampling inside their training loops.
The Schedule (Curriculum):
Phase 1 (Warm-up, Epochs 0–2): Teacher Forcing = 100%. Always feed the Ground Truth of t-1 as input for t.
Goal: Model learns physics quickly.
Phase 2 (Weaning, Epochs 3–32): Linear Decay. Decay Teacher Forcing ratio from 1.0 → 0.0.
Mechanism: For every batch, flip a coin. If Heads, use Ground Truth. If Tails, use the Model's own prediction from the previous step.
Phase 3 (Realism, Epochs 33+): Student Forcing = 100%. Always use the Model's own prediction.
Goal: Simulates the actual Kaggle test environment.

> **REVISION NOTE — Aggressive TF Warmup (Run 1–2 Analysis):**
> Original plan: warmup=10 epochs, decay=30 epochs.
> Production config: **warmup=3 epochs, decay=30 epochs**.
> **Justification:** Run 1 achieved its best val SRMSE at epoch 3 (during
> pure TF) then collapsed when TF started decaying at epoch 10.  Run 2
> tried warmup=5 and still collapsed.  The model was learning to rely on
> perfect ground-truth inputs for too long, creating a distribution shift
> when it finally faced its own predictions.  A 3-epoch warmup forces
> the model to face autoregressive errors from epoch 3 onward, preventing
> this dependency.

Deliverable: A train_step() function that accepts a teacher_forcing_ratio argument and toggles input sources dynamically.

### Task 3.1b: Push-Forward Training (NEW — Not in Original Plan)
Status: COMPLETED.
Owner: Member C.
Objective: Close the gap between training (short-horizon, teacher-forced) and validation (full autoregressive rollout, 40+ steps).

> **REVISION NOTE — Push-Forward Training (Run 1 Analysis):**
> This entire sub-task was added after Run 1 forensics revealed the root
> cause of autoregressive collapse: the model trained on single-step
> predictions but was validated on 40+ step rollouts.  Each step's error
> compounds through the feedback loop.
>
> **Implementation:**
> Instead of computing loss on a single next-step prediction, the model
> rolls out K steps autoregressively and computes loss on the full
> trajectory.  Temporal weighting (linear scheme) penalises later steps
> more heavily to explicitly fight drift.
>
> **Current Production Settings:**
> - `pushforward_K = 20` (target rollout length)
> - `temporal_scheme = "linear"` (later steps weighted more)
> - `use_push_forward = True`

### Task 3.1c: Progressive K Curriculum (NEW — Not in Original Plan)
Status: COMPLETED.
Owner: Member C.
Objective: Prevent the model from being overwhelmed by long rollout losses during early training.

> **REVISION NOTE — Progressive K Curriculum (Run 2 Analysis):**
> Run 2 used a fixed K=15 from epoch 0.  This **tripled** epoch time
> (33s → 193s) and produced worse results (best val 3.53 vs 1.77)
> because early-epoch predictions at steps 10-15 are pure noise,
> flooding the loss with uninformative gradients.
>
> **Implementation:**
> K starts at `K_start=3` and ramps linearly to `pushforward_K=20` over
> `K_ramp_epochs=30` epochs.  This creates a smooth curriculum:
> ```
> Epoch 0:  K=3   (learn single-step accuracy first)
> Epoch 10: K=10  (extend to medium horizons)
> Epoch 30: K=20  (match validation rollout length)
> ```
> **Justification:** The model must master short-horizon prediction
> before being challenged on longer trajectories.  This mirrors
> how the TF curriculum (3.1a) gradually reduces teacher forcing —
> both curricula work together to smoothly transition from "easy
> training" to "realistic evaluation conditions."
>
> **Config fields:** `progressive_K=True`, `K_start=3`, `K_ramp_epochs=30`.

### Task 3.1d: Randomized Spinup Length (NEW — Not in Original Plan)
Status: COMPLETED.
Owner: Member C.
Objective: Prevent the model from overfitting to "perfect 10-step spinup" hidden states.

> **REVISION NOTE — Randomized Spinup (Run 2 Analysis):**
> During training, the model always received a perfect 10-step spinup
> with ground-truth inputs.  During validation, the GRU hidden states
> degrade as errors accumulate.  This is an exposure bias: the model
> has never seen an imperfect hidden state during training.
>
> **Implementation:**
> During training, the spinup length is randomly sampled from
> `[spinup_min, spinup_max]` = `[3, 10]` for each event.
> Sometimes the model gets only 3 ground-truth steps of warm-up,
> forcing it to be robust to under-initialised hidden states.
>
> **Justification:** In the competition test set, the model receives
> exactly 10 spinup steps — but even those 10 steps build hidden states
> that are subtly different from training because the model parameters
> have never seen the test event's patterns before.  By randomizing
> spinup during training, the model learns to produce reasonable
> predictions regardless of hidden state quality.
>
> **Config fields:** `randomize_spinup=True`, `spinup_min=3`, `spinup_max=10`.

### Task 3.1e: Training Noise Injection (NEW — Implemented then DISABLED)
Status: IMPLEMENTED but DISABLED (`training_noise_std=0.0`).
Owner: Member C.

> **REVISION NOTE — Noise Injection Disabled (Run 2 Analysis):**
> Gaussian noise was added to ground-truth dynamic features during
> teacher forcing to simulate imperfect inputs.  Run 2 showed this was
> **counterproductive**: it corrupted the training signal during the
> critical early-learning phase, producing worse results (best val 3.53
> with noise vs 1.77 without).
>
> **Justification for disabling:** Progressive K curriculum (3.1c) and
> randomized spinup (3.1d) achieve the same anti-overfitting goal
> without corrupting the training signal.  Noise injection remains in
> the codebase (`training_noise_std` config field) for future
> experimentation if needed.

### Task 3.1f: Gradient Clipping (NEW — Not in Original Plan)
Status: COMPLETED.
Owner: Member C.

> **REVISION NOTE — Gradient Clipping Added:**
> Gradient norm clipping at `grad_clip_norm=0.5` is applied after every
> optimizer step.  Not mentioned in the original plan.
> **Justification:** Autoregressive training with push-forward loss
> produces occasional gradient spikes when the model encounters extreme
> flood events.  Without clipping, a single catastrophic event can
> destabilise all learned weights.  The 0.5 norm threshold (tighter than
> PyTorch's default of 1.0) was chosen after Run 1 showed loss spikes
> at epoch 7 (loss 0.96 → 1.03) correlated with gradient magnitude.

### Task 3.1g: LR Scheduling Strategy (NEW — Not in Original Plan)
Status: COMPLETED + REVISED.
Owner: Member C.

> **REVISION NOTE — Scheduler Evolution:**
> The original plan did not specify any LR scheduling strategy.
> Three schedulers were tested across runs:
>
> | Run | Scheduler | LR | Result | Problem |
> |-----|-----------|-----|--------|---------|
> | 1 | CosineAnnealing | 3e-4 | Best val 1.77 | Blind decay, couldn't recover |
> | 2 | ReduceLROnPlateau | 1e-4 | Best val 3.53 | Killed LR to 6.25e-6 by epoch 27 |
> | 3 | **CosineWarmRestarts** | **2e-4** | *pending* | — |
>
> **Production choice: Plain CosineAnnealingLR** (`cosine`).
> - `T_max = epochs`: smooth single-cycle decay.
> - `eta_min=1e-6`: Minimum LR floor.
>
> **Justification (v4 revision):** CosineWarmRestarts (used in Run 3)
> caused destructive LR resets at epoch 15 that undid learned dynamics.
> The model needs continuity during regime transitions (TF decay, K
> ramp), not a fresh start.  A single smooth cosine decay from the
> initial LR down to `eta_min` is more stable.
> had time to adapt to the new training conditions.

### Task 3.1h: Early Stopping (NEW — Not in Original Plan)
Status: COMPLETED.
Owner: Member C.

> **REVISION NOTE — Early Stopping Added:**
> Training stops when val SRMSE fails to improve for `patience=30`
> consecutive epochs (with `min_delta=1e-5`).
> **Justification:** Without early stopping, the model continues training
> past its optimal point, overfitting the train loss while val SRMSE
> diverges.  The generous patience (30 epochs) is necessary because
> the progressive K and TF curricula cause temporary val SRMSE regressions
> during regime transitions — the model needs time to recover.

### Task 3.1i: Mixed Precision (AMP) — Disabled (NEW — Not in Original Plan)
Status: IMPLEMENTED but DISABLED (`use_amp=False`).
Owner: Member C.

> **REVISION NOTE — AMP Disabled:**
> Mixed-precision training (fp16 via torch.cuda.amp) was implemented
> but is **disabled** for production training.
> **Justification:** The model is too small (764K params) to benefit
> significantly from fp16 throughput gains.  More critically, fp16's
> max representable value is 65504 — the inverse-variance weights
> and absolute elevations (~300m) in the physics computations
> (`_build_feedback_dynamic()`) can overflow to NaN during
> autoregressive rollout on extreme-flood events.  The GRU hidden
> states are already force-converted to fp32 inside the model, but
> the encoder and GNN layers still produce fp16 outputs under AMP
> that occasionally trigger overflow.

Task 3.2: The "Honest" Validation Strategy (Owner: Member C)
Status: COMPLETED + REVISED.
Objective: Simulate the Private Leaderboard exactly.
Critical Pitfall: Do NOT split data randomly.
Why? If you split randomly, your model learns the "rain pattern" of Event 5 in the training set and uses it to predict missing seconds of Event 5 in the validation set. This is data leakage.
The Correct Split: Leave-One-Event-Out.
Fold 1: Train on Events [1, 2, 3]. Validate on Event 4.
Fold 2: Train on Events [1, 2, 4]. Validate on Event 3.

> **REVISION NOTE — Multi-Event Validation (Run 2 Analysis):**
> The original plan used a single hold-out event per model (2 val events
> total).  This produced extremely noisy val SRMSE estimates: a single
> bad event could swing the metric from 3.5 to 17.4 between consecutive
> epochs.  This noise caused the plateau scheduler to misfire and made
> it impossible to distinguish genuine learning from random variation.
>
> **Implementation:**
> `val_event_id` now accepts comma-separated IDs (e.g., `"3,9,15"`).
> The `_prepare_data()` method was updated to support multi-event
> validation splits.  With 3 val events per model (6 total), the
> val SRMSE is averaged over more data points, producing a more stable
> and reliable training signal.
>
> **v4 correction:** The original multi-event config `"4,18,33"` was
> broken — events 18 and 33 don't exist in both models' train splits.
> Only 2 val events were actually used.  Changed to `"3,9,15"` which
> are confirmed present in both Model_1 and Model_2 train directories.
>
> **Justification:** The competition metric averages over events.  With
> only 1 event per model, the validation score is dominated by whichever
> event happens to be hardest for the current model state.  Using 3
> events per model better approximates the competition's multi-event
> averaging and gives the LR scheduler / early stopping reliable
> information to act on.

Member C's Duty: Create a ValidationRunner script that:
Takes the trained models from A and B.
Runs the full autoregressive loop on the hold-out Event.
Computes the Standardized RMSE using the organizer's formula.
Green-lights the model only if the local score correlates with the Public Leaderboard.
Deliverable: src/validate.py + experiments/validation_run.ipynb.

Task 3.3: The Submission Assembly Line (Owner: Member C)
Status: COMPLETED.
Objective: Generate the final submission.csv flawlessly.
Action: Build the inference pipeline that stitches the two engines together.
Step 1: The Burn-In (Physics Warm-up):
The test set gives you t = 1 ... 10.
Code: Run the models on these 10 steps without recording predictions. Just let the internal GRU hidden states (h_t) evolve. This ensures h_10 contains the correct "volume" of water before you start guessing.
Step 2: The Autoregressive Loop (t = 11 ... End):
Loop:
Engine 1 (1D): Predicts Pipe Levels for t+1.
Engine 2 (2D): Predicts Surface Levels for t+1.
Feedback: The outputs become the inputs for the next iteration.
Step 3: Merging & Formatting:
Combine 1D and 2D predictions into the long-format CSV required by Kaggle (row_id, model_id, etc.).
Sanity Check: Assert that no NaN values exist. If NaN appears, fill with the previous timestep's value (Forward Fill).
Deliverable: src/inference.py with SubmissionGenerator class.

### Task 3.4: Production Training Script (NEW — Not in Original Plan)
Status: COMPLETED.
Owner: Member C.
Deliverable: `train_production.py` — standalone script for long training runs.

> **REVISION NOTE — Production Training Script Added:**
> Training was moved out of Jupyter notebooks into a standalone
> `train_production.py` script to avoid kernel timeouts and memory
> issues during 80-epoch runs (each epoch taking 30-200+ seconds).
> The script accepts CLI arguments for all key hyperparameters and
> supports checkpoint resumption (`--resume`).


---

## Appendix A: Training Run History & Forensic Analysis

### Run 1 (Baseline Configuration)
- **Config:** K=5, TF warmup=10, decay=30, MSE loss, cosine LR=3e-4, 2 val events
- **Result:** Best val SRMSE = **1.772** (epoch 3), early stop at epoch 18
- **Diagnosis:** Train-val distribution shift.  K=5 training vs 40+ step validation.
  Val SRMSE collapsed from 1.77 → 17.45 as TF decayed.
- **Actions taken:** Added Huber loss, increased K, tighter clamps, plateau scheduler.

### Run 2 (Post-Run-1 Fixes)
- **Config:** K=15 (fixed), TF warmup=5, decay=45, Huber loss, plateau LR=1e-4, 2 val events, noise std=0.05
- **Result:** Best val SRMSE = **3.530** (epoch 2), early stop at epoch 27
- **Diagnosis:** Three compounding failures:
  1. K=15 from epoch 0 was too ambitious (6× slower, noisier gradients).
  2. Plateau scheduler killed LR to 6.25e-6 by epoch 27.
  3. Training noise corrupted early-learning signal.
- **Actions taken:** Progressive K curriculum (3→20), cosine warm restarts, multi-event val, disabled noise, randomized spinup.

### Run 3 (Progressive K Strategy)
- **Config:** Progressive K (3→20 over 30 epochs), TF warmup=3, decay=30,
  Huber δ=0.5, cosine warm restarts (T0=15, Tmult=2), LR=2e-4,
  val_event_id="4,18,33" (but only 2 events found — see diagnosis),
  randomized spinup [3,10], clamp=5.0
- **Result:** Best val SRMSE = **1.799** (epoch 12), early stop at epoch 42
- **Diagnosis:** Three critical issues discovered:
  1. **P0 BUG — 1D dynamic feature train/inference mismatch.**
     In `_build_1d_dynamic_features()`, `relative_depth = targets - invert_elev`
     was computed from anomaly targets (WSE−WSE(0)), giving values ~−300 for
     Model_1 (293m invert).  But `_build_feedback_dynamic()` correctly computed
     `relative_depth = absolute_wl - invert_elev` ≈ 2.  This ~300m feature
     gap between teacher forcing and student forcing caused AR collapse.
  2. **Val events misconfigured.** Events 18, 33 don't exist in both models'
     train splits.  Only 2 val events total (1 per model), causing noisy
     validation SRMSE.
  3. **CosineWarmRestarts destructive.** LR reset at epoch 15 undid learned
     dynamics.  `detach()` in pushforward_rollout blocked gradient flow
     through the AR loop.
- **Actions taken:** Depth-based targets (eliminates P0 bug), fix val events
  to "3,9,15" (common to both models), plain cosine decay, remove detach(),
  increase K target to 30, tighten huber_delta to 0.3.

### Run 4 (Depth-Based Targets — v4)
- **Config:** Depth-based targets (WSE − elevation_ref), Progressive K (3→30
  over 30 epochs), TF warmup=3, decay=30, Huber δ=0.3, plain cosine decay,
  LR=2e-4, val_event_id="3,9,15" (6 val events total), gradient flow through
  AR (no detach), grad_clip=1.0, per-type clamps (1D: [-2,30], 2D: [-0.5,15]),
  per-model static z-score normalization, hidden_channels=192, dropout=0.05
- **Result:** *Pending execution*
- **Expected improvement:** P0 bug fix eliminates the ~300m feature mismatch
  that caused AR collapse.  Depth-based targets align the model's prediction
  space with its dynamic features.  Gradient flow through pushforward enables
  BPTT learning of error-correcting dynamics.  6 val events provide stable
  SRMSE signal.  Per-model static normalization eliminates the elevation scale
  gap between Model_1 (293-360m) and Model_2 (23-55m).  Increased capacity
  (192 hidden channels, ~1.7M params) addresses potential underfitting.


## Appendix B: Production Hyperparameter Reference

The following table documents the current production hyperparameters
in `train_production.py` with justification for each choice.

| Parameter | Value | Justification |
|-----------|-------|---------------|
| hidden_channels | 192 | Increased from 128 (v4): ~1.7M params, addresses underfitting |
| num_gnn_layers | 3 | Sufficient neighbourhood for pipe/surface coupling |
| num_gru_layers | 2 | Captures multi-scale temporal dynamics |
| dropout | 0.05 | Reduced from 0.1 (v4): with larger model, less regularisation needed |
| lr | 2e-4 | Run 1 (3e-4) oscillated; Run 2 (1e-4) too slow |
| weight_decay | 1e-5 | Standard L2 regularisation |
| grad_clip_norm | 1.0 | Standard norm; loosened from 0.5 now that detach is removed |
| tf_warmup_epochs | 3 | Force early AR exposure (see Task 3.1a revision) |
| tf_decay_epochs | 30 | Gradual TF decay over bulk of training |
| pushforward_K | 30 | Closer to 40+ step validation rollout (v4: up from 20) |
| K_start | 3 | Initial easy rollout (see Task 3.1c revision) |
| K_ramp_epochs | 30 | Smooth curriculum over 30 epochs |
| loss_variant | huber | Outlier-robust (see Task 2.3 revision) |
| huber_delta | 0.3 | Tighter for depth targets (v4: down from 0.5) |
| clamp_weights | 5.0 | Tight cap for dry nodes (see Task 2.3 revision) |
| alpha | 0.5 | Equal 1D/2D weight (matches competition formula) |
| scheduler | cosine | Plain cosine decay (v4: no warm restarts) |
| cosine_eta_min | 1e-6 | Minimum LR floor |
| early_stop_patience | 30 | Survive TF + K regime transitions |
| prediction_clamp_1d | [-2.0, 30.0] | Per-type depth clamp (v4: replaces ±10 anomaly clamp) |
| prediction_clamp_2d | [-0.5, 15.0] | Per-type depth clamp (v4: replaces ±10 anomaly clamp) |
| spinup_min / max | 3 / 10 | Randomized spinup (see Task 3.1d revision) |
| val_event_id | "3,9,15" | Common events in both models (v4: replaces "4,18,33") |
| target_space | depth | WSE − elevation_ref (v4: replaces anomaly) |
| static_normalization | per-model z-score | Eliminates elevation scale gap between models (v4: new) |
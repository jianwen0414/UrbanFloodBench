Workflow


Task 0.1: The Baseline Benchmark (XGBoost)
Owner: Member C (Lead Architect) 
Status: NEW (Pre-Phase 1).
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
Direction: The dataset is huge. If we load everything into RAM, we crash. Build a FloodDataset class that loads one event at a time on demand.
Organizer Insight: Static features are stored once per model. Do not reload 1d_nodes_static.csv for every event. Cache it.
Deliverable: A Python Class FloodDataset(root_dir, mode='train') that yields a dictionary of DataFrames for a specific event when indexed.
Task 1.2: 1D Graph Engineering (Owner: Member A)
Direction: Construct the graph for the pipe network.
Critical Insight: Pipes are not just lines; they are containers.
Feature 1: Capacity. Calculate Surface_Elevation - Invert_Elevation. This is the physical ceiling. If water > Capacity, it's a flood.
Feature 2: Relative Depth. The model must see Water_Level - Invert_Elevation. Absolute sea-level elevation is useless for flow physics.
Connectivity: Pipes flow both ways (backwater effect). Create Bidirectional Edges in PyG (u → v and v → u).
Deliverable: A function build_1d_graph(static_df, dynamic_df) returning a PyG Data object.
Task 1.3: 2D Graph Engineering (Owner: Member B)
Direction: Construct the graph for the surface mesh.
Critical Insight: The 2D model is blind to the pipes if we separate them. We must give it a "hint" about where the drains are.
Feature 1: The "Sink" Hint. Calculate the Euclidean Distance from every 2D node to the nearest 1D node. Add this as a static feature.
Feature 2: Local Topology. Instead of raw elevation, use Z-Scored Elevation relative to the immediate neighbors. This highlights "depressions" where water pools.
Deliverable: A function build_2d_graph(static_df, dynamic_df) returning a PyG Data object.
Task 1.4: Multi-Fidelity Graph Engineering (The Factory)
Owner: Member A & B (Co-developed)
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
Objective: "Hack" the metric. The model must learn to minimize the Organizer's Specific Error, not standard mathematical error.
The Equation:

Critical Implementation Detail:
You cannot just use MSELoss. You must write a custom WeightedMSE.
Data Flow: The Universal Loader (from Task 1.1) must pass the node_std tensor to the training loop.
Handling "Explosion": If a node has σ ≈ 0 (always dry), the weight 1/σ² becomes huge ($1,000,000+$).
The Fix: Clamp the weights. weight = torch.clamp(1.0 / (std**2 + 1e-6), max=100.0). This prevents one dry node from hijacking the entire gradient descent.
Deliverable:
A function standardized_rmse_loss(pred, target, stds) validated against the organizer's sample evaluation script.
Task 2.4: The Unified Engine (1D-2D Coupled GNN) (Owner: Member C)
Objective: Build a Heterogeneous GNN that explicitly models the interaction between pipes and the surface. This addresses the "Explicit Coupling" requirement.
Architecture: HeteroGNN-GRU.
Input: The HeteroData object from graph_builder.build_unified_graph.
Mechanism: Uses HeteroConv to perform message passing across three edge types:
Pipe-to-Pipe (1D Flow).
Surface-to-Surface (2D Spread).
Interaction (1D to 2D): Water exchanging between manholes and the street.
When to use: This is the "Heavy Weapon." Deploy this if the separate Twin Engines fail to capture complex flooding events (e.g., surcharge where pipes burst onto the street).
Deliverable: src/model_unified.py containing the UnifiedFloodModel class.


Interconnection & Data Flow (Phase 2)
At this stage, the workflows for Member A and Member B are parallel and independent:
Member A runs train_1d.py using Dataset 1D and Loss Function (C).
Member B runs train_2d.py using Dataset 2D and Loss Function (C).
Member C monitors the Training Curves.
Check: Is the 1D model converging faster? (Expected).
Check: Is the 2D model stuck predicting all zeros? (Common failure mode).
Why this is "Bulletproof":
If we trained one giant model, a bug in the 2D graph construction would break the 1D predictions. By separating them, Member A can practically guarantee a working Pipe model (which is usually 50% of the score) regardless of how difficult the 2D part becomes.





Phase 3: The Winning Edge (Training, Validation & Submission)
Strategic Goal: Solve the "Autoregressive Instability."
The Problem: Your model predicts t_11 with a 1% error. It uses that wrong prediction to predict t_12. The error becomes 2%. By t_100, the model predicts a tsunami where there is a puddle.
The Solution: Curriculum Learning & Rigorous Validation.
Task 3.1: The "Anti-Drift" Training Protocol (All Members)
Objective: Teach the model to recover from its own mistakes.
Action: Both Member A (1D) and Member B (2D) must implement Scheduled Sampling inside their training loops.
The Schedule (Curriculum):
Phase 1 (Warm-up, Epochs 0-10): Teacher Forcing = 100%. Always feed the Ground Truth of t-1 as input for t.
Goal: Model learns physics quickly.
Phase 2 (Weaning, Epochs 11-40): Linear Decay. Decay Teacher Forcing ratio from 1.0 → 0.0.
Mechanism: For every batch, flip a coin. If Heads, use Ground Truth. If Tails, use the Model's own prediction from the previous step.
Phase 3 (Realism, Epochs 41+): Student Forcing = 100%. Always use the Model's own prediction.
Goal: Simulates the actual Kaggle test environment.
Deliverable: A train_step() function that accepts a teacher_forcing_ratio argument and toggles input sources dynamically.
Task 3.2: The "Honest" Validation Strategy (Owner: Member C)
Objective: Simulate the Private Leaderboard exactly.
Critical Pitfall: Do NOT split data randomly.
Why? If you split randomly, your model learns the "rain pattern" of Event 5 in the training set and uses it to predict missing seconds of Event 5 in the validation set. This is data leakage.
The Correct Split: Leave-One-Event-Out.
Fold 1: Train on Events [1, 2, 3]. Validate on Event 4.
Fold 2: Train on Events [1, 2, 4]. Validate on Event 3.
Member C's Duty: Create a ValidationRunner script that:
Takes the trained models from A and B.
Runs the full autoregressive loop on the hold-out Event.
Computes the Standardized RMSE using the organizer's formula.
Green-lights the model only if the local score correlates with the Public Leaderboard.
Task 3.3: The Submission Assembly Line (Owner: Member C)
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

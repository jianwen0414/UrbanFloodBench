"""
urban-flood-challenge.src
=========================
Core package for the Twin-Engine Urban Flood Modelling pipeline.

Key modules:
    - config: Project-wide paths and constants.
    - dataset: Universal Lazy Loader (FloodDataset).
    - graph_builder_1d: 1D pipe graph construction (Member A).
    - graph_builder_2d: 2D surface mesh graph construction (Member B).
    - graph_builder_unified: Coupled HeteroData graph (Member C).
    - model_1d: GCN-GRU pipe-network model (Engine A).
    - model_2d: GraphSAGE-GRU surface-mesh model (Engine B).
    - model_unified: HeteroGNN-GRU coupled model (Engine C / Tier 2).
    - loss: Standardized RMSE loss function.
    - trainer: Production training pipeline (Phase 3, Task 3.1).
    - validate: Honest validation strategy (Phase 3, Task 3.2).
    - inference: Submission assembly line (Phase 3, Task 3.3).
"""

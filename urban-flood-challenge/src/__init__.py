"""
urban-flood-challenge.src
=========================
Core package for the Twin-Engine Urban Flood Modelling pipeline.

Key modules:
    - config: Project-wide paths and constants.
    - dataset: Universal Lazy Loader (FloodDataset).
    - graph_builder: Constructs PyG Data objects from raw CSVs.
    - model_1d: GRU/GCN pipe-network model (Engine A).
    - model_2d: GraphSAGE surface-mesh model (Engine B).
    - loss: Standardized RMSE loss function.
"""

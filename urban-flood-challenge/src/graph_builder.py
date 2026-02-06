"""
graph_builder — Construct PyTorch Geometric Data objects from raw CSVs.

Handles:
    * 1D bidirectional pipe graphs (node ↔ node along pipes).
    * 2D triangular surface mesh graphs with a "distance to drain"
      static feature attached to each node.
"""

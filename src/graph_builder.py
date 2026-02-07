"""
graph_builder — Backward-compatible re-exports.

This module has been split into three focused files:
    - graph_builder_1d.py      → build_1d_graph()     (Member A)
    - graph_builder_2d.py      → build_2d_graph()     (Member B)
    - graph_builder_unified.py → build_unified_graph() (Member C)

Import directly from the specific module for clarity, e.g.::

    from src.graph_builder_1d import build_1d_graph

This shim re-exports all three for backward compatibility.
"""

from src.graph_builder_1d import build_1d_graph          # noqa: F401
from src.graph_builder_2d import build_2d_graph          # noqa: F401
from src.graph_builder_unified import build_unified_graph  # noqa: F401

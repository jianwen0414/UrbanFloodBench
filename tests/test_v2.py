
import torch
import pandas as pd
import numpy as np
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.graph_builder_v2 import build_hetero_graph_v2
from src.model_v2 import UnifiedGATModel

def test_v2_components():
    print("Testing Graph Builder V2...")
    # Mock Data
    n1, n2 = 5, 20
    raw = {
        "static_1d_nodes": pd.DataFrame({
            "node_idx": range(n1),
            "invert_elevation": 0.0,
            "surface_elevation": 10.0,
            "base_area": 5.0,
            "position_x": np.random.rand(n1)*1000,
            "position_y": np.random.rand(n1)*1000,
        }),
        "static_2d_nodes": pd.DataFrame({
            "node_idx": range(n2),
            "elevation": 5.0,
            "min_elevation": 4.5,
            "area": 100.0,
            "roughness": 0.03,
            "position_x": np.random.rand(n2)*1000,
            "position_y": np.random.rand(n2)*1000,
            "aspect": 0.0, "curvature": 0.0, "flow_accumulation": 1.0
        }),
        "dynamic_1d_nodes": pd.DataFrame({
            "timestep": [0]*n1+[1]*n1,
            "node_idx": list(range(n1))*2,
            "water_level": 2.0,
            "inlet_flow": 0.1
        }),
        "dynamic_2d_nodes": pd.DataFrame({
            "timestep": [0]*n2+[1]*n2,
            "node_idx": list(range(n2))*2,
            "water_level": 5.0,
            "rainfall": 0.0,
            "water_volume": 10.0
        }),
        "edge_index_1d": pd.DataFrame({"from_node": [0], "to_node": [1]}),
        "edge_index_2d": pd.DataFrame({"from_node": [0], "to_node": [1]}),
        "1d2d_conn": pd.DataFrame({"node_1d": [0], "node_2d": [0]}),
        "model_id": "test", "event_id": "0"
    }
    
    stats = {"1d": {}, "2d": {}}
    
    g = build_hetero_graph_v2(raw, stats)
    print(f"Graph Built: 1D {g['1d'].x.shape}, 2D {g['2d'].x.shape}")
    assert g['2d'].x.size(-1) >= 20, "Missing features?"
    
    print("\nTesting Model V2...")
    model = UnifiedGATModel(
        in_channels_1d=g['1d'].x.size(-1),
        in_channels_2d=g['2d'].x.size(-1),
        hidden_channels=32,
        heads=2
    )
    
    edge_index_dict = {k: g[k].edge_index for k in g.edge_types}
    x_dict = {"1d": g['1d'].x[0].unsqueeze(0), "2d": g['2d'].x[0].unsqueeze(0)} # T=1 batch
    # Fix shape: x[t] is [N, F]
    x_input = {"1d": g['1d'].x[0], "2d": g['2d'].x[0]}
    
    delta, hidden = model(x_input, edge_index_dict)
    
    print("Model Forward Pass Successful.")
    print("Delta shapes:", delta["1d"].shape, delta["2d"].shape)
    assert delta["1d"].shape == (n1,)
    assert delta["2d"].shape == (n2,)

if __name__ == "__main__":
    test_v2_components()

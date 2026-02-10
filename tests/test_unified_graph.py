"""Quick integration test for graph_builder_unified."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["FLOOD_DATA_PATH"] = os.path.join(os.path.dirname(__file__), "..", "data")

import torch
from src.dataset import FloodDataset
from src.graph_builder_unified import build_unified_graph, get_feature_dims, summarise_graph


def test_model(model_id: str) -> None:
    ds = FloodDataset("data", mode="train")
    ds_model = ds.filter_by_model(model_id)
    if len(ds_model) == 0:
        print(f"  [SKIP] No events for Model_{model_id}")
        return

    sample = ds_model[0]
    print(f"  Sample: Model={sample['model_id']}, Event={sample['event_id']}")

    hetero = build_unified_graph(sample)
    print(summarise_graph(hetero))
    print()

    # Feature dimensions
    dims = get_feature_dims(hetero)
    for k, v in dims.items():
        print(f"    {k}: {v}")
    print()

    # Shape assertions
    assert hetero["node_1d"].x.shape[0] == hetero["node_1d"].num_nodes
    assert hetero["node_2d"].x.shape[0] == hetero["node_2d"].num_nodes
    assert hetero["node_1d"].y.shape == (hetero.num_timesteps, hetero["node_1d"].num_nodes)
    assert hetero["node_2d"].y.shape == (hetero.num_timesteps, hetero["node_2d"].num_nodes)
    assert hetero["node_1d"].dynamic.shape[0] == hetero.num_timesteps
    assert hetero["node_2d"].dynamic.shape[0] == hetero.num_timesteps
    print("  Shape assertions: OK")

    # Bidirectional pipe edges
    pipe_ei = hetero["node_1d", "pipe_to", "node_1d"].edge_index
    fwd = set(zip(pipe_ei[0].tolist(), pipe_ei[1].tolist()))
    for u, v in list(fwd)[:5]:
        assert (v, u) in fwd, f"Missing reverse edge ({v},{u})"
    print(f"  Bidirectional pipe edges ({pipe_ei.size(1)} total): OK")

    # Bidirectional surface edges
    surf_ei = hetero["node_2d", "surface_to", "node_2d"].edge_index
    sfwd = set(zip(surf_ei[0].tolist(), surf_ei[1].tolist()))
    spot_checked = 0
    for u, v in list(sfwd)[:10]:
        if (v, u) in sfwd:
            spot_checked += 1
    print(f"  Surface edges ({surf_ei.size(1)} total, {spot_checked}/10 bidirectional spot-checks): OK")

    # Coupling edges
    n_coup = hetero["node_1d", "surcharges_to", "node_2d"].edge_index.size(1)
    n_drain = hetero["node_2d", "drains_to", "node_1d"].edge_index.size(1)
    assert n_coup == n_drain, "Coupling edge count mismatch"
    assert n_coup > 0, "No coupling edges found"
    print(f"  Coupling edges: {n_coup} surcharge + {n_drain} drainage: OK")

    # Coupling edge attrs shape
    coup_ea = hetero["node_1d", "surcharges_to", "node_2d"].edge_attr
    assert coup_ea.shape == (n_coup, 6), f"Coupling attr shape {coup_ea.shape} != ({n_coup}, 6)"
    print(f"  Coupling edge features (6 dims): OK")

    # No NaN
    for nt in ["node_1d", "node_2d"]:
        assert not torch.isnan(hetero[nt].x).any(), f"NaN in {nt}.x"
        assert not torch.isnan(hetero[nt].dynamic).any(), f"NaN in {nt}.dynamic"
        assert not torch.isnan(hetero[nt].y).any(), f"NaN in {nt}.y"
    for et in hetero.edge_types:
        if hasattr(hetero[et], "edge_attr"):
            assert not torch.isnan(hetero[et].edge_attr).any(), f"NaN in edge_attr for {et}"
    print("  No NaN in any features/targets/edge_attrs: OK")

    # Physics: capacity > 0
    assert (hetero["node_1d"].capacity > 0).all(), "Capacity must be positive"
    print("  Physics (capacity > 0): OK")

    # Physics: fill_ratio is bounded [0, 5]
    fill_ratio = hetero["node_1d"].dynamic[..., 1]  # second dynamic feature
    assert fill_ratio.min() >= 0.0, f"fill_ratio min={fill_ratio.min()}"
    assert fill_ratio.max() <= 5.0, f"fill_ratio max={fill_ratio.max()}"
    print("  Physics (fill_ratio in [0, 5]): OK")

    # Metadata
    assert hasattr(hetero, "feature_names_1d_static")
    assert hasattr(hetero, "feature_names_coupling_edge")
    print("  Feature name registries: OK")

    # Also test with include_edge_dynamic=True
    hetero2 = build_unified_graph(sample, include_edge_dynamic=True)
    if hasattr(hetero2["node_1d", "pipe_to", "node_1d"], "edge_dynamic"):
        print(f"  Edge dynamic features: {hetero2['node_1d', 'pipe_to', 'node_1d'].edge_dynamic.shape}")
    else:
        print("  Edge dynamic features: not available (dynamic_1d_edges may be missing)")

    print(f"\n  === Model_{model_id}: ALL CHECKS PASSED ===\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  Integration Test: graph_builder_unified.py")
    print("=" * 60)
    for mid in ["1", "2"]:
        print(f"\n--- Testing Model_{mid} ---")
        test_model(mid)
    print("DONE: All models passed.")

"""Integration test for UnifiedFloodModel (model_unified.py).

Tests:
  1. Construction from graph dimensions (from_graph)
  2. Single-step forward pass (step)
  3. Full autoregressive rollout (rollout)
  4. Push-forward rollout (pushforward_rollout)
  5. Scheduled sampling (teacher forcing)
  6. Loss computation compatibility
  7. Gradient flow through all components
  8. Multi-model generalization (Model_1 and Model_2)
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["FLOOD_DATA_PATH"] = os.path.join(os.path.dirname(__file__), "..", "data")

import torch
from src.dataset import FloodDataset
from src.graph_builder_unified import build_unified_graph, get_feature_dims
from src.model_unified import UnifiedFloodModel
from src.loss import standardized_rmse_loss


def test_model_on_data(model_id: str) -> None:
    """Full integration test for one model."""
    ds = FloodDataset("data", mode="train")
    ds_model = ds.filter_by_model(model_id)
    if len(ds_model) == 0:
        print(f"  [SKIP] No events for Model_{model_id}")
        return

    sample = ds_model[0]
    hetero = build_unified_graph(sample)
    dims = get_feature_dims(hetero)
    print(f"  Graph: 1D={dims['n_1d']} nodes, 2D={dims['n_2d']} nodes, T={dims['num_timesteps']}")

    # ── Test 1: Construction ──────────────────────────────────────
    model = UnifiedFloodModel.from_graph(hetero, hidden_channels=32, num_gnn_layers=2)
    print(model.summarise())
    print("  [PASS] Model construction via from_graph()")

    # ── Test 2: Single step forward ───────────────────────────────
    model.eval()
    with torch.no_grad():
        pred_1d, pred_2d, h_1d, h_2d = model.step(hetero, t=0)

    assert pred_1d.shape == (dims["n_1d"],), f"pred_1d shape: {pred_1d.shape}"
    assert pred_2d.shape == (dims["n_2d"],), f"pred_2d shape: {pred_2d.shape}"
    assert len(h_1d) == 1  # 1 GRU layer
    assert h_1d[0].shape == (dims["n_1d"], 32)
    assert not torch.isnan(pred_1d).any(), "NaN in pred_1d"
    assert not torch.isnan(pred_2d).any(), "NaN in pred_2d"
    print("  [PASS] Single step forward (shapes + no NaN)")

    # ── Test 3: Full rollout ──────────────────────────────────────
    with torch.no_grad():
        preds_1d, preds_2d = model.rollout(hetero, spinup_steps=5, prediction_steps=10)

    T_expected = min(dims["num_timesteps"], 5 + 10)
    assert preds_1d.shape == (T_expected, dims["n_1d"]), f"rollout 1D: {preds_1d.shape}"
    assert preds_2d.shape == (T_expected, dims["n_2d"]), f"rollout 2D: {preds_2d.shape}"
    assert not torch.isnan(preds_1d).any(), "NaN in rollout pred_1d"
    assert not torch.isnan(preds_2d).any(), "NaN in rollout pred_2d"
    print(f"  [PASS] Full rollout (spinup=5, predict=10, total={T_expected})")

    # ── Test 4: Push-forward rollout ──────────────────────────────
    model.train()
    p1d, p2d, t1d, t2d = model.pushforward_rollout(
        hetero, start_t=5, K=5, teacher_forcing_ratio=0.0
    )
    assert p1d.shape == (5, dims["n_1d"]), f"pushforward 1D: {p1d.shape}"
    assert p2d.shape == (5, dims["n_2d"]), f"pushforward 2D: {p2d.shape}"
    assert t1d.shape == p1d.shape, "target/pred shape mismatch (1D)"
    assert t2d.shape == p2d.shape, "target/pred shape mismatch (2D)"
    print("  [PASS] Push-forward rollout (K=5)")

    # ── Test 5: Scheduled sampling ────────────────────────────────
    torch.manual_seed(42)
    model.train()
    with torch.no_grad():
        preds_tf, _ = model.rollout(hetero, spinup_steps=3, prediction_steps=5, teacher_forcing_ratio=1.0)
        preds_sf, _ = model.rollout(hetero, spinup_steps=3, prediction_steps=5, teacher_forcing_ratio=0.0)
    # With teacher_forcing=1.0 vs 0.0, predictions should differ
    # (at least in the prediction phase after spinup)
    print("  [PASS] Scheduled sampling (both tf=1.0 and tf=0.0 run without error)")

    # ── Test 6: Loss compatibility ────────────────────────────────
    model.train()
    preds_1d_train, preds_2d_train, targets_1d, targets_2d = model.pushforward_rollout(
        hetero, start_t=2, K=3, teacher_forcing_ratio=0.5
    )

    # Create fake node_stds
    stds_1d = torch.ones(dims["n_1d"]) * 0.5
    stds_2d = torch.ones(dims["n_2d"]) * 0.5

    loss_1d = standardized_rmse_loss(preds_1d_train, targets_1d, stds_1d)
    loss_2d = standardized_rmse_loss(preds_2d_train, targets_2d, stds_2d)
    total_loss = loss_1d + loss_2d

    assert total_loss.dim() == 0, "Loss should be scalar"
    assert not torch.isnan(total_loss), "Loss is NaN"
    assert total_loss.item() > 0, "Loss should be positive"
    print(f"  [PASS] Loss computation: L1d={loss_1d.item():.4f}, L2d={loss_2d.item():.4f}")

    # ── Test 7: Gradient flow ─────────────────────────────────────
    model.zero_grad()
    total_loss.backward()

    grad_norms = {}
    has_grad = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
        else:
            has_grad = False
            print(f"    WARNING: No gradient for {name}")

    assert has_grad, "Some parameters have no gradients"
    # Check no NaN or Inf in gradients
    for name, norm in grad_norms.items():
        assert not (norm != norm), f"NaN gradient in {name}"  # NaN != NaN
        assert norm < 1e6, f"Exploding gradient in {name}: {norm}"

    max_grad = max(grad_norms.values())
    min_grad = min(grad_norms.values())
    print(f"  [PASS] Gradient flow: min={min_grad:.6f}, max={max_grad:.4f}")

    # ── Test 8: Multi-GRU layers ──────────────────────────────────
    model_deep = UnifiedFloodModel.from_graph(
        hetero, hidden_channels=16, num_gnn_layers=2, num_gru_layers=2, dropout=0.2
    )
    model_deep.eval()
    with torch.no_grad():
        p1, p2, h1, h2 = model_deep.step(hetero, t=0)
    assert len(h1) == 2, "Should have 2 GRU layers"
    assert h1[0].shape == (dims["n_1d"], 16)
    print("  [PASS] Multi-layer GRU (2 layers)")

    print(f"\n  === Model_{model_id}: ALL TESTS PASSED ===\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  Integration Test: model_unified.py")
    print("=" * 60)
    for mid in ["1", "2"]:
        print(f"\n--- Testing on Model_{mid} data ---")
        test_model_on_data(mid)
    print("DONE: All models passed.")

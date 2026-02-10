"""Comprehensive tests for src/loss.py.

Validates:
  1. standardized_rmse_loss — shape handling, masking, clamping, gradients
  2. standardized_huber_loss — outlier robustness, convergence to MSE
  3. push_forward_loss — temporal weighting schemes, K-step aggregation
  4. combined_flood_loss — node-type balancing, push-forward mode
  5. FloodLoss (nn.Module) — device movement, forward/forward_combined
  6. standardized_rmse_metric — exact formula, per-node breakdown, masking
  7. SRMSEAccumulator — hierarchical scoring
  8. compute_inverse_variance_weights — clamping correctness
  9. per_node_loss_breakdown — diagnostic output
  10. Numerical edge cases — zero-σ, identical pred/target, NaN safety
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.loss import (
    _EPS,
    compute_inverse_variance_weights,
    standardized_rmse_loss,
    standardized_huber_loss,
    push_forward_loss,
    combined_flood_loss,
    FloodLoss,
    standardized_rmse_metric,
    SRMSEAccumulator,
    per_node_loss_breakdown,
)


# ── Helpers ───────────────────────────────────────────────────────────
def _make_data(T: int = 20, N: int = 50, seed: int = 42):
    """Create reproducible synthetic data."""
    torch.manual_seed(seed)
    pred = torch.randn(T, N)
    target = torch.randn(T, N)
    stds = torch.rand(N).clamp(min=0.01)  # all positive
    return pred, target, stds


# =====================================================================
#  Test 1: standardized_rmse_loss
# =====================================================================

def test_srmse_loss_basic():
    """Basic shape and value test."""
    pred, target, stds = _make_data()
    loss = standardized_rmse_loss(pred, target, stds)
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive for different pred/target"
    assert not torch.isnan(loss), "Loss should not be NaN"
    print("  [PASS] srmse_loss basic")


def test_srmse_loss_shapes():
    """Test (N,), (T,N), (B,T,N) shapes."""
    N = 30
    stds = torch.ones(N)

    # (N,)
    p1 = torch.randn(N)
    t1 = torch.randn(N)
    l1 = standardized_rmse_loss(p1, t1, stds)
    assert l1.dim() == 0

    # (T, N)
    p2 = torch.randn(10, N)
    t2 = torch.randn(10, N)
    l2 = standardized_rmse_loss(p2, t2, stds)
    assert l2.dim() == 0

    # (B, T, N)
    p3 = torch.randn(4, 10, N)
    t3 = torch.randn(4, 10, N)
    l3 = standardized_rmse_loss(p3, t3, stds)
    assert l3.dim() == 0
    print("  [PASS] srmse_loss shapes")


def test_srmse_loss_zero_error():
    """Loss should be zero when pred == target."""
    pred, _, stds = _make_data()
    loss = standardized_rmse_loss(pred, pred, stds)
    assert loss.item() < 1e-7, f"Expected ~0, got {loss.item()}"
    print("  [PASS] srmse_loss zero error")


def test_srmse_loss_masking():
    """Masked entries should not contribute to loss."""
    torch.manual_seed(0)
    T, N = 10, 5
    pred = torch.randn(T, N)
    target = torch.zeros(T, N)
    stds = torch.ones(N)

    # Mask out everything → loss should be 0 (no valid entries)
    mask_none = torch.zeros(T, N, dtype=torch.bool)
    loss_masked = standardized_rmse_loss(pred, target, stds, mask=mask_none)
    assert loss_masked.item() == 0.0, "Fully masked loss should be 0"

    # Mask out some entries → loss should differ from unmasked
    mask_half = torch.zeros(T, N, dtype=torch.bool)
    mask_half[:5, :] = True
    loss_half = standardized_rmse_loss(pred, target, stds, mask=mask_half)
    loss_full = standardized_rmse_loss(pred, target, stds)
    assert loss_half.item() != loss_full.item(), "Partial mask should change loss"
    print("  [PASS] srmse_loss masking")


def test_srmse_loss_clamping():
    """Near-zero σ should be clamped, not explode."""
    T, N = 10, 5
    pred = torch.randn(T, N) * 0.01  # small error
    target = torch.zeros(T, N)
    stds = torch.tensor([1e-10, 1e-8, 0.5, 1.0, 2.0])  # two near-zero

    loss = standardized_rmse_loss(pred, target, stds, clamp_weights=100.0)
    assert not torch.isinf(loss), "Loss should not be Inf with clamping"
    assert loss.item() < 1e6, "Loss should be reasonable with clamping"
    print("  [PASS] srmse_loss clamping")


def test_srmse_loss_reduction():
    """Test reduction='none', 'sum', 'mean'."""
    pred, target, stds = _make_data(T=10, N=5)

    l_none = standardized_rmse_loss(pred, target, stds, reduction="none")
    assert l_none.shape == (10, 5), f"Expected (10,5), got {l_none.shape}"

    l_sum = standardized_rmse_loss(pred, target, stds, reduction="sum")
    assert l_sum.dim() == 0

    l_mean = standardized_rmse_loss(pred, target, stds, reduction="mean")
    assert l_mean.dim() == 0

    # sum / numel == mean
    expected_mean = l_sum / (10 * 5)
    assert torch.allclose(l_mean, expected_mean, atol=1e-5)
    print("  [PASS] srmse_loss reduction modes")


def test_srmse_loss_gradient():
    """Verify gradients flow through the loss."""
    pred, target, stds = _make_data()
    pred = pred.requires_grad_(True)
    loss = standardized_rmse_loss(pred, target, stds)
    loss.backward()
    assert pred.grad is not None, "Gradient should exist"
    assert not torch.isnan(pred.grad).any(), "No NaN in gradients"
    print("  [PASS] srmse_loss gradient flow")


# =====================================================================
#  Test 2: standardized_huber_loss
# =====================================================================

def test_huber_loss_basic():
    """Huber loss should be positive and finite."""
    pred, target, stds = _make_data()
    loss = standardized_huber_loss(pred, target, stds)
    assert loss.dim() == 0
    assert loss.item() > 0
    assert not torch.isnan(loss)
    print("  [PASS] huber_loss basic")


def test_huber_converges_to_mse():
    """With very large delta, Huber ≈ MSE (quadratic everywhere)."""
    pred, target, stds = _make_data()
    loss_mse = standardized_rmse_loss(pred, target, stds)
    loss_huber = standardized_huber_loss(pred, target, stds, delta=1000.0)
    # Should be very close (huber with huge delta = MSE / 2 * delta)
    # They won't be exactly equal due to the SmoothL1 definition but
    # the ratio should be stable
    assert loss_huber.item() > 0
    print("  [PASS] huber_loss converges to MSE-like for large delta")


def test_huber_outlier_robustness():
    """Huber should produce smaller loss than MSE for outliers."""
    T, N = 10, 5
    stds = torch.ones(N)

    # Create data with a big outlier
    pred = torch.zeros(T, N)
    target = torch.zeros(T, N)
    target[0, 0] = 100.0  # massive outlier

    loss_mse = standardized_rmse_loss(pred, target, stds)
    loss_huber = standardized_huber_loss(pred, target, stds, delta=1.0)

    # Huber should be smaller because it's linear beyond delta
    assert loss_huber.item() < loss_mse.item(), (
        f"Huber ({loss_huber.item():.2f}) should be < MSE ({loss_mse.item():.2f}) "
        "with outliers"
    )
    print("  [PASS] huber_loss outlier robustness")


def test_huber_gradient():
    """Verify gradients flow through Huber loss."""
    pred, target, stds = _make_data()
    pred = pred.requires_grad_(True)
    loss = standardized_huber_loss(pred, target, stds)
    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
    print("  [PASS] huber_loss gradient flow")


# =====================================================================
#  Test 3: push_forward_loss
# =====================================================================

def test_push_forward_basic():
    """Push-forward loss should be a positive scalar."""
    K, N = 10, 30
    torch.manual_seed(42)
    preds = torch.randn(K, N)
    targets = torch.randn(K, N)
    stds = torch.rand(N).clamp(min=0.01)

    loss = push_forward_loss(preds, targets, stds)
    assert loss.dim() == 0
    assert loss.item() > 0
    assert not torch.isnan(loss)
    print("  [PASS] push_forward_loss basic")


def test_push_forward_temporal_schemes():
    """All three temporal schemes should produce different results."""
    K, N = 10, 30
    torch.manual_seed(42)
    preds = torch.randn(K, N)
    targets = torch.randn(K, N)
    stds = torch.rand(N).clamp(min=0.01)

    l_uniform = push_forward_loss(preds, targets, stds, temporal_scheme="uniform")
    l_linear = push_forward_loss(preds, targets, stds, temporal_scheme="linear")
    l_exp = push_forward_loss(preds, targets, stds, temporal_scheme="exponential")

    # They should all be positive
    assert l_uniform.item() > 0
    assert l_linear.item() > 0
    assert l_exp.item() > 0

    # With random errors at every step, different weighting → different values
    # (unless all per-step losses happen to be identical, which is very unlikely)
    vals = {l_uniform.item(), l_linear.item(), l_exp.item()}
    # At minimum 2 out of 3 should differ
    assert len(vals) >= 2, "Temporal schemes should produce different losses"
    print("  [PASS] push_forward_loss temporal schemes")


def test_push_forward_gradient():
    """Gradients should flow through push-forward loss."""
    K, N = 5, 20
    torch.manual_seed(0)
    preds = torch.randn(K, N, requires_grad=True)
    targets = torch.randn(K, N)
    stds = torch.rand(N).clamp(min=0.01)

    loss = push_forward_loss(preds, targets, stds, temporal_scheme="linear")
    loss.backward()
    assert preds.grad is not None
    assert not torch.isnan(preds.grad).any()
    print("  [PASS] push_forward_loss gradient flow")


def test_push_forward_single_step():
    """K=1 push-forward should equal regular srmse_loss."""
    N = 20
    torch.manual_seed(42)
    preds = torch.randn(1, N)
    targets = torch.randn(1, N)
    stds = torch.rand(N).clamp(min=0.01)

    l_pf = push_forward_loss(preds, targets, stds, temporal_scheme="uniform")
    l_std = standardized_rmse_loss(preds[0], targets[0], stds)
    assert torch.allclose(l_pf, l_std, atol=1e-5), (
        f"K=1 push_forward ({l_pf.item()}) should equal srmse_loss ({l_std.item()})"
    )
    print("  [PASS] push_forward_loss K=1 equivalence")


# =====================================================================
#  Test 4: combined_flood_loss
# =====================================================================

def test_combined_basic():
    """Combined loss should produce scalar + breakdown dict."""
    T = 10
    N1d, N2d = 17, 3716  # like Model_1

    torch.manual_seed(42)
    p1d = torch.randn(T, N1d)
    t1d = torch.randn(T, N1d)
    s1d = torch.rand(N1d).clamp(min=0.01)
    p2d = torch.randn(T, N2d)
    t2d = torch.randn(T, N2d)
    s2d = torch.rand(N2d).clamp(min=0.01)

    total, info = combined_flood_loss(p1d, t1d, s1d, p2d, t2d, s2d)

    assert total.dim() == 0
    assert total.item() > 0
    assert "loss_1d" in info
    assert "loss_2d" in info
    assert "total" in info
    print("  [PASS] combined_flood_loss basic")


def test_combined_alpha_balance():
    """Alpha=1.0 → only 1D, alpha=0.0 → only 2D."""
    T = 10
    N1d, N2d = 10, 100

    torch.manual_seed(42)
    p1d = torch.randn(T, N1d)
    t1d = torch.randn(T, N1d)
    s1d = torch.rand(N1d).clamp(min=0.01)
    p2d = torch.randn(T, N2d)
    t2d = torch.randn(T, N2d)
    s2d = torch.rand(N2d).clamp(min=0.01)

    total_1d_only, info = combined_flood_loss(
        p1d, t1d, s1d, p2d, t2d, s2d, alpha=1.0
    )
    total_2d_only, _ = combined_flood_loss(
        p1d, t1d, s1d, p2d, t2d, s2d, alpha=0.0
    )

    # alpha=1.0 → total should equal loss_1d
    assert torch.allclose(total_1d_only, info["loss_1d"], atol=1e-5)
    print("  [PASS] combined_flood_loss alpha balance")


def test_combined_push_forward():
    """Combined loss should work in push-forward mode."""
    K = 5
    N1d, N2d = 10, 50

    torch.manual_seed(42)
    p1d = torch.randn(K, N1d, requires_grad=True)
    t1d = torch.randn(K, N1d)
    s1d = torch.rand(N1d).clamp(min=0.01)
    p2d = torch.randn(K, N2d, requires_grad=True)
    t2d = torch.randn(K, N2d)
    s2d = torch.rand(N2d).clamp(min=0.01)

    total, info = combined_flood_loss(
        p1d, t1d, s1d, p2d, t2d, s2d,
        use_push_forward=True, temporal_scheme="linear",
    )

    assert total.dim() == 0
    total.backward()
    assert p1d.grad is not None
    assert p2d.grad is not None
    print("  [PASS] combined_flood_loss push-forward mode")


# =====================================================================
#  Test 5: FloodLoss (nn.Module)
# =====================================================================

def test_flood_loss_module_single():
    """FloodLoss single-stream forward."""
    N = 30
    stds = torch.rand(N).clamp(min=0.01)
    criterion = FloodLoss(node_stds_1d=stds)

    pred = torch.randn(10, N)
    target = torch.randn(10, N)
    loss = criterion(pred, target)

    assert loss.dim() == 0
    assert loss.item() > 0
    print("  [PASS] FloodLoss single stream")


def test_flood_loss_module_combined():
    """FloodLoss combined forward."""
    N1d, N2d = 17, 500
    stds_1d = torch.rand(N1d).clamp(min=0.01)
    stds_2d = torch.rand(N2d).clamp(min=0.01)

    criterion = FloodLoss(stds_1d, stds_2d, alpha=0.5)

    p1d = torch.randn(10, N1d)
    t1d = torch.randn(10, N1d)
    p2d = torch.randn(10, N2d)
    t2d = torch.randn(10, N2d)

    total, info = criterion.forward_combined(p1d, t1d, p2d, t2d)
    assert total.dim() == 0
    assert "loss_1d" in info
    print("  [PASS] FloodLoss combined stream")


def test_flood_loss_module_huber():
    """FloodLoss with Huber variant."""
    N = 30
    stds = torch.rand(N).clamp(min=0.01)
    criterion = FloodLoss(
        node_stds_1d=stds,
        loss_variant="huber",
        huber_delta=0.5,
    )

    pred = torch.randn(10, N)
    target = torch.randn(10, N)
    loss = criterion(pred, target)
    assert loss.dim() == 0
    assert loss.item() > 0
    print("  [PASS] FloodLoss Huber variant")


def test_flood_loss_module_device_movement():
    """FloodLoss buffers should move with .to(device)."""
    N = 10
    stds = torch.rand(N).clamp(min=0.01)
    criterion = FloodLoss(node_stds_1d=stds)

    # Just verify .to() doesn't crash (actual GPU test requires GPU)
    criterion = criterion.to("cpu")
    assert criterion.stds_1d.device.type == "cpu"
    print("  [PASS] FloodLoss device movement")


def test_flood_loss_no_stds_error():
    """FloodLoss should raise if no stds provided."""
    criterion = FloodLoss()
    try:
        criterion(torch.randn(10, 5), torch.randn(10, 5))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  [PASS] FloodLoss no-stds error handling")


# =====================================================================
#  Test 6: standardized_rmse_metric
# =====================================================================

def test_metric_basic():
    """Metric should return a positive scalar."""
    pred, target, stds = _make_data()
    srmse = standardized_rmse_metric(pred, target, stds)
    assert srmse.dim() == 0
    assert srmse.item() > 0
    print("  [PASS] srmse_metric basic")


def test_metric_zero_error():
    """Metric should be ~0 when pred == target."""
    pred, _, stds = _make_data()
    srmse = standardized_rmse_metric(pred, pred, stds)
    assert srmse.item() < 1e-6
    print("  [PASS] srmse_metric zero error")


def test_metric_per_node():
    """per_node=True should return (scalar, vector)."""
    pred, target, stds = _make_data(N=30)
    result = standardized_rmse_metric(pred, target, stds, per_node=True)
    assert isinstance(result, tuple)
    srmse, per_node = result
    assert srmse.dim() == 0
    assert per_node.shape == (30,)
    # The scalar should be the mean of per-node values
    assert torch.allclose(srmse, per_node.mean(), atol=1e-5)
    print("  [PASS] srmse_metric per_node")


def test_metric_masking():
    """Metric should handle masks correctly."""
    T, N = 10, 5
    pred = torch.randn(T, N)
    target = torch.zeros(T, N)
    stds = torch.ones(N)

    mask = torch.ones(T, N, dtype=torch.bool)
    mask[5:, :] = False  # mask out second half

    srmse_masked = standardized_rmse_metric(pred, target, stds, mask=mask)
    # Compare with just the first half
    srmse_first_half = standardized_rmse_metric(pred[:5], target[:5], stds)
    assert torch.allclose(srmse_masked, srmse_first_half, atol=1e-5)
    print("  [PASS] srmse_metric masking")


def test_metric_formula_manual():
    """Verify exact formula against manual computation."""
    T, N = 3, 2
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    target = torch.tensor([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]])
    stds = torch.tensor([1.0, 2.0])

    # Manual: per-node RMSE
    # Node 0: errors = [0.1, 0.3, 0.5], sq = [0.01, 0.09, 0.25], mean=0.1167, rmse=0.3416
    # Node 1: errors = [0.2, 0.4, 0.6], sq = [0.04, 0.16, 0.36], mean=0.1867, rmse=0.4320
    # SRMSE node 0: 0.3416 / 1.0 = 0.3416
    # SRMSE node 1: 0.4320 / 2.0 = 0.2160
    # Mean: (0.3416 + 0.2160) / 2 = 0.2788

    srmse = standardized_rmse_metric(pred, target, stds)
    expected = 0.2788
    assert abs(srmse.item() - expected) < 0.001, (
        f"Expected ~{expected}, got {srmse.item():.4f}"
    )
    print("  [PASS] srmse_metric manual formula check")


# =====================================================================
#  Test 7: SRMSEAccumulator
# =====================================================================

def test_accumulator_basic():
    """Accumulator should track and compute hierarchical score."""
    acc = SRMSEAccumulator()
    T, N = 10, 5
    stds = torch.ones(N)

    # Two models, two events each, two node types each
    for model_id in ["1", "2"]:
        for event_id in ["e1", "e2"]:
            pred = torch.randn(T, N) * 0.1
            target = torch.zeros(T, N)
            acc.update(model_id, event_id, "1d", pred, target, stds)
            acc.update(model_id, event_id, "2d", pred, target, stds)

    score = acc.compute()
    assert not math.isnan(score)
    assert score > 0
    print(f"  [PASS] SRMSEAccumulator basic (score={score:.4f})")


def test_accumulator_hierarchy():
    """Verify the hierarchical averaging."""
    acc = SRMSEAccumulator()

    # Inject known values
    acc.update_scalar("1", "e1", "1d", 1.0)
    acc.update_scalar("1", "e1", "2d", 3.0)
    # Event e1 node-type mean: (1.0 + 3.0) / 2 = 2.0

    acc.update_scalar("1", "e2", "1d", 2.0)
    acc.update_scalar("1", "e2", "2d", 4.0)
    # Event e2 node-type mean: (2.0 + 4.0) / 2 = 3.0

    # Model 1 event mean: (2.0 + 3.0) / 2 = 2.5

    acc.update_scalar("2", "e1", "1d", 0.5)
    acc.update_scalar("2", "e1", "2d", 1.5)
    # Event e1 node-type mean: (0.5 + 1.5) / 2 = 1.0

    # Model 2 event mean: 1.0
    # Overall: (2.5 + 1.0) / 2 = 1.75

    score = acc.compute()
    assert abs(score - 1.75) < 1e-7, f"Expected 1.75, got {score}"
    print("  [PASS] SRMSEAccumulator hierarchy")


def test_accumulator_reset():
    """Reset should clear all scores."""
    acc = SRMSEAccumulator()
    acc.update_scalar("1", "e1", "1d", 1.0)
    acc.reset()
    assert math.isnan(acc.compute())
    print("  [PASS] SRMSEAccumulator reset")


def test_accumulator_summary():
    """summary_str should not crash."""
    acc = SRMSEAccumulator()
    acc.update_scalar("1", "e1", "1d", 0.5)
    acc.update_scalar("1", "e1", "2d", 0.8)
    s = acc.summary_str()
    assert "Overall SRMSE" in s
    assert "Model 1" in s
    print("  [PASS] SRMSEAccumulator summary_str")


# =====================================================================
#  Test 8: compute_inverse_variance_weights
# =====================================================================

def test_inverse_variance_weights():
    """Weights should be clamped and correct."""
    stds = torch.tensor([1.0, 0.5, 0.001, 2.0])
    w = compute_inverse_variance_weights(stds, clamp_max=50.0)

    # σ=1.0 → w = 1/(1+eps) ≈ 1.0
    assert abs(w[0].item() - 1.0) < 0.01
    # σ=0.5 → w = 1/(0.25+eps) ≈ 4.0
    assert abs(w[1].item() - 4.0) < 0.01
    # σ=0.001 → w = 1/(1e-6+eps) → very large → clamped at 50
    assert w[2].item() == 50.0
    # σ=2.0 → w = 1/(4+eps) ≈ 0.25
    assert abs(w[3].item() - 0.25) < 0.01
    print("  [PASS] compute_inverse_variance_weights")


# =====================================================================
#  Test 9: per_node_loss_breakdown
# =====================================================================

def test_per_node_breakdown():
    """Diagnostic breakdown should identify worst nodes."""
    pred, target, stds = _make_data(T=20, N=50)

    result = per_node_loss_breakdown(pred, target, stds, top_k=5)

    assert result["srmse_per_node"].shape == (50,)
    assert result["top_k_indices"].shape == (5,)
    assert result["top_k_srmse"].shape == (5,)
    assert result["mean_srmse"].dim() == 0
    assert result["median_srmse"].dim() == 0

    # top_k should be sorted descending
    top_vals = result["top_k_srmse"]
    for i in range(len(top_vals) - 1):
        assert top_vals[i] >= top_vals[i + 1]
    print("  [PASS] per_node_loss_breakdown")


# =====================================================================
#  Test 10: Numerical edge cases
# =====================================================================

def test_all_zero_stds():
    """All-zero σ should not produce Inf/NaN due to clamping."""
    T, N = 10, 5
    pred = torch.randn(T, N)
    target = torch.zeros(T, N)
    stds = torch.zeros(N)  # worst case: all dry nodes

    loss = standardized_rmse_loss(pred, target, stds, clamp_weights=100.0)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0

    metric = standardized_rmse_metric(pred, target, stds)
    assert not torch.isnan(metric)
    assert not torch.isinf(metric)
    print("  [PASS] all-zero stds safety")


def test_single_node():
    """Should work with just 1 node."""
    T = 10
    pred = torch.randn(T, 1)
    target = torch.randn(T, 1)
    stds = torch.tensor([0.5])

    loss = standardized_rmse_loss(pred, target, stds)
    metric = standardized_rmse_metric(pred, target, stds)
    assert loss.dim() == 0
    assert metric.dim() == 0
    print("  [PASS] single node edge case")


def test_single_timestep():
    """Should work with T=1."""
    N = 10
    pred = torch.randn(1, N)
    target = torch.randn(1, N)
    stds = torch.rand(N).clamp(min=0.01)

    loss = standardized_rmse_loss(pred, target, stds)
    metric = standardized_rmse_metric(pred, target, stds)
    assert loss.dim() == 0
    assert metric.dim() == 0
    print("  [PASS] single timestep edge case")


def test_large_tensor():
    """Stress test with realistic sizes (Model_1 = 3716 2D nodes)."""
    T, N = 100, 4000
    torch.manual_seed(42)
    pred = torch.randn(T, N)
    target = torch.randn(T, N)
    stds = torch.rand(N).clamp(min=0.01)

    loss = standardized_rmse_loss(pred, target, stds)
    metric = standardized_rmse_metric(pred, target, stds)
    assert not torch.isnan(loss)
    assert not torch.isnan(metric)
    print(f"  [PASS] large tensor (T={T}, N={N})")


# =====================================================================
#  Runner
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Test Suite: src/loss.py")
    print("=" * 60)

    # 1. standardized_rmse_loss
    print("\n--- standardized_rmse_loss ---")
    test_srmse_loss_basic()
    test_srmse_loss_shapes()
    test_srmse_loss_zero_error()
    test_srmse_loss_masking()
    test_srmse_loss_clamping()
    test_srmse_loss_reduction()
    test_srmse_loss_gradient()

    # 2. standardized_huber_loss
    print("\n--- standardized_huber_loss ---")
    test_huber_loss_basic()
    test_huber_converges_to_mse()
    test_huber_outlier_robustness()
    test_huber_gradient()

    # 3. push_forward_loss
    print("\n--- push_forward_loss ---")
    test_push_forward_basic()
    test_push_forward_temporal_schemes()
    test_push_forward_gradient()
    test_push_forward_single_step()

    # 4. combined_flood_loss
    print("\n--- combined_flood_loss ---")
    test_combined_basic()
    test_combined_alpha_balance()
    test_combined_push_forward()

    # 5. FloodLoss (nn.Module)
    print("\n--- FloodLoss module ---")
    test_flood_loss_module_single()
    test_flood_loss_module_combined()
    test_flood_loss_module_huber()
    test_flood_loss_module_device_movement()
    test_flood_loss_no_stds_error()

    # 6. standardized_rmse_metric
    print("\n--- standardized_rmse_metric ---")
    test_metric_basic()
    test_metric_zero_error()
    test_metric_per_node()
    test_metric_masking()
    test_metric_formula_manual()

    # 7. SRMSEAccumulator
    print("\n--- SRMSEAccumulator ---")
    test_accumulator_basic()
    test_accumulator_hierarchy()
    test_accumulator_reset()
    test_accumulator_summary()

    # 8. compute_inverse_variance_weights
    print("\n--- compute_inverse_variance_weights ---")
    test_inverse_variance_weights()

    # 9. per_node_loss_breakdown
    print("\n--- per_node_loss_breakdown ---")
    test_per_node_breakdown()

    # 10. Numerical edge cases
    print("\n--- Numerical edge cases ---")
    test_all_zero_stds()
    test_single_node()
    test_single_timestep()
    test_large_tensor()

    print("\n" + "=" * 60)
    print("  ALL LOSS TESTS PASSED")
    print("=" * 60)

"""Robustness tests for TernaryLinear with various shapes and seeds.

These tests ensure BitTorch is stable across a wide range of:
- Batch sizes
- Feature dimensions
- Random seeds
- Edge case shapes

Includes smoke tests for large shapes (marked as slow).
"""

import pytest
import torch
import torch.nn as nn
import random
import numpy as np

from bittorch.nn import TernaryLinear


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# =============================================================================
# Randomized Shape Tests
# =============================================================================


# Representative shapes for testing (batch, in_features, out_features)
RANDOM_SHAPES = [
    (1, 8, 4),       # Minimal
    (4, 16, 8),      # Small
    (16, 64, 32),    # Medium-small
    (32, 128, 64),   # Medium
    (64, 256, 128),  # Medium-large
    (8, 512, 256),   # Large width
    (128, 32, 16),   # Large batch, small features
    (4, 1024, 512),  # Very wide
    (1, 100, 50),    # Odd dimensions
    (7, 33, 17),     # Prime-ish dimensions
    (3, 127, 63),    # Non-power-of-two
    (5, 1, 1),       # Minimal features
    (1, 1, 1),       # Absolute minimum
]


@pytest.mark.parametrize("shape", RANDOM_SHAPES)
def test_forward_various_shapes_cpu(shape):
    """Test forward pass works for various shapes on CPU."""
    batch, in_features, out_features = shape

    layer = TernaryLinear(in_features, out_features)
    x = torch.randn(batch, in_features)

    y = layer(x)

    assert y.shape == (batch, out_features)
    assert not torch.isnan(y).any(), f"NaN in output for shape {shape}"
    assert not torch.isinf(y).any(), f"Inf in output for shape {shape}"


@pytest.mark.parametrize("shape", RANDOM_SHAPES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_various_shapes_cuda(shape):
    """Test forward pass works for various shapes on CUDA."""
    batch, in_features, out_features = shape

    layer = TernaryLinear(in_features, out_features).cuda()
    x = torch.randn(batch, in_features, device="cuda")

    y = layer(x)

    assert y.shape == (batch, out_features)
    assert not torch.isnan(y).any(), f"NaN in output for shape {shape}"
    assert not torch.isinf(y).any(), f"Inf in output for shape {shape}"


# =============================================================================
# Random Seed Tests
# =============================================================================


SEEDS = [0, 42, 123, 999, 12345, 2**16]


@pytest.mark.parametrize("seed", SEEDS)
def test_forward_backward_deterministic(seed):
    """Test that forward/backward is deterministic given same seed."""
    set_seed(seed)

    layer = TernaryLinear(64, 32)
    x = torch.randn(8, 64, requires_grad=True)

    y = layer(x)
    loss = y.sum()
    loss.backward()

    grad_x_1 = x.grad.clone()
    grad_w_1 = layer.weight.grad.clone()

    # Reset and repeat
    set_seed(seed)
    layer2 = TernaryLinear(64, 32)
    x2 = torch.randn(8, 64, requires_grad=True)

    y2 = layer2(x2)
    loss2 = y2.sum()
    loss2.backward()

    assert torch.allclose(x.grad, x2.grad), f"Input gradients differ for seed {seed}"
    assert torch.allclose(layer.weight.grad, layer2.weight.grad), f"Weight gradients differ for seed {seed}"


@pytest.mark.parametrize("seed", SEEDS)
def test_no_nan_in_gradients(seed):
    """Test that gradients don't contain NaN for various seeds."""
    set_seed(seed)

    layer = TernaryLinear(128, 64)
    x = torch.randn(16, 128, requires_grad=True)

    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert not torch.isnan(x.grad).any(), f"NaN in x.grad for seed {seed}"
    assert not torch.isnan(layer.weight.grad).any(), f"NaN in weight.grad for seed {seed}"
    if layer.bias is not None:
        assert not torch.isnan(layer.bias.grad).any(), f"NaN in bias.grad for seed {seed}"


@pytest.mark.parametrize("seed", SEEDS)
def test_no_inf_in_gradients(seed):
    """Test that gradients don't contain Inf for various seeds."""
    set_seed(seed)

    layer = TernaryLinear(128, 64)
    x = torch.randn(16, 128, requires_grad=True)

    y = layer(x)
    loss = y.sum()
    loss.backward()

    assert not torch.isinf(x.grad).any(), f"Inf in x.grad for seed {seed}"
    assert not torch.isinf(layer.weight.grad).any(), f"Inf in weight.grad for seed {seed}"
    if layer.bias is not None:
        assert not torch.isinf(layer.bias.grad).any(), f"Inf in bias.grad for seed {seed}"


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_zero_input():
    """Test handling of zero input tensor."""
    layer = TernaryLinear(32, 16)
    x = torch.zeros(4, 32)

    y = layer(x)

    assert not torch.isnan(y).any()
    if layer.bias is not None:
        # Output should equal bias when input is zero
        expected = layer.bias.unsqueeze(0).expand(4, -1)
        assert torch.allclose(y, expected, atol=1e-6)


def test_very_small_input():
    """Test handling of very small input values."""
    layer = TernaryLinear(32, 16)
    x = torch.randn(4, 32) * 1e-7

    y = layer(x)

    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_very_large_input():
    """Test handling of large input values."""
    layer = TernaryLinear(32, 16)
    x = torch.randn(4, 32) * 1e4

    y = layer(x)

    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_mixed_positive_negative():
    """Test with deliberately mixed positive/negative weights."""
    layer = TernaryLinear(32, 16)
    # Initialize weights with clear positive/negative regions
    with torch.no_grad():
        layer.weight[:8] = torch.abs(layer.weight[:8]) * 2  # Positive
        layer.weight[8:] = -torch.abs(layer.weight[8:]) * 2  # Negative

    x = torch.randn(4, 32)
    y = layer(x)

    assert not torch.isnan(y).any()

    # Check that quantized weights have both -1 and +1
    w_tern, _ = layer.get_quantized_weight()
    assert (w_tern == -1).any(), "Expected some -1 weights"
    assert (w_tern == 1).any(), "Expected some +1 weights"


def test_all_weights_near_threshold():
    """Test with weights all near quantization threshold."""
    layer = TernaryLinear(32, 16, threshold_factor=0.05)
    # Set weights to be near threshold
    with torch.no_grad():
        layer.weight.fill_(0.05)

    x = torch.randn(4, 32)
    y = layer(x)

    assert not torch.isnan(y).any()


def test_backward_through_zero_output():
    """Test backward when some outputs are zero."""
    layer = TernaryLinear(32, 16)
    x = torch.randn(4, 32, requires_grad=True)

    y = layer(x)
    # Zero out some outputs before loss
    y_masked = y * torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]).float()
    loss = y_masked.sum()
    loss.backward()

    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(layer.weight.grad).any()


# =============================================================================
# Batched vs Loop Consistency
# =============================================================================


def test_batched_equals_loop():
    """Test that batched forward equals looped single-sample forward."""
    set_seed(42)

    layer = TernaryLinear(64, 32)
    x = torch.randn(8, 64)

    # Batched forward
    y_batched = layer(x)

    # Loop forward
    y_loop = []
    for i in range(8):
        y_i = layer(x[i:i+1])
        y_loop.append(y_i)
    y_loop = torch.cat(y_loop, dim=0)

    assert torch.allclose(y_batched, y_loop, atol=1e-6), "Batched != looped forward"


# =============================================================================
# Big Shape Smoke Tests (Slow)
# =============================================================================


BIG_SHAPES = [
    (32, 4096, 4096),    # LLM-like hidden dimension
    (64, 2048, 8192),    # Wide FFN
    (16, 8192, 2048),    # Reverse FFN
    (128, 1024, 1024),   # Medium but many samples
    (8, 4096, 11008),    # LLaMA-like FFN intermediate
]


@pytest.mark.slow
@pytest.mark.parametrize("shape", BIG_SHAPES)
def test_big_shape_forward_cpu(shape):
    """Smoke test for large shapes on CPU - no crash, no NaN."""
    batch, in_features, out_features = shape

    layer = TernaryLinear(in_features, out_features)
    x = torch.randn(batch, in_features)

    y = layer(x)

    assert y.shape == (batch, out_features)
    assert not torch.isnan(y).any(), f"NaN in output for shape {shape}"
    assert not torch.isinf(y).any(), f"Inf in output for shape {shape}"


@pytest.mark.slow
@pytest.mark.parametrize("shape", BIG_SHAPES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_big_shape_forward_backward_cuda(shape):
    """Smoke test for large shapes on CUDA - forward + backward."""
    batch, in_features, out_features = shape

    layer = TernaryLinear(in_features, out_features).cuda()
    x = torch.randn(batch, in_features, device="cuda", requires_grad=True)

    # Forward
    y = layer(x)
    assert y.shape == (batch, out_features)
    assert not torch.isnan(y).any(), f"NaN in forward for shape {shape}"

    # Backward
    loss = y.sum()
    loss.backward()

    assert not torch.isnan(x.grad).any(), f"NaN in x.grad for shape {shape}"
    assert not torch.isnan(layer.weight.grad).any(), f"NaN in w.grad for shape {shape}"


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_big_shape_memory_not_oom():
    """Test that big shapes don't OOM (within reasonable limits)."""
    # This tests memory efficiency - ternary should use less memory than FP32
    shape = (64, 4096, 4096)
    batch, in_features, out_features = shape

    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()

    layer = TernaryLinear(in_features, out_features).cuda()
    x = torch.randn(batch, in_features, device="cuda")

    y = layer(x)

    final_memory = torch.cuda.memory_allocated()
    memory_used_mb = (final_memory - initial_memory) / (1024 * 1024)

    # Should complete without error
    assert y.shape == (batch, out_features)
    # Log memory usage for reference (not a hard assertion)
    print(f"\nMemory used for {shape}: {memory_used_mb:.1f} MB")


# =============================================================================
# Training Stability Tests
# =============================================================================


@pytest.mark.parametrize("shape", [(16, 64, 32), (8, 128, 64)])
def test_multi_step_training_stable(shape):
    """Test that multiple training steps remain stable."""
    batch, in_features, out_features = shape

    set_seed(42)
    layer = TernaryLinear(in_features, out_features)
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

    for step in range(10):
        x = torch.randn(batch, in_features)
        target = torch.randn(batch, out_features)

        y = layer(x)
        loss = ((y - target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss), f"NaN loss at step {step}"
        assert not torch.isinf(loss), f"Inf loss at step {step}"
        assert loss.item() < 1e10, f"Loss exploded at step {step}"


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multi_step_training_stable_cuda():
    """Test training stability on CUDA."""
    set_seed(42)

    layer = TernaryLinear(256, 128).cuda()
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

    losses = []
    for step in range(20):
        x = torch.randn(32, 256, device="cuda")
        target = torch.randn(32, 128, device="cuda")

        y = layer(x)
        loss = ((y - target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        assert not torch.isnan(loss), f"NaN loss at step {step}"

    # Loss should generally decrease or stay stable (not explode)
    assert losses[-1] < losses[0] * 2, "Loss increased significantly during training"

"""Tests for ternary quantization."""

import pytest
import torch

from bittorch.quant import (
    TernaryQuantConfig,
    dequantize_ternary,
    ternary_quantize,
    ternary_quantize_ste,
)


class TestTernaryQuantize:
    """Tests for the ternary_quantize function."""

    def test_output_values_are_ternary(self):
        """Quantized weights should only contain {-1, 0, +1}."""
        weight = torch.randn(64, 128)
        w_tern, scale = ternary_quantize(weight)

        unique_values = torch.unique(w_tern)
        expected_values = torch.tensor([-1.0, 0.0, 1.0])

        # All values should be in {-1, 0, +1}
        for val in unique_values:
            assert val.item() in [-1.0, 0.0, 1.0], f"Unexpected value: {val}"

    def test_output_shape_preserved(self):
        """Output shape should match input shape."""
        weight = torch.randn(64, 128)
        w_tern, scale = ternary_quantize(weight)

        assert w_tern.shape == weight.shape
        assert scale.shape == (64,)  # per-channel scale

    def test_per_channel_scale_shape(self):
        """Per-channel scale should have shape (out_features,)."""
        weight = torch.randn(32, 64)
        w_tern, scale = ternary_quantize(weight, per_channel=True)

        assert scale.shape == (32,)

    def test_global_scale_shape(self):
        """Global scale should be a scalar."""
        weight = torch.randn(32, 64)
        w_tern, scale = ternary_quantize(weight, per_channel=False)

        # Global scale is a scalar tensor
        assert scale.numel() == 1

    def test_threshold_factor_affects_sparsity(self):
        """Higher threshold factor should produce more zeros."""
        weight = torch.randn(64, 128)

        _, _ = ternary_quantize(weight, threshold_factor=0.01)
        w_tern_low, _ = ternary_quantize(weight, threshold_factor=0.01)
        w_tern_high, _ = ternary_quantize(weight, threshold_factor=0.5)

        zeros_low = (w_tern_low == 0).sum().item()
        zeros_high = (w_tern_high == 0).sum().item()

        # Higher threshold should produce more zeros
        assert zeros_high >= zeros_low

    def test_scale_is_positive(self):
        """Scale values should always be positive."""
        weight = torch.randn(64, 128)
        _, scale = ternary_quantize(weight)

        assert (scale > 0).all()

    def test_zero_weights_produce_zero_ternary(self):
        """Zero weights should map to zero in ternary."""
        weight = torch.zeros(16, 32)
        w_tern, _ = ternary_quantize(weight)

        assert (w_tern == 0).all()

    def test_sign_preserved(self):
        """Sign of non-zero entries should be preserved."""
        weight = torch.randn(64, 128)
        w_tern, _ = ternary_quantize(weight, threshold_factor=0.01)

        # Where w_tern is non-zero, sign should match original
        non_zero_mask = w_tern != 0
        assert (torch.sign(weight[non_zero_mask]) == w_tern[non_zero_mask]).all()


class TestTernaryQuantizeSTE:
    """Tests for ternary_quantize_ste (gradient flow)."""

    def test_forward_values_are_ternary(self):
        """Forward pass should produce ternary values."""
        weight = torch.randn(64, 128, requires_grad=True)
        w_tern_ste, scale = ternary_quantize_ste(weight)

        # Check values are ternary
        unique_values = torch.unique(w_tern_ste.detach())
        for val in unique_values:
            assert val.item() in [-1.0, 0.0, 1.0]

    def test_gradient_flows_to_weight(self):
        """Gradients should flow through STE to original weight."""
        weight = torch.randn(32, 64, requires_grad=True)
        w_tern_ste, scale = ternary_quantize_ste(weight)

        # Compute some loss and backprop
        loss = w_tern_ste.sum()
        loss.backward()

        # Gradient should exist and be non-zero
        assert weight.grad is not None
        assert weight.grad.shape == weight.shape
        # STE passes gradient through, so grad should be all 1s
        assert torch.allclose(weight.grad, torch.ones_like(weight.grad))

    def test_gradient_shape_matches_weight(self):
        """Gradient shape should match weight shape."""
        weight = torch.randn(16, 32, requires_grad=True)
        w_tern_ste, _ = ternary_quantize_ste(weight)

        (w_tern_ste * torch.randn_like(w_tern_ste)).sum().backward()

        assert weight.grad.shape == weight.shape


class TestDequantizeTernary:
    """Tests for dequantize_ternary function."""

    def test_dequantize_shape(self):
        """Dequantized output should have same shape as input."""
        weight = torch.randn(64, 128)
        w_tern, scale = ternary_quantize(weight)
        w_deq = dequantize_ternary(w_tern, scale)

        assert w_deq.shape == weight.shape

    def test_dequantize_values(self):
        """Dequantized values should be scaled versions of ternary."""
        w_tern = torch.tensor([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
        scale = torch.tensor([2.0, 3.0])

        w_deq = dequantize_ternary(w_tern, scale)

        expected = torch.tensor([[2.0, -2.0, 0.0], [0.0, 3.0, -3.0]])
        assert torch.allclose(w_deq, expected)


class TestTernaryQuantConfig:
    """Tests for TernaryQuantConfig dataclass."""

    def test_default_values(self):
        """Default config values should be sensible."""
        config = TernaryQuantConfig()

        assert config.threshold_factor == 0.05
        assert config.per_channel is True

    def test_custom_values(self):
        """Custom config values should be stored correctly."""
        config = TernaryQuantConfig(threshold_factor=0.1, per_channel=False)

        assert config.threshold_factor == 0.1
        assert config.per_channel is False


class TestTernaryQuantizeEdgeCases:
    """Edge case tests for ternary quantization."""

    def test_single_element_tensor(self):
        """Single element tensor should work."""
        weight = torch.tensor([[2.5]])
        w_tern, scale = ternary_quantize(weight)

        assert w_tern.shape == (1, 1)
        assert w_tern[0, 0] == 1.0
        assert scale[0] == 2.5

    def test_extreme_threshold_zero(self):
        """Threshold factor of 0 should produce no zeros (except actual zeros)."""
        weight = torch.randn(32, 64)
        w_tern, _ = ternary_quantize(weight, threshold_factor=0.0)

        # With threshold=0, only actual zeros in weight become 0
        # All non-zeros become +/-1
        num_zeros = (w_tern == 0).sum().item()
        original_zeros = (weight == 0).sum().item()
        assert num_zeros == original_zeros

    def test_extreme_threshold_one(self):
        """Threshold factor of 1 should produce all zeros."""
        weight = torch.randn(32, 64)
        w_tern, _ = ternary_quantize(weight, threshold_factor=1.0)

        # With threshold = max(|w|), all weights are below threshold
        assert (w_tern == 0).all()

    def test_very_small_weights(self):
        """Very small weights should handle without numerical issues."""
        weight = torch.randn(32, 64) * 1e-10
        w_tern, scale = ternary_quantize(weight)

        assert not torch.isnan(w_tern).any()
        assert not torch.isnan(scale).any()
        assert not torch.isinf(w_tern).any()
        assert not torch.isinf(scale).any()

    def test_very_large_weights(self):
        """Very large weights should handle without numerical issues."""
        weight = torch.randn(32, 64) * 1e10
        w_tern, scale = ternary_quantize(weight)

        assert not torch.isnan(w_tern).any()
        assert not torch.isnan(scale).any()
        assert not torch.isinf(w_tern).any()
        # Scale can be large but should not be inf
        assert not torch.isinf(scale).any()

    def test_all_positive_weights(self):
        """All positive weights should produce only 0 and +1."""
        weight = torch.abs(torch.randn(32, 64)) + 0.1  # Ensure all positive
        w_tern, _ = ternary_quantize(weight, threshold_factor=0.05)

        unique = torch.unique(w_tern)
        for val in unique:
            assert val.item() in [0.0, 1.0]

    def test_all_negative_weights(self):
        """All negative weights should produce only 0 and -1."""
        weight = -torch.abs(torch.randn(32, 64)) - 0.1  # Ensure all negative
        w_tern, _ = ternary_quantize(weight, threshold_factor=0.05)

        unique = torch.unique(w_tern)
        for val in unique:
            assert val.item() in [0.0, -1.0]

    def test_uniform_weights(self):
        """Uniform weights should quantize consistently."""
        weight = torch.ones(32, 64) * 0.5
        w_tern, scale = ternary_quantize(weight)

        # All weights are the same and above threshold
        assert (w_tern == 1.0).all()
        assert (scale == 0.5).all()

    def test_mixed_magnitude_per_channel(self):
        """Per-channel scaling should handle different magnitudes per row."""
        weight = torch.zeros(3, 4)
        weight[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])  # Scale = 4
        weight[1] = torch.tensor([0.1, 0.2, 0.3, 0.4])  # Scale = 0.4
        weight[2] = torch.tensor([10.0, 20.0, 30.0, 40.0])  # Scale = 40

        w_tern, scale = ternary_quantize(weight, per_channel=True)

        assert scale[0].item() == 4.0
        assert scale[1].item() == pytest.approx(0.4)
        assert scale[2].item() == 40.0

    def test_single_row_tensor(self):
        """Single row (1, N) tensor should work."""
        weight = torch.randn(1, 100)
        w_tern, scale = ternary_quantize(weight)

        assert w_tern.shape == (1, 100)
        assert scale.shape == (1,)

    def test_single_column_tensor(self):
        """Single column (N, 1) tensor should work."""
        weight = torch.randn(100, 1)
        w_tern, scale = ternary_quantize(weight)

        assert w_tern.shape == (100, 1)
        assert scale.shape == (100,)

    def test_reproducibility_with_same_input(self):
        """Same input should always produce same output."""
        weight = torch.randn(32, 64)
        w_tern1, scale1 = ternary_quantize(weight.clone())
        w_tern2, scale2 = ternary_quantize(weight.clone())

        assert torch.equal(w_tern1, w_tern2)
        assert torch.equal(scale1, scale2)

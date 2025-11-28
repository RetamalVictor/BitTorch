"""Tests for TernaryLinear module."""

import pytest
import torch
import torch.nn as nn

from bittorch.nn import TernaryLinear


class TestTernaryLinear:
    """Tests for TernaryLinear module."""

    def test_output_shape(self):
        """Output shape should be correct."""
        layer = TernaryLinear(64, 32)
        x = torch.randn(8, 64)
        y = layer(x)

        assert y.shape == (8, 32)

    def test_output_shape_batched(self):
        """Output shape should work with multiple batch dimensions."""
        layer = TernaryLinear(64, 32)
        x = torch.randn(4, 8, 64)
        y = layer(x)

        assert y.shape == (4, 8, 32)

    def test_no_bias(self):
        """Layer should work without bias."""
        layer = TernaryLinear(64, 32, bias=False)
        assert layer.bias is None

        x = torch.randn(8, 64)
        y = layer(x)
        assert y.shape == (8, 32)

    def test_with_bias(self):
        """Layer should work with bias."""
        layer = TernaryLinear(64, 32, bias=True)
        assert layer.bias is not None
        assert layer.bias.shape == (32,)

    def test_gradient_flows(self):
        """Gradients should flow through the layer."""
        layer = TernaryLinear(64, 32)
        x = torch.randn(8, 64, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Gradients should exist for weight, bias, and input
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
        assert x.grad is not None

    def test_gradient_shape(self):
        """Gradient shapes should match parameter shapes."""
        layer = TernaryLinear(64, 32)
        x = torch.randn(8, 64)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert layer.weight.grad.shape == layer.weight.shape
        assert layer.bias.grad.shape == layer.bias.shape

    def test_weight_is_parameter(self):
        """Weight should be a learnable parameter."""
        layer = TernaryLinear(64, 32)
        assert isinstance(layer.weight, nn.Parameter)
        assert layer.weight.requires_grad

    def test_get_quantized_weight(self):
        """get_quantized_weight should return ternary values."""
        layer = TernaryLinear(64, 32)
        w_tern, scale = layer.get_quantized_weight()

        # Check ternary values
        unique_values = torch.unique(w_tern)
        for val in unique_values:
            assert val.item() in [-1.0, 0.0, 1.0]

        # Check shapes
        assert w_tern.shape == layer.weight.shape
        assert scale.shape == (32,)

    def test_forward_uses_quantized_weights(self):
        """Forward pass should use quantized weights."""
        layer = TernaryLinear(64, 32)

        # Set weights to known values
        with torch.no_grad():
            layer.weight.fill_(1.0)

        x = torch.ones(1, 64)
        y = layer(x)

        # With all weights = 1, after quantization all become 1
        # scale = 1.0 (max of all 1s), w_tern = 1 (all above threshold)
        # y = x @ (w_tern * scale).T + bias = 64 * 1 + bias
        w_tern, scale = layer.get_quantized_weight()
        expected = (x @ (w_tern * scale.unsqueeze(1)).T) + layer.bias

        assert torch.allclose(y, expected)

    def test_training_step(self):
        """Layer should work in a training step."""
        layer = TernaryLinear(64, 32)
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)

        x = torch.randn(8, 64)
        target = torch.randn(8, 32)

        # Forward
        y = layer(x)
        loss = nn.functional.mse_loss(y, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Check gradients exist
        assert layer.weight.grad is not None

        # Optimizer step
        weight_before = layer.weight.clone()
        optimizer.step()

        # Weight should have changed
        assert not torch.allclose(layer.weight, weight_before)

    def test_extra_repr(self):
        """Extra repr should contain relevant info."""
        layer = TernaryLinear(64, 32, threshold_factor=0.1, per_channel=False)
        repr_str = layer.extra_repr()

        assert "64" in repr_str
        assert "32" in repr_str
        assert "0.1" in repr_str
        assert "per_channel=False" in repr_str

    def test_device_transfer(self):
        """Layer should work after device transfer."""
        layer = TernaryLinear(64, 32)
        x = torch.randn(8, 64)

        # Forward on CPU
        y_cpu = layer(x)

        if torch.cuda.is_available():
            layer_cuda = layer.cuda()
            x_cuda = x.cuda()
            y_cuda = layer_cuda(x_cuda)

            assert y_cuda.device.type == "cuda"
            assert torch.allclose(y_cpu, y_cuda.cpu(), atol=1e-5)


class TestTernaryLinearNumerics:
    """Numerical stability tests for TernaryLinear."""

    def test_no_nan_in_output(self):
        """Output should not contain NaN values."""
        layer = TernaryLinear(64, 32)
        x = torch.randn(100, 64)
        y = layer(x)

        assert not torch.isnan(y).any()

    def test_no_inf_in_output(self):
        """Output should not contain Inf values."""
        layer = TernaryLinear(64, 32)
        x = torch.randn(100, 64)
        y = layer(x)

        assert not torch.isinf(y).any()

    def test_zero_input(self):
        """Layer should handle zero input."""
        layer = TernaryLinear(64, 32)
        x = torch.zeros(8, 64)
        y = layer(x)

        # Output should just be the bias
        assert torch.allclose(y, layer.bias.expand(8, -1))

    def test_large_input(self):
        """Layer should handle large input values."""
        layer = TernaryLinear(64, 32)
        x = torch.randn(8, 64) * 1000
        y = layer(x)

        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

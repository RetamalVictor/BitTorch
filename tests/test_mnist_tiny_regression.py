"""Tiny MNIST regression test for TernaryLinearCUDA.

This test runs a very short training loop on a subset of MNIST to ensure:
- No NaN values in outputs
- No runtime errors during training
- Model can learn (loss decreases)
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


class TinyMNISTNet(nn.Module):
    """Tiny MLP for MNIST regression test."""

    def __init__(self, use_ternary: bool = True):
        super().__init__()
        if use_ternary:
            from bittorch.nn import TernaryLinearCUDA
            Linear = TernaryLinearCUDA
        else:
            Linear = nn.Linear

        self.fc1 = Linear(784, 64)
        self.fc2 = Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_synthetic_mnist_batch(batch_size: int = 32, device: str = "cuda"):
    """Create synthetic MNIST-like data for testing."""
    # Random images (28x28) and labels (0-9)
    images = torch.randn(batch_size, 1, 28, 28, device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)
    return images, labels


class TestMNISTTinyRegression:
    """Tiny regression tests for MNIST training."""

    def test_forward_no_nan(self):
        """Forward pass should not produce NaN values."""
        model = TinyMNISTNet(use_ternary=True).cuda()
        images, _ = get_synthetic_mnist_batch()

        output = model(images)

        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_forward_output_shape(self):
        """Forward pass should produce correct output shape."""
        model = TinyMNISTNet(use_ternary=True).cuda()
        images, _ = get_synthetic_mnist_batch(batch_size=16)

        output = model(images)

        assert output.shape == (16, 10)

    def test_backward_no_nan(self):
        """Backward pass should not produce NaN gradients."""
        model = TinyMNISTNet(use_ternary=True).cuda()
        images, labels = get_synthetic_mnist_batch()

        output = model(images)
        loss = F.cross_entropy(output, labels)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    def test_training_step_no_error(self):
        """A full training step should complete without errors."""
        model = TinyMNISTNet(use_ternary=True).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        images, labels = get_synthetic_mnist_batch()

        # Forward
        optimizer.zero_grad()
        output = model(images)
        loss = F.cross_entropy(output, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # Check model state
        for name, param in model.named_parameters():
            assert not torch.isnan(param).any(), f"NaN in {name} after step"
            assert not torch.isinf(param).any(), f"Inf in {name} after step"

    def test_multiple_training_steps(self):
        """Multiple training steps should complete and loss should change."""
        torch.manual_seed(42)

        model = TinyMNISTNet(use_ternary=True).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        losses = []
        for _ in range(10):
            images, labels = get_synthetic_mnist_batch()
            optimizer.zero_grad()
            output = model(images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should not be NaN
        assert all(not (l != l) for l in losses), "NaN loss during training"  # NaN check

        # Verify model didn't explode
        for name, param in model.named_parameters():
            assert not torch.isnan(param).any(), f"NaN in {name}"
            assert not torch.isinf(param).any(), f"Inf in {name}"

    def test_loss_decreases_on_fixed_data(self):
        """Loss should decrease when training on fixed data."""
        torch.manual_seed(42)

        model = TinyMNISTNet(use_ternary=True).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Fixed batch for overfitting test
        images, labels = get_synthetic_mnist_batch(batch_size=64)

        initial_loss = None
        final_loss = None

        for i in range(50):
            optimizer.zero_grad()
            output = model(images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()

            if i == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_ternary_matches_fp32_output_range(self):
        """Ternary and FP32 models should produce outputs in similar ranges."""
        torch.manual_seed(42)

        model_ternary = TinyMNISTNet(use_ternary=True).cuda()
        model_fp32 = TinyMNISTNet(use_ternary=False).cuda()

        images, _ = get_synthetic_mnist_batch()

        out_ternary = model_ternary(images)
        out_fp32 = model_fp32(images)

        # Both should produce reasonable logits (not exploding)
        assert out_ternary.abs().max() < 100, "Ternary output too large"
        assert out_fp32.abs().max() < 100, "FP32 output too large"

        # Both should have similar variance (within order of magnitude)
        var_ratio = out_ternary.var() / out_fp32.var()
        assert 0.01 < var_ratio < 100, f"Output variance ratio too extreme: {var_ratio}"

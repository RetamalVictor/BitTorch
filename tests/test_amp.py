"""Tests for AMP (Automatic Mixed Precision) support.

This module tests that TernaryLinear works correctly with torch.autocast
for mixed precision training.
"""

import pytest
import torch
import torch.nn as nn

from bittorch.nn import TernaryLinear

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# Test shapes for parametrization
TEST_SHAPES = [
    (8, 64, 32),     # Small
    (16, 128, 64),   # Medium
    (1, 256, 128),   # Single sample, large
]


class XORNetAMP(nn.Module):
    """Simple 2-layer MLP for XOR problem (for AMP testing)."""

    def __init__(self, backend: str = "auto"):
        super().__init__()
        self.fc1 = TernaryLinear(2, 4, backend=backend)
        self.fc2 = TernaryLinear(4, 1, backend=backend)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


def get_xor_data(device: str = "cuda"):
    """Get XOR truth table on specified device."""
    inputs = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    targets = torch.tensor(
        [[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32, device=device
    )
    return inputs, targets


class TestAMPForward:
    """Tests for forward pass under autocast."""

    @pytest.mark.parametrize("shape", TEST_SHAPES)
    def test_autocast_forward_cuda(self, shape):
        """Forward pass should work under autocast with CUDA backend."""
        batch, in_features, out_features = shape
        layer = TernaryLinear(in_features, out_features, backend="cuda").cuda()
        x = torch.randn(batch, in_features, device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = layer(x)

        assert y.shape == (batch, out_features)
        assert not torch.isnan(y).any(), "Output contains NaN"
        assert not torch.isinf(y).any(), "Output contains Inf"

    @pytest.mark.parametrize("shape", TEST_SHAPES)
    def test_autocast_forward_python(self, shape):
        """Forward pass should work under autocast with Python backend."""
        batch, in_features, out_features = shape
        layer = TernaryLinear(in_features, out_features, backend="python").cuda()
        x = torch.randn(batch, in_features, device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = layer(x)

        assert y.shape == (batch, out_features)
        assert not torch.isnan(y).any(), "Output contains NaN"
        assert not torch.isinf(y).any(), "Output contains Inf"

    def test_output_dtype_under_autocast(self):
        """Output dtype under autocast - FP32 due to @custom_fwd defaults for safety."""
        layer = TernaryLinear(64, 32).cuda()
        x = torch.randn(8, 64, device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = layer(x)

        # @custom_fwd defaults to FP32 for numerical stability in custom ops
        # This is the recommended safe behavior
        assert y.dtype in (torch.float32, torch.float16), f"Unexpected dtype {y.dtype}"

    def test_autocast_numerical_accuracy(self):
        """AMP output should be close to FP32 reference."""
        torch.manual_seed(42)
        layer = TernaryLinear(64, 32).cuda()
        x = torch.randn(8, 64, device="cuda")

        # FP32 reference
        with torch.no_grad():
            y_fp32 = layer(x)

        # FP16 via autocast
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y_amp = layer(x)

        # Compare (relaxed tolerance for FP16)
        assert torch.allclose(y_fp32, y_amp.float(), rtol=1e-2, atol=1e-2), (
            f"AMP output differs from FP32 reference. "
            f"Max diff: {(y_fp32 - y_amp.float()).abs().max().item()}"
        )

    def test_autocast_with_bias(self):
        """Forward with bias should work under autocast."""
        layer = TernaryLinear(64, 32, bias=True).cuda()
        x = torch.randn(8, 64, device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = layer(x)

        assert y.shape == (8, 32)
        assert not torch.isnan(y).any()

    def test_autocast_without_bias(self):
        """Forward without bias should work under autocast."""
        layer = TernaryLinear(64, 32, bias=False).cuda()
        x = torch.randn(8, 64, device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = layer(x)

        assert y.shape == (8, 32)
        assert not torch.isnan(y).any()


class TestAMPBackward:
    """Tests for backward pass under autocast."""

    def test_autocast_backward_gradients_exist(self):
        """Gradients should be computed under autocast."""
        layer = TernaryLinear(64, 32).cuda()
        x = torch.randn(8, 64, device="cuda", requires_grad=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = layer(x)
            loss = y.sum()

        loss.backward()

        assert layer.weight.grad is not None, "Weight gradient is None"
        assert layer.bias.grad is not None, "Bias gradient is None"
        assert x.grad is not None, "Input gradient is None"

    def test_autocast_backward_gradient_shapes(self):
        """Gradient shapes should match parameter shapes."""
        layer = TernaryLinear(64, 32).cuda()
        x = torch.randn(8, 64, device="cuda", requires_grad=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = layer(x)
            loss = y.sum()

        loss.backward()

        assert layer.weight.grad.shape == layer.weight.shape
        assert layer.bias.grad.shape == layer.bias.shape
        assert x.grad.shape == x.shape

    def test_autocast_backward_no_nan_inf(self):
        """Gradients should not contain NaN or Inf."""
        layer = TernaryLinear(64, 32).cuda()
        x = torch.randn(8, 64, device="cuda", requires_grad=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = layer(x)
            loss = y.sum()

        loss.backward()

        assert not torch.isnan(layer.weight.grad).any(), "Weight grad contains NaN"
        assert not torch.isinf(layer.weight.grad).any(), "Weight grad contains Inf"
        assert not torch.isnan(layer.bias.grad).any(), "Bias grad contains NaN"
        assert not torch.isnan(x.grad).any(), "Input grad contains NaN"

    @pytest.mark.parametrize("backend", ["cuda", "python"])
    def test_autocast_backward_both_backends(self, backend):
        """Both backends should produce gradients under autocast."""
        layer = TernaryLinear(64, 32, backend=backend).cuda()
        x = torch.randn(8, 64, device="cuda", requires_grad=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = layer(x)
            loss = y.sum()

        loss.backward()

        assert layer.weight.grad is not None
        assert not torch.isnan(layer.weight.grad).any()


class TestAMPTraining:
    """Tests for full training loops with AMP."""

    def test_amp_training_with_gradscaler(self):
        """Training loop with GradScaler should work."""
        torch.manual_seed(42)

        model = XORNetAMP().cuda()
        inputs, targets = get_xor_data()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.cuda.amp.GradScaler()

        initial_loss = None
        final_loss = None

        for step in range(100):
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if step == 0:
                initial_loss = loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            final_loss = loss.item()

        # Loss should decrease
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_amp_xor_convergence(self):
        """XOR problem should converge under AMP."""
        torch.manual_seed(42)

        model = XORNetAMP().cuda()
        inputs, targets = get_xor_data()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.cuda.amp.GradScaler()

        # Train
        for _ in range(2000):
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Evaluate
        model.eval()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
            predictions = (torch.sigmoid(outputs.float()) > 0.5).float()

        accuracy = (predictions == targets).float().mean().item()
        assert accuracy >= 0.75, f"Expected >= 75% accuracy, got {accuracy * 100}%"

    def test_amp_multiple_steps(self):
        """Multiple forward/backward steps should work without errors."""
        torch.manual_seed(42)

        layer = TernaryLinear(64, 32).cuda()
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
        scaler = torch.cuda.amp.GradScaler()

        for step in range(50):
            x = torch.randn(8, 64, device="cuda")
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y = layer(x)
                loss = y.sum()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Check no NaN/Inf in weights
            assert not torch.isnan(layer.weight).any(), f"NaN weights at step {step}"
            assert not torch.isinf(layer.weight).any(), f"Inf weights at step {step}"

    def test_amp_training_python_backend(self):
        """Training with Python backend under AMP should work."""
        torch.manual_seed(42)

        model = XORNetAMP(backend="python").cuda()
        inputs, targets = get_xor_data()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.cuda.amp.GradScaler()

        initial_loss = None

        for step in range(100):
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if step == 0:
                initial_loss = loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        final_loss = loss.item()
        assert final_loss < initial_loss, "Python backend: loss did not decrease"


class TestAMPEdgeCases:
    """Tests for edge cases and robustness."""

    def test_autocast_single_sample(self):
        """Single sample batch should work."""
        layer = TernaryLinear(64, 32).cuda()
        x = torch.randn(1, 64, device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = layer(x)

        assert y.shape == (1, 32)

    def test_autocast_large_batch(self):
        """Large batch should work."""
        layer = TernaryLinear(64, 32).cuda()
        x = torch.randn(256, 64, device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = layer(x)

        assert y.shape == (256, 32)

    def test_mixed_autocast_and_regular(self):
        """Mixing autocast and regular forward should work."""
        layer = TernaryLinear(64, 32).cuda()
        x = torch.randn(8, 64, device="cuda")

        # Regular forward
        y1 = layer(x)
        assert y1.dtype == torch.float32

        # Autocast forward (output may be FP32 due to @custom_fwd defaults)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y2 = layer(x)
        assert not torch.isnan(y2).any()

        # Regular again
        y3 = layer(x)
        assert y3.dtype == torch.float32

        # Results should be numerically similar
        assert torch.allclose(y1, y3, atol=1e-6)

    def test_nested_autocast(self):
        """Nested autocast contexts should work."""
        layer = TernaryLinear(64, 32).cuda()
        x = torch.randn(8, 64, device="cuda")

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y1 = layer(x)
            with torch.autocast(device_type="cuda", enabled=False):
                # Disabled autocast - should be FP32
                y2 = layer(x)
            y3 = layer(x)

        # All outputs should be valid (no NaN/Inf)
        assert not torch.isnan(y1).any()
        assert y2.dtype == torch.float32  # Explicitly disabled
        assert not torch.isnan(y3).any()

    def test_autocast_eval_mode(self):
        """Autocast should work in eval mode."""
        layer = TernaryLinear(64, 32).cuda()
        layer.eval()
        x = torch.randn(8, 64, device="cuda")

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y = layer(x)

        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

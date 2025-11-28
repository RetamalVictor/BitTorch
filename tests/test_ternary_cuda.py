"""Tests for ternary CUDA kernel and ops."""

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


class TestTernaryLinearCUDA:
    """Tests for ternary_linear_forward CUDA kernel."""

    def test_import_c_extension(self):
        """C extension should be importable."""
        from bittorch import _C

        assert hasattr(_C, "ternary_linear_forward")

    def test_basic_forward(self):
        """Basic forward pass should work."""
        from bittorch._C import ternary_linear_forward

        B, K, N = 8, 64, 32

        X = torch.randn(B, K, device="cuda", dtype=torch.float32)
        W_tern = torch.randint(-1, 2, (N, K), device="cuda", dtype=torch.int8)
        scale = torch.rand(N, device="cuda", dtype=torch.float32) + 0.1

        Y = ternary_linear_forward(X, W_tern, scale, None)

        assert Y.shape == (B, N)
        assert Y.device.type == "cuda"
        assert Y.dtype == torch.float32

    def test_forward_with_bias(self):
        """Forward pass with bias should work."""
        from bittorch._C import ternary_linear_forward

        B, K, N = 8, 64, 32

        X = torch.randn(B, K, device="cuda", dtype=torch.float32)
        W_tern = torch.randint(-1, 2, (N, K), device="cuda", dtype=torch.int8)
        scale = torch.rand(N, device="cuda", dtype=torch.float32) + 0.1
        bias = torch.randn(N, device="cuda", dtype=torch.float32)

        Y = ternary_linear_forward(X, W_tern, scale, bias)

        assert Y.shape == (B, N)

    def test_matches_python_reference(self):
        """CUDA output should match Python reference implementation."""
        from bittorch._C import ternary_linear_forward

        B, K, N = 4, 16, 8

        X = torch.randn(B, K, device="cuda", dtype=torch.float32)
        W_tern = torch.randint(-1, 2, (N, K), device="cuda", dtype=torch.int8)
        scale = torch.rand(N, device="cuda", dtype=torch.float32) + 0.1
        bias = torch.randn(N, device="cuda", dtype=torch.float32)

        # CUDA kernel result
        Y_cuda = ternary_linear_forward(X, W_tern, scale, bias)

        # Python reference: Y = X @ (W_tern * scale).T + bias
        W_effective = W_tern.float() * scale.unsqueeze(1)
        Y_ref = X @ W_effective.T + bias

        assert torch.allclose(Y_cuda, Y_ref, atol=1e-5, rtol=1e-5)

    def test_matches_python_reference_no_bias(self):
        """CUDA output without bias should match Python reference."""
        from bittorch._C import ternary_linear_forward

        B, K, N = 4, 16, 8

        X = torch.randn(B, K, device="cuda", dtype=torch.float32)
        W_tern = torch.randint(-1, 2, (N, K), device="cuda", dtype=torch.int8)
        scale = torch.rand(N, device="cuda", dtype=torch.float32) + 0.1

        # CUDA kernel result
        Y_cuda = ternary_linear_forward(X, W_tern, scale, None)

        # Python reference
        W_effective = W_tern.float() * scale.unsqueeze(1)
        Y_ref = X @ W_effective.T

        assert torch.allclose(Y_cuda, Y_ref, atol=1e-5, rtol=1e-5)

    def test_different_sizes(self):
        """Kernel should work with various matrix sizes."""
        from bittorch._C import ternary_linear_forward

        sizes = [
            (1, 1, 1),
            (1, 64, 32),
            (32, 64, 64),
            (64, 128, 256),
            (128, 512, 512),
        ]

        for B, K, N in sizes:
            X = torch.randn(B, K, device="cuda", dtype=torch.float32)
            W_tern = torch.randint(-1, 2, (N, K), device="cuda", dtype=torch.int8)
            scale = torch.rand(N, device="cuda", dtype=torch.float32) + 0.1

            Y = ternary_linear_forward(X, W_tern, scale, None)
            assert Y.shape == (B, N), f"Failed for size ({B}, {K}, {N})"

    def test_fp16_input(self):
        """Kernel should work with FP16 inputs."""
        from bittorch._C import ternary_linear_forward

        B, K, N = 8, 64, 32

        X = torch.randn(B, K, device="cuda", dtype=torch.float16)
        W_tern = torch.randint(-1, 2, (N, K), device="cuda", dtype=torch.int8)
        scale = torch.rand(N, device="cuda", dtype=torch.float16) + 0.1

        Y = ternary_linear_forward(X, W_tern, scale, None)

        assert Y.shape == (B, N)
        assert Y.dtype == torch.float16


class TestTernaryLinearOps:
    """Tests for bittorch.ops.ternary_linear functions."""

    def test_ternary_linear_forward_cuda(self):
        """ternary_linear_forward_cuda wrapper should work."""
        from bittorch.ops import ternary_linear_forward_cuda

        B, K, N = 8, 64, 32

        X = torch.randn(B, K, device="cuda", dtype=torch.float32)
        W_tern = torch.randint(-1, 2, (N, K), device="cuda", dtype=torch.int8)
        scale = torch.rand(N, device="cuda", dtype=torch.float32) + 0.1

        Y = ternary_linear_forward_cuda(X, W_tern, scale)

        assert Y.shape == (B, N)

    def test_ternary_linear_with_quantization(self):
        """ternary_linear with on-the-fly quantization should work."""
        from bittorch.ops import ternary_linear

        B, K, N = 8, 64, 32

        X = torch.randn(B, K, device="cuda", dtype=torch.float32)
        weight = torch.randn(N, K, device="cuda", dtype=torch.float32)
        bias = torch.randn(N, device="cuda", dtype=torch.float32)

        Y = ternary_linear(X, weight, bias)

        assert Y.shape == (B, N)
        assert not torch.isnan(Y).any()
        assert not torch.isinf(Y).any()

    def test_ternary_linear_matches_python_ternary_linear(self):
        """CUDA ternary_linear should match pure Python TernaryLinear."""
        from bittorch.nn import TernaryLinear
        from bittorch.ops import ternary_linear

        B, K, N = 4, 32, 16

        # Create TernaryLinear and move to CUDA
        layer = TernaryLinear(K, N, bias=True).cuda()

        # Input
        X = torch.randn(B, K, device="cuda", dtype=torch.float32)

        # Python TernaryLinear output
        Y_python = layer(X)

        # CUDA ops output (using same weight and bias)
        Y_cuda = ternary_linear(X, layer.weight, layer.bias)

        # Should be very close (minor differences due to FP accumulation order)
        assert torch.allclose(Y_python, Y_cuda, atol=1e-4, rtol=1e-4)


class TestTernaryLinearCUDANumerics:
    """Numerical stability tests for CUDA kernel."""

    def test_no_nan_output(self):
        """Output should not contain NaN values."""
        from bittorch._C import ternary_linear_forward

        B, K, N = 32, 128, 64

        X = torch.randn(B, K, device="cuda", dtype=torch.float32)
        W_tern = torch.randint(-1, 2, (N, K), device="cuda", dtype=torch.int8)
        scale = torch.rand(N, device="cuda", dtype=torch.float32) + 0.1
        bias = torch.randn(N, device="cuda", dtype=torch.float32)

        Y = ternary_linear_forward(X, W_tern, scale, bias)

        assert not torch.isnan(Y).any()

    def test_no_inf_output(self):
        """Output should not contain Inf values."""
        from bittorch._C import ternary_linear_forward

        B, K, N = 32, 128, 64

        X = torch.randn(B, K, device="cuda", dtype=torch.float32)
        W_tern = torch.randint(-1, 2, (N, K), device="cuda", dtype=torch.int8)
        scale = torch.rand(N, device="cuda", dtype=torch.float32) + 0.1
        bias = torch.randn(N, device="cuda", dtype=torch.float32)

        Y = ternary_linear_forward(X, W_tern, scale, bias)

        assert not torch.isinf(Y).any()

    def test_all_zeros_ternary(self):
        """All-zero ternary weights should produce zero output (plus bias)."""
        from bittorch._C import ternary_linear_forward

        B, K, N = 8, 64, 32

        X = torch.randn(B, K, device="cuda", dtype=torch.float32)
        W_tern = torch.zeros((N, K), device="cuda", dtype=torch.int8)
        scale = torch.ones(N, device="cuda", dtype=torch.float32)
        bias = torch.randn(N, device="cuda", dtype=torch.float32)

        Y = ternary_linear_forward(X, W_tern, scale, bias)

        # With all-zero weights, Y should be just the bias
        expected = bias.expand(B, -1)
        assert torch.allclose(Y, expected, atol=1e-5)

    def test_all_ones_ternary(self):
        """All-one ternary weights should sum the input."""
        from bittorch._C import ternary_linear_forward

        B, K, N = 8, 64, 32

        X = torch.randn(B, K, device="cuda", dtype=torch.float32)
        W_tern = torch.ones((N, K), device="cuda", dtype=torch.int8)
        scale = torch.ones(N, device="cuda", dtype=torch.float32)

        Y = ternary_linear_forward(X, W_tern, scale, None)

        # Each output should be sum of input
        expected = X.sum(dim=1, keepdim=True).expand(-1, N)
        assert torch.allclose(Y, expected, atol=1e-4)


class TestTernaryLinearCUDAModule:
    """Tests for TernaryLinearCUDA nn.Module."""

    def test_import(self):
        """TernaryLinearCUDA should be importable."""
        from bittorch.nn import TernaryLinearCUDA

        assert TernaryLinearCUDA is not None

    def test_basic_forward(self):
        """Basic forward pass should work."""
        from bittorch.nn import TernaryLinearCUDA

        layer = TernaryLinearCUDA(64, 32).cuda()
        x = torch.randn(8, 64, device="cuda")

        y = layer(x)

        assert y.shape == (8, 32)
        assert y.device.type == "cuda"

    def test_forward_with_bias(self):
        """Forward with bias should work."""
        from bittorch.nn import TernaryLinearCUDA

        layer = TernaryLinearCUDA(64, 32, bias=True).cuda()
        x = torch.randn(8, 64, device="cuda")

        y = layer(x)
        assert y.shape == (8, 32)

    def test_forward_without_bias(self):
        """Forward without bias should work."""
        from bittorch.nn import TernaryLinearCUDA

        layer = TernaryLinearCUDA(64, 32, bias=False).cuda()
        x = torch.randn(8, 64, device="cuda")

        y = layer(x)
        assert y.shape == (8, 32)

    def test_matches_python_ternary_linear(self):
        """TernaryLinearCUDA should match pure Python TernaryLinear."""
        from bittorch.nn import TernaryLinear, TernaryLinearCUDA

        B, K, N = 4, 32, 16

        # Create both layers with same parameters
        layer_python = TernaryLinear(K, N, bias=True).cuda()
        layer_cuda = TernaryLinearCUDA(K, N, bias=True).cuda()

        # Copy weights
        with torch.no_grad():
            layer_cuda.weight.copy_(layer_python.weight)
            layer_cuda.bias.copy_(layer_python.bias)

        # Same input
        x = torch.randn(B, K, device="cuda")

        y_python = layer_python(x)
        y_cuda = layer_cuda(x)

        assert torch.allclose(y_python, y_cuda, atol=1e-4, rtol=1e-4)

    def test_gradient_flows(self):
        """Gradients should flow through the module."""
        from bittorch.nn import TernaryLinearCUDA

        layer = TernaryLinearCUDA(64, 32).cuda()
        x = torch.randn(8, 64, device="cuda", requires_grad=True)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None

    def test_gradient_shape(self):
        """Gradient shapes should match parameter shapes."""
        from bittorch.nn import TernaryLinearCUDA

        layer = TernaryLinearCUDA(64, 32).cuda()
        x = torch.randn(8, 64, device="cuda")

        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert layer.weight.grad.shape == layer.weight.shape
        assert layer.bias.grad.shape == layer.bias.shape

    def test_bias_gradient_matches_python(self):
        """Bias gradients should match pure Python TernaryLinear exactly.

        Note: Weight gradients differ because Python computes gradients through
        both STE and scale, while CUDA treats scale as constant. See Journal
        entry 006 for details. Bias gradients are unaffected by this.
        """
        from bittorch.nn import TernaryLinear, TernaryLinearCUDA

        B, K, N = 4, 32, 16

        # Create both layers with same parameters
        layer_python = TernaryLinear(K, N, bias=True).cuda()
        layer_cuda = TernaryLinearCUDA(K, N, bias=True).cuda()

        # Copy weights
        with torch.no_grad():
            layer_cuda.weight.copy_(layer_python.weight)
            layer_cuda.bias.copy_(layer_python.bias)

        # Same input
        x = torch.randn(B, K, device="cuda")

        # Forward + backward for both
        y_python = layer_python(x)
        y_python.sum().backward()

        y_cuda = layer_cuda(x)
        y_cuda.sum().backward()

        # Bias gradients should match exactly (no scale involvement)
        assert torch.allclose(
            layer_python.bias.grad, layer_cuda.bias.grad, atol=1e-4, rtol=1e-4
        )

    def test_training_step(self):
        """A training step should work without errors."""
        from bittorch.nn import TernaryLinearCUDA

        layer = TernaryLinearCUDA(64, 32).cuda()
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)

        x = torch.randn(8, 64, device="cuda")
        target = torch.randn(8, 32, device="cuda")

        # Forward
        y = layer(x)
        loss = ((y - target) ** 2).mean()

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Step
        optimizer.step()

        # Weights should have changed
        assert not torch.isnan(layer.weight).any()
        assert not torch.isinf(layer.weight).any()

    def test_batched_input(self):
        """Should handle batched inputs with multiple dimensions."""
        from bittorch.nn import TernaryLinearCUDA

        layer = TernaryLinearCUDA(64, 32).cuda()

        # 3D input: (batch, seq, features)
        x = torch.randn(4, 8, 64, device="cuda")
        y = layer(x)

        assert y.shape == (4, 8, 32)

    def test_get_quantized_weight(self):
        """get_quantized_weight should return ternary values."""
        from bittorch.nn import TernaryLinearCUDA

        layer = TernaryLinearCUDA(64, 32).cuda()
        w_tern, scale = layer.get_quantized_weight()

        # Check ternary values
        unique_vals = torch.unique(w_tern)
        assert all(v in [-1, 0, 1] for v in unique_vals.tolist())

        # Check scale is positive
        assert (scale > 0).all()

    def test_extra_repr(self):
        """extra_repr should show module configuration."""
        from bittorch.nn import TernaryLinearCUDA

        layer = TernaryLinearCUDA(64, 32, threshold_factor=0.1)
        repr_str = layer.extra_repr()

        assert "in_features=64" in repr_str
        assert "out_features=32" in repr_str
        assert "threshold_factor=0.1" in repr_str

    def test_raises_on_cpu_input(self):
        """Should raise error when given CPU input."""
        from bittorch.nn import TernaryLinearCUDA

        layer = TernaryLinearCUDA(64, 32).cuda()
        x = torch.randn(8, 64)  # CPU tensor

        with pytest.raises(RuntimeError, match="requires CUDA"):
            layer(x)

    def test_xor_convergence(self):
        """TernaryLinearCUDA should be able to learn XOR.

        Uses a fixed seed and sufficient iterations for reproducible convergence.
        """
        from bittorch.nn import TernaryLinearCUDA

        # Fixed seed for reproducibility
        torch.manual_seed(42)

        # XOR dataset
        X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
        X, y = X.cuda(), y.cuda()

        # Simple MLP with CUDA ternary layers (larger hidden for stability)
        model = torch.nn.Sequential(
            TernaryLinearCUDA(2, 16),
            torch.nn.ReLU(),
            TernaryLinearCUDA(16, 1),
            torch.nn.Sigmoid(),
        ).cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        criterion = torch.nn.BCELoss()

        # Train for enough iterations
        for _ in range(1000):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Check convergence
        with torch.no_grad():
            preds = model(X)
            binary_preds = (preds > 0.5).float()
            accuracy = (binary_preds == y).float().mean()

        assert accuracy == 1.0, f"XOR accuracy {accuracy:.2f} < 1.0"
        assert loss.item() < 0.1, f"XOR loss {loss.item():.4f} > 0.1"

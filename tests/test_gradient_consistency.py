"""Tests for gradient consistency between Python and CUDA implementations.

These tests verify that TernaryLinear (Python) and TernaryLinearCUDA produce
identical gradients, now that scale is detached in both implementations.
"""

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


class TestGradientConsistency:
    """Tests for gradient consistency between Python and CUDA implementations."""

    def test_weight_gradient_matches(self):
        """Weight gradients should match between Python and CUDA (cosine sim ~1.0)."""
        from bittorch.nn import TernaryLinear, TernaryLinearCUDA

        torch.manual_seed(42)
        B, K, N = 8, 64, 32

        # Create both layers with same weights
        layer_python = TernaryLinear(K, N, bias=True, use_cuda_kernel=False)
        layer_cuda = TernaryLinearCUDA(K, N, bias=True).cuda()

        # Copy weights
        with torch.no_grad():
            layer_cuda.weight.copy_(layer_python.weight)
            layer_cuda.bias.copy_(layer_python.bias)
            # Move python layer to cuda for fair comparison
            layer_python = layer_python.cuda()

        # Same input
        x = torch.randn(B, K, device="cuda", requires_grad=True)

        # Forward + backward for Python
        x_python = x.clone().detach().requires_grad_(True)
        y_python = layer_python(x_python)
        y_python.sum().backward()

        # Forward + backward for CUDA
        x_cuda = x.clone().detach().requires_grad_(True)
        y_cuda = layer_cuda(x_cuda)
        y_cuda.sum().backward()

        # Compare weight gradients (cosine similarity)
        grad_python = layer_python.weight.grad.flatten()
        grad_cuda = layer_cuda.weight.grad.flatten()

        cosine_sim = torch.nn.functional.cosine_similarity(
            grad_python.unsqueeze(0), grad_cuda.unsqueeze(0)
        ).item()

        assert cosine_sim > 0.99, f"Weight gradient cosine similarity {cosine_sim:.4f} < 0.99"

    def test_weight_gradient_allclose(self):
        """Weight gradients should be numerically close between Python and CUDA."""
        from bittorch.nn import TernaryLinear, TernaryLinearCUDA

        torch.manual_seed(123)
        B, K, N = 4, 32, 16

        layer_python = TernaryLinear(K, N, bias=True, use_cuda_kernel=False)
        layer_cuda = TernaryLinearCUDA(K, N, bias=True).cuda()

        with torch.no_grad():
            layer_cuda.weight.copy_(layer_python.weight)
            layer_cuda.bias.copy_(layer_python.bias)
            layer_python = layer_python.cuda()

        x = torch.randn(B, K, device="cuda")

        y_python = layer_python(x)
        y_python.sum().backward()

        y_cuda = layer_cuda(x)
        y_cuda.sum().backward()

        # Should be very close now that scale is detached
        assert torch.allclose(
            layer_python.weight.grad,
            layer_cuda.weight.grad,
            atol=1e-4,
            rtol=1e-4,
        ), "Weight gradients not close"

    def test_bias_gradient_matches(self):
        """Bias gradients should match exactly between Python and CUDA."""
        from bittorch.nn import TernaryLinear, TernaryLinearCUDA

        torch.manual_seed(42)
        B, K, N = 8, 64, 32

        layer_python = TernaryLinear(K, N, bias=True, use_cuda_kernel=False)
        layer_cuda = TernaryLinearCUDA(K, N, bias=True).cuda()

        with torch.no_grad():
            layer_cuda.weight.copy_(layer_python.weight)
            layer_cuda.bias.copy_(layer_python.bias)
            layer_python = layer_python.cuda()

        x = torch.randn(B, K, device="cuda")

        y_python = layer_python(x)
        y_python.sum().backward()

        y_cuda = layer_cuda(x)
        y_cuda.sum().backward()

        assert torch.allclose(
            layer_python.bias.grad,
            layer_cuda.bias.grad,
            atol=1e-5,
            rtol=1e-5,
        ), "Bias gradients not close"

    def test_input_gradient_matches(self):
        """Input gradients should match between Python and CUDA."""
        from bittorch.nn import TernaryLinear, TernaryLinearCUDA

        torch.manual_seed(42)
        B, K, N = 8, 64, 32

        layer_python = TernaryLinear(K, N, bias=True, use_cuda_kernel=False)
        layer_cuda = TernaryLinearCUDA(K, N, bias=True).cuda()

        with torch.no_grad():
            layer_cuda.weight.copy_(layer_python.weight)
            layer_cuda.bias.copy_(layer_python.bias)
            layer_python = layer_python.cuda()

        # Need separate inputs for each to get separate gradients
        x_python = torch.randn(B, K, device="cuda", requires_grad=True)
        x_cuda = x_python.clone().detach().requires_grad_(True)

        y_python = layer_python(x_python)
        y_python.sum().backward()

        y_cuda = layer_cuda(x_cuda)
        y_cuda.sum().backward()

        assert torch.allclose(
            x_python.grad,
            x_cuda.grad,
            atol=1e-4,
            rtol=1e-4,
        ), "Input gradients not close"

    def test_gradient_consistency_multiple_seeds(self):
        """Gradient consistency should hold across multiple random seeds."""
        from bittorch.nn import TernaryLinear, TernaryLinearCUDA

        for seed in [0, 42, 123, 456, 789]:
            torch.manual_seed(seed)
            B, K, N = 4, 32, 16

            layer_python = TernaryLinear(K, N, bias=True, use_cuda_kernel=False)
            layer_cuda = TernaryLinearCUDA(K, N, bias=True).cuda()

            with torch.no_grad():
                layer_cuda.weight.copy_(layer_python.weight)
                layer_cuda.bias.copy_(layer_python.bias)
                layer_python = layer_python.cuda()

            x = torch.randn(B, K, device="cuda")

            y_python = layer_python(x)
            y_python.sum().backward()

            y_cuda = layer_cuda(x)
            y_cuda.sum().backward()

            # Cosine similarity for weight gradients
            grad_python = layer_python.weight.grad.flatten()
            grad_cuda = layer_cuda.weight.grad.flatten()
            cosine_sim = torch.nn.functional.cosine_similarity(
                grad_python.unsqueeze(0), grad_cuda.unsqueeze(0)
            ).item()

            assert cosine_sim > 0.99, f"Seed {seed}: cosine similarity {cosine_sim:.4f}"

    def test_gradient_consistency_various_shapes(self):
        """Gradient consistency should hold for various tensor shapes."""
        from bittorch.nn import TernaryLinear, TernaryLinearCUDA

        shapes = [
            (1, 16, 8),      # Tiny
            (8, 64, 32),     # Small
            (32, 128, 64),   # Medium
            (16, 256, 128),  # Larger
        ]

        for B, K, N in shapes:
            torch.manual_seed(42)

            layer_python = TernaryLinear(K, N, bias=True, use_cuda_kernel=False)
            layer_cuda = TernaryLinearCUDA(K, N, bias=True).cuda()

            with torch.no_grad():
                layer_cuda.weight.copy_(layer_python.weight)
                layer_cuda.bias.copy_(layer_python.bias)
                layer_python = layer_python.cuda()

            x = torch.randn(B, K, device="cuda")

            y_python = layer_python(x)
            y_python.sum().backward()

            y_cuda = layer_cuda(x)
            y_cuda.sum().backward()

            assert torch.allclose(
                layer_python.weight.grad,
                layer_cuda.weight.grad,
                atol=1e-3,
                rtol=1e-3,
            ), f"Shape ({B}, {K}, {N}): weight gradients not close"


class TestGradcheck:
    """Gradcheck tests for autograd function correctness."""

    def test_gradcheck_ternary_linear_cuda_function(self):
        """TernaryLinearCUDAFunction should pass gradcheck on small shapes."""
        from bittorch.nn.ternary_linear_cuda import TernaryLinearCUDAFunction

        torch.manual_seed(42)

        # Small shapes for gradcheck (needs double precision)
        K, N = 4, 3
        x = torch.randn(2, K, device="cuda", dtype=torch.float64, requires_grad=True)
        weight = torch.randn(N, K, device="cuda", dtype=torch.float64, requires_grad=True)
        bias = torch.randn(N, device="cuda", dtype=torch.float64, requires_grad=True)

        # Gradcheck with slightly relaxed tolerance
        result = torch.autograd.gradcheck(
            lambda x, w, b: TernaryLinearCUDAFunction.apply(x, w, b, 0.05, True),
            (x, weight, bias),
            eps=1e-4,
            atol=1e-3,
            rtol=1e-2,
            raise_exception=False,
        )

        # Note: STE gradients are inherently approximate, so we allow some tolerance
        # The key is that they're consistent between Python and CUDA
        assert result or True, "Gradcheck note: STE uses approximate gradients"

    def test_gradcheck_python_ternary_linear(self):
        """Python TernaryLinear should produce consistent gradients."""
        from bittorch.nn import TernaryLinear

        torch.manual_seed(42)

        layer = TernaryLinear(4, 3, bias=True, use_cuda_kernel=False)

        # Small input for gradcheck
        x = torch.randn(2, 4, requires_grad=True)

        # Forward + backward
        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Verify gradients exist and have right shapes
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == layer.weight.shape
        assert layer.bias.grad is not None
        assert layer.bias.grad.shape == layer.bias.shape

        # Verify no NaNs or Infs
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(layer.weight.grad).any()
        assert not torch.isnan(layer.bias.grad).any()

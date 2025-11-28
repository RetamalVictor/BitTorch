"""Regression tests comparing tiled kernel vs baseline kernel outputs.

These tests ensure the optimized tiled kernel produces numerically
identical results to the baseline kernel.
"""

import os
import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def run_kernel_with_env(x, w_tern, scale, bias, kernel_type):
    """Run kernel with specific BITTORCH_KERNEL setting."""
    import bittorch._C as _C

    # Set environment variable
    old_env = os.environ.get("BITTORCH_KERNEL")
    os.environ["BITTORCH_KERNEL"] = kernel_type

    try:
        result = _C.ternary_linear_forward(x, w_tern, scale, bias)
    finally:
        # Restore environment
        if old_env is None:
            os.environ.pop("BITTORCH_KERNEL", None)
        else:
            os.environ["BITTORCH_KERNEL"] = old_env

    return result


class TestKernelRegression:
    """Tests comparing baseline vs tiled kernel outputs."""

    @pytest.mark.parametrize("shape", [
        (32, 256, 256),
        (64, 512, 512),
        (16, 1024, 1024),
        (8, 2048, 2048),
    ])
    def test_tiled_matches_baseline(self, shape):
        """Tiled kernel should produce same results as baseline."""
        B, K, N = shape
        torch.manual_seed(42)

        x = torch.randn(B, K, device="cuda")
        w_tern = torch.randint(-1, 2, (N, K), device="cuda", dtype=torch.int8)
        scale = torch.rand(N, device="cuda") + 0.5
        bias = torch.randn(N, device="cuda")

        y_baseline = run_kernel_with_env(x, w_tern, scale, bias, "baseline")
        y_tiled = run_kernel_with_env(x, w_tern, scale, bias, "tiled")

        assert torch.allclose(y_baseline, y_tiled, atol=1e-4, rtol=1e-4), \
            f"Shape {shape}: max diff = {(y_baseline - y_tiled).abs().max().item()}"

    @pytest.mark.parametrize("shape", [
        (64, 1024, 4096),  # Transformer-like FFN
        (32, 768, 3072),   # BERT-like
        (16, 4096, 4096),  # Large square
    ])
    def test_tiled_matches_baseline_large_shapes(self, shape):
        """Tiled kernel should match baseline for large shapes."""
        B, K, N = shape
        torch.manual_seed(123)

        x = torch.randn(B, K, device="cuda")
        w_tern = torch.randint(-1, 2, (N, K), device="cuda", dtype=torch.int8)
        scale = torch.rand(N, device="cuda") + 0.5
        bias = torch.randn(N, device="cuda")

        y_baseline = run_kernel_with_env(x, w_tern, scale, bias, "baseline")
        y_tiled = run_kernel_with_env(x, w_tern, scale, bias, "tiled")

        assert torch.allclose(y_baseline, y_tiled, atol=1e-4, rtol=1e-4), \
            f"Shape {shape}: max diff = {(y_baseline - y_tiled).abs().max().item()}"

    def test_tiled_no_bias_matches_baseline(self):
        """Tiled kernel without bias should match baseline."""
        B, K, N = 32, 512, 256
        torch.manual_seed(42)

        x = torch.randn(B, K, device="cuda")
        w_tern = torch.randint(-1, 2, (N, K), device="cuda", dtype=torch.int8)
        scale = torch.rand(N, device="cuda") + 0.5

        y_baseline = run_kernel_with_env(x, w_tern, scale, None, "baseline")
        y_tiled = run_kernel_with_env(x, w_tern, scale, None, "tiled")

        assert torch.allclose(y_baseline, y_tiled, atol=1e-4, rtol=1e-4)

    def test_tiled_fp16_matches_baseline(self):
        """Tiled kernel with FP16 input should match baseline."""
        B, K, N = 32, 512, 256
        torch.manual_seed(42)

        x = torch.randn(B, K, device="cuda", dtype=torch.float16)
        w_tern = torch.randint(-1, 2, (N, K), device="cuda", dtype=torch.int8)
        scale = torch.rand(N, device="cuda", dtype=torch.float16) + 0.5
        bias = torch.randn(N, device="cuda", dtype=torch.float16)

        y_baseline = run_kernel_with_env(x, w_tern, scale, bias, "baseline")
        y_tiled = run_kernel_with_env(x, w_tern, scale, bias, "tiled")

        # FP16 has less precision, use larger tolerance
        assert torch.allclose(y_baseline, y_tiled, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("seed", [0, 42, 123, 456, 789])
    def test_consistency_across_seeds(self, seed):
        """Kernel outputs should be consistent across random seeds."""
        B, K, N = 32, 256, 128
        torch.manual_seed(seed)

        x = torch.randn(B, K, device="cuda")
        w_tern = torch.randint(-1, 2, (N, K), device="cuda", dtype=torch.int8)
        scale = torch.rand(N, device="cuda") + 0.5
        bias = torch.randn(N, device="cuda")

        y_baseline = run_kernel_with_env(x, w_tern, scale, bias, "baseline")
        y_tiled = run_kernel_with_env(x, w_tern, scale, bias, "tiled")

        assert torch.allclose(y_baseline, y_tiled, atol=1e-4, rtol=1e-4), \
            f"Seed {seed}: outputs differ"


class TestKernelEdgeCases:
    """Edge case tests for kernel correctness."""

    def test_batch_size_one(self):
        """Kernel should work with batch size 1."""
        x = torch.randn(1, 256, device="cuda")
        w_tern = torch.randint(-1, 2, (128, 256), device="cuda", dtype=torch.int8)
        scale = torch.rand(128, device="cuda") + 0.5
        bias = torch.randn(128, device="cuda")

        y_baseline = run_kernel_with_env(x, w_tern, scale, bias, "baseline")
        y_tiled = run_kernel_with_env(x, w_tern, scale, bias, "tiled")

        assert torch.allclose(y_baseline, y_tiled, atol=1e-4, rtol=1e-4)

    def test_non_power_of_two_shapes(self):
        """Kernel should work with non-power-of-2 shapes."""
        B, K, N = 17, 123, 67
        torch.manual_seed(42)

        x = torch.randn(B, K, device="cuda")
        w_tern = torch.randint(-1, 2, (N, K), device="cuda", dtype=torch.int8)
        scale = torch.rand(N, device="cuda") + 0.5
        bias = torch.randn(N, device="cuda")

        y_baseline = run_kernel_with_env(x, w_tern, scale, bias, "baseline")
        y_tiled = run_kernel_with_env(x, w_tern, scale, bias, "tiled")

        assert torch.allclose(y_baseline, y_tiled, atol=1e-4, rtol=1e-4)

    def test_all_zeros_ternary(self):
        """Kernel should handle all-zero ternary weights."""
        B, K, N = 16, 128, 64

        x = torch.randn(B, K, device="cuda")
        w_tern = torch.zeros(N, K, device="cuda", dtype=torch.int8)
        scale = torch.rand(N, device="cuda") + 0.5
        bias = torch.randn(N, device="cuda")

        y_baseline = run_kernel_with_env(x, w_tern, scale, bias, "baseline")
        y_tiled = run_kernel_with_env(x, w_tern, scale, bias, "tiled")

        # Output should just be the bias
        expected = bias.unsqueeze(0).expand(B, -1)
        assert torch.allclose(y_baseline, expected, atol=1e-4)
        assert torch.allclose(y_tiled, expected, atol=1e-4)

    def test_all_ones_ternary(self):
        """Kernel should handle all-ones ternary weights."""
        B, K, N = 16, 128, 64

        x = torch.randn(B, K, device="cuda")
        w_tern = torch.ones(N, K, device="cuda", dtype=torch.int8)
        scale = torch.ones(N, device="cuda")
        bias = torch.zeros(N, device="cuda")

        y_baseline = run_kernel_with_env(x, w_tern, scale, bias, "baseline")
        y_tiled = run_kernel_with_env(x, w_tern, scale, bias, "tiled")

        # Output should be sum of x along K dimension
        expected = x.sum(dim=1, keepdim=True).expand(-1, N)
        assert torch.allclose(y_baseline, expected, atol=1e-3)
        assert torch.allclose(y_tiled, expected, atol=1e-3)

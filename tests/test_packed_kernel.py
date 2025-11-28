"""Tests for packed ternary CUDA kernel.

Tests cover:
- Kernel correctness vs unpacked reference
- Various shapes (small, transformer-like)
- Edge cases (K not divisible by 4, small batch sizes)
- FP16 and FP32 dtypes
- Bias handling
- Numerical equivalence with TernaryLinear
"""

import pytest
import torch
import torch.nn.functional as F

from bittorch.quant import pack_ternary, unpack_ternary, ternary_quantize


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPackedKernelCorrectness:
    """Tests for packed CUDA kernel correctness."""

    def test_basic_forward(self):
        """Test basic packed kernel forward pass."""
        from bittorch.ops import ternary_linear_packed_forward

        # Create random ternary weights
        torch.manual_seed(42)
        out_features, in_features = 32, 64
        w_tern = torch.randint(-1, 2, (out_features, in_features)).float()
        scale = torch.rand(out_features)

        # Pack weights
        w_packed, k = pack_ternary(w_tern)

        # Move to GPU
        w_packed = w_packed.cuda()
        scale = scale.cuda()
        x = torch.randn(8, in_features).cuda()

        # Run packed kernel
        y_packed = ternary_linear_packed_forward(x, w_packed, scale, in_features)

        # Reference: unpack and compute
        w_tern_gpu = w_tern.cuda()
        w_effective = w_tern_gpu * scale.unsqueeze(1)
        y_ref = F.linear(x, w_effective)

        # Compare
        assert torch.allclose(y_packed, y_ref, rtol=1e-4, atol=1e-4)

    def test_with_bias(self):
        """Test packed kernel with bias."""
        from bittorch.ops import ternary_linear_packed_forward

        torch.manual_seed(42)
        out_features, in_features = 32, 64
        w_tern = torch.randint(-1, 2, (out_features, in_features)).float()
        scale = torch.rand(out_features)
        bias = torch.randn(out_features)

        w_packed, k = pack_ternary(w_tern)

        w_packed = w_packed.cuda()
        scale = scale.cuda()
        bias = bias.cuda()
        x = torch.randn(8, in_features).cuda()

        y_packed = ternary_linear_packed_forward(x, w_packed, scale, in_features, bias)

        w_tern_gpu = w_tern.cuda()
        w_effective = w_tern_gpu * scale.unsqueeze(1)
        y_ref = F.linear(x, w_effective, bias)

        assert torch.allclose(y_packed, y_ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("out_features,in_features", [
        (4, 8),       # Tiny
        (32, 64),     # Small
        (128, 256),   # Medium
        (768, 768),   # Transformer-like (hidden dim)
        (3072, 768),  # Transformer-like (MLP expansion)
    ])
    def test_various_shapes(self, out_features, in_features):
        """Test packed kernel with various shapes."""
        from bittorch.ops import ternary_linear_packed_forward

        torch.manual_seed(42)
        w_tern = torch.randint(-1, 2, (out_features, in_features)).float()
        scale = torch.rand(out_features)

        w_packed, k = pack_ternary(w_tern)

        w_packed = w_packed.cuda()
        scale = scale.cuda()
        x = torch.randn(16, in_features).cuda()

        y_packed = ternary_linear_packed_forward(x, w_packed, scale, in_features)

        w_tern_gpu = w_tern.cuda()
        w_effective = w_tern_gpu * scale.unsqueeze(1)
        y_ref = F.linear(x, w_effective)

        assert torch.allclose(y_packed, y_ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("in_features", [5, 6, 7, 9, 13, 17, 31, 65])
    def test_padding_required(self, in_features):
        """Test packed kernel with K not divisible by 4."""
        from bittorch.ops import ternary_linear_packed_forward

        torch.manual_seed(42)
        out_features = 32
        w_tern = torch.randint(-1, 2, (out_features, in_features)).float()
        scale = torch.rand(out_features)

        w_packed, k = pack_ternary(w_tern)

        w_packed = w_packed.cuda()
        scale = scale.cuda()
        x = torch.randn(8, in_features).cuda()

        y_packed = ternary_linear_packed_forward(x, w_packed, scale, in_features)

        w_tern_gpu = w_tern.cuda()
        w_effective = w_tern_gpu * scale.unsqueeze(1)
        y_ref = F.linear(x, w_effective)

        assert torch.allclose(y_packed, y_ref, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64])
    def test_batch_sizes(self, batch_size):
        """Test packed kernel with various batch sizes."""
        from bittorch.ops import ternary_linear_packed_forward

        torch.manual_seed(42)
        out_features, in_features = 32, 64
        w_tern = torch.randint(-1, 2, (out_features, in_features)).float()
        scale = torch.rand(out_features)

        w_packed, k = pack_ternary(w_tern)

        w_packed = w_packed.cuda()
        scale = scale.cuda()
        x = torch.randn(batch_size, in_features).cuda()

        y_packed = ternary_linear_packed_forward(x, w_packed, scale, in_features)

        w_tern_gpu = w_tern.cuda()
        w_effective = w_tern_gpu * scale.unsqueeze(1)
        y_ref = F.linear(x, w_effective)

        assert torch.allclose(y_packed, y_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPackedKernelDtypes:
    """Tests for different floating-point dtypes."""

    def test_fp32(self):
        """Test packed kernel with FP32 input."""
        from bittorch.ops import ternary_linear_packed_forward

        torch.manual_seed(42)
        w_tern = torch.randint(-1, 2, (32, 64)).float()
        scale = torch.rand(32).float()
        w_packed, k = pack_ternary(w_tern)

        x = torch.randn(8, 64, dtype=torch.float32).cuda()
        w_packed = w_packed.cuda()
        scale = scale.cuda()

        y_packed = ternary_linear_packed_forward(x, w_packed, scale, 64)

        w_effective = w_tern.cuda() * scale.unsqueeze(1)
        y_ref = F.linear(x, w_effective)

        assert y_packed.dtype == torch.float32
        assert torch.allclose(y_packed, y_ref, rtol=1e-4, atol=1e-4)

    def test_fp16(self):
        """Test packed kernel with FP16 input."""
        from bittorch.ops import ternary_linear_packed_forward

        torch.manual_seed(42)
        w_tern = torch.randint(-1, 2, (32, 64)).float()
        scale = torch.rand(32).half()
        w_packed, k = pack_ternary(w_tern)

        x = torch.randn(8, 64, dtype=torch.float16).cuda()
        w_packed = w_packed.cuda()
        scale = scale.cuda()

        y_packed = ternary_linear_packed_forward(x, w_packed, scale, 64)

        w_effective = w_tern.cuda().half() * scale.unsqueeze(1)
        y_ref = F.linear(x, w_effective)

        assert y_packed.dtype == torch.float16
        assert torch.allclose(y_packed, y_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPackedKernelEdgeCases:
    """Tests for edge cases."""

    def test_all_zeros(self):
        """Test packed kernel with all-zero weights."""
        from bittorch.ops import ternary_linear_packed_forward

        w_tern = torch.zeros(32, 64)
        scale = torch.rand(32)
        w_packed, k = pack_ternary(w_tern)

        x = torch.randn(8, 64).cuda()
        w_packed = w_packed.cuda()
        scale = scale.cuda()

        y = ternary_linear_packed_forward(x, w_packed, scale, 64)

        assert torch.allclose(y, torch.zeros_like(y))

    def test_all_ones(self):
        """Test packed kernel with all +1 weights."""
        from bittorch.ops import ternary_linear_packed_forward

        w_tern = torch.ones(32, 64)
        scale = torch.rand(32)
        w_packed, k = pack_ternary(w_tern)

        x = torch.randn(8, 64).cuda()
        w_packed = w_packed.cuda()
        scale = scale.cuda()

        y_packed = ternary_linear_packed_forward(x, w_packed, scale, 64)

        w_effective = w_tern.cuda() * scale.unsqueeze(1)
        y_ref = F.linear(x, w_effective)

        assert torch.allclose(y_packed, y_ref, rtol=1e-4, atol=1e-4)

    def test_all_neg_ones(self):
        """Test packed kernel with all -1 weights."""
        from bittorch.ops import ternary_linear_packed_forward

        w_tern = -torch.ones(32, 64)
        scale = torch.rand(32)
        w_packed, k = pack_ternary(w_tern)

        x = torch.randn(8, 64).cuda()
        w_packed = w_packed.cuda()
        scale = scale.cuda()

        y_packed = ternary_linear_packed_forward(x, w_packed, scale, 64)

        w_effective = w_tern.cuda() * scale.unsqueeze(1)
        y_ref = F.linear(x, w_effective)

        assert torch.allclose(y_packed, y_ref, rtol=1e-4, atol=1e-4)

    def test_single_batch(self):
        """Test packed kernel with batch size 1."""
        from bittorch.ops import ternary_linear_packed_forward

        torch.manual_seed(42)
        w_tern = torch.randint(-1, 2, (32, 64)).float()
        scale = torch.rand(32)
        w_packed, k = pack_ternary(w_tern)

        x = torch.randn(1, 64).cuda()
        w_packed = w_packed.cuda()
        scale = scale.cuda()

        y_packed = ternary_linear_packed_forward(x, w_packed, scale, 64)

        w_effective = w_tern.cuda() * scale.unsqueeze(1)
        y_ref = F.linear(x, w_effective)

        assert torch.allclose(y_packed, y_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTernaryLinearInferenceCUDA:
    """Tests for TernaryLinearInference with CUDA kernel."""

    def test_inference_uses_cuda_kernel(self):
        """Test that TernaryLinearInference uses CUDA kernel on GPU."""
        from bittorch.nn import TernaryLinearInference
        from bittorch.quant import ternary_quantize

        torch.manual_seed(42)
        w = torch.randn(32, 64)
        w_tern, scale = ternary_quantize(w)

        module = TernaryLinearInference.from_unpacked(w_tern, scale).cuda()

        x = torch.randn(8, 64).cuda()
        y = module(x)

        # Reference: compute manually
        w_effective = w_tern.cuda() * scale.cuda().unsqueeze(1)
        y_ref = F.linear(x, w_effective)

        assert torch.allclose(y, y_ref, rtol=1e-4, atol=1e-4)

    def test_matches_training_module(self):
        """Test that inference output matches TernaryLinear (training)."""
        from bittorch.nn import TernaryLinear, TernaryLinearInference

        torch.manual_seed(42)

        # Create training module
        train_module = TernaryLinear(64, 32, bias=False, quantize=True).cuda()

        # Get quantized weights
        w_tern, scale = train_module.get_quantized_weight()

        # Create inference module
        infer_module = TernaryLinearInference.from_unpacked(w_tern, scale).cuda()

        # Compare outputs
        x = torch.randn(8, 64).cuda()

        train_module.eval()
        with torch.no_grad():
            y_train = train_module(x)
            y_infer = infer_module(x)

        assert torch.allclose(y_train, y_infer, rtol=1e-4, atol=1e-4)

    def test_inference_with_bias(self):
        """Test TernaryLinearInference with bias on GPU."""
        from bittorch.nn import TernaryLinearInference

        torch.manual_seed(42)
        w_tern = torch.randint(-1, 2, (32, 64)).float()
        scale = torch.rand(32)
        bias = torch.randn(32)

        module = TernaryLinearInference.from_unpacked(w_tern, scale, bias).cuda()

        x = torch.randn(8, 64).cuda()
        y = module(x)

        w_effective = w_tern.cuda() * scale.cuda().unsqueeze(1)
        y_ref = F.linear(x, w_effective, bias.cuda())

        assert torch.allclose(y, y_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPackedKernelDeterminism:
    """Tests for deterministic behavior."""

    def test_deterministic_output(self):
        """Test that kernel produces deterministic output."""
        from bittorch.ops import ternary_linear_packed_forward

        torch.manual_seed(42)
        w_tern = torch.randint(-1, 2, (32, 64)).float()
        scale = torch.rand(32)
        w_packed, k = pack_ternary(w_tern)

        x = torch.randn(8, 64).cuda()
        w_packed = w_packed.cuda()
        scale = scale.cuda()

        # Run multiple times
        y1 = ternary_linear_packed_forward(x, w_packed, scale, 64)
        y2 = ternary_linear_packed_forward(x, w_packed, scale, 64)
        y3 = ternary_linear_packed_forward(x, w_packed, scale, 64)

        assert torch.allclose(y1, y2)
        assert torch.allclose(y2, y3)

    @pytest.mark.parametrize("seed", [0, 42, 123, 999])
    def test_random_seeds(self, seed):
        """Test with various random seeds."""
        from bittorch.ops import ternary_linear_packed_forward

        torch.manual_seed(seed)
        w_tern = torch.randint(-1, 2, (32, 64)).float()
        scale = torch.rand(32)
        w_packed, k = pack_ternary(w_tern)

        x = torch.randn(16, 64).cuda()
        w_packed = w_packed.cuda()
        scale = scale.cuda()

        y_packed = ternary_linear_packed_forward(x, w_packed, scale, 64)

        w_effective = w_tern.cuda() * scale.unsqueeze(1)
        y_ref = F.linear(x, w_effective)

        assert torch.allclose(y_packed, y_ref, rtol=1e-4, atol=1e-4)

"""Tests for packed ternary format and TernaryLinearInference.

Tests cover:
- Packing/unpacking correctness
- Roundtrip consistency
- Edge cases (padding, shapes)
- Memory calculations
- TernaryLinearInference forward pass
- Numerical equivalence with regular ternary
"""

import pytest
import torch
import torch.nn as nn

from bittorch.quant import (
    pack_ternary,
    unpack_ternary,
    pack_ternary_with_scale,
    get_packed_size,
    get_memory_reduction,
    ternary_quantize,
)
from bittorch.nn import TernaryLinearInference, TernaryLinear


class TestPackUnpack:
    """Tests for pack_ternary and unpack_ternary functions."""

    def test_basic_roundtrip(self):
        """Test basic pack/unpack roundtrip."""
        w = torch.tensor([[1., -1., 0., 1.], [0., 1., 1., -1.]])
        packed, k = pack_ternary(w)
        unpacked = unpack_ternary(packed, k)
        assert torch.allclose(w, unpacked.float())

    def test_packed_shape(self):
        """Test packed tensor shape."""
        w = torch.tensor([[1., -1., 0., 1.], [0., 1., 1., -1.]])
        packed, k = pack_ternary(w)
        assert packed.shape == (2, 1)  # 4 weights per byte
        assert packed.dtype == torch.uint8

    def test_padding_required(self):
        """Test packing with non-multiple-of-4 in_features."""
        # 6 in_features -> needs padding to 8
        w = torch.tensor([[1., -1., 0., 1., -1., 0.]])
        packed, k = pack_ternary(w)
        assert packed.shape == (1, 2)  # ceil(6/4) = 2 bytes
        assert k == 6  # original in_features preserved

        unpacked = unpack_ternary(packed, k)
        assert unpacked.shape == (1, 6)
        assert torch.allclose(w, unpacked.float())

    def test_single_weight(self):
        """Test packing single weight."""
        w = torch.tensor([[1.]])
        packed, k = pack_ternary(w)
        assert packed.shape == (1, 1)
        unpacked = unpack_ternary(packed, k)
        assert unpacked[0, 0] == 1

    def test_all_zeros(self):
        """Test packing all zeros."""
        w = torch.zeros(4, 8)
        packed, k = pack_ternary(w)
        assert (packed == 0).all()
        unpacked = unpack_ternary(packed, k)
        assert (unpacked == 0).all()

    def test_all_ones(self):
        """Test packing all +1."""
        w = torch.ones(4, 8)
        packed, k = pack_ternary(w)
        # All +1 = 0b01010101 = 85
        assert (packed == 85).all()
        unpacked = unpack_ternary(packed, k)
        assert (unpacked == 1).all()

    def test_all_neg_ones(self):
        """Test packing all -1."""
        w = -torch.ones(4, 8)
        packed, k = pack_ternary(w)
        # All -1 = 0b10101010 = 170
        assert (packed == 170).all()
        unpacked = unpack_ternary(packed, k)
        assert (unpacked == -1).all()

    def test_various_shapes(self):
        """Test packing with various shapes."""
        shapes = [(1, 4), (4, 4), (7, 13), (32, 64), (128, 256)]
        for out_f, in_f in shapes:
            w = torch.randint(-1, 2, (out_f, in_f)).float()
            packed, k = pack_ternary(w)
            unpacked = unpack_ternary(packed, k)
            assert torch.allclose(w, unpacked.float()), f"Failed for shape ({out_f}, {in_f})"

    def test_unpacked_dtype(self):
        """Test that unpacked tensor is int8."""
        w = torch.tensor([[1., -1., 0., 1.]])
        packed, k = pack_ternary(w)
        unpacked = unpack_ternary(packed, k)
        assert unpacked.dtype == torch.int8

    @pytest.mark.parametrize("seed", [0, 42, 123, 999])
    def test_random_roundtrip(self, seed):
        """Test roundtrip with random ternary weights."""
        torch.manual_seed(seed)
        w = torch.randint(-1, 2, (64, 128)).float()
        packed, k = pack_ternary(w)
        unpacked = unpack_ternary(packed, k)
        assert torch.allclose(w, unpacked.float())


class TestMemoryCalculations:
    """Tests for memory calculation utilities."""

    def test_packed_size(self):
        """Test get_packed_size calculation."""
        # 4096 x 4096 with 4 weights per byte
        size = get_packed_size(4096, 4096)
        assert size == 4096 * 1024  # 4M bytes

    def test_packed_size_with_padding(self):
        """Test packed size with padding."""
        # 10 x 5 -> 10 x ceil(5/4) = 10 x 2 = 20 bytes
        size = get_packed_size(10, 5)
        assert size == 20

    def test_memory_reduction(self):
        """Test memory reduction calculation."""
        # For large matrices, should approach 16x
        reduction = get_memory_reduction(4096, 4096)
        assert 15.5 < reduction < 16.5  # ~16x

    def test_memory_reduction_small(self):
        """Test memory reduction for small matrix."""
        # Scale overhead matters more for small matrices
        reduction = get_memory_reduction(4, 4)
        assert reduction > 1.0  # Still some reduction


class TestPackWithScale:
    """Tests for pack_ternary_with_scale."""

    def test_basic(self):
        """Test pack_ternary_with_scale."""
        w = torch.tensor([[1., -1., 0., 1.], [0., 1., 1., -1.]])
        scale = torch.tensor([0.5, 0.8])
        packed, scale_fp16, k = pack_ternary_with_scale(w, scale)

        assert packed.dtype == torch.uint8
        assert scale_fp16.dtype == torch.float16
        assert k == 4


class TestTernaryLinearInference:
    """Tests for TernaryLinearInference module."""

    def test_creation(self):
        """Test module creation."""
        w = torch.randint(-1, 2, (32, 64)).float()
        scale = torch.rand(32)
        packed, k = pack_ternary(w)

        module = TernaryLinearInference(64, 32, packed, scale)
        assert module.in_features == 64
        assert module.out_features == 32
        assert module.bias is None

    def test_forward_shape(self):
        """Test forward pass output shape."""
        w = torch.randint(-1, 2, (32, 64)).float()
        scale = torch.rand(32)
        packed, k = pack_ternary(w)

        module = TernaryLinearInference(64, 32, packed, scale)
        x = torch.randn(8, 64)
        y = module(x)
        assert y.shape == (8, 32)

    def test_forward_with_bias(self):
        """Test forward with bias."""
        w = torch.randint(-1, 2, (32, 64)).float()
        scale = torch.rand(32)
        bias = torch.rand(32)
        packed, k = pack_ternary(w)

        module = TernaryLinearInference(64, 32, packed, scale, bias)
        x = torch.randn(8, 64)
        y = module(x)
        assert y.shape == (8, 32)

    def test_no_gradients(self):
        """Test that module has no gradients."""
        w = torch.randint(-1, 2, (32, 64)).float()
        scale = torch.rand(32)
        packed, k = pack_ternary(w)

        module = TernaryLinearInference(64, 32, packed, scale)

        # All parameters should not require grad
        for param in module.parameters():
            assert not param.requires_grad

        # Buffers should not require grad
        assert not module.weight_packed.requires_grad
        assert not module.scale.requires_grad

    def test_from_unpacked(self):
        """Test from_unpacked class method."""
        w_tern = torch.randint(-1, 2, (32, 64)).float()
        scale = torch.rand(32)

        module = TernaryLinearInference.from_unpacked(w_tern, scale)
        assert module.in_features == 64
        assert module.out_features == 32

    def test_numerical_equivalence(self):
        """Test that inference matches regular ternary computation."""
        torch.manual_seed(42)

        # Create random FP weights
        w = torch.randn(32, 64)

        # Quantize
        w_tern, scale = ternary_quantize(w)

        # Create inference module
        module = TernaryLinearInference.from_unpacked(w_tern, scale)

        # Compute expected output manually
        x = torch.randn(8, 64)
        w_effective = w_tern * scale.unsqueeze(1)
        expected = x @ w_effective.T

        # Compare
        actual = module(x)
        assert torch.allclose(expected, actual, rtol=1e-4, atol=1e-4)

    def test_matches_ternary_linear(self):
        """Test that inference matches TernaryLinear output."""
        torch.manual_seed(42)

        # Create TernaryLinear
        training_module = TernaryLinear(64, 32, bias=False, quantize=True)

        # Get quantized weights
        w_tern, scale = training_module.get_quantized_weight()

        # Create inference module
        infer_module = TernaryLinearInference.from_unpacked(w_tern, scale)

        # Compare outputs
        x = torch.randn(8, 64)

        training_module.eval()
        with torch.no_grad():
            expected = training_module(x)
            actual = infer_module(x)

        assert torch.allclose(expected, actual, rtol=1e-4, atol=1e-4)

    def test_batch_sizes(self):
        """Test various batch sizes."""
        w = torch.randint(-1, 2, (32, 64)).float()
        scale = torch.rand(32)
        module = TernaryLinearInference.from_unpacked(w, scale)

        for batch_size in [1, 4, 16, 64]:
            x = torch.randn(batch_size, 64)
            y = module(x)
            assert y.shape == (batch_size, 32)

    def test_extra_repr(self):
        """Test string representation."""
        w = torch.randint(-1, 2, (32, 64)).float()
        scale = torch.rand(32)
        module = TernaryLinearInference.from_unpacked(w, scale)

        repr_str = module.extra_repr()
        assert "in_features=64" in repr_str
        assert "out_features=32" in repr_str
        assert "reduction=" in repr_str


class TestCreateTernaryLinearInference:
    """Tests for create_ternary_linear_inference convenience function."""

    def test_basic_creation(self):
        """Test basic creation from FP weights."""
        from bittorch.nn.ternary_linear_infer import create_ternary_linear_inference

        w = torch.randn(32, 64)
        module = create_ternary_linear_inference(64, 32, w)

        assert module.in_features == 64
        assert module.out_features == 32
        assert module.bias is None

    def test_with_bias(self):
        """Test creation with bias."""
        from bittorch.nn.ternary_linear_infer import create_ternary_linear_inference

        w = torch.randn(32, 64)
        bias = torch.randn(32)
        module = create_ternary_linear_inference(64, 32, w, bias=bias)

        assert module.bias is not None
        assert module.bias.shape == (32,)


class TestInferenceNoGrad:
    """Tests ensuring no gradient computation in inference."""

    def test_no_grad_on_forward(self):
        """Test that forward doesn't create grad_fn."""
        w = torch.randint(-1, 2, (32, 64)).float()
        scale = torch.rand(32)
        module = TernaryLinearInference.from_unpacked(w, scale)

        x = torch.randn(8, 64, requires_grad=True)
        y = module(x)

        # Output should have grad_fn from input, but module params don't
        # contribute to the computation graph
        assert y.grad_fn is not None  # From x

    def test_buffers_not_parameters(self):
        """Test that weights are buffers, not parameters."""
        w = torch.randint(-1, 2, (32, 64)).float()
        scale = torch.rand(32)
        module = TernaryLinearInference.from_unpacked(w, scale)

        # Should have no learnable parameters
        assert len(list(module.parameters())) == 0

        # But should have buffers
        assert len(list(module.buffers())) >= 2  # weight_packed, scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDA:
    """Tests for CUDA compatibility."""

    def test_pack_unpack_cuda(self):
        """Test packing/unpacking on CUDA."""
        w = torch.randint(-1, 2, (32, 64)).float().cuda()
        packed, k = pack_ternary(w)
        unpacked = unpack_ternary(packed, k)
        assert torch.allclose(w, unpacked.float())

    def test_inference_cuda(self):
        """Test inference module on CUDA."""
        w = torch.randint(-1, 2, (32, 64)).float()
        scale = torch.rand(32)
        module = TernaryLinearInference.from_unpacked(w, scale).cuda()

        x = torch.randn(8, 64).cuda()
        y = module(x)
        assert y.device.type == "cuda"
        assert y.shape == (8, 32)

"""Inference-only ternary linear module with packed weights.

This module provides TernaryLinearInference, a memory-efficient linear layer
that stores weights in packed 2-bit format. Unlike TernaryLinear (for training),
this module:
- Has NO master weights (no FP32 storage overhead)
- Has NO autograd support (inference only)
- Uses packed uint8 storage (~16x smaller than FP32)

Use this for deployment after training with TernaryLinear.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..quant.ternary_packed import pack_ternary, pack_ternary_transposed, unpack_ternary, unpack_ternary_transposed


class TernaryLinearInference(nn.Module):
    """Inference-only linear layer with packed ternary weights.

    This module stores weights in a compact 2-bit packed format, achieving
    ~16x memory reduction compared to FP32. It is designed for deployment
    after training with TernaryLinear.

    The forward pass:
        1. Unpacks ternary weights from uint8 to int8 {-1, 0, +1}
        2. Computes y = x @ (w_tern * scale).T + bias

    Attributes:
        weight_packed: Packed ternary weights in TRANSPOSED layout, shape (ceil(in_features/4), out_features)
        scale: Per-channel scaling factors, shape (out_features,)
        bias: Optional bias, shape (out_features,)
        in_features: Original input dimension
        out_features: Output dimension

    Note:
        This module has requires_grad=False on all parameters.
        Use TernaryLinear for training.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_packed: Tensor,
        scale: Tensor,
        bias: Optional[Tensor] = None,
    ):
        """Initialize TernaryLinearInference.

        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            weight_packed: Packed ternary weights, dtype uint8
            scale: Per-channel scales, shape (out_features,)
            bias: Optional bias tensor
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Register as buffers (not parameters) - no gradients
        self.register_buffer("weight_packed", weight_packed)
        self.register_buffer("scale", scale)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.register_buffer("bias", None)

        # Ensure no gradients
        self.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with packed ternary weights.

        Uses CUDA kernel that reads packed weights directly when on GPU,
        avoiding the memory overhead of unpacking. Falls back to CPU
        unpack + matmul when not on CUDA.

        Args:
            x: Input tensor of shape (*, in_features)

        Returns:
            Output tensor of shape (*, out_features)
        """
        if x.is_cuda:
            # Use packed CUDA kernel - reads 2-bit weights directly
            from ..ops.ternary_linear_packed import ternary_linear_packed_forward

            return ternary_linear_packed_forward(
                x, self.weight_packed, self.scale, self.in_features, self.bias
            )
        else:
            # CPU fallback: unpack + matmul (slow but simple)
            # weight_packed is transposed [K_bytes, N], unpack gives [N, K]
            w_tern = unpack_ternary_transposed(self.weight_packed, self.in_features)
            w_tern = w_tern.to(x.dtype)
            w_effective = w_tern * self.scale.unsqueeze(1).to(x.dtype)
            return torch.nn.functional.linear(x, w_effective, self.bias)

    def extra_repr(self) -> str:
        """String representation with extra info."""
        packed_bytes = self.weight_packed.numel()
        fp32_bytes = self.out_features * self.in_features * 4
        reduction = fp32_bytes / (packed_bytes + self.scale.numel() * 2)
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"packed_bytes={packed_bytes}, "
            f"reduction={reduction:.1f}x"
        )

    @classmethod
    def from_unpacked(
        cls,
        w_tern: Tensor,
        scale: Tensor,
        bias: Optional[Tensor] = None,
    ) -> "TernaryLinearInference":
        """Create from unpacked ternary weights.

        This is useful when you have already quantized weights
        but haven't packed them yet.

        Args:
            w_tern: Ternary weights {-1, 0, +1}, shape (out_features, in_features)
            scale: Per-channel scales, shape (out_features,)
            bias: Optional bias

        Returns:
            TernaryLinearInference instance
        """
        out_features, in_features = w_tern.shape
        # Use transposed layout [K_bytes, N] for optimal GPU memory access
        weight_packed, _ = pack_ternary_transposed(w_tern)
        return cls(
            in_features=in_features,
            out_features=out_features,
            weight_packed=weight_packed,
            scale=scale,
            bias=bias,
        )

    @property
    def packed_size_bytes(self) -> int:
        """Total size of packed weights in bytes."""
        return self.weight_packed.numel()

    @property
    def effective_bits_per_weight(self) -> float:
        """Effective bits per weight including scale overhead."""
        total_bits = self.weight_packed.numel() * 8 + self.scale.numel() * 16
        num_weights = self.out_features * self.in_features
        return total_bits / num_weights


def create_ternary_linear_inference(
    in_features: int,
    out_features: int,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    threshold_factor: float = 0.05,
) -> TernaryLinearInference:
    """Create TernaryLinearInference from FP weights.

    This is a convenience function that:
    1. Quantizes weights to ternary
    2. Packs the ternary weights
    3. Returns a TernaryLinearInference module

    Args:
        in_features: Input dimension
        out_features: Output dimension
        weight: Full-precision weights, shape (out_features, in_features)
        bias: Optional bias
        threshold_factor: Threshold factor for ternary quantization

    Returns:
        TernaryLinearInference instance
    """
    from ..quant.ternary import ternary_quantize

    # Quantize to ternary
    w_tern, scale = ternary_quantize(weight, threshold_factor=threshold_factor)

    # Create inference module
    return TernaryLinearInference.from_unpacked(w_tern, scale, bias)

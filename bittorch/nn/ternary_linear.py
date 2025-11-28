"""TernaryLinear module - drop-in replacement for nn.Linear with ternary weights.

This module implements a linear layer where weights are quantized to ternary values
{-1, 0, +1} during the forward pass, with gradients flowing through via STE.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..quant.ternary import TernaryQuantConfig, ternary_quantize_ste


def _cuda_kernel_available() -> bool:
    """Check if CUDA kernel is available."""
    try:
        from .. import _C
        return hasattr(_C, "ternary_linear_forward")
    except ImportError:
        return False


class TernaryLinear(nn.Module):
    """Linear layer with ternary quantized weights.

    This layer stores full-precision master weights and quantizes them to
    ternary values {-1, 0, +1} during forward pass. Gradients flow through
    the quantization via Straight-Through Estimator (STE).

    The effective computation is:
        y = x @ (w_tern * scale).T + bias

    Where w_tern ∈ {-1, 0, +1} and scale is per-channel.

    When running on CUDA with `use_cuda_kernel=True` (default), the forward
    pass automatically uses the optimized CUDA kernel. Falls back to pure
    Python implementation on CPU or when kernel is unavailable.

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If True, adds a learnable bias. Default: True
        threshold_factor: Factor for ternary quantization threshold. Default: 0.05
        per_channel: If True, use per-channel scaling. Default: True
        quantize: If True, apply ternary quantization. If False, behaves like
            standard nn.Linear (useful for debugging/comparison). Default: True
        use_cuda_kernel: If True, automatically use CUDA kernel when on GPU.
            Default: True

    Shape:
        - Input: (*, in_features) where * means any number of batch dimensions
        - Output: (*, out_features)

    Attributes:
        weight: Master weights of shape (out_features, in_features)
        bias: Bias of shape (out_features) if bias=True, else None

    Example:
        >>> layer = TernaryLinear(64, 32)
        >>> x = torch.randn(8, 64)
        >>> y = layer(x)
        >>> y.shape
        torch.Size([8, 32])

        >>> # On CUDA: automatically uses optimized kernel
        >>> layer = TernaryLinear(64, 32).cuda()
        >>> x = torch.randn(8, 64, device='cuda')
        >>> y = layer(x)  # Uses CUDA kernel

        >>> # Debug mode: behaves like nn.Linear
        >>> layer_debug = TernaryLinear(64, 32, quantize=False)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold_factor: float = 0.05,
        per_channel: bool = True,
        quantize: bool = True,
        use_cuda_kernel: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_factor = threshold_factor
        self.per_channel = per_channel
        self.quantize = quantize
        self.use_cuda_kernel = use_cuda_kernel

        # Master weights in full precision
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _should_use_cuda_kernel(self, x: Tensor) -> bool:
        """Check if we should use CUDA kernel for this forward pass."""
        return (
            self.use_cuda_kernel
            and x.is_cuda
            and self.weight.is_cuda
            and _cuda_kernel_available()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with ternary quantized weights.

        Automatically uses CUDA kernel when on GPU and use_cuda_kernel=True.

        Args:
            x: Input tensor of shape (*, in_features)

        Returns:
            Output tensor of shape (*, out_features)
        """
        if not self.quantize:
            # Debug mode: behave like standard nn.Linear
            return F.linear(x, self.weight, self.bias)

        # Use CUDA kernel path when available
        if self._should_use_cuda_kernel(x):
            from .ternary_linear_cuda import TernaryLinearCUDAFunction
            return TernaryLinearCUDAFunction.apply(
                x,
                self.weight,
                self.bias,
                self.threshold_factor,
                self.per_channel,
            )

        # Pure Python fallback
        # Quantize weights with STE for gradient flow
        w_tern, scale = ternary_quantize_ste(
            self.weight,
            threshold_factor=self.threshold_factor,
            per_channel=self.per_channel,
        )

        # Effective weights: w_tern * scale (broadcast scale over in_features)
        # w_tern: (out_features, in_features)
        # scale: (out_features,)
        w_effective = w_tern * scale.unsqueeze(1)

        # Linear transformation
        return F.linear(x, w_effective, self.bias)

    def extra_repr(self) -> str:
        """String representation with extra info."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"threshold_factor={self.threshold_factor}, "
            f"per_channel={self.per_channel}, "
            f"quantize={self.quantize}, "
            f"use_cuda_kernel={self.use_cuda_kernel}"
        )

    def get_quantized_weight(self) -> tuple[Tensor, Tensor]:
        """Get the current quantized weights (for inspection/debugging).

        Returns:
            Tuple of (w_tern, scale) where:
                w_tern: Ternary weights in {-1, 0, +1}
                scale: Per-channel scaling factors
        """
        from ..quant.ternary import ternary_quantize

        with torch.no_grad():
            return ternary_quantize(
                self.weight,
                threshold_factor=self.threshold_factor,
                per_channel=self.per_channel,
            )

"""TernaryLinear module - drop-in replacement for nn.Linear with ternary weights.

This module implements a linear layer where weights are quantized to ternary values
{-1, 0, +1} during the forward pass, with gradients flowing through via STE.
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..quant.ternary import TernaryQuantConfig, ternary_quantize_ste


BackendType = Literal["auto", "cuda", "python"]


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

    Backend selection:
        - "auto" (default): Uses CUDA kernel when on GPU and available,
          otherwise falls back to pure Python implementation.
        - "cuda": Forces CUDA kernel (raises error if unavailable or on CPU).
        - "python": Forces pure Python implementation (useful for debugging).

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If True, adds a learnable bias. Default: True
        threshold_factor: Factor for ternary quantization threshold. Default: 0.05
        per_channel: If True, use per-channel scaling. Default: True
        quantize: If True, apply ternary quantization. If False, behaves like
            standard nn.Linear (useful for debugging/comparison). Default: True
        backend: Backend to use for forward pass. One of:
            - "auto": Automatically select best backend (default)
            - "cuda": Force CUDA kernel (error if unavailable)
            - "python": Force pure Python implementation
        use_cuda_kernel: Deprecated. Use backend="auto" (True) or backend="python"
            (False) instead. Kept for backward compatibility.

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

        >>> # Force Python backend (useful for debugging)
        >>> layer = TernaryLinear(64, 32, backend="python").cuda()
        >>> x = torch.randn(8, 64, device='cuda')
        >>> y = layer(x)  # Uses Python even on CUDA

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
        backend: BackendType = "auto",
        use_cuda_kernel: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_factor = threshold_factor
        self.per_channel = per_channel
        self.quantize = quantize

        # Handle backend parameter with backward compatibility
        if use_cuda_kernel is not None:
            # Legacy parameter: use_cuda_kernel=True -> auto, False -> python
            import warnings
            warnings.warn(
                "use_cuda_kernel is deprecated, use backend='auto'|'cuda'|'python' instead",
                DeprecationWarning,
                stacklevel=2,
            )
            self.backend = "auto" if use_cuda_kernel else "python"
        else:
            if backend not in ("auto", "cuda", "python"):
                raise ValueError(f"backend must be 'auto', 'cuda', or 'python', got {backend!r}")
            self.backend = backend

        # Keep use_cuda_kernel for backward compatibility in extra_repr
        self.use_cuda_kernel = self.backend != "python"

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
        if self.backend == "python":
            return False

        on_cuda = x.is_cuda and self.weight.is_cuda
        kernel_available = _cuda_kernel_available()

        if self.backend == "cuda":
            # Explicit CUDA requested - raise error if not possible
            if not on_cuda:
                raise RuntimeError(
                    "backend='cuda' requires input and weights on CUDA device"
                )
            if not kernel_available:
                raise RuntimeError(
                    "backend='cuda' requires CUDA extension to be built. "
                    "Rebuild with CUDA support or use backend='python'"
                )
            return True

        # backend == "auto": use CUDA if available, otherwise Python
        return on_cuda and kernel_available

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

        # Cast to input dtype for AMP compatibility
        # Under autocast, x may be FP16 while weights are FP32
        w_effective = w_effective.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None

        # Linear transformation
        return F.linear(x, w_effective, bias)

    def extra_repr(self) -> str:
        """String representation with extra info."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"threshold_factor={self.threshold_factor}, "
            f"per_channel={self.per_channel}, "
            f"quantize={self.quantize}, "
            f"backend={self.backend!r}"
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

"""TernaryLinearCUDA module - CUDA-accelerated ternary linear layer.

This module implements a linear layer where weights are quantized to ternary values
{-1, 0, +1} during forward pass, using a custom CUDA kernel for the forward
computation and PyTorch matmuls for backward (STE).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import custom_fwd, custom_bwd

from ..quant.ternary import ternary_quantize

if TYPE_CHECKING:
    from torch.autograd.function import FunctionCtx


class TernaryLinearCUDAFunction(torch.autograd.Function):
    """Autograd function for ternary linear with CUDA forward."""

    @staticmethod
    @custom_fwd
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        threshold_factor: float,
        per_channel: bool,
    ) -> Tensor:
        """Forward pass using CUDA kernel.

        Args:
            ctx: Autograd context
            x: Input tensor (*, K)
            weight: Master weights (N, K)
            bias: Optional bias (N,)
            threshold_factor: Threshold for ternary quantization
            per_channel: Whether to use per-channel scaling

        Returns:
            Output tensor (*, N)
        """
        from ..ops.ternary_linear import ternary_linear_forward_cuda

        # Quantize weights
        w_tern, scale = ternary_quantize(
            weight, threshold_factor=threshold_factor, per_channel=per_channel
        )

        # Convert to int8 for kernel
        w_tern_int8 = w_tern.to(torch.int8)

        # Call CUDA kernel
        output = ternary_linear_forward_cuda(x, w_tern_int8, scale, bias)

        # Save for backward
        # w_effective is needed for input gradient
        # scale is needed for weight gradient (STE pattern)
        w_effective = w_tern * scale.unsqueeze(1)
        ctx.save_for_backward(x, w_effective, scale)
        ctx.has_bias = bias is not None

        return output

    @staticmethod
    @custom_bwd
    def backward(
        ctx: FunctionCtx, grad_output: Tensor
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, None, None]:
        """Backward pass using PyTorch matmuls (STE).

        The STE pattern means gradients flow through quantization as if it
        were identity, but we need to account for the scale factor.

        In forward: w_effective = (w_tern.detach() - weight.detach() + weight) * scale
        So: d(w_effective)/d(weight) = scale (element-wise)
        Thus: dL/d(weight) = dL/d(w_effective) * scale
        """
        x, w_effective, scale = ctx.saved_tensors

        grad_x = grad_weight = grad_bias = None

        # Reshape for matmul if needed
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        # Gradient w.r.t. input: grad_output @ w_effective
        if ctx.needs_input_grad[0]:
            grad_x = grad_output_2d @ w_effective
            grad_x = grad_x.reshape(input_shape)

        # Gradient w.r.t. weight (STE pattern)
        # dL/d(weight) = dL/d(w_effective) * scale
        # where dL/d(w_effective) = grad_output.T @ x
        if ctx.needs_input_grad[1]:
            grad_w_effective = grad_output_2d.T @ x_2d  # (N, K)
            grad_weight = grad_w_effective * scale.unsqueeze(1)  # scale broadcast

        # Gradient w.r.t. bias
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_output_2d.sum(dim=0)

        # Return gradients for all inputs (including threshold_factor and per_channel)
        return grad_x, grad_weight, grad_bias, None, None


class TernaryLinearCUDA(nn.Module):
    """Linear layer with ternary quantized weights using CUDA kernel.

    This layer stores full-precision master weights and quantizes them to
    ternary values {-1, 0, +1} during forward pass. The forward computation
    uses a custom CUDA kernel, while backward uses PyTorch matmuls with STE.

    The effective computation is:
        y = x @ (w_tern * scale).T + bias

    Where w_tern ∈ {-1, 0, +1} and scale is per-channel.

    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If True, adds a learnable bias. Default: True
        threshold_factor: Factor for ternary quantization threshold. Default: 0.05
        per_channel: If True, use per-channel scaling. Default: True

    Shape:
        - Input: (*, in_features) where * means any number of batch dimensions
        - Output: (*, out_features)

    Note:
        Requires CUDA. For CPU fallback, use TernaryLinear instead.

    Example:
        >>> layer = TernaryLinearCUDA(64, 32).cuda()
        >>> x = torch.randn(8, 64, device='cuda')
        >>> y = layer(x)
        >>> y.shape
        torch.Size([8, 32])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold_factor: float = 0.05,
        per_channel: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_factor = threshold_factor
        self.per_channel = per_channel

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

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with ternary quantized weights using CUDA kernel.

        Args:
            x: Input tensor of shape (*, in_features)

        Returns:
            Output tensor of shape (*, out_features)

        Raises:
            RuntimeError: If tensors are not on CUDA device.
        """
        if not x.is_cuda:
            raise RuntimeError(
                "TernaryLinearCUDA requires CUDA tensors. "
                "Use TernaryLinear for CPU."
            )

        return TernaryLinearCUDAFunction.apply(
            x,
            self.weight,
            self.bias,
            self.threshold_factor,
            self.per_channel,
        )

    def extra_repr(self) -> str:
        """String representation with extra info."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"threshold_factor={self.threshold_factor}, "
            f"per_channel={self.per_channel}"
        )

    def get_quantized_weight(self) -> tuple[Tensor, Tensor]:
        """Get the current quantized weights (for inspection/debugging).

        Returns:
            Tuple of (w_tern, scale) where:
                w_tern: Ternary weights in {-1, 0, +1}
                scale: Per-channel scaling factors
        """
        with torch.no_grad():
            return ternary_quantize(
                self.weight,
                threshold_factor=self.threshold_factor,
                per_channel=self.per_channel,
            )

"""Functional interface for ternary linear operations.

This module provides the functional API for ternary linear operations,
wrapping the CUDA kernel with proper tensor preparation and validation.
"""

from typing import Optional

import torch
from torch import Tensor

from ..quant.ternary import ternary_quantize


def ternary_linear_forward_cuda(
    x: Tensor,
    w_tern: Tensor,
    scale: Tensor,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Ternary linear forward pass using CUDA kernel.

    Computes Y = X @ (W_tern * scale).T + bias

    Args:
        x: Input tensor of shape (*, K)
        w_tern: Ternary weights of shape (N, K) as int8 ({-1, 0, +1})
        scale: Per-channel scale of shape (N,)
        bias: Optional bias of shape (N,)

    Returns:
        Output tensor of shape (*, N)

    Raises:
        RuntimeError: If inputs are not CUDA tensors or have invalid shapes.
    """
    from bittorch._C import ternary_linear_forward

    # Ensure contiguous tensors
    x = x.contiguous()
    w_tern = w_tern.contiguous()

    # Cast scale and bias to match input dtype (for mixed precision compatibility)
    scale = scale.to(dtype=x.dtype).contiguous()

    if bias is not None:
        bias = bias.to(dtype=x.dtype).contiguous()

    return ternary_linear_forward(x, w_tern, scale, bias)


def ternary_linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    threshold_factor: float = 0.05,
    per_channel: bool = True,
) -> Tensor:
    """Ternary linear transformation with on-the-fly quantization.

    This is the high-level functional API that:
    1. Quantizes weights to ternary values
    2. Calls the CUDA kernel for the forward pass

    For training, use the TernaryLinear module which handles STE gradients.

    Args:
        x: Input tensor of shape (*, in_features)
        weight: Weight tensor of shape (out_features, in_features)
        bias: Optional bias tensor of shape (out_features,)
        threshold_factor: Factor for ternary quantization threshold
        per_channel: If True, use per-channel scaling

    Returns:
        Output tensor of shape (*, out_features)
    """
    # Quantize weights to ternary
    w_tern, scale = ternary_quantize(
        weight, threshold_factor=threshold_factor, per_channel=per_channel
    )

    # Convert ternary weights to int8 for kernel
    w_tern_int8 = w_tern.to(torch.int8)

    # Call CUDA kernel
    return ternary_linear_forward_cuda(x, w_tern_int8, scale, bias)

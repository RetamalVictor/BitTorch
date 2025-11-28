"""Functional interface for packed ternary linear operations.

This module provides the functional API for packed ternary linear operations,
calling the CUDA kernel that reads 2-bit packed weights directly without
unpacking to full float/int8 tensors.
"""

from typing import Optional

import torch
from torch import Tensor


def ternary_linear_packed_forward(
    x: Tensor,
    w_packed: Tensor,
    scale: Tensor,
    in_features: int,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Packed ternary linear forward pass using CUDA kernel.

    Computes Y = X @ (W_packed * scale).T + bias

    This kernel reads packed 2-bit ternary weights directly without
    materializing full float/int8 weight tensors, achieving ~16x memory
    reduction compared to FP32.

    Args:
        x: Input tensor of shape (*, K) where K = in_features
        w_packed: Packed ternary weights of shape (N, K_bytes) as uint8
                  where K_bytes = ceil(K/4) and each byte contains 4 ternary values
        scale: Per-channel scale of shape (N,)
        in_features: Original K (unpacked) - needed because K_bytes = ceil(K/4)
        bias: Optional bias of shape (N,)

    Returns:
        Output tensor of shape (*, N)

    Raises:
        RuntimeError: If inputs are not CUDA tensors or have invalid shapes.

    Note:
        The packed format uses 2-bit encoding:
        - 00 = 0
        - 01 = +1
        - 10 = -1
        - 11 = reserved

        Layout: bits [1:0] = weight[0], [3:2] = weight[1], etc.
    """
    from bittorch._C import ternary_linear_packed_forward as _cuda_forward

    # Ensure contiguous tensors
    x = x.contiguous()
    w_packed = w_packed.contiguous()
    scale = scale.contiguous()

    if bias is not None:
        bias = bias.contiguous()

    return _cuda_forward(x, w_packed, scale, bias, in_features)


def has_packed_cuda_support() -> bool:
    """Check if packed CUDA kernel is available.

    Returns:
        True if the packed CUDA kernel is available, False otherwise.
    """
    try:
        from bittorch._C import ternary_linear_packed_forward
        return torch.cuda.is_available()
    except ImportError:
        return False

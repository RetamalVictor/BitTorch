"""Functional interface for packed ternary linear operations.

This module provides the functional API for packed ternary linear operations,
calling the CUDA kernel that reads 2-bit packed weights directly without
unpacking to full float/int8 tensors.

The production kernel uses TRANSPOSED weight layout [K_bytes, N] and
automatically selects between:
- Small-batch kernel (B <= 32): optimized for inference
- Large-batch kernel (B > 32): optimized for training

Override kernel selection via BITTORCH_KERNEL env var: "small", "large", "auto"
"""

from typing import Optional

import torch
from torch import Tensor


def ternary_linear_packed_forward(
    x: Tensor,
    w_packed_T: Tensor,
    scale: Tensor,
    in_features: int,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """Packed ternary linear forward pass with automatic kernel selection.

    Computes Y = X @ (W_packed_T * scale).T + bias

    This kernel reads packed 2-bit ternary weights directly without
    materializing full float/int8 weight tensors, achieving ~16x memory
    reduction compared to FP32.

    Uses TRANSPOSED weight layout [K_bytes, N] for optimal memory access.
    Automatically selects kernel based on batch size:
    - B <= 32: small-batch kernel (optimized for inference)
    - B > 32: large-batch kernel (optimized for training)

    Override via BITTORCH_KERNEL env var: "small", "large", or "auto"

    Args:
        x: Input tensor of shape (*, K) where K = in_features
        w_packed_T: Packed ternary weights of shape (K_bytes, N) as uint8
                    where K_bytes = ceil(K/4) and each byte contains 4 ternary values.
                    Use pack_ternary_transposed() to create this tensor.
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
    w_packed_T = w_packed_T.contiguous()
    scale = scale.contiguous()

    if bias is not None:
        bias = bias.contiguous()

    return _cuda_forward(x, w_packed_T, scale, bias, in_features)


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

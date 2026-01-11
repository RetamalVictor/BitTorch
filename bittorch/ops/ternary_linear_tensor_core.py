"""
Tensor Core Ternary Linear Operation.

Uses INT8 Tensor Cores with LUT-based 2-bit to INT8 unpacking.
Experimental kernel for improved training performance.

Kernel versions:
- V1: LUT unpack + float compute (validates data path)
- V2: LUT unpack + mma.sync (actual Tensor Core compute)
"""

from typing import Optional
import torch
from torch import Tensor


def ternary_linear_tensor_core_forward(
    x: Tensor,
    w_packed_T: Tensor,
    scale: Tensor,
    in_features: int,
    bias: Optional[Tensor] = None,
    version: int = 1,
) -> Tensor:
    """
    Forward pass for ternary linear using Tensor Cores.

    Args:
        x: Input tensor [B, K] float32
        w_packed_T: Packed transposed weights [K_bytes, N] uint8
        scale: Per-channel scale [N] float32
        in_features: Number of input features (K)
        bias: Optional bias [N] float32
        version: Kernel version (1=V1 float compute, 2=V2 mma.sync)

    Returns:
        Output tensor [B, N] float32
    """
    from bittorch._C import ternary_linear_tensor_core_forward as _cuda_forward

    return _cuda_forward(x, w_packed_T, scale, bias, in_features, version)

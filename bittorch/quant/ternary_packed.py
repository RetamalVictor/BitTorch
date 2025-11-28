"""Packed ternary weight utilities for inference.

This module provides functions to pack and unpack ternary weights {-1, 0, +1}
into a compact 2-bit representation for memory-efficient inference.

Encoding:
    00 = 0
    01 = +1
    10 = -1
    11 = reserved

Four weights are packed per byte, achieving ~16x memory reduction vs FP32.
See docs/ternary_format.md for the complete specification.
"""

from typing import Tuple

import torch
from torch import Tensor


def pack_ternary(w_tern: Tensor) -> Tuple[Tensor, int]:
    """Pack ternary weights {-1, 0, +1} into uint8 tensor.

    Each byte contains 4 ternary values using 2-bit encoding:
        00 = 0, 01 = +1, 10 = -1

    Args:
        w_tern: Ternary weights, shape (out_features, in_features).
                Values must be in {-1, 0, +1}.

    Returns:
        Tuple of (packed, original_in_features) where:
            packed: Packed weights, shape (out_features, ceil(in_features/4)),
                    dtype uint8
            original_in_features: Original in_features dimension (for unpacking)

    Example:
        >>> w = torch.tensor([[1, -1, 0, 1], [0, 1, 1, -1]], dtype=torch.float32)
        >>> packed, orig_k = pack_ternary(w)
        >>> packed.shape
        torch.Size([2, 1])
    """
    if w_tern.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {w_tern.dim()}D")

    out_features, in_features = w_tern.shape
    device = w_tern.device

    # Pad in_features to multiple of 4
    pad_size = (4 - in_features % 4) % 4
    if pad_size > 0:
        w_padded = torch.nn.functional.pad(w_tern, (0, pad_size), value=0)
    else:
        w_padded = w_tern

    # Encode: -1 -> 2, 0 -> 0, +1 -> 1
    # Use integer arithmetic for encoding
    encoded = torch.zeros_like(w_padded, dtype=torch.uint8)
    encoded[w_padded == 1] = 1
    encoded[w_padded == -1] = 2

    # Reshape to (out_features, num_bytes, 4) for packing
    num_bytes = w_padded.shape[1] // 4
    encoded = encoded.view(out_features, num_bytes, 4)

    # Pack 4 values per byte: w0 in bits 0-1, w1 in bits 2-3, w2 in bits 4-5, w3 in bits 6-7
    packed = (
        encoded[:, :, 0]
        | (encoded[:, :, 1] << 2)
        | (encoded[:, :, 2] << 4)
        | (encoded[:, :, 3] << 6)
    )

    return packed, in_features


def unpack_ternary(packed: Tensor, in_features: int) -> Tensor:
    """Unpack uint8 tensor to ternary weights {-1, 0, +1}.

    Args:
        packed: Packed weights, shape (out_features, packed_in_features),
                dtype uint8
        in_features: Original input dimension (removes padding)

    Returns:
        Ternary weights, shape (out_features, in_features), dtype int8.
        Values are in {-1, 0, +1}.

    Example:
        >>> packed = torch.tensor([[0x49]], dtype=torch.uint8)  # [1, -1, 0, 1]
        >>> w = unpack_ternary(packed, 4)
        >>> w
        tensor([[ 1, -1,  0,  1]], dtype=torch.int8)
    """
    if packed.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {packed.dim()}D")

    out_features, packed_in = packed.shape
    device = packed.device

    # Extract 2-bit values
    w0 = packed & 0x03  # bits 0-1
    w1 = (packed >> 2) & 0x03  # bits 2-3
    w2 = (packed >> 4) & 0x03  # bits 4-5
    w3 = (packed >> 6) & 0x03  # bits 6-7

    # Interleave back to original order
    unpacked = torch.stack([w0, w1, w2, w3], dim=2)  # (out, packed_in, 4)
    unpacked = unpacked.view(out_features, packed_in * 4)

    # Decode: 0 -> 0, 1 -> +1, 2 -> -1, 3 -> 0 (reserved treated as 0)
    decoded = torch.zeros_like(unpacked, dtype=torch.int8)
    decoded[unpacked == 1] = 1
    decoded[unpacked == 2] = -1

    # Remove padding
    decoded = decoded[:, :in_features]

    return decoded


def pack_ternary_with_scale(
    w_tern: Tensor, scale: Tensor
) -> Tuple[Tensor, Tensor, int]:
    """Pack ternary weights along with scale for complete inference state.

    This is a convenience function that packs weights and returns the scale
    unchanged (but ensures it's in the correct dtype for inference).

    Args:
        w_tern: Ternary weights, shape (out_features, in_features)
        scale: Per-channel scales, shape (out_features,)

    Returns:
        Tuple of (packed_weights, scale_fp16, original_in_features)
    """
    packed, in_features = pack_ternary(w_tern)
    # Convert scale to FP16 for inference efficiency
    scale_fp16 = scale.to(torch.float16)
    return packed, scale_fp16, in_features


def get_packed_size(out_features: int, in_features: int) -> int:
    """Calculate packed tensor size in bytes.

    Args:
        out_features: Number of output features
        in_features: Number of input features

    Returns:
        Total bytes for packed weights (excluding scale)
    """
    packed_in = (in_features + 3) // 4  # ceil division
    return out_features * packed_in


def get_memory_reduction(out_features: int, in_features: int) -> float:
    """Calculate memory reduction factor vs FP32.

    Args:
        out_features: Number of output features
        in_features: Number of input features

    Returns:
        Reduction factor (e.g., 16.0 means 16x smaller)
    """
    fp32_bytes = out_features * in_features * 4
    packed_bytes = get_packed_size(out_features, in_features)
    scale_bytes = out_features * 2  # FP16 scale
    return fp32_bytes / (packed_bytes + scale_bytes)

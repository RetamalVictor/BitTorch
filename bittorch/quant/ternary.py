"""Ternary quantization utilities for BitNet-style 1.58-bit weights.

This module implements ternary quantization where weights are mapped to {-1, 0, +1}
with per-channel scaling factors. The quantization follows the BitNet b1.58 approach.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass
class TernaryQuantConfig:
    """Configuration for ternary quantization.

    Args:
        threshold_factor: Factor λ for computing threshold τ = λ * scale.
            Values in range [0.05, 0.1] work well. Default: 0.05
        per_channel: If True, compute scale per output channel.
            If False, use a single global scale. Default: True
    """

    threshold_factor: float = 0.05
    per_channel: bool = True


def ternary_quantize(
    weight: Tensor,
    threshold_factor: float = 0.05,
    per_channel: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Quantize weights to ternary values {-1, 0, +1}.

    Implements the quantization scheme:
        scale = max(|w|) per channel (or global)
        threshold = threshold_factor * scale
        w_tern = sign(w) where |w| > threshold, else 0

    Args:
        weight: Input weight tensor of shape (out_features, in_features)
        threshold_factor: Factor for computing threshold. Default: 0.05
        per_channel: If True, compute scale per output channel. Default: True

    Returns:
        Tuple of (w_tern, scale) where:
            w_tern: Ternary weights in {-1, 0, +1}, same shape as input
            scale: Scaling factors, shape (out_features,) if per_channel else (1,)
    """
    if per_channel:
        # Per-channel scale: max absolute value along input dimension
        # weight shape: (out_features, in_features)
        scale = weight.abs().max(dim=1, keepdim=True).values
        # Avoid division by zero
        scale = scale.clamp(min=1e-8)
    else:
        # Global scale
        scale = weight.abs().max().unsqueeze(0)
        scale = scale.clamp(min=1e-8)

    # Compute threshold
    threshold = threshold_factor * scale

    # Quantize: sign(w) where |w| > threshold, else 0
    abs_weight = weight.abs()
    w_tern = torch.where(
        abs_weight > threshold,
        torch.sign(weight),
        torch.zeros_like(weight),
    )

    # Squeeze scale to (out_features,) if per_channel
    if per_channel:
        scale = scale.squeeze(1)
    else:
        scale = scale.squeeze(0)

    return w_tern, scale


def ternary_quantize_ste(
    weight: Tensor,
    threshold_factor: float = 0.05,
    per_channel: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Quantize weights with Straight-Through Estimator (STE) for training.

    This version allows gradients to flow through the quantization operation
    by using the STE trick: forward uses quantized values, backward treats
    quantization as identity.

    The STE pattern used:
        w_q = (quantize(w)).detach() - w.detach() + w

    This gives:
        Forward: w_q (quantized values)
        Backward: gradient flows to w as if quantization was identity

    Note: Scale is treated as a non-differentiable calibration statistic.
    Gradients do NOT flow through scale, matching the CUDA kernel behavior.
    This ensures consistent gradients between Python and CUDA implementations.

    Args:
        weight: Input weight tensor (master weights in FP32/FP16)
        threshold_factor: Factor for computing threshold. Default: 0.05
        per_channel: If True, compute scale per output channel. Default: True

    Returns:
        Tuple of (w_tern_ste, scale) where:
            w_tern_ste: Ternary weights with STE gradient passthrough
            scale: Scaling factors (detached, no gradient)
    """
    w_tern, scale = ternary_quantize(weight, threshold_factor, per_channel)

    # STE: forward uses quantized, backward flows through original
    # w_tern_ste has the value of w_tern but gradients flow to weight
    w_tern_ste = w_tern.detach() - weight.detach() + weight

    # Detach scale - it's a calibration statistic, not a learnable path.
    # This matches CUDA kernel behavior where scale is treated as constant.
    scale = scale.detach()

    return w_tern_ste, scale


def dequantize_ternary(w_tern: Tensor, scale: Tensor) -> Tensor:
    """Dequantize ternary weights back to full precision.

    Args:
        w_tern: Ternary weights in {-1, 0, +1}, shape (out_features, in_features)
        scale: Scaling factors, shape (out_features,)

    Returns:
        Dequantized weights, shape (out_features, in_features)
    """
    # Broadcast scale: (out_features,) -> (out_features, 1)
    return w_tern * scale.unsqueeze(1)

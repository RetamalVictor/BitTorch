"""Ternary quantization utilities for BitNet-style 1.58-bit weights.

This module implements ternary quantization where weights are mapped to {-1, 0, +1}
with per-channel scaling factors. The quantization follows the BitNet b1.58 approach.
"""

from dataclasses import dataclass, field
from typing import Tuple

import torch
from torch import Tensor

from .base import QuantConfig, Quantizer, QuantType, ScaleType


@dataclass
class TernaryQuantConfig(QuantConfig):
    """Configuration for ternary quantization.

    Ternary quantization maps weights to {-1, 0, +1} with per-channel scaling.
    This achieves approximately 1.58 bits per weight (log2(3) ≈ 1.58).

    Args:
        threshold_factor: Factor λ for computing threshold τ = λ * scale.
            Values in range [0.05, 0.1] work well. Default: 0.05
        per_channel: If True, use per-channel scaling. If False, use global.
            Default: True
    """
    threshold_factor: float = 0.05
    per_channel: bool = True

    def __post_init__(self):
        # Sync scale_type with per_channel for internal consistency
        if self.per_channel:
            self.scale_type = ScaleType.PER_CHANNEL
        else:
            self.scale_type = ScaleType.PER_TENSOR

    def validate(self) -> None:
        """Validate ternary-specific parameters."""
        super().validate()
        if not 0.0 <= self.threshold_factor <= 1.0:
            raise ValueError(
                f"threshold_factor must be in [0, 1], got {self.threshold_factor}"
            )


class TernaryQuantizer(Quantizer):
    """Quantizer for ternary (1.58-bit) weights.

    Maps weights to {-1, 0, +1} with per-channel or global scaling.

    Example:
        >>> config = TernaryQuantConfig(threshold_factor=0.05)
        >>> quantizer = TernaryQuantizer(config)
        >>> w_tern, scale = quantizer.quantize(weight)
        >>> w_recon = quantizer.dequantize(w_tern, scale)
    """

    def __init__(self, config: TernaryQuantConfig = None):
        """Initialize ternary quantizer.

        Args:
            config: Ternary quantization config. If None, uses defaults.
        """
        if config is None:
            config = TernaryQuantConfig()
        super().__init__(config)
        self._config: TernaryQuantConfig = config

    @property
    def quant_type(self) -> QuantType:
        return QuantType.TERNARY

    @property
    def bits_per_weight(self) -> float:
        return 1.58  # log2(3)

    def quantize(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantize weights to ternary {-1, 0, +1}."""
        return ternary_quantize(
            weight,
            threshold_factor=self._config.threshold_factor,
            per_channel=self._config.per_channel,
        )

    def dequantize(self, w_tern: Tensor, scale: Tensor) -> Tensor:
        """Dequantize ternary weights back to full precision."""
        return dequantize_ternary(w_tern, scale)

    def quantize_ste(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantize with STE for training."""
        return ternary_quantize_ste(
            weight,
            threshold_factor=self._config.threshold_factor,
            per_channel=self._config.per_channel,
        )


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

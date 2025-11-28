"""Base classes for quantization configurations and operations.

This module provides the extensible foundation for different quantization schemes
(ternary, INT4, etc.) to plug into a common interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional, Any

import torch
from torch import Tensor


class QuantType(Enum):
    """Supported quantization types."""
    TERNARY = "ternary"  # {-1, 0, +1} with 1.58 bits
    INT4 = "int4"        # 4-bit signed integers {-8, ..., +7}
    # Future: FP4 = "fp4"


class ScaleType(Enum):
    """How scaling factors are computed."""
    PER_TENSOR = "per_tensor"    # Single scale for entire tensor
    PER_CHANNEL = "per_channel"  # Scale per output channel
    PER_GROUP = "per_group"      # Scale per group of channels (for INT4)


@dataclass
class QuantConfig:
    """Base configuration for quantization schemes.

    This is the base class that all quantization configs inherit from.
    It provides common parameters shared across different quantization types.

    Args:
        scale_type: How to compute scaling factors.
        group_size: Size of groups for per-group scaling. Only used when
            scale_type is PER_GROUP. Default: 128
    """
    scale_type: ScaleType = ScaleType.PER_CHANNEL
    group_size: int = 128

    @property
    def per_channel(self) -> bool:
        """Convenience property for backward compatibility."""
        return self.scale_type == ScaleType.PER_CHANNEL

    def validate(self) -> None:
        """Validate configuration parameters. Override in subclasses."""
        if self.scale_type == ScaleType.PER_GROUP and self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")


class Quantizer(ABC):
    """Abstract base class for quantizers.

    A quantizer handles the conversion between full-precision and quantized
    representations. Each quantization scheme (ternary, INT4, etc.) implements
    this interface.

    Subclasses must implement:
        - quantize(): Convert FP weights to quantized representation
        - dequantize(): Convert quantized weights back to FP
        - quantize_ste(): Quantize with STE for training
    """

    def __init__(self, config: QuantConfig):
        """Initialize quantizer with configuration.

        Args:
            config: Quantization configuration
        """
        self.config = config
        config.validate()

    @abstractmethod
    def quantize(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantize weights to low-precision representation.

        Args:
            weight: Input weight tensor, typically (out_features, in_features)

        Returns:
            Tuple of (quantized_weight, scale) where:
                quantized_weight: Low-precision weights
                scale: Scaling factors for dequantization
        """
        pass

    @abstractmethod
    def dequantize(self, w_quant: Tensor, scale: Tensor) -> Tensor:
        """Dequantize weights back to full precision.

        Args:
            w_quant: Quantized weights
            scale: Scaling factors

        Returns:
            Dequantized weights in full precision
        """
        pass

    @abstractmethod
    def quantize_ste(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantize with Straight-Through Estimator for training.

        The STE allows gradients to flow through the quantization
        operation by treating it as identity in the backward pass.

        Args:
            weight: Input weight tensor (master weights)

        Returns:
            Tuple of (quantized_weight_ste, scale) where:
                quantized_weight_ste: Quantized weights with STE gradient
                scale: Scaling factors (typically detached)
        """
        pass

    @property
    @abstractmethod
    def quant_type(self) -> QuantType:
        """Return the quantization type this quantizer implements."""
        pass

    @property
    @abstractmethod
    def bits_per_weight(self) -> float:
        """Return the effective bits per weight for this quantization scheme."""
        pass


def compute_scale(
    weight: Tensor,
    scale_type: ScaleType,
    group_size: int = 128,
    eps: float = 1e-8,
) -> Tensor:
    """Compute scaling factors for a weight tensor.

    This is a shared utility for computing scales across different
    quantization schemes.

    Args:
        weight: Input weight tensor of shape (out_features, in_features)
        scale_type: How to compute scales
        group_size: Size of groups for per-group scaling
        eps: Small value to avoid division by zero

    Returns:
        Scaling factors tensor
    """
    if scale_type == ScaleType.PER_TENSOR:
        scale = weight.abs().max()
        return scale.clamp(min=eps).unsqueeze(0)

    elif scale_type == ScaleType.PER_CHANNEL:
        # Max absolute value along input dimension
        scale = weight.abs().max(dim=1).values
        return scale.clamp(min=eps)

    elif scale_type == ScaleType.PER_GROUP:
        out_features, in_features = weight.shape
        # Pad if necessary
        if in_features % group_size != 0:
            pad_size = group_size - (in_features % group_size)
            weight = torch.nn.functional.pad(weight, (0, pad_size))
            in_features = weight.shape[1]

        # Reshape to (out_features, num_groups, group_size)
        num_groups = in_features // group_size
        weight_grouped = weight.view(out_features, num_groups, group_size)

        # Max per group
        scale = weight_grouped.abs().max(dim=2).values
        return scale.clamp(min=eps)

    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")

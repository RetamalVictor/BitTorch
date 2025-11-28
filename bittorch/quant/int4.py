"""INT4 quantization utilities (placeholder for future implementation).

This module will implement 4-bit integer quantization where weights are mapped
to signed 4-bit values {-8, ..., +7} with per-channel or per-group scaling.

Status: PLACEHOLDER - Not yet implemented. This shows the structure for
future INT4 support.
"""

from dataclasses import dataclass, field
from typing import Tuple

import torch
from torch import Tensor

from .base import QuantConfig, Quantizer, QuantType, ScaleType


@dataclass
class Int4QuantConfig(QuantConfig):
    """Configuration for INT4 quantization.

    INT4 quantization maps weights to signed 4-bit integers {-8, ..., +7}
    with per-channel or per-group scaling. This achieves 4 bits per weight.

    Args:
        scale_type: How to compute scales. Default: PER_GROUP (common for INT4)
        group_size: Size of groups for per-group scaling. Default: 128
        symmetric: If True, use symmetric quantization around zero.
            If False, use asymmetric with zero-point. Default: True
    """
    scale_type: ScaleType = field(default=ScaleType.PER_GROUP)
    group_size: int = 128
    symmetric: bool = True

    def validate(self) -> None:
        """Validate INT4-specific parameters."""
        super().validate()
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")


class Int4Quantizer(Quantizer):
    """Quantizer for INT4 (4-bit) weights.

    Maps weights to {-8, ..., +7} with per-channel or per-group scaling.

    Status: NOT YET IMPLEMENTED - This is a placeholder showing the interface.

    Example (future):
        >>> config = Int4QuantConfig(group_size=128)
        >>> quantizer = Int4Quantizer(config)
        >>> w_int4, scale = quantizer.quantize(weight)
        >>> w_recon = quantizer.dequantize(w_int4, scale)
    """

    def __init__(self, config: Int4QuantConfig = None):
        """Initialize INT4 quantizer.

        Args:
            config: INT4 quantization config. If None, uses defaults.
        """
        if config is None:
            config = Int4QuantConfig()
        super().__init__(config)
        self._config: Int4QuantConfig = config

    @property
    def quant_type(self) -> QuantType:
        return QuantType.INT4

    @property
    def bits_per_weight(self) -> float:
        return 4.0

    def quantize(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantize weights to INT4 {-8, ..., +7}.

        NOT YET IMPLEMENTED.
        """
        raise NotImplementedError(
            "INT4 quantization is not yet implemented. "
            "Coming in a future version of BitTorch."
        )

    def dequantize(self, w_int4: Tensor, scale: Tensor) -> Tensor:
        """Dequantize INT4 weights back to full precision.

        NOT YET IMPLEMENTED.
        """
        raise NotImplementedError(
            "INT4 dequantization is not yet implemented. "
            "Coming in a future version of BitTorch."
        )

    def quantize_ste(self, weight: Tensor) -> Tuple[Tensor, Tensor]:
        """Quantize with STE for training.

        NOT YET IMPLEMENTED.
        """
        raise NotImplementedError(
            "INT4 quantization with STE is not yet implemented. "
            "Coming in a future version of BitTorch."
        )


# Functional API (placeholders)

def int4_quantize(
    weight: Tensor,
    group_size: int = 128,
    symmetric: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Quantize weights to INT4 values {-8, ..., +7}.

    NOT YET IMPLEMENTED.

    Args:
        weight: Input weight tensor of shape (out_features, in_features)
        group_size: Size of groups for per-group scaling. Default: 128
        symmetric: If True, use symmetric quantization. Default: True

    Returns:
        Tuple of (w_int4, scale) where:
            w_int4: INT4 weights (stored as int8), same shape as input
            scale: Scaling factors
    """
    raise NotImplementedError(
        "INT4 quantization is not yet implemented. "
        "Coming in a future version of BitTorch."
    )


def int4_quantize_ste(
    weight: Tensor,
    group_size: int = 128,
    symmetric: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Quantize weights to INT4 with STE for training.

    NOT YET IMPLEMENTED.
    """
    raise NotImplementedError(
        "INT4 quantization with STE is not yet implemented. "
        "Coming in a future version of BitTorch."
    )


def dequantize_int4(w_int4: Tensor, scale: Tensor) -> Tensor:
    """Dequantize INT4 weights back to full precision.

    NOT YET IMPLEMENTED.
    """
    raise NotImplementedError(
        "INT4 dequantization is not yet implemented. "
        "Coming in a future version of BitTorch."
    )

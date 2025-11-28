"""Conversion utilities for transforming models to ternary inference.

This module provides functions to convert:
- nn.Linear -> TernaryLinearInference
- Full models with recursive layer replacement
- Checkpoint save/load for packed ternary models
"""

from typing import Callable, Dict, Optional, Any

import torch
import torch.nn as nn
from torch import Tensor

from ..nn import TernaryLinear, TernaryLinearInference
from ..quant import ternary_quantize


def fp_linear_to_ternary_infer(
    linear: nn.Linear,
    threshold_factor: float = 0.05,
) -> TernaryLinearInference:
    """Convert nn.Linear to TernaryLinearInference.

    Takes a full-precision linear layer, quantizes its weights to ternary,
    packs them, and returns an inference-only module.

    Args:
        linear: Full-precision linear layer
        threshold_factor: Threshold factor for ternary quantization

    Returns:
        TernaryLinearInference with packed weights

    Example:
        >>> linear = nn.Linear(64, 32)
        >>> infer = fp_linear_to_ternary_infer(linear)
        >>> infer.in_features
        64
    """
    # Get weight and optional bias
    weight = linear.weight.data
    bias = linear.bias.data if linear.bias is not None else None

    # Quantize to ternary
    w_tern, scale = ternary_quantize(weight, threshold_factor=threshold_factor)

    # Create inference module
    return TernaryLinearInference.from_unpacked(w_tern, scale, bias)


def ternary_linear_to_infer(
    ternary_linear: TernaryLinear,
) -> TernaryLinearInference:
    """Convert TernaryLinear (training) to TernaryLinearInference (deployment).

    Takes a trained TernaryLinear module and creates an inference module
    with the same quantized weights but packed storage.

    Args:
        ternary_linear: Trained TernaryLinear module

    Returns:
        TernaryLinearInference with packed weights
    """
    # Get quantized weights from training module
    w_tern, scale = ternary_linear.get_quantized_weight()

    # Get bias if present
    bias = ternary_linear.bias.data if ternary_linear.bias is not None else None

    # Create inference module
    return TernaryLinearInference.from_unpacked(w_tern, scale, bias)


def convert_linear_layers(
    module: nn.Module,
    threshold_factor: float = 0.05,
    skip_names: Optional[list] = None,
    inplace: bool = False,
) -> nn.Module:
    """Recursively convert nn.Linear layers to TernaryLinearInference.

    Walks the module tree and replaces nn.Linear layers with
    TernaryLinearInference. Optionally skip certain layers by name.

    Args:
        module: PyTorch module to convert
        threshold_factor: Threshold for ternary quantization
        skip_names: List of layer name patterns to skip (e.g., ["head", "lm_head"])
        inplace: If True, modify module in place. If False, return a copy.

    Returns:
        Module with converted layers

    Example:
        >>> model = nn.Sequential(
        ...     nn.Linear(64, 32),
        ...     nn.ReLU(),
        ...     nn.Linear(32, 10)
        ... )
        >>> model_infer = convert_linear_layers(model, skip_names=["2"])
        >>> type(model_infer[0])
        <class 'bittorch.nn.ternary_linear_infer.TernaryLinearInference'>
    """
    if skip_names is None:
        skip_names = []

    if not inplace:
        import copy
        module = copy.deepcopy(module)

    def should_skip(name: str) -> bool:
        return any(pattern in name for pattern in skip_names)

    def convert_recursive(parent: nn.Module, prefix: str = ""):
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear) and not should_skip(full_name):
                # Convert to inference module
                infer_module = fp_linear_to_ternary_infer(child, threshold_factor)
                setattr(parent, name, infer_module)
            elif isinstance(child, TernaryLinear) and not should_skip(full_name):
                # Convert training module to inference
                infer_module = ternary_linear_to_infer(child)
                setattr(parent, name, infer_module)
            else:
                # Recurse into children
                convert_recursive(child, full_name)

    convert_recursive(module)
    return module


def count_ternary_params(module: nn.Module) -> Dict[str, int]:
    """Count parameters by type in a module.

    Args:
        module: Module to analyze

    Returns:
        Dict with counts: {"fp32": n, "fp16": n, "packed_ternary": n, "total": n}
    """
    counts = {
        "fp32": 0,
        "fp16": 0,
        "packed_ternary": 0,
        "total": 0,
    }

    for name, child in module.named_modules():
        if isinstance(child, TernaryLinearInference):
            # Packed weights: each byte holds 4 ternary values
            num_weights = child.out_features * child.in_features
            counts["packed_ternary"] += num_weights
            counts["total"] += num_weights
        elif isinstance(child, (nn.Linear, TernaryLinear)):
            num_weights = child.weight.numel()
            if child.weight.dtype == torch.float16:
                counts["fp16"] += num_weights
            else:
                counts["fp32"] += num_weights
            counts["total"] += num_weights

    return counts


def get_model_memory_bytes(module: nn.Module) -> Dict[str, int]:
    """Calculate memory usage by type.

    Args:
        module: Module to analyze

    Returns:
        Dict with byte counts: {"fp32": n, "fp16": n, "packed_ternary": n, "total": n}
    """
    bytes_count = {
        "fp32": 0,
        "fp16": 0,
        "packed_ternary": 0,
        "scales": 0,
        "total": 0,
    }

    for name, child in module.named_modules():
        if isinstance(child, TernaryLinearInference):
            # Packed weights + scales
            packed_bytes = child.weight_packed.numel()
            scale_bytes = child.scale.numel() * child.scale.element_size()
            bytes_count["packed_ternary"] += packed_bytes
            bytes_count["scales"] += scale_bytes
            bytes_count["total"] += packed_bytes + scale_bytes
            if child.bias is not None:
                bias_bytes = child.bias.numel() * child.bias.element_size()
                bytes_count["total"] += bias_bytes
        elif isinstance(child, (nn.Linear, TernaryLinear)):
            weight_bytes = child.weight.numel() * child.weight.element_size()
            if child.weight.dtype == torch.float16:
                bytes_count["fp16"] += weight_bytes
            else:
                bytes_count["fp32"] += weight_bytes
            bytes_count["total"] += weight_bytes
            if child.bias is not None:
                bytes_count["total"] += child.bias.numel() * child.bias.element_size()

    return bytes_count


def save_ternary_checkpoint(
    module: nn.Module,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a model with TernaryLinearInference layers.

    Saves packed weights, scales, and module structure for later loading.

    Args:
        module: Module to save (should contain TernaryLinearInference layers)
        path: Path to save checkpoint
        metadata: Optional metadata dict to include
    """
    checkpoint = {
        "state_dict": module.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)


def load_ternary_checkpoint(
    path: str,
    module: nn.Module,
) -> Dict[str, Any]:
    """Load a ternary checkpoint into a module.

    Args:
        path: Path to checkpoint
        module: Module with matching architecture

    Returns:
        Metadata dict from checkpoint
    """
    checkpoint = torch.load(path, map_location="cpu")
    module.load_state_dict(checkpoint["state_dict"])
    return checkpoint.get("metadata", {})
